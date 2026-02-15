# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Launch a single-env navigation simulation with a ROS2 bridge.

Operating modes:
    --manual              Zero actions unless /cmd_vel override (no trained agent).
    --checkpoint=<path>   Trained PPO agent controls the robot.
    --checkpoint + --allow-override   Agent controls, but /cmd_vel can override.
"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="ROS2 bridge for Create 3 navigation.")
parser.add_argument("--task", type=str, default="Create3-Navigation-Direct-v0", help="Gym task ID.")
parser.add_argument("--checkpoint", type=str, default=None, help="Path to a trained SB3 PPO model .zip.")
parser.add_argument("--manual", action="store_true", default=False, help="Manual control via /cmd_vel only.")
parser.add_argument(
    "--allow-override", action="store_true", default=False, help="Allow /cmd_vel to override agent actions."
)
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Force single env and enable cameras
args_cli.num_envs = 1
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import time

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

import rl_navigation.tasks  # noqa: F401
from rl_navigation.ros2_bridge import SimBridgeNode

import rclpy


def main():
    """Run single-env simulation with ROS2 bridge."""
    # Parse environment config
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1, use_fabric=not args_cli.disable_fabric)
    env_cfg.enable_camera = True

    # Create environment
    env = gym.make(args_cli.task, cfg=env_cfg)
    unwrapped = env.unwrapped

    # Optionally load trained agent
    agent = None
    if args_cli.checkpoint is not None:
        from stable_baselines3 import PPO

        from isaaclab_tasks.utils.wrappers.sb3 import Sb3VecEnvWrapper

        sb3_env = Sb3VecEnvWrapper(env)
        agent = PPO.load(args_cli.checkpoint, env=sb3_env, device=unwrapped.device)
        print(f"[INFO]: Loaded checkpoint: {args_cli.checkpoint}")

    # Initialise ROS2
    rclpy.init()
    bridge = SimBridgeNode(unwrapped, max_wheel_vel=env_cfg.max_wheel_vel)

    print("[INFO]: ROS2 bridge started. Topics:")
    print("  Publishers:  /scan, /camera/rgb/image_raw, /camera/depth/image_raw, /occupancy_grid")
    print("  Subscribers: /cmd_vel, /goal_pose")

    # Reset environment
    obs, _ = env.reset()
    sim_dt = env_cfg.sim.dt * env_cfg.decimation

    try:
        while simulation_app.is_running():
            step_start = time.perf_counter()

            with torch.inference_mode():
                # Process ROS2 callbacks (non-blocking)
                rclpy.spin_once(bridge, timeout_sec=0.0)

                # Apply external goal if received
                bridge.apply_external_goal()

                # Determine action
                action_override = bridge.get_action_override()

                if args_cli.manual:
                    # Manual mode: use /cmd_vel or zero
                    if action_override is not None:
                        actions = action_override
                    else:
                        actions = torch.zeros(1, 2, device=unwrapped.device)
                elif agent is not None:
                    if args_cli.allow_override and action_override is not None:
                        # Override mode: /cmd_vel takes precedence
                        actions = action_override
                    else:
                        # Agent mode
                        actions, _ = agent.predict(obs, deterministic=True)
                else:
                    # No agent, no manual flag — just zero actions
                    actions = torch.zeros(1, 2, device=unwrapped.device)

                # Step the environment
                obs, _, _, _, _ = env.step(actions)

                # Publish sensor data
                sim_time = unwrapped.episode_length_buf[0].item() * sim_dt
                bridge.publish_sensor_data(sim_time)

            # Real-time pacing
            elapsed = time.perf_counter() - step_start
            sleep_time = sim_dt - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("\n[INFO]: Shutting down...")
    finally:
        bridge.destroy_node()
        rclpy.shutdown()
        env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
