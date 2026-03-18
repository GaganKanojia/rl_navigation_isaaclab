# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Launch a single-env navigation simulation with Nav2 integration.

Uses Isaac Sim's built-in OmniGraph ROS2 bridge — no direct rclpy dependency.
The robot is driven by Nav2 via ``/cmd_vel``. This script publishes sensor data
(``/scan``, ``/odom``, TF) that Nav2 and SLAM Toolbox consume for mapping,
planning, and control.

Modes:
    --mode exploration   Frontier-based exploration — use with SLAM + Nav2 + explore_lite.
    --mode navigation    Goal-directed navigation — use with SLAM + Nav2, send goals via /goal_pose.
"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Nav2 integration bridge for Create 3 navigation in Isaac Sim.")
parser.add_argument("--task", type=str, default="Create3-Navigation-Direct-v0", help="Gym task ID.")
parser.add_argument(
    "--mode",
    type=str,
    choices=["exploration", "navigation"],
    default="navigation",
    help="Operating mode: 'exploration' for frontier-based mapping, 'navigation' for goal-directed.",
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

NAV2_INSTRUCTIONS = {
    "exploration": """
================================================================================
  Nav2 Frontier Exploration Mode
================================================================================
  Run these commands in separate terminals:

  Terminal 2 — SLAM Toolbox:
    ros2 launch slam_toolbox online_async_launch.py \\
      params_file:=$(pwd)/config/nav2/slam_toolbox_params.yaml

  Terminal 3 — Nav2 Stack:
    ros2 launch nav2_bringup navigation_launch.py \\
      params_file:=$(pwd)/config/nav2/nav2_params.yaml use_sim_time:=false

  Terminal 4 — Frontier Explorer:
    ros2 launch explore_lite explore.launch.py

  Terminal 5 (optional) — RViz2 Visualization:
    ros2 launch nav2_bringup rviz_launch.py
================================================================================
""",
    "navigation": """
================================================================================
  Nav2 Goal Navigation Mode
================================================================================
  Run these commands in separate terminals:

  Terminal 2 — SLAM Toolbox:
    ros2 launch slam_toolbox online_async_launch.py \\
      params_file:=$(pwd)/config/nav2/slam_toolbox_params.yaml

  Terminal 3 — Nav2 Stack:
    ros2 launch nav2_bringup navigation_launch.py \\
      params_file:=$(pwd)/config/nav2/nav2_params.yaml use_sim_time:=false

  Terminal 4 (optional) — RViz2 Visualization:
    ros2 launch nav2_bringup rviz_launch.py

  Send a goal via CLI:
    ros2 topic pub --once /goal_pose geometry_msgs/PoseStamped \\
      "{header: {frame_id: 'map'}, pose: {position: {x: 3.0, y: 2.0, z: 0.0}, orientation: {w: 1.0}}}"

  Or use the RViz2 "2D Goal Pose" button to set goals interactively.
================================================================================
""",
}


def main():
    """Run single-env simulation with Nav2 integration."""
    # Parse environment config
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1, use_fabric=not args_cli.disable_fabric)
    env_cfg.enable_camera = True
    # Nav2 mode: SLAM builds the map, no precomputed grid needed
    env_cfg.require_occupancy_grid = False
    # Nav2 handles obstacle avoidance — don't terminate on collision
    env_cfg.enable_collision_termination = False
    # Set very long episode to prevent auto-resets during Nav2 operation
    env_cfg.episode_length_s = 99999.0

    # Create environment
    env = gym.make(args_cli.task, cfg=env_cfg)
    unwrapped = env.unwrapped

    # Reset environment first (scene must exist before creating OmniGraph)
    obs, _ = env.reset()
    sim_dt = env_cfg.sim.dt * env_cfg.decimation

    # Create OmniGraph ROS2 bridge (no rclpy.init() needed)
    bridge = SimBridgeNode(unwrapped, max_wheel_vel=env_cfg.max_wheel_vel)

    print(NAV2_INSTRUCTIONS[args_cli.mode])
    print("[INFO]: OmniGraph ROS2 bridge started. Publishing:")
    print("  /scan (LaserScan), /odom (Odometry), /tf, /tf_static, /clock")
    print("  Subscribing: /cmd_vel")
    print(f"\n[INFO]: Mode: {args_cli.mode}")
    print("[INFO]: Robot is controlled by Nav2 via /cmd_vel. Waiting for Nav2 stack...\n")

    try:
        while simulation_app.is_running():
            step_start = time.perf_counter()

            with torch.inference_mode():
                # Read /cmd_vel from OmniGraph subscriber
                action_override = bridge.get_action_override()
                if action_override is not None:
                    actions = action_override
                else:
                    actions = torch.zeros(1, 2, device=unwrapped.device)

                # Step the environment (triggers OnPlaybackTick -> OmniGraph publishes odom, TF, clock)
                obs, _, terminated, truncated, _ = env.step(actions)

                # Update sensor data that OmniGraph can't read automatically (LaserScan from RayCaster)
                bridge.publish_sensor_data(0.0)

                # Handle episode reset (collision or timeout)
                if terminated.any() or truncated.any():
                    print("[INFO]: Episode ended (collision or timeout). Resetting...")
                    obs, _ = env.reset()

            # Real-time pacing
            elapsed = time.perf_counter() - step_start
            sleep_time = sim_dt - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("\n[INFO]: Shutting down...")
    finally:
        bridge.destroy_node()
        env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
