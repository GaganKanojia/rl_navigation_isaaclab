# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the Create 3 navigation environment."""

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import CameraCfg, RayCasterCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass

from rl_navigation.robots.create3 import CREATE3_CFG
from rl_navigation.sensors.camera_cfg import CAMERA_CFG
from rl_navigation.sensors.lidar_cfg import LIDAR_CFG


@configclass
class NavigationEnvCfg(DirectRLEnvCfg):
    # --- Environment timing ---
    decimation = 2
    episode_length_s = 60.0

    # --- Spaces (dict observation for MultiInputPolicy) ---
    observation_space = {
        "lidar": 360,
        "occupancy_grid": [1, 50, 50],
        "goal_pose": 3,
        "velocity": 3,
    }
    action_space = 2
    state_space = 0

    # --- Simulation ---
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 120,
        render_interval=2,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=0.8,
            dynamic_friction=0.6,
        ),
    )

    # --- Terrain / room ---
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="usd",
        usd_path="PLACEHOLDER_ROOM_USD_PATH",
    )

    # --- Scene ---
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=4096,
        env_spacing=15.0,
        replicate_physics=True,
    )

    # --- Robot ---
    robot_cfg: ArticulationCfg = CREATE3_CFG.replace(prim_path="/World/envs/env_.*/Robot")

    # --- Sensors ---
    lidar_cfg: RayCasterCfg = LIDAR_CFG
    camera_cfg: CameraCfg | None = CAMERA_CFG
    enable_camera: bool = True

    # --- Wheel joint names (must match USD) ---
    left_wheel_joint: str = "left_wheel_joint"
    right_wheel_joint: str = "right_wheel_joint"

    # --- Action / dynamics ---
    max_wheel_vel: float = 6.28  # rad/s (~0.5 m/s for 80mm radius wheel)

    # --- Occupancy grid ---
    rooms_txt_path: str = "config/rooms.txt"
    room_index: int = 0
    grid_resolution: float = 0.1
    local_patch_size: int = 50

    # --- Goal ---
    goal_radius: float = 0.3
    min_goal_distance: float = 1.0
    max_goal_distance: float = 8.0

    # --- Reward scales ---
    rew_scale_goal_reached: float = 10.0
    rew_scale_distance_progress: float = 1.0
    rew_scale_collision: float = -5.0
    rew_scale_time_penalty: float = -0.01
    rew_scale_smoothness: float = -0.1
    collision_threshold: float = 0.25
