# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""RTX Lidar sensor configuration for the Create 3.

The RTX Lidar ray-traces against *all* scene geometry (walls, floor, obstacles)
using Isaac Sim's GPU-accelerated RTX pipeline.  It is the single lidar source
in this codebase — no Isaac Lab RayCaster is used.

The sensor prim is created at runtime by ``SimBridgeNode._setup_rtx_lidar()``
using the constants defined here.
"""

# Prim path of the RTX Lidar sensor (absolute, for env_0 in Nav2 / single-env mode)
RTX_LIDAR_PRIM_PATH: str = "/World/envs/env_0/Robot/create_3/base_link/rtx_lidar"

# Isaac Sim sensor preset — "Example_Rotary_2D" is a single-channel 360° rotating lidar
RTX_LIDAR_CONFIG: str = "Example_Rotary_2D"

# Height of the lidar above the robot's base_link origin (metres)
RTX_LIDAR_HEIGHT_OFFSET: float = 0.12

# ROS2 frame ID used in the published LaserScan message
RTX_LIDAR_FRAME_ID: str = "laser_frame"

# TF: translation of laser_frame relative to base_link [x, y, z] (metres)
RTX_LIDAR_TF_TRANSLATION: list[float] = [0.0, 0.0, RTX_LIDAR_HEIGHT_OFFSET]

# TF: rotation of laser_frame relative to base_link in IJKR (x, y, z, w) — identity
RTX_LIDAR_TF_ROTATION: list[float] = [0.0, 0.0, 0.0, 1.0]
