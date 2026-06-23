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

import os

# Prim path of the RTX Lidar sensor (absolute, for env_0 in Nav2 / single-env mode)
RTX_LIDAR_PRIM_PATH: str = "/World/envs/env_0/Robot/create_3/base_link/rtx_lidar"

# Custom RTX Lidar profile shipped with this package.
#
# The stock Isaac Sim preset "Example_Rotary_2D" has nearRangeM=1.0 (a 1 m blind
# ring — unusable for a 0.17 m-radius indoor robot) and a beam tilted 2 deg down
# into the floor.  "Create3_Planar_Lidar" (see lidar_configs/) fixes both:
# nearRangeM=0.05, horizontal beam, farRangeM=20 m (above the SLAM max_laser_range
# of 15 m and costmap raytrace_max_range of 12 m, so those YAML values stay the
# effective limit).
#
# Isaac Sim resolves a config by name against the folders listed in the carb
# setting ``app.sensors.nv.lidar.profileBaseFolder``; ``SimBridgeNode`` appends
# RTX_LIDAR_CONFIG_DIR to that list before creating the sensor.
RTX_LIDAR_CONFIG_DIR: str = os.path.join(os.path.dirname(__file__), "lidar_configs")
RTX_LIDAR_CONFIG: str = "Create3_Planar_Lidar"

# Height of the lidar above the robot's base_link origin (metres)
RTX_LIDAR_HEIGHT_OFFSET: float = 0.12

# ROS2 frame ID used in the published LaserScan message
RTX_LIDAR_FRAME_ID: str = "laser_frame"

# TF: translation of laser_frame relative to base_link [x, y, z] (metres)
RTX_LIDAR_TF_TRANSLATION: list[float] = [0.0, 0.0, RTX_LIDAR_HEIGHT_OFFSET]

# TF: rotation of laser_frame relative to base_link in IJKR (x, y, z, w) — identity
RTX_LIDAR_TF_ROTATION: list[float] = [0.0, 0.0, 0.0, 1.0]
