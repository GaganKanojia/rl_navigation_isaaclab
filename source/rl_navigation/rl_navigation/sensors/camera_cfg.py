# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Front-facing RGB-D camera configuration for the Create 3."""

import isaaclab.sim as sim_utils
from isaaclab.sensors import CameraCfg

# ROS2 frame ID used in TF and published image messages
CAMERA_FRAME_ID: str = "camera_link"

# TF: translation of camera_link relative to base_link [x, y, z] (metres)
CAMERA_TF_TRANSLATION: list[float] = [0.12, 0.0, 0.10]

# TF: rotation of camera_link relative to base_link in IJKR (x, y, z, w) format.
# Isaac/ROS convention (w,x,y,z) = (0.5, -0.5, 0.5, -0.5) → IJKR (x,y,z,w) = (-0.5, 0.5, -0.5, 0.5)
CAMERA_TF_ROTATION_IJKR: list[float] = [-0.5, 0.5, -0.5, 0.5]

CAMERA_CFG = CameraCfg(
    prim_path="/World/envs/env_.*/Robot/create_3/base_link/front_camera",
    offset=CameraCfg.OffsetCfg(
        pos=(0.12, 0.0, 0.10),
        rot=(0.5, -0.5, 0.5, -0.5),
        convention="ros",
    ),
    data_types=["rgb", "distance_to_camera"],
    width=320,
    height=240,
    spawn=sim_utils.PinholeCameraCfg(
        focal_length=24.0,
        horizontal_aperture=20.955,
        clipping_range=(0.1, 10.0),
    ),
    update_period=0.0,
)
