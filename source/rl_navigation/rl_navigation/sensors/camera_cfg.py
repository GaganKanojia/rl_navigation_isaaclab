# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Front-facing RGB-D camera configuration for the Create 3."""

import isaaclab.sim as sim_utils
from isaaclab.sensors import CameraCfg

CAMERA_CFG = CameraCfg(
    prim_path="/World/envs/env_.*/Robot/base_link/front_camera",
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
