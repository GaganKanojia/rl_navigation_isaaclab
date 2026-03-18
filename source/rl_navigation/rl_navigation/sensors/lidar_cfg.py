# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""2D planar lidar configuration using Isaac Lab RayCaster."""

from isaaclab.sensors import RayCasterCfg
from isaaclab.sensors.ray_caster import patterns

LIDAR_CFG = RayCasterCfg(
    prim_path="/World/envs/env_.*/Robot/create_3/base_link",
    offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.12)),
    ray_alignment="yaw",
    pattern_cfg=patterns.LidarPatternCfg(
        channels=1,
        vertical_fov_range=(0.0, 0.0),
        horizontal_fov_range=(0.0, 360.0),
        horizontal_res=1.0,
    ),
    max_distance=12.0,
    mesh_prim_paths=["/World/ground"],
    debug_vis=False,
)
