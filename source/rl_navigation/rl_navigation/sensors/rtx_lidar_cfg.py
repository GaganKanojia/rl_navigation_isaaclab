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

**Isaac Sim 5.0 profile model.**  Isaac Sim 5.0 replaced the old JSON lidar
profiles (resolved by name against ``app.sensors.nv.lidar.profileBaseFolder``)
with the ``OmniLidar`` prim, whose profile is authored directly as
``omni:sensor:Core:*`` USD attributes via the ``OmniSensorGenericLidarCoreAPI``
schema.  The Create 3 planar profile below is therefore expressed as attribute
values rather than a JSON file.
"""

# Prim path of the RTX Lidar sensor (absolute, for env_0 in Nav2 / single-env mode)
RTX_LIDAR_PRIM_PATH: str = "/World/envs/env_0/Robot/create_3/base_link/rtx_lidar"

# Core lidar profile, authored as OmniSensorGenericLidarCoreAPI USD attributes.
#
# Tuned for a small indoor robot (radius ~0.17 m).  The stock "Example_Rotary_2D"
# preset has nearRangeM=0.3 (a blind ring unusable up close) and a beam tilted
# 2 deg down into the floor; this profile drops nearRangeM to 0.05, keeps the beam
# horizontal (see RTX_LIDAR_EMITTER_STATE), and sets farRangeM=20 m — above the SLAM
# max_laser_range (15 m) and costmap raytrace_max_range (12 m), so those YAML values
# remain the effective limit.
RTX_LIDAR_CORE_PROFILE: dict = {
    # Scan geometry
    "omni:sensor:Core:scanType": "ROTARY",
    "omni:sensor:Core:rotationDirection": "CW",
    "omni:sensor:Core:rayType": "IDEALIZED",
    "omni:sensor:Core:startAzimuthOffsetDeg": 0.0,
    # Range
    "omni:sensor:Core:nearRangeM": 0.05,
    "omni:sensor:Core:farRangeM": 20.0,
    "omni:sensor:Core:rangeResolutionM": 0.004,
    "omni:sensor:Core:rangeAccuracyM": 0.02,
    # Single planar beam
    "omni:sensor:Core:numberOfEmitters": 1,
    "omni:sensor:Core:numberOfChannels": 1,
    "omni:sensor:Core:maxReturns": 1,
    # Rates
    "omni:sensor:Core:scanRateBaseHz": 30,
    "omni:sensor:Core:reportRateBaseHz": 32000,
    # Optics / returns
    "omni:sensor:Core:avgPowerW": 0.002,
    "omni:sensor:Core:minReflectance": 0.1,
    "omni:sensor:Core:minReflectionRangeM": 20.0,
    "omni:sensor:Core:waveLengthNm": 903.0,
    "omni:sensor:Core:pulseTimeNs": 6,
    # Noise
    "omni:sensor:Core:azimuthErrorMean": 0.0,
    "omni:sensor:Core:azimuthErrorStd": 0.015,
    "omni:sensor:Core:elevationErrorMean": 0.0,
    "omni:sensor:Core:elevationErrorStd": 0.0,
    # Intensity
    "omni:sensor:Core:intensityProcessing": "NORMALIZATION",
    "omni:sensor:Core:intensityMappingType": "LINEAR",
}

# Emitter-state instance applied by OmniSensorGenericLidarCoreAPI (schema default).
RTX_LIDAR_EMITTER_STATE_NAME: str = "s001"

# Single horizontal emitter on channel 1.  elevationDeg=0 keeps the planar scan
# horizontal; the Example_Rotary_2D preset used -2 deg, tilting the beam into the
# floor (phantom ring at ~3.4 m for a 0.12 m mount height).  These arrays have
# numberOfEmitters (= 1) elements each.
RTX_LIDAR_EMITTER_STATE: dict = {
    "azimuthDeg": [0.0],
    "elevationDeg": [0.0],
    "fireTimeNs": [0],
    "channelId": [1],
}

# Height of the lidar above the robot's base_link origin (metres)
RTX_LIDAR_HEIGHT_OFFSET: float = 0.12

# ROS2 frame ID used in the published LaserScan message
RTX_LIDAR_FRAME_ID: str = "laser_frame"

# TF: translation of laser_frame relative to base_link [x, y, z] (metres)
RTX_LIDAR_TF_TRANSLATION: list[float] = [0.0, 0.0, RTX_LIDAR_HEIGHT_OFFSET]

# TF: rotation of laser_frame relative to base_link in IJKR (x, y, z, w) — identity
RTX_LIDAR_TF_ROTATION: list[float] = [0.0, 0.0, 0.0, 1.0]
