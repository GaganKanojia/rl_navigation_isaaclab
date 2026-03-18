# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""ROS2 bridge module using Isaac Sim's built-in OmniGraph ROS2 bridge.

This module does NOT depend on rclpy directly.  It uses OmniGraph nodes from
the ``isaacsim.ros2.bridge`` extension, avoiding Python version conflicts.
"""

from .sim_bridge_node import SimBridgeNode  # noqa: F401
