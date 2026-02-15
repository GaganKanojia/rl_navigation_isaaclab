# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""ROS2 bridge module — gracefully degrades when rclpy is not available."""

try:
    from .sim_bridge_node import SimBridgeNode  # noqa: F401
except ImportError:
    import warnings

    warnings.warn(
        "ROS2 packages (rclpy) not found. The ROS2 bridge is unavailable. "
        "Source your ROS2 workspace (e.g. `source /opt/ros/jazzy/setup.bash`) to enable it.",
        ImportWarning,
        stacklevel=2,
    )
