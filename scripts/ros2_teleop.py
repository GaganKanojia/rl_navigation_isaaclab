#!/usr/bin/env python3
# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Keyboard teleop for the Create 3 — publishes ``/cmd_vel`` Twist messages.

Controls:
    W / Up      — forward
    S / Down    — backward
    A / Left    — turn left
    D / Right   — turn right
    Space       — stop
    Q           — quit

Alternative: ``ros2 run teleop_twist_keyboard teleop_twist_keyboard``
"""

from __future__ import annotations

import sys
import termios
import tty

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy

from geometry_msgs.msg import Twist

LINEAR_VEL = 0.3  # m/s
ANGULAR_VEL = 1.0  # rad/s

SENSOR_QOS = QoSProfile(
    reliability=QoSReliabilityPolicy.BEST_EFFORT,
    history=QoSHistoryPolicy.KEEP_LAST,
    depth=1,
)

KEY_BINDINGS = {
    "w": (LINEAR_VEL, 0.0),
    "\x1b[A": (LINEAR_VEL, 0.0),  # Up arrow
    "s": (-LINEAR_VEL, 0.0),
    "\x1b[B": (-LINEAR_VEL, 0.0),  # Down arrow
    "a": (0.0, ANGULAR_VEL),
    "\x1b[D": (0.0, ANGULAR_VEL),  # Left arrow
    "d": (0.0, -ANGULAR_VEL),
    "\x1b[C": (0.0, -ANGULAR_VEL),  # Right arrow
    " ": (0.0, 0.0),
}


def get_key() -> str:
    """Read a single keypress from stdin (non-blocking-ish)."""
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ch = sys.stdin.read(1)
        if ch == "\x1b":
            ch += sys.stdin.read(2)
        return ch
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)


def main():
    rclpy.init()
    node = Node("teleop_keyboard")
    pub = node.create_publisher(Twist, "/cmd_vel", SENSOR_QOS)

    print("Keyboard Teleop for Create 3")
    print("----------------------------")
    print("  W/Up    — forward")
    print("  S/Down  — backward")
    print("  A/Left  — turn left")
    print("  D/Right — turn right")
    print("  Space   — stop")
    print("  Q       — quit")
    print()

    try:
        while True:
            key = get_key()
            if key in ("q", "\x03"):  # q or Ctrl+C
                break

            if key in KEY_BINDINGS:
                linear, angular = KEY_BINDINGS[key]
            else:
                continue

            msg = Twist()
            msg.linear.x = linear
            msg.angular.z = angular
            pub.publish(msg)
    except KeyboardInterrupt:
        pass
    finally:
        # Send stop command
        msg = Twist()
        pub.publish(msg)
        node.destroy_node()
        rclpy.shutdown()
        print("\nTeleop stopped.")


if __name__ == "__main__":
    main()
