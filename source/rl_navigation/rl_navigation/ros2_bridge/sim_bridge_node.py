# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""ROS2 bridge node that connects a single-env NavigationEnv to ROS2 topics."""

from __future__ import annotations

import math

import numpy as np
import torch

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy

from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import OccupancyGrid as OccupancyGridMsg
from sensor_msgs.msg import Image, LaserScan
from std_msgs.msg import Header

# Create 3 kinematic parameters
WHEEL_BASE = 0.233  # metres between wheel centres
WHEEL_RADIUS = 0.036  # metres

SENSOR_QOS = QoSProfile(
    reliability=QoSReliabilityPolicy.BEST_EFFORT,
    history=QoSHistoryPolicy.KEEP_LAST,
    depth=1,
)

TRANSIENT_LOCAL_QOS = QoSProfile(
    reliability=QoSReliabilityPolicy.RELIABLE,
    durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
    history=QoSHistoryPolicy.KEEP_LAST,
    depth=1,
)


class SimBridgeNode(Node):
    """Bridges a single-env ``NavigationEnv`` instance to ROS2 topics.

    **Publishers** (sensor QoS, best-effort, depth 1):
        - ``/scan`` — ``sensor_msgs/LaserScan``
        - ``/camera/rgb/image_raw`` — ``sensor_msgs/Image`` (encoding ``rgb8``)
        - ``/camera/depth/image_raw`` — ``sensor_msgs/Image`` (encoding ``32FC1``)

    **Publisher** (transient-local QoS, published once):
        - ``/occupancy_grid`` — ``nav_msgs/OccupancyGrid``

    **Subscribers** (sensor QoS):
        - ``/cmd_vel`` — ``geometry_msgs/Twist``
        - ``/goal_pose`` — ``geometry_msgs/PoseStamped``

    Args:
        env: The unwrapped ``NavigationEnv`` instance (``num_envs=1``).
        max_wheel_vel: Maximum wheel angular velocity in rad/s (from env config).
    """

    def __init__(self, env, max_wheel_vel: float = 6.28) -> None:
        super().__init__("sim_bridge")
        self._env = env
        self._max_wheel_vel = max_wheel_vel

        # --- Publishers ---
        self._scan_pub = self.create_publisher(LaserScan, "/scan", SENSOR_QOS)
        self._rgb_pub = self.create_publisher(Image, "/camera/rgb/image_raw", SENSOR_QOS)
        self._depth_pub = self.create_publisher(Image, "/camera/depth/image_raw", SENSOR_QOS)
        self._grid_pub = self.create_publisher(OccupancyGridMsg, "/occupancy_grid", TRANSIENT_LOCAL_QOS)

        # --- Subscribers ---
        self.create_subscription(Twist, "/cmd_vel", self._cmd_vel_cb, SENSOR_QOS)
        self.create_subscription(PoseStamped, "/goal_pose", self._goal_pose_cb, SENSOR_QOS)

        # --- State ---
        self._cmd_vel_override: torch.Tensor | None = None
        self._external_goal: tuple[float, float] | None = None

        # Publish static occupancy grid once
        self._publish_occupancy_grid()

        self.get_logger().info("SimBridgeNode initialised — publishing sensor data to ROS2.")

    # ------------------------------------------------------------------
    # Subscriber callbacks
    # ------------------------------------------------------------------

    def _cmd_vel_cb(self, msg: Twist) -> None:
        """Convert ``(linear.x, angular.z)`` to differential-drive wheel velocities."""
        v = msg.linear.x
        omega = msg.angular.z

        v_left = (v - omega * WHEEL_BASE / 2.0) / WHEEL_RADIUS
        v_right = (v + omega * WHEEL_BASE / 2.0) / WHEEL_RADIUS

        # Normalise to [-1, 1] by dividing by max_wheel_vel
        action = torch.tensor(
            [[v_left / self._max_wheel_vel, v_right / self._max_wheel_vel]],
            dtype=torch.float32,
            device=self._env.device,
        ).clamp(-1.0, 1.0)
        self._cmd_vel_override = action

    def _goal_pose_cb(self, msg: PoseStamped) -> None:
        """Store ``(x, y)`` from an incoming ``/goal_pose``."""
        self._external_goal = (msg.pose.position.x, msg.pose.position.y)
        self.get_logger().info(f"Received goal: ({msg.pose.position.x:.2f}, {msg.pose.position.y:.2f})")

    # ------------------------------------------------------------------
    # Interface methods called from the main simulation loop
    # ------------------------------------------------------------------

    def publish_sensor_data(self, sim_time_s: float) -> None:
        """Publish all sensor data for env index 0."""
        stamp = self._time_to_stamp(sim_time_s)
        self._publish_scan(stamp)
        if self._env.camera is not None:
            self._publish_rgb(stamp)
            self._publish_depth(stamp)

    def get_action_override(self) -> torch.Tensor | None:
        """Return the latest ``/cmd_vel`` action tensor, or ``None`` if none received."""
        return self._cmd_vel_override

    def apply_external_goal(self) -> None:
        """If a ``/goal_pose`` was received, write it to the environment."""
        if self._external_goal is not None:
            gx, gy = self._external_goal
            goal_tensor = torch.tensor([[gx, gy]], dtype=torch.float32, device=self._env.device)
            self._env.goal_pos = goal_tensor
            self._external_goal = None

    # ------------------------------------------------------------------
    # Publisher helpers
    # ------------------------------------------------------------------

    def _publish_scan(self, stamp: rclpy.time.Time) -> None:
        ray_hits = self._env.lidar.data.ray_hits_w[0]  # (num_rays, 3)
        lidar_pos = self._env.lidar.data.pos_w[0]  # (3,)
        ranges_tensor = torch.norm(ray_hits - lidar_pos, dim=-1)  # (num_rays,)
        ranges_np = ranges_tensor.cpu().numpy().astype(np.float32)

        msg = LaserScan()
        msg.header = Header(stamp=stamp, frame_id="base_link")
        msg.angle_min = 0.0
        msg.angle_max = 2.0 * math.pi
        msg.angle_increment = 2.0 * math.pi / len(ranges_np)
        msg.range_min = 0.0
        msg.range_max = 12.0
        msg.ranges = ranges_np.tolist()
        self._scan_pub.publish(msg)

    def _publish_rgb(self, stamp: rclpy.time.Time) -> None:
        rgb = self._env._last_rgb[0]  # (H, W, 3) uint8
        data = rgb.cpu().numpy().tobytes()

        msg = Image()
        msg.header = Header(stamp=stamp, frame_id="camera_link")
        msg.height = rgb.shape[0]
        msg.width = rgb.shape[1]
        msg.encoding = "rgb8"
        msg.is_bigendian = False
        msg.step = rgb.shape[1] * 3
        msg.data = list(data)
        self._rgb_pub.publish(msg)

    def _publish_depth(self, stamp: rclpy.time.Time) -> None:
        depth = self._env._last_depth[0]  # (H, W, 1) float32
        depth_2d = depth.squeeze(-1)  # (H, W)
        data = depth_2d.cpu().numpy().tobytes()

        msg = Image()
        msg.header = Header(stamp=stamp, frame_id="camera_link")
        msg.height = depth_2d.shape[0]
        msg.width = depth_2d.shape[1]
        msg.encoding = "32FC1"
        msg.is_bigendian = False
        msg.step = depth_2d.shape[1] * 4
        msg.data = list(data)
        self._depth_pub.publish(msg)

    def _publish_occupancy_grid(self) -> None:
        """Publish the static occupancy grid once with transient-local QoS."""
        grid = self._env.occ_grid
        grid_np = grid.grid.cpu().numpy()  # (H, W) binary: 0=free, 1=obstacle

        msg = OccupancyGridMsg()
        msg.header = Header(frame_id="map")
        msg.info.resolution = grid.resolution
        msg.info.width = grid_np.shape[1]
        msg.info.height = grid_np.shape[0]
        msg.info.origin.position.x = float(grid.origin[0])
        msg.info.origin.position.y = float(grid.origin[1])
        msg.info.origin.position.z = 0.0
        msg.info.origin.orientation.w = 1.0

        # Convert: 0 (free) -> 0, 1 (obstacle) -> 100
        occ_data = (grid_np.flatten() * 100).astype(np.int8)
        msg.data = occ_data.tolist()

        self._grid_pub.publish(msg)
        self.get_logger().info(
            f"Published occupancy grid ({grid_np.shape[1]}x{grid_np.shape[0]}, "
            f"res={grid.resolution:.3f}m)"
        )

    @staticmethod
    def _time_to_stamp(sim_time_s: float):
        """Convert a float timestamp to a ROS2 ``Time`` message."""
        from builtin_interfaces.msg import Time

        sec = int(sim_time_s)
        nanosec = int((sim_time_s - sec) * 1e9)
        return Time(sec=sec, nanosec=nanosec)
