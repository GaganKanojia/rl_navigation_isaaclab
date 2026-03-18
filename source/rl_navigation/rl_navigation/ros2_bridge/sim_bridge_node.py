# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""OmniGraph-based ROS2 bridge node for NavigationEnv.

Uses Isaac Sim's built-in ROS2 bridge extension (isaacsim.ros2.bridge) via
OmniGraph instead of direct rclpy usage.  This avoids Python version conflicts
between Isaac Sim (Python 3.11) and system ROS2 Jazzy (Python 3.12).
"""

from __future__ import annotations

import numpy as np
import torch

import omni.graph.core as og
import usdrt.Sdf

# Create 3 kinematic parameters
WHEEL_BASE = 0.233  # metres between wheel centres
WHEEL_RADIUS = 0.036  # metres

# OmniGraph path for the ROS2 bridge graph
GRAPH_PATH = "/ROS2Bridge"


class SimBridgeNode:
    """OmniGraph-based bridge between a single-env ``NavigationEnv`` and ROS2.

    Uses Isaac Sim's built-in OmniGraph ROS2 nodes to publish sensor data and
    subscribe to commands, completely avoiding direct rclpy imports.

    **Automatic publishers** (driven by OmniGraph on each simulation tick):
        - ``/odom``  — ``nav_msgs/Odometry``  (from ``IsaacComputeOdometry``)
        - ``/tf``    — dynamic ``odom → base_link``
        - ``/tf_static`` — ``base_link → laser_frame``, ``base_link → camera_link``
        - ``/clock`` — ``rosgraph_msgs/Clock``

    **Python-driven publishers** (data set from env each step):
        - ``/scan``  — ``sensor_msgs/LaserScan``  (from RayCaster)

    **Subscribers** (read from OmniGraph outputs):
        - ``/cmd_vel`` — ``geometry_msgs/Twist``

    Args:
        env: The unwrapped ``NavigationEnv`` instance (``num_envs=1``).
        max_wheel_vel: Maximum wheel angular velocity in rad/s.
    """

    def __init__(self, env, max_wheel_vel: float = 6.28) -> None:
        self._env = env
        self._max_wheel_vel = max_wheel_vel

        # OmniGraph subscriber holds the last received value persistently — we
        # cannot detect individual message arrivals.  We track whether a non-zero
        # command has ever been seen to distinguish "no publisher" from "commanded stop".
        self._ever_received: bool = False

        # Build the OmniGraph
        self._setup_graph()

        print("[OmniGraph ROS2 Bridge] Initialised — publishing sensor data, TF, and odometry.")

    # ------------------------------------------------------------------
    # Graph setup
    # ------------------------------------------------------------------

    def _setup_graph(self) -> None:
        """Create the OmniGraph action graph with all ROS2 nodes."""
        # Ensure required extensions are loaded
        from isaacsim.core.utils.extensions import enable_extension

        enable_extension("isaacsim.ros2.bridge")

        keys = og.Controller.Keys

        # Resolve robot prim path for env 0
        robot_prim_path = self._env._robot.root_physx_view.prim_paths[0]

        # Get actual ray count from the lidar sensor (may differ from config due to endpoint handling)
        num_rays = self._env.lidar.data.ray_hits_w.shape[1]

        (self._graph, self._nodes, _, _) = og.Controller.edit(
            {"graph_path": GRAPH_PATH, "evaluator_name": "execution"},
            {
                keys.CREATE_NODES: [
                    # --- Infrastructure ---
                    ("OnPlaybackTick", "omni.graph.action.OnPlaybackTick"),
                    ("Context", "isaacsim.ros2.bridge.ROS2Context"),
                    ("ReadSimTime", "isaacsim.core.nodes.IsaacReadSimulationTime"),
                    # --- Odometry (automatic from USD) ---
                    ("ComputeOdom", "isaacsim.core.nodes.IsaacComputeOdometry"),
                    ("PublishOdom", "isaacsim.ros2.bridge.ROS2PublishOdometry"),
                    # --- TF dynamic: odom -> base_link ---
                    ("PublishTF", "isaacsim.ros2.bridge.ROS2PublishRawTransformTree"),
                    # --- TF static: base_link -> laser_frame ---
                    ("PublishTFLidar", "isaacsim.ros2.bridge.ROS2PublishRawTransformTree"),
                    # --- TF static: base_link -> camera_link ---
                    ("PublishTFCamera", "isaacsim.ros2.bridge.ROS2PublishRawTransformTree"),
                    # --- LaserScan (data set from Python) ---
                    ("PublishScan", "isaacsim.ros2.bridge.ROS2PublishLaserScan"),
                    # --- Clock ---
                    ("PublishClock", "isaacsim.ros2.bridge.ROS2PublishClock"),
                    # --- Cmd_vel subscriber ---
                    ("SubscribeTwist", "isaacsim.ros2.bridge.ROS2SubscribeTwist"),
                ],
                keys.SET_VALUES: [
                    # Context
                    ("Context.inputs:useDomainIDEnvVar", True),
                    # Sim time
                    ("ReadSimTime.inputs:resetOnStop", False),
                    # Compute odometry from robot chassis
                    ("ComputeOdom.inputs:chassisPrim", [usdrt.Sdf.Path(robot_prim_path)]),
                    # Odometry publisher
                    ("PublishOdom.inputs:topicName", "odom"),
                    ("PublishOdom.inputs:chassisFrameId", "base_link"),
                    ("PublishOdom.inputs:odomFrameId", "odom"),
                    ("PublishOdom.inputs:publishRawVelocities", False),
                    ("PublishOdom.inputs:robotFront", [1.0, 0.0, 0.0]),
                    ("PublishOdom.inputs:queueSize", 1),
                    # Dynamic TF: odom -> base_link
                    ("PublishTF.inputs:topicName", "tf"),
                    ("PublishTF.inputs:parentFrameId", "odom"),
                    ("PublishTF.inputs:childFrameId", "base_link"),
                    # Static TF: base_link -> laser_frame (z=0.12m)
                    ("PublishTFLidar.inputs:topicName", "tf_static"),
                    ("PublishTFLidar.inputs:parentFrameId", "base_link"),
                    ("PublishTFLidar.inputs:childFrameId", "laser_frame"),
                    ("PublishTFLidar.inputs:translation", [0.0, 0.0, 0.12]),
                    ("PublishTFLidar.inputs:rotation", [0.0, 0.0, 0.0, 1.0]),  # IJKR identity
                    ("PublishTFLidar.inputs:staticPublisher", True),
                    # Static TF: base_link -> camera_link
                    ("PublishTFCamera.inputs:topicName", "tf_static"),
                    ("PublishTFCamera.inputs:parentFrameId", "base_link"),
                    ("PublishTFCamera.inputs:childFrameId", "camera_link"),
                    ("PublishTFCamera.inputs:translation", [0.12, 0.0, 0.10]),
                    # Isaac camera rotation (w,x,y,z)=(0.5,-0.5,0.5,-0.5) -> IJKR (x,y,z,w)=(-0.5,0.5,-0.5,0.5)
                    ("PublishTFCamera.inputs:rotation", [-0.5, 0.5, -0.5, 0.5]),
                    ("PublishTFCamera.inputs:staticPublisher", True),
                    # LaserScan publisher
                    ("PublishScan.inputs:topicName", "scan"),
                    ("PublishScan.inputs:frameId", "laser_frame"),
                    ("PublishScan.inputs:horizontalFov", 360.0),
                    ("PublishScan.inputs:horizontalResolution", 360.0 / num_rays),
                    ("PublishScan.inputs:depthRange", [0.0, 12.0]),
                    ("PublishScan.inputs:azimuthRange", [0.0, 360.0]),
                    ("PublishScan.inputs:numRows", 1),
                    ("PublishScan.inputs:numCols", num_rays),
                    ("PublishScan.inputs:rotationRate", 0.0),
                    ("PublishScan.inputs:queueSize", 1),
                    # Clock publisher
                    ("PublishClock.inputs:topicName", "clock"),
                    # Twist subscriber
                    ("SubscribeTwist.inputs:topicName", "cmd_vel"),
                    ("SubscribeTwist.inputs:queueSize", 1),
                ],
                keys.CONNECT: [
                    # --- Execution flow ---
                    # Tick -> compute odometry
                    ("OnPlaybackTick.outputs:tick", "ComputeOdom.inputs:execIn"),
                    # Odometry -> publish odom
                    ("ComputeOdom.outputs:execOut", "PublishOdom.inputs:execIn"),
                    # Tick -> publish TF
                    ("OnPlaybackTick.outputs:tick", "PublishTF.inputs:execIn"),
                    # Tick -> static TFs (they only publish once due to staticPublisher=True)
                    ("OnPlaybackTick.outputs:tick", "PublishTFLidar.inputs:execIn"),
                    ("OnPlaybackTick.outputs:tick", "PublishTFCamera.inputs:execIn"),
                    # Tick -> publish scan
                    ("OnPlaybackTick.outputs:tick", "PublishScan.inputs:execIn"),
                    # Tick -> publish clock
                    ("OnPlaybackTick.outputs:tick", "PublishClock.inputs:execIn"),
                    # Tick -> subscribe twist
                    ("OnPlaybackTick.outputs:tick", "SubscribeTwist.inputs:execIn"),
                    # --- Context propagation ---
                    ("Context.outputs:context", "PublishOdom.inputs:context"),
                    ("Context.outputs:context", "PublishTF.inputs:context"),
                    ("Context.outputs:context", "PublishTFLidar.inputs:context"),
                    ("Context.outputs:context", "PublishTFCamera.inputs:context"),
                    ("Context.outputs:context", "PublishScan.inputs:context"),
                    ("Context.outputs:context", "PublishClock.inputs:context"),
                    ("Context.outputs:context", "SubscribeTwist.inputs:context"),
                    # --- Timestamp propagation ---
                    ("ReadSimTime.outputs:simulationTime", "PublishOdom.inputs:timeStamp"),
                    ("ReadSimTime.outputs:simulationTime", "PublishTF.inputs:timeStamp"),
                    ("ReadSimTime.outputs:simulationTime", "PublishTFLidar.inputs:timeStamp"),
                    ("ReadSimTime.outputs:simulationTime", "PublishTFCamera.inputs:timeStamp"),
                    ("ReadSimTime.outputs:simulationTime", "PublishScan.inputs:timeStamp"),
                    ("ReadSimTime.outputs:simulationTime", "PublishClock.inputs:timeStamp"),
                    # --- Odometry data connections ---
                    ("ComputeOdom.outputs:position", "PublishOdom.inputs:position"),
                    ("ComputeOdom.outputs:orientation", "PublishOdom.inputs:orientation"),
                    ("ComputeOdom.outputs:linearVelocity", "PublishOdom.inputs:linearVelocity"),
                    ("ComputeOdom.outputs:angularVelocity", "PublishOdom.inputs:angularVelocity"),
                    # --- TF data from odometry ---
                    ("ComputeOdom.outputs:position", "PublishTF.inputs:translation"),
                    ("ComputeOdom.outputs:orientation", "PublishTF.inputs:rotation"),
                ],
            },
        )

        print(f"[OmniGraph ROS2 Bridge] Created graph at {GRAPH_PATH} with {len(self._nodes)} nodes")

    # ------------------------------------------------------------------
    # Interface methods called from the main simulation loop
    # ------------------------------------------------------------------

    def publish_sensor_data(self, sim_time_s: float) -> None:
        """Update OmniGraph inputs that can't be driven automatically.

        Currently this sets LaserScan depth data from the RayCaster sensor.
        Odometry, TF, and clock are driven automatically via OmniGraph connections.

        Args:
            sim_time_s: Simulation time (unused; timestamps come from ReadSimTime node).
        """
        self._update_scan_data()

    def get_action_override(self) -> torch.Tensor | None:
        """Read ``/cmd_vel`` from the OmniGraph subscriber and convert to wheel velocities.

        OmniGraph subscribers hold the last received value persistently — we cannot
        detect individual message arrivals.  Once any non-zero command is seen, we
        treat subsequent values (including zero) as intentional commands.

        Returns:
            Action tensor ``(1, 2)`` with normalised ``[left, right]`` wheel velocities,
            or ``None`` if no command has ever been received.
        """
        try:
            linear_vel = og.Controller.attribute(
                GRAPH_PATH + "/SubscribeTwist.outputs:linearVelocity"
            ).get()
            angular_vel = og.Controller.attribute(
                GRAPH_PATH + "/SubscribeTwist.outputs:angularVelocity"
            ).get()
        except Exception:
            return None

        if linear_vel is None or angular_vel is None:
            return None

        v = float(linear_vel[0])  # linear.x
        omega = float(angular_vel[2])  # angular.z

        # Mark as received once we see any non-zero value
        if abs(v) > 1e-6 or abs(omega) > 1e-6:
            self._ever_received = True

        # If we've never received a real command, return None
        if not self._ever_received:
            return None

        # Differential drive: convert (v, omega) to wheel velocities
        v_left = (v - omega * WHEEL_BASE / 2.0) / WHEEL_RADIUS
        v_right = (v + omega * WHEEL_BASE / 2.0) / WHEEL_RADIUS

        # Normalise to [-1, 1]
        action = torch.tensor(
            [[v_left / self._max_wheel_vel, v_right / self._max_wheel_vel]],
            dtype=torch.float32,
            device=self._env.device,
        ).clamp(-1.0, 1.0)

        return action

    def apply_external_goal(self) -> None:
        """No-op in OmniGraph mode.

        Nav2 handles goal management internally and drives the robot via /cmd_vel.
        """
        pass

    def destroy_node(self) -> None:
        """Remove the OmniGraph from the USD stage."""
        try:
            import omni.usd

            stage = omni.usd.get_context().get_stage()
            if stage and stage.GetPrimAtPath(GRAPH_PATH):
                stage.RemovePrim(GRAPH_PATH)
                print("[OmniGraph ROS2 Bridge] Graph removed.")
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _update_scan_data(self) -> None:
        """Set LaserScan depth and intensity data on the OmniGraph node from RayCaster sensor."""
        ray_hits = self._env.lidar.data.ray_hits_w[0]  # (num_rays, 3)
        lidar_pos = self._env.lidar.data.pos_w[0]  # (3,)
        ranges = torch.norm(ray_hits - lidar_pos, dim=-1)  # (num_rays,)
        ranges_np = ranges.cpu().numpy().astype(np.float32)
        num_rays = len(ranges_np)

        try:
            og.Controller.attribute(
                GRAPH_PATH + "/PublishScan.inputs:linearDepthData"
            ).set(ranges_np.tolist())
            # Intensities must match depth data size — use uniform intensity
            og.Controller.attribute(
                GRAPH_PATH + "/PublishScan.inputs:intensitiesData"
            ).set([255] * num_rays)
        except Exception as e:
            print(f"[OmniGraph ROS2 Bridge] Warning: Failed to set scan data: {e}")
