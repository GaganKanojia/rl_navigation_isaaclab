# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""OmniGraph-based ROS2 bridge node for NavigationEnv.

Uses Isaac Sim's built-in ROS2 bridge extension (isaacsim.ros2.bridge) via
OmniGraph instead of direct rclpy usage.  This avoids Python version conflicts
between Isaac Sim (Python 3.11) and system ROS2 Jazzy (Python 3.12).

The ``/scan`` topic is published by an **RTX Lidar** sensor which ray-traces
against *all* scene geometry (walls, obstacles, furniture, etc.).  This is the
single lidar source — no RayCaster is used anywhere in this codebase.
"""

from __future__ import annotations

import torch

import omni.graph.core as og
import omni.kit.commands
import omni.replicator.core as rep
import usdrt.Sdf
from pxr import Gf

from rl_navigation.sensors.camera_cfg import (
    CAMERA_FRAME_ID,
    CAMERA_TF_ROTATION_IJKR,
    CAMERA_TF_TRANSLATION,
)
from rl_navigation.sensors.rtx_lidar_cfg import (
    RTX_LIDAR_CONFIG,
    RTX_LIDAR_CONFIG_DIR,
    RTX_LIDAR_FRAME_ID,
    RTX_LIDAR_HEIGHT_OFFSET,
    RTX_LIDAR_PRIM_PATH,
    RTX_LIDAR_TF_ROTATION,
    RTX_LIDAR_TF_TRANSLATION,
)

# Create 3 kinematic parameters
WHEEL_BASE = 0.233  # metres between wheel centres
WHEEL_RADIUS = 0.036  # metres

# OmniGraph path for the ROS2 bridge graph
GRAPH_PATH = "/ROS2Bridge"


class SimBridgeNode:
    """OmniGraph-based bridge between a single-env ``NavigationEnv`` and ROS2.

    Uses Isaac Sim's built-in OmniGraph ROS2 nodes to publish sensor data and
    subscribe to commands, completely avoiding direct rclpy imports.

    **Fully-automatic publishers** (driven by OmniGraph / RTX pipeline each tick):
        - ``/scan``  — ``sensor_msgs/LaserScan``  (from RTX Lidar via ROS2RtxLidarHelper)
        - ``/odom``  — ``nav_msgs/Odometry``  (from ``IsaacComputeOdometry``)
        - ``/tf``    — dynamic ``odom -> base_link``
        - ``/tf_static`` — ``base_link -> laser_frame``, ``base_link -> camera_link``
        - ``/clock`` — ``rosgraph_msgs/Clock``

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

        # Build the RTX Lidar sensor and OmniGraph
        self._render_product = None
        self._render_product_path: str | None = None
        self._setup_rtx_lidar()
        self._setup_graph()

        print("[OmniGraph ROS2 Bridge] Initialised — RTX Lidar publishes /scan, OmniGraph publishes odom/TF/clock.")

    # ------------------------------------------------------------------
    # Graph setup
    # ------------------------------------------------------------------

    def _setup_rtx_lidar(self) -> None:
        """Create an RTX Lidar sensor prim and its render product.

        The RTX Lidar ray-traces against all scene geometry (walls, floor, obstacles).
        The sensor is created at the robot's base_link at z=0.12 m above the base.
        """
        from isaacsim.core.utils.extensions import enable_extension

        enable_extension("isaacsim.sensors.rtx")

        # Register this package's lidar_configs folder so the custom
        # "Create3_Planar_Lidar" profile is resolvable by name.  Isaac Sim looks up
        # configs in the folders listed in app.sensors.nv.lidar.profileBaseFolder, so
        # we append ours rather than overwrite the built-in search paths.
        import carb

        settings = carb.settings.get_settings()
        profile_setting = "/app/sensors/nv/lidar/profileBaseFolder"
        profile_folders = list(settings.get(profile_setting) or [])
        if RTX_LIDAR_CONFIG_DIR not in profile_folders:
            profile_folders.append(RTX_LIDAR_CONFIG_DIR)
            settings.set_string_array(profile_setting, profile_folders)

        # Create RTX Lidar prim using Isaac Sim command
        _, self._lidar_prim = omni.kit.commands.execute(
            "IsaacSensorCreateRtxLidar",
            path=RTX_LIDAR_PRIM_PATH,
            parent=None,
            config=RTX_LIDAR_CONFIG,
            translation=Gf.Vec3d(0.0, 0.0, RTX_LIDAR_HEIGHT_OFFSET),
            orientation=Gf.Quatd(1, 0, 0, 0),  # identity
        )

        # Create a render product so the RTX pipeline processes this sensor
        self._render_product = rep.create.render_product(
            RTX_LIDAR_PRIM_PATH, resolution=(1, 1)
        )
        self._render_product_path = self._render_product.path

        # RTX temporal interpolation emits two warnings per step in step-by-step mode
        # (no adjacent time samples exist). Functionally benign — timestamps use current
        # sim time — but the flood obscures real errors, so suppress to ERROR level.
        import omni.log as omni_log
        omni_log.get_log().set_channel_level(
            "isaacsim.core.simulation_manager.plugin",
            omni_log.Level.ERROR,
            omni_log.SettingBehavior.OVERRIDE,
        )

        print(f"[OmniGraph ROS2 Bridge] RTX Lidar created at {RTX_LIDAR_PRIM_PATH}")
        print(f"[OmniGraph ROS2 Bridge] Render product at {self._render_product_path}")

    def _setup_graph(self) -> None:
        """Create the OmniGraph action graph with all ROS2 nodes.

        The ``/scan`` topic is published automatically by the ``ROS2RtxLidarHelper``
        node which reads from the RTX Lidar render product — no Python-side data
        extraction is needed.
        """
        from isaacsim.core.utils.extensions import enable_extension

        enable_extension("isaacsim.ros2.bridge")

        keys = og.Controller.Keys

        # Resolve robot prim path for env 0
        robot_prim_path = self._env._robot.root_physx_view.prim_paths[0]

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
                    # --- RTX Lidar -> LaserScan (fully automatic) ---
                    ("RtxLidarHelper", "isaacsim.ros2.bridge.ROS2RtxLidarHelper"),
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
                    # Static TF: base_link -> laser_frame
                    ("PublishTFLidar.inputs:topicName", "tf_static"),
                    ("PublishTFLidar.inputs:parentFrameId", "base_link"),
                    ("PublishTFLidar.inputs:childFrameId", RTX_LIDAR_FRAME_ID),
                    ("PublishTFLidar.inputs:translation", RTX_LIDAR_TF_TRANSLATION),
                    ("PublishTFLidar.inputs:rotation", RTX_LIDAR_TF_ROTATION),
                    ("PublishTFLidar.inputs:staticPublisher", True),
                    # Static TF: base_link -> camera_link
                    ("PublishTFCamera.inputs:topicName", "tf_static"),
                    ("PublishTFCamera.inputs:parentFrameId", "base_link"),
                    ("PublishTFCamera.inputs:childFrameId", CAMERA_FRAME_ID),
                    ("PublishTFCamera.inputs:translation", CAMERA_TF_TRANSLATION),
                    ("PublishTFCamera.inputs:rotation", CAMERA_TF_ROTATION_IJKR),
                    ("PublishTFCamera.inputs:staticPublisher", True),
                    # RTX Lidar Helper — publishes /scan as LaserScan automatically
                    ("RtxLidarHelper.inputs:renderProductPath", self._render_product_path),
                    ("RtxLidarHelper.inputs:topicName", "scan"),
                    ("RtxLidarHelper.inputs:frameId", RTX_LIDAR_FRAME_ID),
                    ("RtxLidarHelper.inputs:type", "laser_scan"),
                    ("RtxLidarHelper.inputs:queueSize", 1),
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
                    # Tick -> RTX Lidar helper (publishes /scan)
                    ("OnPlaybackTick.outputs:tick", "RtxLidarHelper.inputs:execIn"),
                    # Tick -> publish clock
                    ("OnPlaybackTick.outputs:tick", "PublishClock.inputs:execIn"),
                    # Tick -> subscribe twist
                    ("OnPlaybackTick.outputs:tick", "SubscribeTwist.inputs:execIn"),
                    # --- Context propagation ---
                    ("Context.outputs:context", "PublishOdom.inputs:context"),
                    ("Context.outputs:context", "PublishTF.inputs:context"),
                    ("Context.outputs:context", "PublishTFLidar.inputs:context"),
                    ("Context.outputs:context", "PublishTFCamera.inputs:context"),
                    ("Context.outputs:context", "RtxLidarHelper.inputs:context"),
                    ("Context.outputs:context", "PublishClock.inputs:context"),
                    ("Context.outputs:context", "SubscribeTwist.inputs:context"),
                    # --- Timestamp propagation ---
                    ("ReadSimTime.outputs:simulationTime", "PublishOdom.inputs:timeStamp"),
                    ("ReadSimTime.outputs:simulationTime", "PublishTF.inputs:timeStamp"),
                    ("ReadSimTime.outputs:simulationTime", "PublishTFLidar.inputs:timeStamp"),
                    ("ReadSimTime.outputs:simulationTime", "PublishTFCamera.inputs:timeStamp"),
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
        """No-op — all sensor publishing is now fully automatic via OmniGraph.

        The RTX Lidar sensor publishes ``/scan`` through the ``ROS2RtxLidarHelper``
        node.  Odometry, TF, and clock are driven by other OmniGraph connections.

        Args:
            sim_time_s: Unused (kept for API compatibility).
        """
        pass

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
        """Remove the OmniGraph and RTX Lidar from the USD stage."""
        try:
            # Destroy render product first
            if self._render_product is not None:
                self._render_product.destroy()
                self._render_product = None

            import omni.usd

            stage = omni.usd.get_context().get_stage()
            if stage:
                # Remove OmniGraph
                if stage.GetPrimAtPath(GRAPH_PATH):
                    stage.RemovePrim(GRAPH_PATH)
                # Remove RTX Lidar prim
                if stage.GetPrimAtPath(RTX_LIDAR_PRIM_PATH):
                    stage.RemovePrim(RTX_LIDAR_PRIM_PATH)
                print("[OmniGraph ROS2 Bridge] Graph and RTX Lidar removed.")
        except Exception:
            pass
