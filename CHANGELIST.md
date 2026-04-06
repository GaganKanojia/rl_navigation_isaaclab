# Changelist

## [Unreleased] — 2026-04-06

### Changed (RTX Lidar — Single Lidar Source)

- **Replaced RayCaster with RTX Lidar** as the single lidar source throughout the codebase.
  - Isaac Lab `RayCaster` only ray-casts against explicitly specified mesh prims, causing it to miss walls when floor and walls are separate meshes in the room USD.
  - RTX Lidar (`isaacsim.sensors.rtx`) ray-traces against *all* scene geometry via the GPU RTX pipeline, correctly detecting walls, obstacles, and the floor.
  - The RTX Lidar creates a `sensor_msgs/LaserScan` on `/scan` fully automatically via the `ROS2RtxLidarHelper` OmniGraph node — no Python-side data extraction needed.
- **Deleted `sensors/lidar_cfg.py`**: `RayCasterCfg` configuration file removed; no longer needed.
- **Removed `lidar` from RL observation space**: The RL policy no longer receives raw lidar ranges. Instead it uses the SLAM occupancy map (published by Nav2 to `/map`) as the spatial input. This keeps training coherent with real-device deployment where the same Nav2 SLAM pipeline runs.
- **Collision detection uses occupancy grid**: `_get_rewards()` and `_get_dones()` now detect collisions by checking whether the robot's grid cell is occupied (`OccupancyGrid.is_occupied()`), replacing the lidar min-range proximity check.
- **Removed unused `collision_threshold` config field** from `NavigationEnvCfg` (was used for lidar proximity; no longer applicable).

### Added

- **`sensors/rtx_lidar_cfg.py`**: Central configuration for the RTX Lidar sensor.
  - `RTX_LIDAR_PRIM_PATH` — absolute USD prim path for env_0.
  - `RTX_LIDAR_CONFIG` — Isaac Sim sensor preset (`"Example_Rotary_2D"`, single-channel 360° rotating lidar).
  - `RTX_LIDAR_HEIGHT_OFFSET` — mounting height above `base_link` (0.12 m).
  - `RTX_LIDAR_FRAME_ID`, `RTX_LIDAR_TF_TRANSLATION`, `RTX_LIDAR_TF_ROTATION` — ROS2 TF constants.
  - Imported by `SimBridgeNode`; no lidar values are hardcoded in the bridge.
- **Camera TF constants in `sensors/camera_cfg.py`**: `CAMERA_FRAME_ID`, `CAMERA_TF_TRANSLATION`, `CAMERA_TF_ROTATION_IJKR` added and imported by `SimBridgeNode`, replacing previously hardcoded TF values.
- **`OccupancyGrid.is_occupied()`**: New method (inverse of existing `is_free()`) for binary collision detection from world positions.

### Fixed

- **`_reset_idx()` crash on full reset**: `self._robot._ALL_INDICES` does not exist on Isaac Lab's `Articulation` class. Replaced with `torch.arange(self.num_envs, device=self.device)`.
- **`SimBridgeNode.destroy_node()` `AttributeError`**: `self._render_product` was only assigned inside `_setup_rtx_lidar()`. If setup failed partway through, `destroy_node()` would raise `AttributeError`. Fixed by initializing `self._render_product = None` in `__init__`.
- **Stale `NavigationEnv` docstring**: Class docstring still referenced "lidar scans" as an observation after the RayCaster removal. Updated to describe the SLAM-map-based observation pipeline.
- **Robot floating above ground**: Robot USD was converted from URDF with `fix_base=True`, creating a fixed world joint. Added `fix_root_link=False` to `ArticulationRootPropertiesCfg` in `CREATE3_CFG`.
- **Missing collision geometry on robot**: Added `collision_props=CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0)` and `activate_contact_sensors=True` to `CREATE3_CFG` so the physics engine generates collision shapes from the USD mesh geometry.

## [Unreleased] — 2026-03-18

### Changed (OmniGraph ROS2 Bridge Rewrite)

- **Replaced rclpy with OmniGraph**: `SimBridgeNode` no longer imports `rclpy`, `tf2_ros`, or ROS2 message types directly. Instead it creates an OmniGraph action graph using Isaac Sim's built-in `isaacsim.ros2.bridge` extension. This resolves the Python 3.11 (Isaac Sim) vs 3.12 (ROS2 Jazzy) version mismatch.
  - Odometry computed automatically via `IsaacComputeOdometry` OmniGraph node from the robot USD prim.
  - TF (odom→base_link) driven by OmniGraph data connections from the odometry node.
  - Static TF (base_link→laser_frame, base_link→camera_link) via `ROS2PublishRawTransformTree` with `staticPublisher=True`.
  - LaserScan depth data set from Python each step (RayCaster sensor data → `ROS2PublishLaserScan` node).
  - `/cmd_vel` read from `ROS2SubscribeTwist` OmniGraph node outputs.
  - Simulation clock published via `ROS2PublishClock` node.
- **Removed rclpy dependency from scripts**: `ros2_sim.py` and `ros2_nav2_sim.py` no longer call `rclpy.init()`, `rclpy.spin_once()`, or `rclpy.shutdown()`.
- **Simplified `ros2_bridge/__init__.py`**: No conditional rclpy import guard needed; direct import of `SimBridgeNode`.
- **Environment reset before bridge creation**: Scripts now call `env.reset()` before creating the bridge, ensuring the USD scene exists for OmniGraph prim references.

### Added

- **Nav2 Integration**: Full integration with the Nav2 navigation stack for autonomous navigation and exploration in Isaac Sim.
  - **TF Broadcasting**: `SimBridgeNode` now publishes dynamic `odom → base_link` transforms and static `base_link → laser_frame` / `base_link → camera_link` transforms via `tf2_ros`.
  - **Odometry Publishing**: New `/odom` topic (`nav_msgs/Odometry`) with robot pose and body-frame velocity from Isaac Sim ground truth.
  - **Wall-clock Timestamps**: All published messages use monotonic wall-clock timestamps (instead of simulation time) to ensure compatibility with Nav2 and SLAM Toolbox.
  - **Cmd_vel Safety Timeout**: `/cmd_vel` commands expire after 0.5 seconds of inactivity, zeroing velocity to prevent uncontrolled drift.
  - **Nav2 Configuration Files**: Pre-tuned configs for Create 3 kinematics in `config/nav2/`:
    - `nav2_params.yaml` — DWB controller, NavFn planner, costmaps, recovery behaviors.
    - `slam_toolbox_params.yaml` — Online async SLAM with 360° lidar support.
    - `exploration_params.yaml` — Frontier-based exploration (m-explore-next).
  - **Nav2 Simulation Script** (`scripts/ros2_nav2_sim.py`): Dedicated script with `--mode exploration` and `--mode navigation` flags, prints parallel terminal commands for the user.

### Changed

- **LaserScan frame_id**: Changed from `"base_link"` to `"laser_frame"` to match the static TF transform and Nav2 conventions.
- **`config/ros2_bridge.yaml`**: Added `odom` topic, `odom` and `laser_frame` frame IDs, and `cmd_vel_timeout` parameter.

### Documentation

- Renamed `CLAUDE.md` to `README.md`.
- Added Nav2 integration sections: prerequisites, frontier exploration workflow, goal navigation workflow, TF tree diagram, data flow diagram, and configuration reference.
