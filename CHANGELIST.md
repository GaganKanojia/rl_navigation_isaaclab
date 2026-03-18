# Changelist

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
