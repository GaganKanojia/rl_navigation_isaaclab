# RL Navigation — Isaac Lab Extension

Isaac Lab extension for RL navigation. Implements two environments using NVIDIA Isaac Sim and Isaac Lab:

1. **CartPole baseline** (`Template-Rl-Navigation-Direct-v0`) — simple 1-DOF cart-pole balancing task.
2. **Create 3 Navigation** (`Create3-Navigation-Direct-v0`) — goal-conditioned navigation for an iRobot Create 3 differential-drive robot with lidar, occupancy grid, and a custom CNN feature extractor.

The project is structured as an isolated extension outside the core Isaac Lab repository. Requires Python 3.10+ and Isaac Sim 4.5+.

## Commands

All commands should be run from `rl_navigation/` directory. Requires Isaac Lab conda environment to be active.

```bash
# Install the extension (editable mode)
python -m pip install -e source/rl_navigation

# List available environments
python scripts/list_envs.py

# Test with dummy agents (CartPole)
python scripts/zero_agent.py --task=Template-Rl-Navigation-Direct-v0
python scripts/random_agent.py --task=Template-Rl-Navigation-Direct-v0

# Test with dummy agents (Create 3 Navigation)
python scripts/zero_agent.py --task=Create3-Navigation-Direct-v0
python scripts/random_agent.py --task=Create3-Navigation-Direct-v0

# Train with Stable-Baselines3 PPO (CartPole)
python scripts/sb3/train.py --task=Template-Rl-Navigation-Direct-v0 --num_envs=4096 --max_iterations=100

# Train with Stable-Baselines3 PPO (Create 3 Navigation)
python scripts/sb3/train.py --task=Create3-Navigation-Direct-v0 --num_envs=4096 --max_iterations=100

# Evaluate trained agent
python scripts/sb3/play.py --task=Create3-Navigation-Direct-v0 --checkpoint=<path/to/model.zip>

# Precompute occupancy grid from room USD
python scripts/precompute_grid.py --usd_path=/path/to/room.usd --output_path=/path/to/room_grid.npy

# Code formatting (pre-commit hooks: black, flake8, isort, pyupgrade, codespell)
pre-commit run --all-files
```

## Code Style

- **black** formatter, line length 120, `--unstable` flag
- **isort** with black profile
- **flake8** with flake8-simplify and flake8-return plugins
- **pyupgrade** targeting Python 3.10+
- BSD-3-Clause license headers required on `.py` and `.yaml` files (auto-inserted by pre-commit via `.github/LICENSE_HEADER.txt`)
- **codespell** for spell checking; trailing whitespace and EOF fixer also enforced
- Pre-commit rejects files >2MB (use git-lfs for large assets)

## Architecture

Extension metadata (version, author, etc.) lives in `source/rl_navigation/config/extension.toml`, read by `setup.py`.

```
rl_navigation/
├── config/
│   ├── rooms.txt                          # Room USD + grid path pairs
│   ├── ros2_bridge.yaml                   # ROS2 topic names, kinematic params
│   └── nav2/                              # Nav2 stack configuration
│       ├── nav2_params.yaml               # Nav2 controller, planner, costmaps
│       ├── slam_toolbox_params.yaml       # SLAM Toolbox parameters
│       └── exploration_params.yaml        # Frontier exploration parameters
├── scripts/
│   ├── sb3/train.py                       # SB3 PPO training loop
│   ├── sb3/play.py                        # Inference/playback
│   ├── precompute_grid.py                 # Offline occupancy grid generation
│   ├── ros2_sim.py                        # Single-env sim + ROS2 bridge
│   ├── ros2_nav2_sim.py                   # Nav2 integration (exploration + navigation)
│   ├── ros2_teleop.py                     # Keyboard teleop (/cmd_vel)
│   ├── zero_agent.py                      # Zero-action baseline
│   └── random_agent.py                    # Random-action baseline
└── source/rl_navigation/
    └── rl_navigation/
        ├── robots/
        │   └── create3.py                 # CREATE3_CFG ArticulationCfg
        ├── sensors/
        │   ├── lidar_cfg.py               # LIDAR_CFG RayCasterCfg (360° planar)
        │   └── camera_cfg.py              # CAMERA_CFG CameraCfg (front RGB-D)
        ├── ros2_bridge/
        │   ├── __init__.py                # Conditional rclpy import
        │   └── sim_bridge_node.py         # SimBridgeNode ROS2 node
        ├── feature_extractor/
        │   └── nav_feature_extractor.py   # CNN feature extractor for SB3
        ├── utils/
        │   ├── occupancy_grid.py          # GPU-accelerated grid ops
        │   └── scene_loader.py            # Room/grid file loading
        └── tasks/direct/rl_navigation/
            ├── __init__.py                # Gymnasium env registration
            ├── rl_navigation_env.py       # CartPole env (unchanged)
            ├── rl_navigation_env_cfg.py   # CartPole config (unchanged)
            ├── navigation_env.py          # NavigationEnv (Create 3)
            ├── navigation_env_cfg.py      # NavigationEnvCfg
            └── agents/
                ├── sb3_ppo_cfg.yaml       # CartPole PPO config
                └── sb3_nav_ppo_cfg.yaml   # Navigation PPO config
```

**Key patterns:**
- Environments subclass `isaaclab.envs.DirectRLEnv` and implement: `_setup_scene`, `_pre_physics_step`, `_apply_action`, `_get_observations`, `_get_rewards`, `_get_dones`, `_reset_idx`
- Environment configs use `@configclass` decorator (Isaac Lab's dataclass variant) and subclass `DirectRLEnvCfg`
- Reward functions are extracted as standalone `@torch.jit.script` functions for performance
- Environments are registered via `gymnasium.register()` in `__init__.py`
- Simulation runs 4096 parallel environments by default on GPU with PyTorch/CUDA
- Physics at 120 Hz with decimation=2 (control at 60 Hz)

**Navigation environment specifics:**
- Dict observation space: lidar(360), occupancy_grid(1×50×50), goal_pose(3), velocity(3)
- Custom `NavigationFeaturesExtractor` with 2D CNN (grid) + 1D CNN (lidar) → 134-dim features
- SB3 `MultiInputPolicy` with custom feature extractor resolved from YAML string path
- Occupancy grids precomputed offline from room USD meshes via `scripts/precompute_grid.py`
- All parallel envs share the same room (required by `replicate_physics=True`); different rooms via `room_index` config
- User must provide Create 3 USD file and update joint names in `robots/create3.py`
- Front-facing RGB-D camera is opt-in via `enable_camera` config flag (disabled during large-scale training to save GPU memory)

## ROS2 Bridge (Inference / Visualization)

The ROS2 bridge runs a single-env simulation connected to ROS2 topics via Isaac Sim's built-in **OmniGraph ROS2 bridge** (`isaacsim.ros2.bridge` extension). It does **not** import `rclpy` directly, avoiding Python version conflicts between Isaac Sim (Python 3.11) and ROS2 Jazzy (Python 3.12). It is **not** part of the training pipeline.

### Prerequisites

- Isaac Sim 5.1+ with `isaacsim.ros2.bridge` extension enabled
- ROS2 Jazzy installed for Nav2/SLAM nodes (`source /opt/ros/jazzy/setup.bash`)
- Extension installed (`pip install -e source/rl_navigation`)

### Commands

```bash
# Manual control via /cmd_vel (no trained agent)
python scripts/ros2_sim.py --task=Create3-Navigation-Direct-v0 --manual --disable_fabric

# Trained agent controls the robot
python scripts/ros2_sim.py --task=Create3-Navigation-Direct-v0 --checkpoint=logs/sb3/Create3-Navigation-Direct-v0/model.zip --disable_fabric

# Agent controls, but /cmd_vel can override
python scripts/ros2_sim.py --checkpoint=<path> --allow-override --disable_fabric

# Keyboard teleop (run in a separate terminal)
python scripts/ros2_teleop.py
```

### ROS2 Topics

| Topic | Type | Direction | QoS |
|---|---|---|---|
| `/scan` | `sensor_msgs/LaserScan` | Published | Best-effort |
| `/odom` | `nav_msgs/Odometry` | Published | Best-effort |
| `/tf` | `tf2_msgs/TFMessage` | Published | Best-effort |
| `/tf_static` | `tf2_msgs/TFMessage` | Published (once) | Transient-local |
| `/camera/rgb/image_raw` | `sensor_msgs/Image` (rgb8) | Published | Best-effort |
| `/camera/depth/image_raw` | `sensor_msgs/Image` (32FC1) | Published | Best-effort |
| `/occupancy_grid` | `nav_msgs/OccupancyGrid` | Published once | Transient-local |
| `/cmd_vel` | `geometry_msgs/Twist` | Subscribed | Best-effort |
| `/goal_pose` | `geometry_msgs/PoseStamped` | Subscribed | Best-effort |

### TF Tree

The bridge publishes the following TF tree for Nav2 compatibility:

```
map → odom              (published by slam_toolbox)
  odom → base_link      (published by SimBridgeNode, dynamic)
    base_link → laser_frame   (published by SimBridgeNode, static)
    base_link → camera_link   (published by SimBridgeNode, static)
```

### Architecture

```
source/rl_navigation/rl_navigation/
├── ros2_bridge/
│   ├── __init__.py              # Direct import (no rclpy dependency)
│   └── sim_bridge_node.py       # SimBridgeNode (OmniGraph-based)
├── sensors/
│   ├── lidar_cfg.py             # LIDAR_CFG (360° planar)
│   └── camera_cfg.py            # CAMERA_CFG (front RGB-D, 320×240)
scripts/
├── ros2_sim.py                  # Single-env sim + ROS2 bridge
├── ros2_nav2_sim.py             # Nav2 integration bridge
└── ros2_teleop.py               # Keyboard teleop (/cmd_vel)
config/
├── ros2_bridge.yaml             # Topic names, kinematic params
└── nav2/                        # Nav2 stack configuration
    ├── nav2_params.yaml
    ├── slam_toolbox_params.yaml
    └── exploration_params.yaml
```

**Key design decisions:**
- ROS2 bridge is standalone — runs with `num_envs=1`, not part of training
- Camera excluded from training observation space (320×240 RGBD × 4096 envs = ~2.4GB GPU memory)
- Bridge uses OmniGraph (`isaacsim.ros2.bridge` extension) — no direct `rclpy` import, avoids Python version conflicts
- Odometry and TF driven automatically by `IsaacComputeOdometry` OmniGraph node from USD prim data
- LaserScan depth data fed from Isaac Lab's RayCaster sensor each step
- `/cmd_vel` persists last command with safety timeout (0.5s) — zeros velocity if no new command received
- TF, odometry, and clock published every tick by OmniGraph; only scan data requires Python update
- Create 3 kinematics: wheel_base=0.233m, wheel_radius=0.036m (configured in `config/ros2_bridge.yaml`)

## Nav2 Integration

The project integrates with the Nav2 stack for autonomous navigation and exploration. Nav2 runs as separate ROS2 processes and communicates with the Isaac Sim bridge via standard ROS2 topics.

### Prerequisites

Install the required ROS2 packages:

```bash
# Nav2 and SLAM
sudo apt install ros-jazzy-navigation2 ros-jazzy-nav2-bringup
sudo apt install ros-jazzy-slam-toolbox
sudo apt install ros-jazzy-tf2-ros

# Frontier explorer (build from source if not available as binary)
cd ~/ros2_ws/src
git clone https://github.com/robo-friends/m-explore-next.git
cd ~/ros2_ws
colcon build --packages-select explore_lite
source install/setup.bash
```

### Important: `--disable_fabric` Flag

The OmniGraph ROS2 nodes (`IsaacComputeOdometry`, `ROS2PublishRawTransformTree`, etc.) read robot pose data from the **USD stage**. By default, Isaac Sim uses **Fabric** — a high-performance GPU runtime layer that bypasses USD for faster physics/rendering. With Fabric enabled, prim transforms may not be written back to the USD stage, causing OmniGraph nodes to read stale or zero values for odometry and TF.

**Use `--disable_fabric` when running with the ROS2 bridge** to ensure OmniGraph nodes can read up-to-date prim data. This is not needed for RL training (which doesn't use OmniGraph).

### Frontier-Based Exploration

Autonomously explore and map the environment using frontier-based exploration. The robot uses SLAM to build a map from lidar data, and the frontier explorer selects unexplored regions as goals for Nav2.

Run each command in a separate terminal:

```bash
# Terminal 1 — Isaac Sim bridge (--disable_fabric required for OmniGraph ROS2 nodes)
python scripts/ros2_nav2_sim.py --mode exploration --disable_fabric

# Terminal 2 — SLAM Toolbox (builds map from lidar)
ros2 launch slam_toolbox online_async_launch.py \
  params_file:=$(pwd)/config/nav2/slam_toolbox_params.yaml

# Terminal 3 — Nav2 stack (path planning + obstacle avoidance)
ros2 launch nav2_bringup navigation_launch.py \
  params_file:=$(pwd)/config/nav2/nav2_params.yaml use_sim_time:=false

# Terminal 4 — Frontier explorer (selects exploration goals)
ros2 launch explore_lite explore.launch.py

# Terminal 5 (optional) — RViz2 visualization
ros2 launch nav2_bringup rviz_launch.py
```

**Data flow:**
```
Isaac Sim → /scan, /odom, /tf → SLAM Toolbox → /map
                                                 ↓
Frontier Explorer ← /map                    Nav2 ← /map, /scan, /tf
        ↓                                         ↓
    /goal_pose → Nav2 → /cmd_vel → Isaac Sim (robot moves)
```

### Goal-Directed Navigation

Navigate the robot to a specific goal location using Nav2 for path planning.

Run each command in a separate terminal:

```bash
# Terminal 1 — Isaac Sim bridge (--disable_fabric required for OmniGraph ROS2 nodes)
python scripts/ros2_nav2_sim.py --mode navigation --disable_fabric

# Terminal 2 — SLAM Toolbox
ros2 launch slam_toolbox online_async_launch.py \
  params_file:=$(pwd)/config/nav2/slam_toolbox_params.yaml

# Terminal 3 — Nav2 stack
ros2 launch nav2_bringup navigation_launch.py \
  params_file:=$(pwd)/config/nav2/nav2_params.yaml use_sim_time:=false

# Terminal 4 (optional) — RViz2 visualization
ros2 launch nav2_bringup rviz_launch.py
```

Send a goal via CLI:

```bash
ros2 topic pub --once /goal_pose geometry_msgs/PoseStamped \
  "{header: {frame_id: 'map'}, pose: {position: {x: 3.0, y: 2.0, z: 0.0}, orientation: {w: 1.0}}}"
```

Or use the RViz2 **"2D Goal Pose"** button to set goals interactively.

### Nav2 Configuration

All Nav2 configuration files are in `config/nav2/`:

| File | Description |
|---|---|
| `nav2_params.yaml` | DWB controller, NavFn planner, costmaps, recovery behaviors — tuned for Create 3 kinematics |
| `slam_toolbox_params.yaml` | Online async SLAM with 360° lidar, 12m range, loop closure enabled |
| `exploration_params.yaml` | Frontier explorer (m-explore-next) parameters |

Key parameters matching Create 3 kinematics:
- `max_vel_x: 0.5 m/s`, `max_vel_theta: 1.9 rad/s`
- `robot_radius: 0.17 m`
- Costmap resolution: 0.05 m/cell
- Inflation radius: 0.30 m

## Training Logs

Training outputs go to `logs/sb3/<task_name>/` including model checkpoints and optional video recordings.
