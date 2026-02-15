# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Isaac Lab extension for RL navigation. Implements two environments using NVIDIA Isaac Sim and Isaac Lab:

1. **CartPole baseline** (`Template-Rl-Navigation-Direct-v0`) — simple 1-DOF cart-pole balancing task.
2. **Create 3 Navigation** (`Create3-Navigation-Direct-v0`) — goal-conditioned navigation for an iRobot Create 3 differential-drive robot with lidar, occupancy grid, and a custom CNN feature extractor.

The project is structured as an isolated extension outside the core Isaac Lab repository. Requires Python 3.10+, Isaac Sim 4.5+, and an active Isaac Lab conda environment.

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
│   └── ros2_bridge.yaml                   # ROS2 topic names, kinematic params
├── scripts/
│   ├── sb3/train.py                       # SB3 PPO training loop
│   ├── sb3/play.py                        # Inference/playback
│   ├── precompute_grid.py                 # Offline occupancy grid generation
│   ├── ros2_sim.py                        # Single-env sim + ROS2 bridge
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

The ROS2 bridge runs a single-env simulation connected to ROS2 topics. It is **not** part of the training pipeline.

### Prerequisites

- ROS2 Jazzy installed (`source /opt/ros/jazzy/setup.bash`)
- Isaac Lab conda environment active
- Extension installed (`pip install -e source/rl_navigation`)

### Commands

```bash
# Manual control via /cmd_vel (no trained agent)
python scripts/ros2_sim.py --task=Create3-Navigation-Direct-v0 --manual

# Trained agent controls the robot
python scripts/ros2_sim.py --task=Create3-Navigation-Direct-v0 --checkpoint=logs/sb3/Create3-Navigation-Direct-v0/model.zip

# Agent controls, but /cmd_vel can override
python scripts/ros2_sim.py --checkpoint=<path> --allow-override

# Keyboard teleop (run in a separate terminal)
python scripts/ros2_teleop.py
```

### ROS2 Topics

| Topic | Type | Direction | QoS |
|---|---|---|---|
| `/scan` | `sensor_msgs/LaserScan` | Published | Best-effort |
| `/camera/rgb/image_raw` | `sensor_msgs/Image` (rgb8) | Published | Best-effort |
| `/camera/depth/image_raw` | `sensor_msgs/Image` (32FC1) | Published | Best-effort |
| `/occupancy_grid` | `nav_msgs/OccupancyGrid` | Published once | Transient-local |
| `/cmd_vel` | `geometry_msgs/Twist` | Subscribed | Best-effort |
| `/goal_pose` | `geometry_msgs/PoseStamped` | Subscribed | Best-effort |

### Architecture

```
source/rl_navigation/rl_navigation/
├── ros2_bridge/
│   ├── __init__.py              # Conditional import (graceful if no rclpy)
│   └── sim_bridge_node.py       # SimBridgeNode ROS2 node
├── sensors/
│   ├── lidar_cfg.py             # LIDAR_CFG (360° planar)
│   └── camera_cfg.py            # CAMERA_CFG (front RGB-D, 320×240)
scripts/
├── ros2_sim.py                  # Single-env sim + ROS2 bridge
└── ros2_teleop.py               # Keyboard teleop (/cmd_vel)
config/
└── ros2_bridge.yaml             # Topic names, kinematic params
```

**Key design decisions:**
- ROS2 bridge is standalone — runs with `num_envs=1`, not part of training
- Camera excluded from training observation space (320×240 RGBD × 4096 envs = ~2.4GB GPU memory)
- `rclpy` import is conditional — training works without ROS2 installed
- `/cmd_vel` persists last command (continuous velocity, matching ROS conventions)
- Occupancy grid published once with transient-local QoS (late subscribers get it immediately)
- Create 3 kinematics: wheel_base=0.233m, wheel_radius=0.036m (configured in `config/ros2_bridge.yaml`)

## Training Logs

Training outputs go to `logs/sb3/<task_name>/` including model checkpoints and optional video recordings.
