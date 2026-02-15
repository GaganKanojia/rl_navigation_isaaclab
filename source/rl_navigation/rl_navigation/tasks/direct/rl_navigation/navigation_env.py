# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Navigation environment for the iRobot Create 3 using a differential drive."""

from __future__ import annotations

import math
import torch
from collections.abc import Sequence

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sensors import Camera, RayCaster

from rl_navigation.utils import load_occupancy_grid, load_room_list

from .navigation_env_cfg import NavigationEnvCfg


class NavigationEnv(DirectRLEnv):
    """Goal-conditioned navigation environment for a differential-drive robot.

    The robot receives lidar scans, a local occupancy grid patch, goal relative
    pose, and its own velocity as observations. Actions are left/right wheel
    velocity targets. The reward encourages reaching the goal efficiently while
    avoiding collisions.
    """

    cfg: NavigationEnvCfg

    def __init__(self, cfg: NavigationEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # Find wheel joint indices
        self._left_wheel_idx, _ = self._robot.find_joints(self.cfg.left_wheel_joint)
        self._right_wheel_idx, _ = self._robot.find_joints(self.cfg.right_wheel_joint)
        self._wheel_ids = self._left_wheel_idx + self._right_wheel_idx

        # Action buffers
        self._actions = torch.zeros(self.num_envs, 2, device=self.device)
        self._previous_actions = torch.zeros(self.num_envs, 2, device=self.device)

        # Goal buffers (x, y in room-local coordinates)
        self._goal_pos = torch.zeros(self.num_envs, 2, device=self.device)
        self._prev_goal_dist = torch.zeros(self.num_envs, device=self.device)

        # Load occupancy grid
        rooms = load_room_list(self.cfg.rooms_txt_path)
        room = rooms[self.cfg.room_index]
        self._occ_grid = load_occupancy_grid(room["grid_path"], self.device)

        # Episode reward tracking for logging
        self._reward_keys = ["goal_reached", "distance_progress", "collision", "time_penalty", "smoothness"]
        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device) for key in self._reward_keys
        }

    def _setup_scene(self):
        # Robot
        self._robot = Articulation(self.cfg.robot_cfg)
        self.scene.articulations["robot"] = self._robot

        # Lidar
        self._lidar = RayCaster(self.cfg.lidar_cfg)
        self.scene.sensors["lidar"] = self._lidar

        # Camera (opt-in, excluded from training observation space)
        self._camera = None
        if self.cfg.enable_camera and self.cfg.camera_cfg is not None:
            self._camera = Camera(self.cfg.camera_cfg)
            self.scene.sensors["camera"] = self._camera

        # Terrain (room)
        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)

        # Clone environments
        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])

        # Lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self._previous_actions[:] = self._actions
        self._actions[:] = actions.clamp(-1.0, 1.0)

    def _apply_action(self) -> None:
        wheel_vel_targets = self._actions * self.cfg.max_wheel_vel
        self._robot.set_joint_velocity_target(wheel_vel_targets, joint_ids=self._wheel_ids)

    def _get_observations(self) -> dict:
        # --- Lidar ranges ---
        ray_hits = self._lidar.data.ray_hits_w  # (N, num_rays, 3)
        lidar_pos = self._lidar.data.pos_w  # (N, 3)
        ranges = torch.norm(ray_hits - lidar_pos.unsqueeze(1), dim=-1)  # (N, num_rays)
        ranges = ranges.clamp(0.0, self.cfg.lidar_cfg.max_distance)
        lidar_obs = ranges / self.cfg.lidar_cfg.max_distance  # Normalize to [0, 1]

        # --- Robot pose ---
        robot_pos_w = self._robot.data.root_pos_w  # (N, 3)
        robot_quat = self._robot.data.root_quat_w  # (N, 4) as (w, x, y, z)

        # Extract yaw from quaternion
        yaw = torch.atan2(
            2.0 * (robot_quat[:, 0] * robot_quat[:, 3] + robot_quat[:, 1] * robot_quat[:, 2]),
            1.0 - 2.0 * (robot_quat[:, 2] ** 2 + robot_quat[:, 3] ** 2),
        )

        # Robot position in room-local coordinates
        robot_local_xy = robot_pos_w[:, :2] - self.scene.env_origins[:, :2]

        # --- Local occupancy grid patch ---
        occ_patch = self._occ_grid.get_local_patch(
            robot_local_xy, yaw, self.cfg.local_patch_size
        )  # (N, 1, patch_size, patch_size)

        # --- Goal relative pose in robot frame ---
        goal_delta = self._goal_pos - robot_local_xy  # (N, 2)
        dist = torch.norm(goal_delta, dim=-1, keepdim=True)  # (N, 1)
        angle_to_goal = torch.atan2(goal_delta[:, 1], goal_delta[:, 0]) - yaw  # (N,)
        goal_obs = torch.cat(
            [
                dist,
                torch.cos(angle_to_goal).unsqueeze(-1),
                torch.sin(angle_to_goal).unsqueeze(-1),
            ],
            dim=-1,
        )  # (N, 3)

        # --- Robot velocity in body frame ---
        velocity = torch.cat(
            [
                self._robot.data.root_lin_vel_b[:, :2],
                self._robot.data.root_ang_vel_b[:, 2:3],
            ],
            dim=-1,
        )  # (N, 3)

        # Cache camera data for ROS2 bridge (not part of policy observations)
        if self._camera is not None:
            self._last_rgb = self._camera.data.output["rgb"]  # (N, H, W, 3) uint8
            self._last_depth = self._camera.data.output["distance_to_camera"]  # (N, H, W, 1) float32

        observations = {
            "policy": {
                "lidar": lidar_obs,
                "occupancy_grid": occ_patch,
                "goal_pose": goal_obs,
                "velocity": velocity,
            }
        }
        return observations

    def _get_rewards(self) -> torch.Tensor:
        # Robot position in room-local coordinates
        robot_local_xy = self._robot.data.root_pos_w[:, :2] - self.scene.env_origins[:, :2]
        goal_dist = torch.norm(self._goal_pos - robot_local_xy, dim=-1)

        # Lidar min range for collision detection
        ray_hits = self._lidar.data.ray_hits_w
        lidar_pos = self._lidar.data.pos_w
        ranges = torch.norm(ray_hits - lidar_pos.unsqueeze(1), dim=-1)
        min_range = ranges.min(dim=-1).values

        # Compute individual reward components
        rew_goal = self.cfg.rew_scale_goal_reached * (goal_dist < self.cfg.goal_radius).float()
        rew_progress = self.cfg.rew_scale_distance_progress * (self._prev_goal_dist - goal_dist)
        collision_proximity = (self.cfg.collision_threshold - min_range).clamp(min=0.0)
        rew_collision = self.cfg.rew_scale_collision * collision_proximity
        rew_time = self.cfg.rew_scale_time_penalty * torch.ones_like(goal_dist)
        rew_smooth = self.cfg.rew_scale_smoothness * torch.sum(
            torch.square(self._actions - self._previous_actions), dim=-1
        )

        # Track for logging
        self._episode_sums["goal_reached"] += rew_goal
        self._episode_sums["distance_progress"] += rew_progress
        self._episode_sums["collision"] += rew_collision
        self._episode_sums["time_penalty"] += rew_time
        self._episode_sums["smoothness"] += rew_smooth

        # Update previous distance
        self._prev_goal_dist[:] = goal_dist

        total_reward = rew_goal + rew_progress + rew_collision + rew_time + rew_smooth
        return total_reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        robot_local_xy = self._robot.data.root_pos_w[:, :2] - self.scene.env_origins[:, :2]
        goal_dist = torch.norm(self._goal_pos - robot_local_xy, dim=-1)

        # Termination conditions
        goal_reached = goal_dist < self.cfg.goal_radius

        # Hard collision detection
        ray_hits = self._lidar.data.ray_hits_w
        lidar_pos = self._lidar.data.pos_w
        ranges = torch.norm(ray_hits - lidar_pos.unsqueeze(1), dim=-1)
        collision = ranges.min(dim=-1).values < 0.1

        terminated = goal_reached | collision
        time_out = self.episode_length_buf >= self.max_episode_length - 1

        return terminated, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES
        super()._reset_idx(env_ids)

        num_resets = len(env_ids)

        # Sample robot start positions in free space
        start_positions = self._occ_grid.sample_free_positions(num_resets, self.device)

        # Sample goal positions at valid distances
        goal_positions = self._sample_goals(start_positions)
        self._goal_pos[env_ids] = goal_positions

        # Set robot root state
        default_root_state = self._robot.data.default_root_state[env_ids]
        default_root_state[:, 0] = start_positions[:, 0] + self.scene.env_origins[env_ids, 0]
        default_root_state[:, 1] = start_positions[:, 1] + self.scene.env_origins[env_ids, 1]

        # Random yaw
        random_yaw = torch.rand(num_resets, device=self.device) * 2 * math.pi - math.pi
        default_root_state[:, 3] = torch.cos(random_yaw / 2)  # quat w
        default_root_state[:, 4] = 0.0  # quat x
        default_root_state[:, 5] = 0.0  # quat y
        default_root_state[:, 6] = torch.sin(random_yaw / 2)  # quat z

        # Zero velocities
        default_root_state[:, 7:] = 0.0

        self._robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)

        # Reset joint states
        joint_pos = self._robot.data.default_joint_pos[env_ids]
        joint_vel = torch.zeros_like(joint_pos)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        # Reset action buffers
        self._actions[env_ids] = 0.0
        self._previous_actions[env_ids] = 0.0

        # Initialize previous goal distance
        self._prev_goal_dist[env_ids] = torch.norm(goal_positions - start_positions, dim=-1)

        # Log episode reward sums and reset
        extras = {}
        for key in self._reward_keys:
            extras[f"Episode_Reward/{key}"] = torch.mean(self._episode_sums[key][env_ids])
            self._episode_sums[key][env_ids] = 0.0
        self.extras["log"] = extras

    def _sample_goals(self, start_positions: torch.Tensor) -> torch.Tensor:
        """Sample goal positions in free space within [min_dist, max_dist] of start.

        Uses rejection sampling: generates many candidate positions and selects
        the first valid one per environment.

        Args:
            start_positions: (N, 2) robot start positions in room-local coords.

        Returns:
            (N, 2) goal positions in room-local coords.
        """
        num = start_positions.shape[0]
        goals = start_positions.clone()  # Fallback: goal at start
        remaining = torch.ones(num, dtype=torch.bool, device=self.device)
        candidates_per_env = 10

        for _ in range(100):
            if not remaining.any():
                break

            n_remaining = remaining.sum().item()
            candidates = self._occ_grid.sample_free_positions(n_remaining * candidates_per_env, self.device)
            candidates = candidates.view(n_remaining, candidates_per_env, 2)

            starts_expanded = start_positions[remaining].unsqueeze(1).expand(-1, candidates_per_env, -1)
            dists = torch.norm(candidates - starts_expanded, dim=-1)
            valid = (dists >= self.cfg.min_goal_distance) & (dists <= self.cfg.max_goal_distance)

            # Pick first valid candidate per environment
            has_valid = valid.any(dim=1)
            first_valid_idx = valid.float().argmax(dim=1)

            remaining_indices = torch.where(remaining)[0]
            for i in range(n_remaining):
                if has_valid[i]:
                    goals[remaining_indices[i]] = candidates[i, first_valid_idx[i]]

            remaining[remaining.clone()] = ~has_valid

        return goals

    # --- Property accessors for ROS2 bridge ---

    @property
    def camera(self) -> Camera | None:
        """Front-facing RGB-D camera sensor, or ``None`` if disabled."""
        return self._camera

    @property
    def lidar(self) -> RayCaster:
        """Planar lidar ray-caster sensor."""
        return self._lidar

    @property
    def occ_grid(self):
        """Static occupancy grid for the current room."""
        return self._occ_grid

    @property
    def goal_pos(self) -> torch.Tensor:
        """Current goal positions ``(N, 2)`` in room-local coordinates."""
        return self._goal_pos

    @goal_pos.setter
    def goal_pos(self, value: torch.Tensor) -> None:
        self._goal_pos[:] = value
        robot_local_xy = self._robot.data.root_pos_w[:, :2] - self.scene.env_origins[:, :2]
        self._prev_goal_dist[:] = torch.norm(self._goal_pos - robot_local_xy, dim=-1)
