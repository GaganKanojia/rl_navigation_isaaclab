# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Custom SB3 feature extractor for the navigation environment.

Processes dict observations with a CNN branch for the occupancy grid,
concatenated with goal pose and velocity vectors.
"""

from __future__ import annotations

import gymnasium as gym
import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class NavigationFeaturesExtractor(BaseFeaturesExtractor):
    """Feature extractor for navigation observations.

    Architecture::

        occupancy_grid (1, 50, 50) -> 2D CNN -> 64-dim
        goal_pose (3,)             -> passthrough
        velocity (3,)              -> passthrough
        ───────────────────────────────────────────
        concat -> 70-dim output

    Args:
        observation_space: Dict observation space from the environment.
        features_dim: Output feature dimension (default: 70).
    """

    def __init__(self, observation_space: gym.spaces.Dict, features_dim: int = 70):
        super().__init__(observation_space, features_dim)

        # 2D CNN for occupancy grid: (N, 1, 50, 50) -> (N, 64)
        self.grid_cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 64),
            nn.ReLU(),
        )

        # goal_pose(3) + velocity(3) pass through directly
        # Total output: 64 + 3 + 3 = 70

    def forward(self, observations: dict[str, torch.Tensor]) -> torch.Tensor:
        grid_features = self.grid_cnn(observations["occupancy_grid"])
        goal = observations["goal_pose"]
        vel = observations["velocity"]
        return torch.cat([grid_features, goal, vel], dim=-1)
