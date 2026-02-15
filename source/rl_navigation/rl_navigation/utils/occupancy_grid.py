# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""GPU-accelerated occupancy grid utilities for navigation."""

from __future__ import annotations

import torch
import torch.nn.functional as F


class OccupancyGrid:
    """Manages a 2D binary occupancy grid for goal sampling and local patch extraction.

    The grid is stored on GPU and all operations are batched for use with
    parallel Isaac Lab environments.

    Attributes:
        grid: Binary tensor (H, W) where 1=occupied, 0=free.
        resolution: Meters per grid cell.
        origin: World (x, y) coordinates of grid cell (0, 0).
    """

    def __init__(self, grid: torch.Tensor, resolution: float, origin: tuple[float, float]):
        """Initialize occupancy grid.

        Args:
            grid: Binary tensor (H, W), 1=occupied, 0=free.
            resolution: Meters per cell.
            origin: World (x, y) of grid cell (0, 0) corner.
        """
        self.grid = grid.float()
        self.resolution = resolution
        self.origin = origin
        self.device = grid.device

        # Precompute free cell indices for fast sampling
        free_mask = self.grid == 0.0
        self._free_indices = torch.nonzero(free_mask, as_tuple=False)  # (M, 2) row, col
        self._num_free = self._free_indices.shape[0]

        # Prepare grid for grid_sample: needs (1, 1, H, W)
        self._grid_for_sample = self.grid.unsqueeze(0).unsqueeze(0)

    def world_to_grid(self, positions: torch.Tensor) -> torch.Tensor:
        """Convert world (x, y) coordinates to grid (row, col) indices.

        Args:
            positions: (N, 2) world coordinates.

        Returns:
            (N, 2) integer grid indices (row, col).
        """
        # x -> col, y -> row
        col = ((positions[:, 0] - self.origin[0]) / self.resolution).long()
        row = ((positions[:, 1] - self.origin[1]) / self.resolution).long()
        return torch.stack([row, col], dim=-1)

    def grid_to_world(self, indices: torch.Tensor) -> torch.Tensor:
        """Convert grid (row, col) indices to world (x, y) center of cell.

        Args:
            indices: (N, 2) grid indices (row, col).

        Returns:
            (N, 2) world coordinates.
        """
        x = indices[:, 1].float() * self.resolution + self.origin[0] + self.resolution / 2
        y = indices[:, 0].float() * self.resolution + self.origin[1] + self.resolution / 2
        return torch.stack([x, y], dim=-1)

    def get_local_patch(
        self,
        positions: torch.Tensor,
        headings: torch.Tensor,
        patch_size: int = 50,
    ) -> torch.Tensor:
        """Extract a local occupancy grid patch centered on each robot, rotated to robot frame.

        Uses affine_grid + grid_sample for batched GPU-accelerated extraction.

        Args:
            positions: (N, 2) robot world positions (x, y).
            headings: (N,) robot yaw angles in radians.
            patch_size: Grid cells per side of the square output patch.

        Returns:
            (N, 1, patch_size, patch_size) binary tensor suitable for CNN input.
        """
        n = positions.shape[0]
        h, w = self.grid.shape

        # Compute the physical size of the patch in world units
        patch_world_size = patch_size * self.resolution

        # Normalize positions to [-1, 1] range for grid_sample
        # grid_sample expects coordinates in [-1, 1] where -1 is left/top, +1 is right/bottom
        cx = (positions[:, 0] - self.origin[0]) / (w * self.resolution) * 2.0 - 1.0
        cy = (positions[:, 1] - self.origin[1]) / (h * self.resolution) * 2.0 - 1.0

        # Scale factor: ratio of patch size to full grid size
        sx = patch_world_size / (w * self.resolution)
        sy = patch_world_size / (h * self.resolution)

        cos_h = torch.cos(headings)
        sin_h = torch.sin(headings)

        # Build 2x3 affine matrix for each environment
        # This rotates by -heading (to align with robot frame) and translates to robot position
        theta = torch.zeros(n, 2, 3, device=self.device)
        theta[:, 0, 0] = sx * cos_h
        theta[:, 0, 1] = sx * sin_h
        theta[:, 0, 2] = cx
        theta[:, 1, 0] = -sy * sin_h
        theta[:, 1, 1] = sy * cos_h
        theta[:, 1, 2] = cy

        # Generate sampling grid
        sample_grid = F.affine_grid(theta, (n, 1, patch_size, patch_size), align_corners=False)

        # Expand the occupancy grid to batch dimension
        grid_batch = self._grid_for_sample.expand(n, -1, -1, -1)

        # Sample the grid
        patches = F.grid_sample(grid_batch, sample_grid, mode="nearest", padding_mode="border", align_corners=False)

        return patches

    def sample_free_positions(self, num_samples: int, device: str | torch.device | None = None) -> torch.Tensor:
        """Sample random positions in free space.

        Args:
            num_samples: Number of positions to sample.
            device: Target device (defaults to grid device).

        Returns:
            (num_samples, 2) world positions guaranteed to be in free cells.
        """
        if device is None:
            device = self.device

        if self._num_free == 0:
            raise RuntimeError("No free cells available in the occupancy grid.")

        # Random indices into the free cells list
        rand_idx = torch.randint(0, self._num_free, (num_samples,), device=self.device)
        grid_indices = self._free_indices[rand_idx]  # (num_samples, 2) as (row, col)

        # Convert to world coordinates
        positions = self.grid_to_world(grid_indices)

        if device != self.device:
            positions = positions.to(device)

        return positions

    def is_free(self, positions: torch.Tensor) -> torch.Tensor:
        """Check if world positions are in free space.

        Args:
            positions: (N, 2) world coordinates.

        Returns:
            (N,) boolean tensor, True if the position is in a free cell.
        """
        grid_idx = self.world_to_grid(positions)
        h, w = self.grid.shape

        # Clamp to grid bounds
        row = grid_idx[:, 0].clamp(0, h - 1)
        col = grid_idx[:, 1].clamp(0, w - 1)

        # Check if in bounds
        in_bounds = (grid_idx[:, 0] >= 0) & (grid_idx[:, 0] < h) & (grid_idx[:, 1] >= 0) & (grid_idx[:, 1] < w)

        # Check if free
        cell_values = self.grid[row, col]
        return in_bounds & (cell_values == 0.0)
