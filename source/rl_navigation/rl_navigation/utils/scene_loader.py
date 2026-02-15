# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Utilities for loading room scenes and precomputed occupancy grids."""

from __future__ import annotations

import json
import numpy as np
import torch
from pathlib import Path

from .occupancy_grid import OccupancyGrid


def load_room_list(rooms_txt_path: str) -> list[dict[str, str]]:
    """Parse a rooms.txt file listing room USD and grid file pairs.

    The file format is one room per line::

        /path/to/room1.usd /path/to/room1_grid.npy
        /path/to/room2.usd /path/to/room2_grid.npy

    Args:
        rooms_txt_path: Path to the rooms.txt file.

    Returns:
        List of dicts with keys ``usd_path`` and ``grid_path``.

    Raises:
        FileNotFoundError: If rooms_txt_path does not exist.
        ValueError: If a line does not contain exactly two whitespace-separated paths.
    """
    path = Path(rooms_txt_path)
    if not path.exists():
        raise FileNotFoundError(f"Rooms file not found: {rooms_txt_path}")

    rooms = []
    for line_num, line in enumerate(path.read_text().strip().splitlines(), start=1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) != 2:
            raise ValueError(f"Line {line_num} in {rooms_txt_path}: expected 2 paths, got {len(parts)}")
        rooms.append({"usd_path": parts[0], "grid_path": parts[1]})

    return rooms


def load_occupancy_grid(grid_path: str, device: str | torch.device = "cuda:0") -> OccupancyGrid:
    """Load a precomputed occupancy grid from .npy data and .json metadata.

    Expects two files:
    - ``<grid_path>`` — a ``.npy`` file containing a 2D binary numpy array (H, W).
    - ``<grid_path_stem>.json`` — metadata with ``resolution`` (float) and ``origin`` ([x, y]).

    Args:
        grid_path: Path to the ``.npy`` grid file.
        device: Torch device to load the grid onto.

    Returns:
        An :class:`OccupancyGrid` instance.

    Raises:
        FileNotFoundError: If the grid or metadata file is missing.
    """
    grid_path = Path(grid_path)
    meta_path = grid_path.with_suffix(".json")

    if not grid_path.exists():
        raise FileNotFoundError(f"Grid file not found: {grid_path}")
    if not meta_path.exists():
        raise FileNotFoundError(f"Grid metadata file not found: {meta_path}")

    # Load grid data
    grid_np = np.load(str(grid_path))
    grid_tensor = torch.from_numpy(grid_np).to(device=device, dtype=torch.float32)

    # Load metadata
    with open(meta_path) as f:
        meta = json.load(f)

    resolution = float(meta["resolution"])
    origin = (float(meta["origin"][0]), float(meta["origin"][1]))

    return OccupancyGrid(grid=grid_tensor, resolution=resolution, origin=origin)
