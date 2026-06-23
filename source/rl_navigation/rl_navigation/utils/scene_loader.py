# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Utilities for loading room scenes and SLAM-generated occupancy maps."""

from __future__ import annotations

import numpy as np
import torch
import yaml
from pathlib import Path

from .occupancy_grid import OccupancyGrid


def load_room_list(rooms_txt_path: str) -> list[dict[str, str]]:
    """Parse a rooms.txt file listing room USD and SLAM-map file pairs.

    The file format is one room per line::

        /path/to/room1.usd /path/to/room1.yaml
        /path/to/room2.usd /path/to/room2.yaml

    The map token is the slam_toolbox / nav2_map_saver map: either the ``.yaml``
    or the ``.pgm`` (or the basename). Only one of the pair needs to be listed —
    the sibling file is derived by basename in :func:`load_occupancy_grid`.

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


def _read_pgm(path: Path) -> np.ndarray:
    """Read a binary (P5) or ASCII (P2) PGM image into a uint8 ``(H, W)`` array.

    Minimal self-contained parser so we don't pull in PIL/OpenCV. Row 0 of the
    returned array is the top of the image (the PGM storage order).

    Args:
        path: Path to the ``.pgm`` file.

    Returns:
        ``(H, W)`` array of pixel values scaled to the 0-255 range.

    Raises:
        ValueError: If the file is not a valid P2/P5 PGM.
    """
    with open(path, "rb") as f:
        data = f.read()

    # Tokenize the header, skipping '#' comment lines, until we have magic,
    # width, height, maxval. The binary pixel block follows the byte after maxval.
    tokens: list[bytes] = []
    idx = 0
    n = len(data)
    while len(tokens) < 4 and idx < n:
        # skip whitespace
        while idx < n and data[idx : idx + 1].isspace():
            idx += 1
        # skip comment line
        if idx < n and data[idx : idx + 1] == b"#":
            while idx < n and data[idx : idx + 1] not in (b"\n", b"\r"):
                idx += 1
            continue
        start = idx
        while idx < n and not data[idx : idx + 1].isspace():
            idx += 1
        tokens.append(data[start:idx])

    if len(tokens) < 4:
        raise ValueError(f"Malformed PGM header: {path}")

    magic = tokens[0]
    width, height, maxval = int(tokens[1]), int(tokens[2]), int(tokens[3])

    if magic == b"P5":
        # single whitespace byte separates maxval from the pixel block
        idx += 1
        pixels = np.frombuffer(data, dtype=np.uint8, count=width * height, offset=idx)
        img = pixels.reshape(height, width)
    elif magic == b"P2":
        vals = data[idx:].split()
        pixels = np.array([int(v) for v in vals[: width * height]], dtype=np.uint16)
        img = pixels.reshape(height, width)
    else:
        raise ValueError(f"Unsupported PGM magic {magic!r} (expected P2 or P5): {path}")

    if maxval != 255:
        img = (img.astype(np.float32) * (255.0 / maxval)).round().astype(np.uint8)
    return img.astype(np.uint8)


def load_occupancy_grid(
    map_path: str,
    device: str | torch.device = "cuda:0",
    unknown_as_occupied: bool = True,
) -> OccupancyGrid:
    """Load an occupancy grid from a slam_toolbox / nav2_map_saver map.

    Reads the standard ROS map pair:
    - ``<map>.yaml`` — metadata (``resolution``, ``origin``, ``negate``,
      ``occupied_thresh``, ``free_thresh``, ``image``).
    - ``<image>.pgm`` — the occupancy image referenced by the yaml (falls back to
      ``<map>.pgm`` if the yaml has no ``image`` field).

    ``map_path`` may be the ``.yaml``, the ``.pgm``, or the basename — the sibling
    is derived from the ``.yaml`` stem.

    Args:
        map_path: Path to the map (``.yaml``/``.pgm``/basename).
        device: Torch device to load the grid onto.
        unknown_as_occupied: If True (default), unmapped cells are treated as
            occupied (conservative for collision checking and goal sampling).

    Returns:
        An :class:`OccupancyGrid` instance (``1=occupied``, ``0=free``).

    Raises:
        FileNotFoundError: If the yaml or image file is missing.
    """
    yaml_path = Path(map_path).with_suffix(".yaml")
    if not yaml_path.exists():
        raise FileNotFoundError(f"Map metadata (.yaml) not found: {yaml_path}")

    with open(yaml_path) as f:
        meta = yaml.safe_load(f)

    resolution = float(meta["resolution"])
    origin = (float(meta["origin"][0]), float(meta["origin"][1]))  # ignore yaw (origin[2])
    negate = int(meta.get("negate", 0))
    occupied_thresh = float(meta.get("occupied_thresh", 0.65))
    free_thresh = float(meta.get("free_thresh", 0.196))

    image_field = meta.get("image")
    image_path = (yaml_path.parent / image_field) if image_field else yaml_path.with_suffix(".pgm")
    if not image_path.exists():
        raise FileNotFoundError(f"Map image not found: {image_path}")

    img = _read_pgm(image_path).astype(np.float32)

    # ROS map_server occupancy semantics: occupancy probability p in [0, 1].
    p = img / 255.0 if negate else (255.0 - img) / 255.0

    occupied = p > occupied_thresh
    free = p < free_thresh
    unknown = ~(occupied | free)

    grid_np = occupied.astype(np.float32)  # 1=occupied, 0=free
    if unknown_as_occupied:
        grid_np[unknown] = 1.0

    # PGM row 0 is the top (max y); OccupancyGrid.world_to_grid has row increase
    # with +y from a bottom-left origin, so flip vertically to align.
    grid_np = np.flipud(grid_np).copy()

    grid_tensor = torch.from_numpy(grid_np).to(device=device, dtype=torch.float32)

    return OccupancyGrid(grid=grid_tensor, resolution=resolution, origin=origin)
