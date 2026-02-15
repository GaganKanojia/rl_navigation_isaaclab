# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Offline script to precompute 2D occupancy grids from room USD meshes.

This script loads a room USD file, extracts the floor-level collision
geometry, rasterizes it into a 2D binary occupancy grid, and saves the
result as a .npy file with companion .json metadata.

Usage::

    python scripts/precompute_grid.py \
        --usd_path /path/to/room.usd \
        --output_path /path/to/room_grid.npy \
        --resolution 0.1 \
        --height 0.05
"""

from __future__ import annotations

import argparse
import json
import numpy as np
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Precompute occupancy grid from room USD mesh.")
    parser.add_argument("--usd_path", type=str, required=True, help="Path to the room USD file.")
    parser.add_argument("--output_path", type=str, required=True, help="Output path for the .npy grid file.")
    parser.add_argument("--resolution", type=float, default=0.1, help="Grid resolution in meters per cell.")
    parser.add_argument(
        "--height", type=float, default=0.05, help="Height above floor to slice for occupancy detection."
    )
    parser.add_argument(
        "--robot_radius", type=float, default=0.17, help="Robot radius in meters for inflating obstacles."
    )
    args = parser.parse_args()

    try:
        import trimesh
    except ImportError:
        print("ERROR: trimesh is required. Install with: pip install trimesh")
        return

    try:
        from pxr import Usd, UsdGeom
    except ImportError:
        print("WARNING: pxr (USD) not available. Attempting to load as mesh directly with trimesh.")
        mesh = trimesh.load(args.usd_path)
        if isinstance(mesh, trimesh.Scene):
            mesh = mesh.dump(concatenate=True)
        _process_mesh(mesh, args)
        return

    # Load USD and extract mesh geometry
    stage = Usd.Stage.Open(args.usd_path)
    meshes = []

    for prim in stage.Traverse():
        if prim.IsA(UsdGeom.Mesh):
            usd_mesh = UsdGeom.Mesh(prim)
            points = np.array(usd_mesh.GetPointsAttr().Get(), dtype=np.float64)
            face_counts = np.array(usd_mesh.GetFaceVertexCountsAttr().Get(), dtype=np.int32)
            face_indices = np.array(usd_mesh.GetFaceVertexIndicesAttr().Get(), dtype=np.int32)

            # Convert to trimesh faces (triangulate if needed)
            faces = []
            idx = 0
            for count in face_counts:
                if count == 3:
                    faces.append(face_indices[idx : idx + 3])
                elif count == 4:
                    # Triangulate quad
                    faces.append(face_indices[[idx, idx + 1, idx + 2]])
                    faces.append(face_indices[[idx, idx + 2, idx + 3]])
                else:
                    # Fan triangulation for polygons
                    for i in range(1, count - 1):
                        faces.append(face_indices[[idx, idx + i, idx + i + 1]])
                idx += count

            if faces:
                faces = np.array(faces, dtype=np.int32)
                tri_mesh = trimesh.Trimesh(vertices=points, faces=faces)
                meshes.append(tri_mesh)

    if not meshes:
        print("ERROR: No mesh geometry found in USD file.")
        return

    combined = trimesh.util.concatenate(meshes)
    _process_mesh(combined, args)


def _process_mesh(mesh: "trimesh.Trimesh", args: argparse.Namespace) -> None:
    """Rasterize a mesh into a 2D occupancy grid at the specified height."""
    import trimesh

    bounds = mesh.bounds  # (2, 3): [[xmin, ymin, zmin], [xmax, ymax, zmax]]
    x_min, y_min = bounds[0, 0], bounds[0, 1]
    x_max, y_max = bounds[1, 0], bounds[1, 1]

    # Add margin around the grid
    margin = args.robot_radius * 2
    x_min -= margin
    y_min -= margin
    x_max += margin
    y_max += margin

    # Create grid dimensions
    width = int(np.ceil((x_max - x_min) / args.resolution))
    height = int(np.ceil((y_max - y_min) / args.resolution))

    print(f"Grid dimensions: {height} x {width} ({height * width} cells)")
    print(f"World bounds: x=[{x_min:.2f}, {x_max:.2f}], y=[{y_min:.2f}, {y_max:.2f}]")

    # Create a horizontal slice at the specified height
    grid = np.zeros((height, width), dtype=np.uint8)

    # Cast rays downward from above the slice height to detect obstacles
    origins = []
    for row in range(height):
        for col in range(width):
            x = x_min + (col + 0.5) * args.resolution
            y = y_min + (row + 0.5) * args.resolution
            origins.append([x, y, args.height])

    origins = np.array(origins)
    directions = np.tile([0.0, 0.0, -1.0], (len(origins), 1))

    # Ray-cast to find occupied cells
    locations, index_ray, _ = mesh.ray.intersects_location(ray_origins=origins, ray_directions=directions)

    if len(index_ray) > 0:
        occupied_cells = set(index_ray.tolist())
        for cell_idx in occupied_cells:
            row = cell_idx // width
            col = cell_idx % width
            if 0 <= row < height and 0 <= col < width:
                grid[row, col] = 1

    # Inflate obstacles by robot radius
    if args.robot_radius > 0:
        inflate_cells = int(np.ceil(args.robot_radius / args.resolution))
        from scipy.ndimage import binary_dilation

        struct = np.ones((2 * inflate_cells + 1, 2 * inflate_cells + 1), dtype=bool)
        grid = binary_dilation(grid, structure=struct).astype(np.uint8)

    # Set border as occupied
    grid[0, :] = 1
    grid[-1, :] = 1
    grid[:, 0] = 1
    grid[:, -1] = 1

    # Save grid
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(output_path), grid)

    # Save metadata
    meta_path = output_path.with_suffix(".json")
    metadata = {
        "resolution": args.resolution,
        "origin": [float(x_min), float(y_min)],
        "width": width,
        "height": height,
        "robot_radius": args.robot_radius,
        "source_usd": args.usd_path,
    }
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    num_free = int(np.sum(grid == 0))
    num_occupied = int(np.sum(grid == 1))
    print(f"Saved grid to: {output_path}")
    print(f"Saved metadata to: {meta_path}")
    print(f"Free cells: {num_free}, Occupied cells: {num_occupied}")


if __name__ == "__main__":
    main()
