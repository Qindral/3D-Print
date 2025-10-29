"""Distance-map based voxel filling for organic shapes."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Callable, Optional, Tuple

import numpy as np
from scipy import ndimage

from .constants import (
    BASE_ROD_THICKNESS,
    CUBE_HALF,
    CUBE_SIZE,
    ROD_LENGTH,
    ROLLING_EXPORT_PATH,
)
from .models import RenderGeometry, RollingBallResult


def sphere_offsets(radius: int) -> List[Tuple[int, int, int]]:
    if radius <= 0:
        return [(0, 0, 0)]
    offsets: List[Tuple[int, int, int]] = []
    radius_sq = radius * radius
    for dx in range(-radius, radius + 1):
        for dy in range(-radius, radius + 1):
            for dz in range(-radius, radius + 1):
                if dx * dx + dy * dy + dz * dz <= radius_sq:
                    offsets.append((dx, dy, dz))
    if not offsets:
        offsets.append((0, 0, 0))
    return offsets


def extract_surface_points(occupancy: np.ndarray, voxel_size: float) -> np.ndarray:
    coords = np.argwhere(occupancy)
    if coords.size == 0:
        return np.empty((0, 3), dtype=float)
    max_x, max_y, max_z = occupancy.shape
    neighbor_offsets = [
        (-1, 0, 0),
        (1, 0, 0),
        (0, -1, 0),
        (0, 1, 0),
        (0, 0, -1),
        (0, 0, 1),
    ]
    surface_indices: List[Tuple[int, int, int]] = []
    occupancy_bool = occupancy.astype(bool, copy=False)
    for x, y, z in coords:
        for ox, oy, oz in neighbor_offsets:
            nx, ny, nz = x + ox, y + oy, z + oz
            if not (0 <= nx < max_x and 0 <= ny < max_y and 0 <= nz < max_z):
                surface_indices.append((x, y, z))
                break
            if not occupancy_bool[nx, ny, nz]:
                surface_indices.append((x, y, z))
                break
    if not surface_indices:
        surface_indices = [tuple(coord) for coord in coords]
    surface_array = np.array(surface_indices, dtype=float)
    centers = (surface_array + 0.5) * voxel_size - CUBE_HALF
    return centers


def run_distance_fill(
    render_data: RenderGeometry,
    threshold: float,
    grid_resolution: int = 256,
    progress_cb: Optional[Callable[[float, str], None]] = None,
) -> RollingBallResult:
    def report(progress: float, message: str) -> None:
        if progress_cb is not None:
            progress_cb(max(0.0, min(1.0, progress)), message)

    grid_shape = (grid_resolution, grid_resolution, grid_resolution)
    occupancy = np.zeros(grid_shape, dtype=bool)
    voxel_size = CUBE_SIZE / grid_resolution
    half = CUBE_HALF

    if render_data.is_empty():
        report(1.0, "Keine Rod-Daten vorhanden")
        return RollingBallResult(
            surface_points=np.empty((0, 3), dtype=float),
            filled_voxel_count=0,
            total_voxel_count=int(np.prod(grid_shape)),
            fill_fraction=0.0,
            threshold=max(threshold, 0.0),
            voxel_size=voxel_size,
            grid_resolution=grid_resolution,
        )

    samples_per_rod = max(4, int(math.ceil(ROD_LENGTH / max(voxel_size * 0.5, 1e-6))))
    thickness_radius = max(
        1, int(math.ceil((BASE_ROD_THICKNESS * 0.5) / max(voxel_size, 1e-6)))
    )
    thickness_offsets = sphere_offsets(thickness_radius)

    report(0.0, f"Voxelisiere Rods ({grid_resolution}³)")
    total_rods = len(render_data)
    next_progress_update = 0.05
    points_a = render_data.points_a
    points_b = render_data.points_b
    for index in range(total_rods):
        start = points_a[index]
        end = points_b[index]
        for t in np.linspace(0.0, 1.0, samples_per_rod):
            position = start * (1.0 - t) + end * t
            idx = np.floor((position + half) / voxel_size).astype(int)
            ix, iy, iz = np.clip(idx, 0, grid_resolution - 1)
            for ox, oy, oz in thickness_offsets:
                nx, ny, nz = ix + ox, iy + oy, iz + oz
                if 0 <= nx < grid_resolution and 0 <= ny < grid_resolution and 0 <= nz < grid_resolution:
                    occupancy[nx, ny, nz] = True
        if total_rods > 0:
            progress = 0.6 * ((index + 1) / total_rods)
            if progress >= next_progress_update:
                report(progress, f"Voxelisiere Rods ({index + 1}/{total_rods})")
                next_progress_update += 0.05

    report(0.62, "Berechne Distanzkarte der Freiräume")

    sampling = (voxel_size, voxel_size, voxel_size)
    empty_distance = ndimage.distance_transform_edt(~occupancy, sampling=sampling)

    report(0.8, "Fülle Regionen unterhalb des Schwellenwerts")

    threshold = max(threshold, 0.0)
    filled = occupancy | ((~occupancy) & (empty_distance <= threshold))

    report(0.92, "Extrahiere Oberfläche")

    filled_voxel_count = int(np.count_nonzero(filled))
    total_voxel_count = int(filled.size)
    fill_fraction = (filled_voxel_count / total_voxel_count) if total_voxel_count else 0.0
    surface_points = extract_surface_points(filled, voxel_size)

    report(1.0, "Distanzfüllung abgeschlossen")

    return RollingBallResult(
        surface_points=surface_points,
        filled_voxel_count=filled_voxel_count,
        total_voxel_count=total_voxel_count,
        fill_fraction=fill_fraction,
        threshold=threshold,
        voxel_size=voxel_size,
        grid_resolution=grid_resolution,
    )


def export_distance_fill(result: RollingBallResult, path: Optional[Path] = None) -> Path:
    """Export the filled surface as an OBJ file."""

    export_path = path or ROLLING_EXPORT_PATH
    tmp_path = export_path.parent / f"{export_path.name}.tmp"
    with tmp_path.open("w", encoding="utf-8") as handle:
        handle.write("# Distance-fill surface export\n")
        handle.write(f"# threshold={result.threshold:.3f}\n")
        handle.write(f"# voxel_size={result.voxel_size:.3f}\n")
        handle.write(f"# fill_fraction={result.fill_fraction:.6f}\n")
        handle.write(f"# grid_resolution={result.grid_resolution}\n")
        for point in result.surface_points:
            handle.write(f"v {point[0]:.6f} {point[1]:.6f} {point[2]:.6f}\n")
    tmp_path.replace(export_path)
    return export_path


__all__ = ["run_distance_fill", "export_distance_fill", "extract_surface_points", "sphere_offsets"]
