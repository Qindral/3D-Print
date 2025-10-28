import json
import math
import multiprocessing as mp
import queue
import random
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pygame
from matplotlib.widgets import Button, TextBox
from scipy import ndimage


# Simulation constants
CUBE_SIZE = 100.0
ROD_LENGTH = 5.0
NUM_RODS = 5000
TIME_STEP = 0.05
TRANSLATION_SCALE = 2.5
ROTATION_SCALE = 0.7
CONNECTION_DISTANCE = 1.5
MAX_GROUP_SIZE = 10
BACKGROUND_COLOR = (15, 15, 25)
ROD_COLOR = (180, 220, 255)
CONNECTED_COLOR = (255, 150, 80)
TEXT_COLOR = (240, 240, 240)
BASE_ROD_THICKNESS = 1.0
CUBE_HALF = CUBE_SIZE / 2.0
CUBE_BOUNDING_RADIUS = CUBE_HALF * math.sqrt(3.0)
DEFAULT_CAMERA_DISTANCE = CUBE_SIZE * 2.2
FOV = max(math.radians(30.0), 2.0 * math.atan(CUBE_BOUNDING_RADIUS / DEFAULT_CAMERA_DISTANCE))
NEAR_PLANE = 5.0
FAR_PLANE = 1200.0
CUBE_VERTICES = [
    np.array([x, y, z])
    for x in (-CUBE_HALF, CUBE_HALF)
    for y in (-CUBE_HALF, CUBE_HALF)
    for z in (-CUBE_HALF, CUBE_HALF)
]
CUBE_EDGES = [
    (0, 1),
    (0, 2),
    (0, 4),
    (1, 3),
    (1, 5),
    (2, 3),
    (2, 6),
    (3, 7),
    (4, 5),
    (4, 6),
    (5, 7),
    (6, 7),
]

STATE_FILE_PATH = Path("rod_state.json")
ROLLING_EXPORT_PATH = Path("rolling_ball_surface.obj")


def random_unit_vector() -> np.ndarray:
    """Return a random unit vector uniformly distributed on the sphere."""

    phi = random.uniform(0.0, 2.0 * math.pi)
    costheta = random.uniform(-1.0, 1.0)
    sintheta = math.sqrt(1.0 - costheta * costheta)
    return np.array([math.cos(phi) * sintheta, math.sin(phi) * sintheta, costheta])


@dataclass
class Rod:
    center: np.ndarray
    orientation: np.ndarray
    connections: Dict[str, int] = field(default_factory=dict)

    def endpoints(self) -> Dict[str, np.ndarray]:
        offset = self.orientation * (ROD_LENGTH / 2.0)
        return {
            "A": self.center + offset,
            "B": self.center - offset,
        }

    def has_free_end(self, end: str) -> bool:
        return end not in self.connections

    def apply_translation(self, delta: np.ndarray) -> None:
        self.center += delta

    def apply_rotation(self, axis: np.ndarray, angle: float) -> None:
        if np.linalg.norm(axis) < 1e-6:
            return
        axis = axis / np.linalg.norm(axis)
        cos_theta = math.cos(angle)
        sin_theta = math.sin(angle)
        ux, uy, uz = axis
        rot = np.array(
            [
                [
                    cos_theta + ux * ux * (1 - cos_theta),
                    ux * uy * (1 - cos_theta) - uz * sin_theta,
                    ux * uz * (1 - cos_theta) + uy * sin_theta,
                ],
                [
                    uy * ux * (1 - cos_theta) + uz * sin_theta,
                    cos_theta + uy * uy * (1 - cos_theta),
                    uy * uz * (1 - cos_theta) - ux * sin_theta,
                ],
                [
                    uz * ux * (1 - cos_theta) - uy * sin_theta,
                    uz * uy * (1 - cos_theta) + ux * sin_theta,
                    cos_theta + uz * uz * (1 - cos_theta),
                ],
            ]
        )
        self.orientation = rot @ self.orientation
        self.orientation /= np.linalg.norm(self.orientation)

    def enforce_single_anchor(self, end: str, anchor: np.ndarray) -> None:
        direction = 1.0 if end == "A" else -1.0
        self.center = anchor - direction * self.orientation * (ROD_LENGTH / 2.0)

    def enforce_dual_anchors(self, anchor_a: np.ndarray, anchor_b: np.ndarray) -> None:
        direction = anchor_a - anchor_b
        norm = np.linalg.norm(direction)
        if norm < 1e-6:
            return
        direction /= norm
        self.orientation = direction
        midpoint = (anchor_a + anchor_b) / 2.0
        self.center = midpoint


@dataclass
class ConnectionGroup:
    members: List[Tuple[int, str]] = field(default_factory=list)


class ConnectionManager:
    def __init__(self) -> None:
        self.groups: Dict[int, ConnectionGroup] = {}
        self._next_id = 0

    def group_size(self, group_id: int) -> int:
        group = self.groups.get(group_id)
        if group is None:
            return 0
        return len(group.members)

    def is_full(self, group_id: Optional[int]) -> bool:
        if group_id is None:
            return False
        return self.group_size(group_id) >= MAX_GROUP_SIZE

    def create_group(self, member_a: Tuple[int, str], member_b: Tuple[int, str]) -> int:
        group_id = self._next_id
        self._next_id += 1
        self.groups[group_id] = ConnectionGroup(members=[member_a, member_b])
        return group_id

    def add_member(self, group_id: int, member: Tuple[int, str]) -> None:
        group = self.groups.get(group_id)
        if group is None:
            return
        if len(group.members) >= MAX_GROUP_SIZE:
            return
        if member not in group.members:
            group.members.append(member)

    def merge_groups(self, target_id: int, source_id: int, rods: List[Rod]) -> None:
        if target_id == source_id:
            return
        target = self.groups.get(target_id)
        if target is None:
            return
        source = self.groups.pop(source_id, None)
        if source is None:
            return
        if len(target.members) + len(source.members) > MAX_GROUP_SIZE:
            self.groups[source_id] = source
            return
        for member in source.members:
            rod_idx, end = member
            rods[rod_idx].connections[end] = target_id
            if member not in target.members:
                target.members.append(member)

    def anchors(self, rods: List[Rod]) -> Dict[int, np.ndarray]:
        anchor_map: Dict[int, np.ndarray] = {}
        for group_id, group in self.groups.items():
            if not group.members:
                continue
            accumulator = np.zeros(3)
            for rod_idx, end in group.members:
                accumulator += rods[rod_idx].endpoints()[end]
            anchor_map[group_id] = accumulator / len(group.members)
        return anchor_map

    def next_group_id(self) -> int:
        return self._next_id


def create_rods(num_rods: int) -> List[Rod]:
    rods = []
    for _ in range(num_rods):
        center = np.array(
            [
                random.uniform(-CUBE_SIZE / 2.0, CUBE_SIZE / 2.0),
                random.uniform(-CUBE_SIZE / 2.0, CUBE_SIZE / 2.0),
                random.uniform(-CUBE_SIZE / 2.0, CUBE_SIZE / 2.0),
            ]
        )
        orientation = random_unit_vector()
        rods.append(Rod(center=center, orientation=orientation))
    return rods


def random_displacement(scale: float) -> np.ndarray:
    return np.random.normal(scale=scale, size=3)


def random_rotation() -> Tuple[np.ndarray, float]:
    axis = random_unit_vector()
    angle = np.random.normal(scale=ROTATION_SCALE)
    return axis, angle


def clamp_to_cube(center: np.ndarray) -> np.ndarray:
    half = CUBE_SIZE / 2.0
    return np.clip(center, -half, half)


def distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def attempt_connections(rods: List[Rod], manager: ConnectionManager) -> None:
    cell_size = CONNECTION_DISTANCE * 2.0
    inv_cell = 1.0 / cell_size
    half = CUBE_SIZE / 2.0

    endpoints: List[Tuple[int, str, Tuple[float, float, float], Tuple[int, int, int]]] = []
    grid: Dict[Tuple[int, int, int], List[int]] = {}

    for idx, rod in enumerate(rods):
        rod_endpoints = rod.endpoints()
        for end, pos in rod_endpoints.items():
            pos_tuple = (float(pos[0]), float(pos[1]), float(pos[2]))
            cell = (
                int((pos_tuple[0] + half) * inv_cell),
                int((pos_tuple[1] + half) * inv_cell),
                int((pos_tuple[2] + half) * inv_cell),
            )
            entry_index = len(endpoints)
            endpoints.append((idx, end, pos_tuple, cell))
            grid.setdefault(cell, []).append(entry_index)

    order = list(range(len(endpoints)))
    random.shuffle(order)
    max_dist_sq = CONNECTION_DISTANCE * CONNECTION_DISTANCE

    for idx_i in order:
        rod_i, end_i, pos_i, cell_i = endpoints[idx_i]
        group_i = rods[rod_i].connections.get(end_i)
        if manager.is_full(group_i):
            continue

        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    neighbor_cell = (
                        cell_i[0] + dx,
                        cell_i[1] + dy,
                        cell_i[2] + dz,
                    )
                    candidates = grid.get(neighbor_cell)
                    if not candidates:
                        continue
                    for idx_j in candidates:
                        if idx_j == idx_i or idx_j < idx_i:
                            continue
                        rod_j, end_j, pos_j, _ = endpoints[idx_j]
                        if rod_i == rod_j or end_i == end_j:
                            continue

                        group_j = rods[rod_j].connections.get(end_j)
                        if manager.is_full(group_j):
                            continue
                        if group_i is not None and group_j is not None and group_i == group_j:
                            continue

                        dxp = pos_i[0] - pos_j[0]
                        dyp = pos_i[1] - pos_j[1]
                        dzp = pos_i[2] - pos_j[2]
                        dist_sq = dxp * dxp + dyp * dyp + dzp * dzp
                        if dist_sq > max_dist_sq:
                            continue

                        if group_i is None and group_j is None:
                            group_id = manager.create_group((rod_i, end_i), (rod_j, end_j))
                            rods[rod_i].connections[end_i] = group_id
                            rods[rod_j].connections[end_j] = group_id
                            break
                        if group_i is not None and group_j is None:
                            if manager.is_full(group_i):
                                continue
                            manager.add_member(group_i, (rod_j, end_j))
                            rods[rod_j].connections[end_j] = group_i
                            break
                        if group_i is None and group_j is not None:
                            if manager.is_full(group_j):
                                continue
                            manager.add_member(group_j, (rod_i, end_i))
                            rods[rod_i].connections[end_i] = group_j
                            break
                        if group_i is not None and group_j is not None and group_i != group_j:
                            combined = manager.group_size(group_i) + manager.group_size(group_j)
                            if combined > MAX_GROUP_SIZE:
                                continue
                            target, source = (group_i, group_j)
                            if source < target:
                                target, source = source, target
                            manager.merge_groups(target, source, rods)
                            break
                    else:
                        continue
                    break
                else:
                    continue
                break


def enforce_connections(rods: List[Rod], manager: ConnectionManager) -> None:
    anchor_map = manager.anchors(rods)
    for idx, rod in enumerate(rods):
        group_a = rod.connections.get("A")
        group_b = rod.connections.get("B")
        anchor_a = anchor_map.get(group_a) if group_a is not None else None
        anchor_b = anchor_map.get(group_b) if group_b is not None else None

        if anchor_a is not None and anchor_b is not None and group_a != group_b:
            rod.enforce_dual_anchors(anchor_a, anchor_b)
        elif anchor_a is not None:
            rod.enforce_single_anchor("A", anchor_a)
        elif anchor_b is not None:
            rod.enforce_single_anchor("B", anchor_b)


def update_rods(rods: List[Rod], manager: ConnectionManager) -> None:
    for rod in rods:
        delta = random_displacement(TRANSLATION_SCALE) * TIME_STEP
        rod.apply_translation(delta)
        axis, angle = random_rotation()
        rod.apply_rotation(axis, angle * TIME_STEP)
        rod.center = clamp_to_cube(rod.center)

    enforce_connections(rods, manager)
    attempt_connections(rods, manager)
    enforce_connections(rods, manager)


def project_point(
    point: np.ndarray,
    view_matrix: np.ndarray,
    screen_size: Tuple[int, int],
    camera_position: np.ndarray,
) -> Tuple[Optional[Tuple[int, int]], float]:
    relative = point - camera_position
    view = view_matrix @ relative
    depth = float(view[2])
    if depth <= NEAR_PLANE or depth >= FAR_PLANE:
        return (None, depth)
    width, height = screen_size
    aspect = width / height if height else 1.0
    f = 1.0 / math.tan(FOV / 2.0)
    x_ndc = (view[0] * f / aspect) / depth
    y_ndc = (view[1] * f) / depth
    screen_x = int((x_ndc + 1.0) * 0.5 * width)
    screen_y = int((1.0 - y_ndc) * 0.5 * height)
    return (screen_x, screen_y), depth


RenderRod = Tuple[Tuple[float, float, float], Tuple[float, float, float], bool]


@dataclass
class CameraState:
    position: np.ndarray
    yaw: float
    pitch: float
    move_speed: float = 120.0
    mouse_sensitivity: float = 0.005

    def rotation_matrix(self) -> np.ndarray:
        cos_y = math.cos(self.yaw)
        sin_y = math.sin(self.yaw)
        cos_p = math.cos(self.pitch)
        sin_p = math.sin(self.pitch)

        rot_yaw = np.array(
            [
                [cos_y, 0.0, sin_y],
                [0.0, 1.0, 0.0],
                [-sin_y, 0.0, cos_y],
            ]
        )
        rot_pitch = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, cos_p, -sin_p],
                [0.0, sin_p, cos_p],
            ]
        )
        return rot_pitch @ rot_yaw

    def forward(self) -> np.ndarray:
        cos_p = math.cos(self.pitch)
        sin_p = math.sin(self.pitch)
        cos_y = math.cos(self.yaw)
        sin_y = math.sin(self.yaw)
        return np.array([sin_y * cos_p, -sin_p, cos_y * cos_p])

    def right(self) -> np.ndarray:
        cos_y = math.cos(self.yaw)
        sin_y = math.sin(self.yaw)
        return np.array([cos_y, 0.0, -sin_y])

    def up(self) -> np.ndarray:
        up_vector = np.cross(self.right(), self.forward())
        norm = np.linalg.norm(up_vector)
        if norm < 1e-6:
            return np.array([0.0, 1.0, 0.0])
        return up_vector / norm


def depth_factor_from_distance(depth: float) -> float:
    return min(max((depth - NEAR_PLANE) / (FAR_PLANE - NEAR_PLANE), 0.0), 1.0)


def shade_color(base: Tuple[int, int, int], depth: float) -> Tuple[int, int, int]:
    factor = depth_factor_from_distance(depth)
    brightness = 1.0 - 0.6 * factor
    return tuple(max(0, min(255, int(c * brightness))) for c in base)


def thickness_for_depth(depth: float) -> int:
    factor = depth_factor_from_distance(depth)
    thickness = BASE_ROD_THICKNESS * (1.0 - 0.7 * factor)
    return max(1, int(thickness))


@dataclass
class SimulationStatePayload:
    centers: List[Tuple[float, float, float]]
    orientations: List[Tuple[float, float, float]]
    connections: List[Dict[str, int]]
    groups: Dict[int, List[Tuple[int, str]]]
    next_group_id: int


@dataclass
class SimulationSnapshot:
    render_data: List[RenderRod]
    free_end_count: int
    largest_cluster_percent: float
    state_payload: Optional[SimulationStatePayload] = None


@dataclass
class RollingBallResult:
    surface_points: np.ndarray
    filled_voxel_count: int
    total_voxel_count: int
    fill_fraction: float
    threshold: float
    voxel_size: float
    grid_resolution: int


@dataclass
class HeadlessUIState:
    abort_requested: bool = False
    distance_threshold: float = 1.5
    grid_resolution: int = 256
    rolling_result: Optional[RollingBallResult] = None
    launch_rolling_viewer: bool = False
    latest_snapshot: Optional[SimulationSnapshot] = None
    status_message: str = "Bereit"


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


def run_rolling_ball_simulation(
    render_data: List[RenderRod],
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

    if not render_data:
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
    # behandle die Rods als ausgefülltes Medium, indem jede Probe um den tatsächlichen Radius dilatiert wird
    thickness_radius = max(1, int(math.ceil((BASE_ROD_THICKNESS * 0.5) / max(voxel_size, 1e-6))))
    thickness_offsets = sphere_offsets(thickness_radius)

    report(0.0, f"Voxelisiere Rods ({grid_resolution}³)")
    total_rods = len(render_data)
    next_progress_update = 0.05
    for index, (point_a, point_b, _) in enumerate(render_data):
        start = np.array(point_a, dtype=float)
        end = np.array(point_b, dtype=float)
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


def export_rolling_ball_result(
    result: RollingBallResult, path: Path = ROLLING_EXPORT_PATH
) -> Path:
    tmp_path = path.parent / f"{path.name}.tmp"
    with tmp_path.open("w", encoding="utf-8") as handle:
        handle.write("# Distance-fill surface export\n")
        handle.write(f"# threshold={result.threshold:.3f}\n")
        handle.write(f"# voxel_size={result.voxel_size:.3f}\n")
        handle.write(f"# fill_fraction={result.fill_fraction:.6f}\n")
        handle.write(f"# grid_resolution={result.grid_resolution}\n")
        for point in result.surface_points:
            handle.write(f"v {point[0]:.6f} {point[1]:.6f} {point[2]:.6f}\n")
    tmp_path.replace(path)
    return path


def compute_metrics(rods: List[Rod], manager: ConnectionManager) -> Tuple[int, float]:
    free_ends = 0
    adjacency: List[set[int]] = [set() for _ in rods]

    for idx, rod in enumerate(rods):
        if "A" not in rod.connections:
            free_ends += 1
        if "B" not in rod.connections:
            free_ends += 1

    for group in manager.groups.values():
        if not group.members:
            continue
        rod_indices = {member[0] for member in group.members}
        if len(rod_indices) <= 1:
            continue
        for rod_idx in rod_indices:
            adjacency[rod_idx].update(rod_indices - {rod_idx})

    visited = set()
    largest_component = 0
    for idx in range(len(rods)):
        if idx in visited:
            continue
        stack = [idx]
        component_size = 0
        while stack:
            node = stack.pop()
            if node in visited:
                continue
            visited.add(node)
            component_size += 1
            stack.extend(neigh for neigh in adjacency[node] if neigh not in visited)
        largest_component = max(largest_component, component_size)

    percent = (largest_component / len(rods) * 100.0) if rods else 0.0
    return free_ends, percent


def build_snapshot(
    rods: List[Rod], manager: ConnectionManager, include_state: bool = False
) -> SimulationSnapshot:
    render_data: List[RenderRod] = []
    for rod in rods:
        endpoints = rod.endpoints()
        render_data.append(
            (
                tuple(float(v) for v in endpoints["A"]),
                tuple(float(v) for v in endpoints["B"]),
                bool(rod.connections),
            )
        )
    free_ends, largest_percent = compute_metrics(rods, manager)
    state_payload: Optional[SimulationStatePayload] = None
    if include_state:
        centers = [tuple(float(value) for value in rod.center) for rod in rods]
        orientations = [tuple(float(value) for value in rod.orientation) for rod in rods]
        connections: List[Dict[str, int]] = []
        for rod in rods:
            conn_dict = {end: int(group) for end, group in rod.connections.items()}
            connections.append(conn_dict)
        groups: Dict[int, List[Tuple[int, str]]] = {}
        for group_id, group in manager.groups.items():
            groups[int(group_id)] = [(int(idx), end) for idx, end in group.members]
        state_payload = SimulationStatePayload(
            centers=centers,
            orientations=orientations,
            connections=connections,
            groups=groups,
            next_group_id=manager.next_group_id(),
        )
    return SimulationSnapshot(
        render_data=render_data,
        free_end_count=free_ends,
        largest_cluster_percent=largest_percent,
        state_payload=state_payload,
    )


def collect_cube_segments(
    view_matrix: np.ndarray,
    screen_size: Tuple[int, int],
    camera: CameraState,
) -> List[Tuple[Tuple[int, int], Tuple[int, int], float]]:
    segments: List[Tuple[Tuple[int, int], Tuple[int, int], float]] = []
    for start_idx, end_idx in CUBE_EDGES:
        start_proj = project_point(CUBE_VERTICES[start_idx], view_matrix, screen_size, camera.position)
        end_proj = project_point(CUBE_VERTICES[end_idx], view_matrix, screen_size, camera.position)
        if start_proj[0] is None or end_proj[0] is None:
            continue
        depth = max(start_proj[1], end_proj[1])
        segments.append((start_proj[0], end_proj[0], depth))
    segments.sort(key=lambda item: item[2], reverse=True)
    return segments


def save_simulation_state(
    snapshot: SimulationSnapshot, path: Path = STATE_FILE_PATH
) -> Optional[Path]:
    payload = snapshot.state_payload
    if payload is None:
        return None
    groups_serializable = {
        str(group_id): [[int(idx), end] for idx, end in members]
        for group_id, members in payload.groups.items()
    }
    data = {
        "centers": payload.centers,
        "orientations": payload.orientations,
        "connections": payload.connections,
        "groups": groups_serializable,
        "next_group_id": payload.next_group_id,
        "timestamp": time.time(),
        "metadata": {
            "rod_length": ROD_LENGTH,
            "cube_size": CUBE_SIZE,
            "max_group_size": MAX_GROUP_SIZE,
        },
    }
    tmp_path = path.parent / f"{path.name}.tmp"
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)
    tmp_path.replace(path)
    return path


def load_simulation_state(
    path: Path = STATE_FILE_PATH,
) -> Optional[Tuple[List[Rod], ConnectionManager]]:
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None

    centers = data.get("centers")
    orientations = data.get("orientations")
    connections = data.get("connections")
    groups_data = data.get("groups", {})
    next_group_id = int(data.get("next_group_id", 0))

    if not isinstance(centers, list) or not isinstance(orientations, list):
        return None
    if len(centers) != len(orientations):
        return None
    if connections is None:
        connections = [{} for _ in centers]

    rods: List[Rod] = []
    for idx, (center_values, orientation_values) in enumerate(zip(centers, orientations)):
        center_array = np.array(center_values, dtype=float)
        center_array = clamp_to_cube(center_array)
        orientation_array = np.array(orientation_values, dtype=float)
        if np.linalg.norm(orientation_array) < 1e-6:
            orientation_array = random_unit_vector()
        else:
            orientation_array = orientation_array / np.linalg.norm(orientation_array)
        rod = Rod(center=center_array, orientation=orientation_array)
        conn_dict = connections[idx] if idx < len(connections) else {}
        for end, group_id in conn_dict.items():
            try:
                rod.connections[end] = int(group_id)
            except (TypeError, ValueError):
                continue
        rods.append(rod)

    manager = ConnectionManager()
    manager.groups = {}
    for key, members in groups_data.items():
        try:
            group_id = int(key)
        except (TypeError, ValueError):
            continue
        group_members: List[Tuple[int, str]] = []
        if isinstance(members, list):
            for entry in members:
                if (
                    isinstance(entry, (list, tuple))
                    and len(entry) == 2
                    and isinstance(entry[0], (int, float))
                    and isinstance(entry[1], str)
                ):
                    rod_idx = int(entry[0])
                    if 0 <= rod_idx < len(rods):
                        group_members.append((rod_idx, entry[1]))
        manager.groups[group_id] = ConnectionGroup(members=group_members)

    if manager.groups:
        max_existing = max(manager.groups.keys())
        next_group_id = max(next_group_id, max_existing + 1)
    manager._next_id = next_group_id

    return rods, manager


def wait_for_state_snapshot(
    state_queue: mp.Queue,
    current_snapshot: SimulationSnapshot,
    timeout: float = 5.0,
) -> SimulationSnapshot:
    deadline = time.perf_counter() + timeout
    snapshot = current_snapshot
    while time.perf_counter() < deadline:
        if snapshot.state_payload is not None:
            break
        try:
            snapshot = state_queue.get(timeout=0.2)
        except queue.Empty:
            continue
    return snapshot


def draw_scene(
    screen: pygame.Surface,
    render_data: List[RenderRod],
    camera: CameraState,
    overlay_text: str | None = None,
    font: pygame.font.Font | None = None,
) -> None:
    screen.fill(BACKGROUND_COLOR)
    screen_size = screen.get_size()
    view_matrix = camera.rotation_matrix()

    for start, end, depth in collect_cube_segments(view_matrix, screen_size, camera):
        color = shade_color((90, 90, 130), depth)
        pygame.draw.line(screen, color, start, end, 1)

    rod_segments: List[Tuple[Tuple[int, int], Tuple[int, int], float, Tuple[int, int, int], int]] = []
    for point_a_3d, point_b_3d, connected in render_data:
        proj_a = project_point(np.array(point_a_3d), view_matrix, screen_size, camera.position)
        proj_b = project_point(np.array(point_b_3d), view_matrix, screen_size, camera.position)
        if proj_a[0] is None or proj_b[0] is None:
            continue
        avg_depth = (proj_a[1] + proj_b[1]) / 2.0
        base_color = CONNECTED_COLOR if connected else ROD_COLOR
        color = shade_color(base_color, avg_depth)
        thickness = thickness_for_depth(avg_depth)
        rod_segments.append((proj_a[0], proj_b[0], avg_depth, color, thickness))

    rod_segments.sort(key=lambda item: item[2], reverse=True)
    for start, end, _, color, thickness in rod_segments:
        pygame.draw.line(screen, color, start, end, thickness)

    if overlay_text and font:
        text_surface = font.render(overlay_text, True, TEXT_COLOR)
        screen.blit(text_surface, (10, 10))
    pygame.display.flip()


def draw_voxel_scene(
    screen: pygame.Surface,
    surface_points: np.ndarray,
    camera: CameraState,
    overlay_text: str | None = None,
    font: pygame.font.Font | None = None,
) -> None:
    screen.fill(BACKGROUND_COLOR)
    screen_size = screen.get_size()
    view_matrix = camera.rotation_matrix()

    for start, end, depth in collect_cube_segments(view_matrix, screen_size, camera):
        color = shade_color((90, 90, 130), depth)
        pygame.draw.line(screen, color, start, end, 1)

    point_entries: List[Tuple[float, Tuple[int, int], Tuple[int, int, int], int]] = []
    for point in surface_points:
        proj = project_point(np.array(point), view_matrix, screen_size, camera.position)
        screen_pos, depth = proj
        if screen_pos is None:
            continue
        color = shade_color(CONNECTED_COLOR, depth)
        radius = max(1, int(BASE_ROD_THICKNESS * 0.4 * (1.0 - depth_factor_from_distance(depth))))
        point_entries.append((depth, screen_pos, color, radius))

    point_entries.sort(key=lambda item: item[0], reverse=True)
    for depth, screen_pos, color, radius in point_entries:
        pygame.draw.circle(screen, color, screen_pos, radius)

    if overlay_text and font:
        text_surface = font.render(overlay_text, True, TEXT_COLOR)
        screen.blit(text_surface, (10, 10))
    pygame.display.flip()


def simulation_worker(state_queue: mp.Queue, stop_event: mp.Event) -> None:
    loaded_state = load_simulation_state()
    if loaded_state is not None:
        rods, manager = loaded_state
        print(
            f"Lade gespeicherten Zustand mit {len(rods)} Rods aus {STATE_FILE_PATH}",
            flush=True,
        )
        if len(rods) != NUM_RODS:
            print(
                f"Hinweis: gespeicherter Zustand enthält {len(rods)} Rods (Standard {NUM_RODS})",
                flush=True,
            )
    else:
        rods = create_rods(NUM_RODS)
        manager = ConnectionManager()
    try:
        while not stop_event.is_set():
            update_rods(rods, manager)
            try:
                state_queue.put_nowait(build_snapshot(rods, manager))
            except queue.Full:
                pass
    finally:
        final_snapshot = build_snapshot(rods, manager, include_state=True)
        state_queue.put(final_snapshot)


def run_headless_phase(
    state_queue: mp.Queue,
    stop_event: mp.Event,
    current_snapshot: SimulationSnapshot,
) -> Tuple[SimulationSnapshot, HeadlessUIState]:
    plt.ion()
    fig, (ax_free, ax_cluster) = plt.subplots(2, 1, figsize=(7, 7), sharex=True)
    fig.subplots_adjust(bottom=0.28)
    ax_free.set_ylabel("Free Ends")
    ax_cluster.set_ylabel("Largest Cluster %")
    ax_cluster.set_xlabel("Simulation Samples")
    ax_free.grid(True, alpha=0.3)
    ax_cluster.grid(True, alpha=0.3)
    line_free, = ax_free.plot([], [], color="tab:blue", label="Free Ends")
    line_cluster, = ax_cluster.plot([], [], color="tab:orange", label="Largest %")
    ax_free.legend(loc="upper right")
    ax_cluster.legend(loc="upper right")
    fig.tight_layout(rect=(0.02, 0.28, 0.98, 0.98))

    ui_state = HeadlessUIState(latest_snapshot=current_snapshot)

    status_text = fig.text(0.02, 0.02, f"Status: {ui_state.status_message}")

    def update_status(message: str) -> None:
        ui_state.status_message = message
        status_text.set_text(f"Status: {message}")
        fig.canvas.draw_idle()

    threshold_ax = fig.add_axes([0.1, 0.18, 0.2, 0.05])
    threshold_box = TextBox(
        threshold_ax,
        "Schwelle",
        initial=f"{ui_state.distance_threshold:.1f}",
    )

    def handle_threshold_submit(text: str) -> None:
        try:
            value = float(text)
            if value < 0:
                raise ValueError
        except ValueError:
            update_status("Ungültige Schwelle – bitte eine nichtnegative Zahl eingeben")
            threshold_box.set_val(f"{ui_state.distance_threshold:.1f}")
            return
        ui_state.distance_threshold = value
        update_status(f"Schwellenwert gesetzt auf {value:.2f}")

    threshold_box.on_submit(handle_threshold_submit)

    resolution_ax = fig.add_axes([0.1, 0.12, 0.2, 0.05])
    resolution_box = TextBox(
        resolution_ax,
        "Voxel-Kante",
        initial=str(ui_state.grid_resolution),
    )

    def handle_resolution_submit(text: str) -> None:
        try:
            value = int(text)
            if value <= 0:
                raise ValueError
        except ValueError:
            update_status("Ungültige Voxelzahl – bitte eine positive ganze Zahl eingeben")
            resolution_box.set_val(str(ui_state.grid_resolution))
            return

        ui_state.grid_resolution = value
        voxels = value**3
        estimated_bytes = voxels  # bool array ~1 byte pro Voxel
        estimated_gib = estimated_bytes / (1024**3)
        if estimated_gib >= 1.0:
            update_status(
                f"Auflösung gesetzt auf {value}³ (~{estimated_gib:.1f} GiB Speicher)"
            )
        else:
            update_status(
                f"Auflösung gesetzt auf {value}³ (~{estimated_bytes / (1024**2):.1f} MiB Speicher)"
            )

    resolution_box.on_submit(handle_resolution_submit)

    initial_voxels = ui_state.grid_resolution**3
    initial_gib = initial_voxels / (1024**3)
    memory_hint = (
        f"~{initial_gib:.1f} GiB"
        if initial_gib >= 1.0
        else f"~{initial_voxels / (1024**2):.1f} MiB"
    )
    update_status(
        "Bereit – "
        f"Schwelle {ui_state.distance_threshold:.2f}, Auflösung {ui_state.grid_resolution}³ ({memory_hint})"
    )

    def perform_distance_fill() -> bool:
        snapshot = ui_state.latest_snapshot or current_snapshot
        if snapshot is None:
            update_status("Keine Snapshot-Daten verfügbar")
            return False
        if not snapshot.render_data:
            update_status("Snapshot enthält keine Rod-Geometrie")
            return False
        update_status(
            "Fülle basierend auf Distanzkarte … "
            f"(Auflösung {ui_state.grid_resolution}³)"
        )

        def progress_callback(progress: float, message: str) -> None:
            update_status(f"{message} – {progress * 100:.0f}%")
            fig.canvas.draw_idle()
            fig.canvas.flush_events()

        fig.canvas.draw()
        result = run_rolling_ball_simulation(
            snapshot.render_data,
            ui_state.distance_threshold,
            grid_resolution=ui_state.grid_resolution,
            progress_cb=progress_callback,
        )
        ui_state.rolling_result = result
        filled_percent = result.fill_fraction * 100.0
        update_status(
            "Distanzfüllung fertig – "
            f"{result.filled_voxel_count} von {result.total_voxel_count} Voxeln "
            f"({filled_percent:.2f}%) bei {result.grid_resolution}³"
        )
        return True

    roll_ax = fig.add_axes([0.35, 0.17, 0.25, 0.06])
    roll_button = Button(roll_ax, "Distanzfüllung ausführen")

    def handle_roll_click(_: object) -> None:
        perform_distance_fill()

    roll_button.on_clicked(handle_roll_click)

    view_ax = fig.add_axes([0.64, 0.17, 0.26, 0.06])
    view_button = Button(view_ax, "3D-Ansicht anzeigen")

    def handle_view_click(_: object) -> None:
        if ui_state.rolling_result is None:
            if not perform_distance_fill():
                update_status("Distanzfüllung konnte nicht erzeugt werden")
                return
        ui_state.launch_rolling_viewer = True
        update_status("3D-Ansicht wird geöffnet")
        plt.close(fig)

    view_button.on_clicked(handle_view_click)

    export_ax = fig.add_axes([0.64, 0.08, 0.26, 0.06])
    export_button = Button(export_ax, "Distanzfüllung exportieren")

    def handle_export_click(_: object) -> None:
        if ui_state.rolling_result is None:
            update_status("Bitte zuerst die Distanzfüllung ausführen")
            return
        export_path = export_rolling_ball_result(ui_state.rolling_result)
        update_status(f"Export gespeichert in {export_path}")

    export_button.on_clicked(handle_export_click)

    abort_ax = fig.add_axes([0.1, 0.08, 0.2, 0.06])
    abort_button = Button(abort_ax, "Simulation abbrechen", color="#c94c4c", hovercolor="#d96c6c")

    def handle_abort_click(_: object) -> None:
        ui_state.abort_requested = True
        update_status("Abbruch angefordert – schließe Simulation")
        stop_event.set()
        plt.close(fig)

    abort_button.on_clicked(handle_abort_click)

    history_indices: deque[int] = deque(maxlen=600)
    history_free: deque[int] = deque(maxlen=600)
    history_cluster: deque[float] = deque(maxlen=600)
    sample_index = 0
    last_chart_update = 0.0

    while plt.fignum_exists(fig.number) and not stop_event.is_set():
        updated = False
        try:
            while True:
                current_snapshot = state_queue.get_nowait()
                ui_state.latest_snapshot = current_snapshot
                updated = True
        except queue.Empty:
            pass

        if updated:
            history_indices.append(sample_index)
            history_free.append(current_snapshot.free_end_count)
            history_cluster.append(current_snapshot.largest_cluster_percent)
            sample_index += 1

        now = time.perf_counter()
        if updated and now - last_chart_update >= 0.05:
            x_data = list(history_indices)
            line_free.set_data(x_data, list(history_free))
            line_cluster.set_data(x_data, list(history_cluster))
            if x_data:
                ax_free.set_xlim(
                    x_data[0], x_data[-1] if x_data[-1] > x_data[0] else x_data[0] + 1
                )
            ax_free.relim()
            ax_free.autoscale_view()
            ax_cluster.relim()
            ax_cluster.autoscale_view()
            fig.canvas.draw_idle()
            fig.canvas.flush_events()
            last_chart_update = now

        plt.pause(0.001)

    plt.ioff()
    if plt.fignum_exists(fig.number):
        plt.close(fig)

    return current_snapshot, ui_state


def run_interactive_viewer(
    state_queue: mp.Queue,
    stop_event: mp.Event,
    worker: mp.Process,
    current_snapshot: SimulationSnapshot,
) -> SimulationSnapshot:
    screen = pygame.display.set_mode((960, 720))
    pygame.display.set_caption("Rod Brownian Motion Simulation")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("consolas", 20)

    initial_yaw = math.radians(45)
    initial_pitch = math.radians(20)
    camera = CameraState(position=np.zeros(3), yaw=initial_yaw, pitch=initial_pitch)
    camera.position = -camera.forward() * DEFAULT_CAMERA_DISTANCE
    mouse_rotating = False
    last_caption_update = 0
    update_fps_display = 0.0
    last_update_tick: int | None = None

    running = True
    while running:
        dt_ms = clock.tick(60)
        render_fps = 1000.0 / dt_ms if dt_ms > 0 else 0.0
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                mouse_rotating = True
                pygame.mouse.get_rel()
                pygame.event.set_grab(True)
                pygame.mouse.set_visible(False)
            elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                mouse_rotating = False
                pygame.event.set_grab(False)
                pygame.mouse.set_visible(True)

        received_update = False
        try:
            while True:
                current_snapshot = state_queue.get_nowait()
                received_update = True
        except queue.Empty:
            pass

        if mouse_rotating:
            delta_x, delta_y = pygame.mouse.get_rel()
            camera.yaw += delta_x * camera.mouse_sensitivity
            camera.pitch -= delta_y * camera.mouse_sensitivity
            camera.pitch = max(min(camera.pitch, math.radians(89.0)), math.radians(-89.0))
            camera.yaw = (camera.yaw + math.pi) % (2 * math.pi) - math.pi
        else:
            pygame.mouse.get_rel()

        keys = pygame.key.get_pressed()
        move_direction = np.zeros(3)
        if keys[pygame.K_w]:
            move_direction += camera.forward()
        if keys[pygame.K_s]:
            move_direction -= camera.forward()
        if keys[pygame.K_d]:
            move_direction += camera.right()
        if keys[pygame.K_a]:
            move_direction -= camera.right()
        if keys[pygame.K_SPACE] or keys[pygame.K_r]:
            move_direction += camera.up()
        if keys[pygame.K_LCTRL] or keys[pygame.K_f]:
            move_direction -= camera.up()

        if np.linalg.norm(move_direction) > 1e-6:
            move_direction = move_direction / np.linalg.norm(move_direction)
            speed_multiplier = 2.5 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 1.0
            camera.position += (
                move_direction * camera.move_speed * speed_multiplier * (dt_ms / 1000.0)
            )

        if received_update:
            now_ticks = pygame.time.get_ticks()
            if last_update_tick is not None:
                delta = now_ticks - last_update_tick
                if delta > 0:
                    update_fps_display = 1000.0 / delta
            last_update_tick = now_ticks

        draw_scene(
            screen,
            current_snapshot.render_data,
            camera,
            overlay_text=(
                "Render FPS: "
                f"{render_fps:5.1f} | Sim updates/s: {update_fps_display:5.1f} | "
                f"Free ends: {current_snapshot.free_end_count:4d} | Largest cluster: {current_snapshot.largest_cluster_percent:5.1f}%"
            ),
            font=font,
        )

        now_ticks = pygame.time.get_ticks()
        if now_ticks - last_caption_update >= 250:
            pygame.display.set_caption(
                "Rod Brownian Motion Simulation - "
                f"Render FPS: {render_fps:5.1f} | Sim updates/s: {update_fps_display:5.1f}"
            )
            last_caption_update = now_ticks

    stop_event.set()
    current_snapshot = wait_for_state_snapshot(state_queue, current_snapshot)
    worker.join(timeout=2.0)
    if worker.is_alive():
        worker.terminate()

    return current_snapshot


def run_rolling_ball_viewer(result: RollingBallResult) -> None:
    screen = pygame.display.set_mode((960, 720))
    pygame.display.set_caption("Distanzfüllung-Organik")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("consolas", 20)

    initial_yaw = math.radians(45)
    initial_pitch = math.radians(20)
    camera = CameraState(position=np.zeros(3), yaw=initial_yaw, pitch=initial_pitch)
    camera.position = -camera.forward() * DEFAULT_CAMERA_DISTANCE
    mouse_rotating = False

    running = True
    while running:
        dt_ms = clock.tick(60)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                mouse_rotating = True
                pygame.mouse.get_rel()
                pygame.event.set_grab(True)
                pygame.mouse.set_visible(False)
            elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                mouse_rotating = False
                pygame.event.set_grab(False)
                pygame.mouse.set_visible(True)

        if mouse_rotating:
            delta_x, delta_y = pygame.mouse.get_rel()
            camera.yaw += delta_x * camera.mouse_sensitivity
            camera.pitch -= delta_y * camera.mouse_sensitivity
            camera.pitch = max(min(camera.pitch, math.radians(89.0)), math.radians(-89.0))
            camera.yaw = (camera.yaw + math.pi) % (2 * math.pi) - math.pi
        else:
            pygame.mouse.get_rel()

        keys = pygame.key.get_pressed()
        move_direction = np.zeros(3)
        if keys[pygame.K_w]:
            move_direction += camera.forward()
        if keys[pygame.K_s]:
            move_direction -= camera.forward()
        if keys[pygame.K_d]:
            move_direction += camera.right()
        if keys[pygame.K_a]:
            move_direction -= camera.right()
        if keys[pygame.K_SPACE] or keys[pygame.K_r]:
            move_direction += camera.up()
        if keys[pygame.K_LCTRL] or keys[pygame.K_f]:
            move_direction -= camera.up()

        if np.linalg.norm(move_direction) > 1e-6:
            move_direction = move_direction / np.linalg.norm(move_direction)
            speed_multiplier = 2.5 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 1.0
            camera.position += move_direction * camera.move_speed * speed_multiplier * (dt_ms / 1000.0)

        overlay_text = (
            f"Schwelle: {result.threshold:.2f} | Füllung: {result.fill_fraction * 100.0:5.2f}% | "
            f"Auflösung: {result.grid_resolution}³ (Voxel {result.voxel_size:.2f}) | "
            f"Oberflächenpunkte: {len(result.surface_points)}"
        )

        draw_voxel_scene(screen, result.surface_points, camera, overlay_text=overlay_text, font=font)

    pygame.event.set_grab(False)
    pygame.mouse.set_visible(True)


def run_simulation() -> None:
    ctx = mp.get_context("spawn")
    state_queue: mp.Queue = ctx.Queue(maxsize=2)
    stop_event = ctx.Event()
    worker = ctx.Process(target=simulation_worker, args=(state_queue, stop_event))
    worker.start()

    try:
        current_snapshot: SimulationSnapshot = state_queue.get(timeout=5.0)
    except queue.Empty:
        current_snapshot = SimulationSnapshot(render_data=[], free_end_count=0, largest_cluster_percent=0.0)

    current_snapshot, ui_state = run_headless_phase(state_queue, stop_event, current_snapshot)

    if ui_state.abort_requested:
        stop_event.set()
        current_snapshot = wait_for_state_snapshot(state_queue, current_snapshot)
        worker.join(timeout=2.0)
        if worker.is_alive():
            worker.terminate()
        saved_path = save_simulation_state(current_snapshot)
        if saved_path is not None:
            print(f"Simulationzustand in {saved_path} gespeichert", flush=True)
        return

    pygame.init()

    if ui_state.launch_rolling_viewer and ui_state.rolling_result is not None:
        run_rolling_ball_viewer(ui_state.rolling_result)

    current_snapshot = run_interactive_viewer(state_queue, stop_event, worker, current_snapshot)
    saved_path = save_simulation_state(current_snapshot)
    if saved_path is not None:
        print(f"Simulationzustand in {saved_path} gespeichert", flush=True)

    pygame.mouse.set_visible(True)
    pygame.quit()


if __name__ == "__main__":
    run_simulation()

