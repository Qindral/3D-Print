import math
import multiprocessing as mp
import queue
import random
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pygame


# Simulation constants
CUBE_SIZE = 100.0
ROD_LENGTH = 10.0
NUM_RODS = 1200
TIME_STEP = 0.05
TRANSLATION_SCALE = 2.5
ROTATION_SCALE = 0.2
CONNECTION_DISTANCE = 2.5
BACKGROUND_COLOR = (15, 15, 25)
ROD_COLOR = (180, 220, 255)
CONNECTED_COLOR = (255, 150, 80)
TEXT_COLOR = (240, 240, 240)


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

    def create_group(self, member_a: Tuple[int, str], member_b: Tuple[int, str]) -> int:
        group_id = self._next_id
        self._next_id += 1
        self.groups[group_id] = ConnectionGroup(members=[member_a, member_b])
        return group_id

    def add_member(self, group_id: int, member: Tuple[int, str]) -> None:
        group = self.groups.get(group_id)
        if group is None:
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
                            manager.add_member(group_i, (rod_j, end_j))
                            rods[rod_j].connections[end_j] = group_i
                            break
                        if group_i is None and group_j is not None:
                            manager.add_member(group_j, (rod_i, end_i))
                            rods[rod_i].connections[end_i] = group_j
                            break
                        if group_i is not None and group_j is not None and group_i != group_j:
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


def project_point(point: np.ndarray, angle_x: float, angle_y: float, scale: float, screen_size: Tuple[int, int]) -> Tuple[int, int]:
    rot_x = np.array(
        [
            [1, 0, 0],
            [0, math.cos(angle_x), -math.sin(angle_x)],
            [0, math.sin(angle_x), math.cos(angle_x)],
        ]
    )
    rot_y = np.array(
        [
            [math.cos(angle_y), 0, math.sin(angle_y)],
            [0, 1, 0],
            [-math.sin(angle_y), 0, math.cos(angle_y)],
        ]
    )
    rotated = rot_y @ (rot_x @ point)
    x = rotated[0] * scale + screen_size[0] // 2
    y = rotated[1] * scale + screen_size[1] // 2
    return int(x), int(y)


RenderRod = Tuple[Tuple[float, float, float], Tuple[float, float, float], bool]


def snapshot_rods(rods: Iterable[Rod]) -> List[RenderRod]:
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
    return render_data


def draw_rods(
    screen: pygame.Surface,
    render_data: List[RenderRod],
    angle_x: float,
    angle_y: float,
    overlay_text: str | None = None,
    font: pygame.font.Font | None = None,
) -> None:
    screen.fill(BACKGROUND_COLOR)
    scale = screen.get_width() / (CUBE_SIZE * 0.8)
    for point_a_3d, point_b_3d, connected in render_data:
        color = CONNECTED_COLOR if connected else ROD_COLOR
        point_a = project_point(np.array(point_a_3d), angle_x, angle_y, scale, screen.get_size())
        point_b = project_point(np.array(point_b_3d), angle_x, angle_y, scale, screen.get_size())
        pygame.draw.line(screen, color, point_a, point_b, 2)
    if overlay_text and font:
        text_surface = font.render(overlay_text, True, TEXT_COLOR)
        screen.blit(text_surface, (10, 10))
    pygame.display.flip()


def simulation_worker(state_queue: mp.Queue, stop_event: mp.Event) -> None:
    rods = create_rods(NUM_RODS)
    manager = ConnectionManager()
    try:
        while not stop_event.is_set():
            update_rods(rods, manager)
            try:
                state_queue.put_nowait(snapshot_rods(rods))
            except queue.Full:
                pass
    finally:
        try:
            state_queue.put_nowait(snapshot_rods(rods))
        except queue.Full:
            pass


def run_simulation() -> None:
    pygame.init()
    screen = pygame.display.set_mode((960, 720))
    pygame.display.set_caption("Rod Brownian Motion Simulation")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("consolas", 20)

    ctx = mp.get_context("spawn")
    state_queue: mp.Queue = ctx.Queue(maxsize=2)
    stop_event = ctx.Event()
    worker = ctx.Process(target=simulation_worker, args=(state_queue, stop_event))
    worker.start()

    try:
        current_state: List[RenderRod] = state_queue.get(timeout=5.0)
    except queue.Empty:
        current_state = []

    angle_x = math.radians(35)
    angle_y = math.radians(45)
    last_caption_update = 0
    update_fps_display = 0.0
    last_update_tick: int | None = None

    running = True
    s = 0
    while running:
        dt_ms = clock.tick(60)
        render_fps = 1000.0 / dt_ms if dt_ms > 0 else 0.0
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
        received_update = False
        try:
            while True:
                current_state = state_queue.get_nowait()
                received_update = True
        except queue.Empty:
            pass

        if received_update:
            now = pygame.time.get_ticks()
            if last_update_tick is not None:
                delta = now - last_update_tick
                if delta > 0:
                    update_fps_display = 1000.0 / delta
            last_update_tick = now

        draw_rods(
            screen,
            current_state,
            angle_x,
            angle_y,
            overlay_text=(
                f"Render FPS: {render_fps:5.1f} | Sim updates/s: {update_fps_display:5.1f}"
            ),
            font=font,
        )
        now = pygame.time.get_ticks()
        if now - last_caption_update >= 250:
            pygame.display.set_caption(
                "Rod Brownian Motion Simulation - "
                f"Render FPS: {render_fps:5.1f} | Sim updates/s: {update_fps_display:5.1f}"
            )
            last_caption_update = now

    stop_event.set()
    worker.join(timeout=2.0)
    if worker.is_alive():
        worker.terminate()

    pygame.quit()


if __name__ == "__main__":
    run_simulation()

