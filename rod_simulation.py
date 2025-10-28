import math
import multiprocessing as mp
import queue
import random
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pygame


# Simulation constants
CUBE_SIZE = 200.0
ROD_LENGTH = 20.0
NUM_RODS = 500
TIME_STEP = 0.05
TRANSLATION_SCALE = 2.5
ROTATION_SCALE = 0.2
CONNECTION_DISTANCE = 2.5
BACKGROUND_COLOR = (15, 15, 25)
ROD_COLOR = (180, 220, 255)
CONNECTED_COLOR = (255, 150, 80)


def random_unit_vector() -> np.ndarray:
    """Return a random unit vector uniformly distributed on the sphere."""

    phi = random.uniform(0.0, 2.0 * math.pi)
    costheta = random.uniform(-1.0, 1.0)
    sintheta = math.sqrt(1.0 - costheta * costheta)
    return np.array([math.cos(phi) * sintheta, math.sin(phi) * sintheta, costheta])


@dataclass
class Connection:
    other_id: int
    other_end: str


@dataclass
class Rod:
    center: np.ndarray
    orientation: np.ndarray
    connections: Dict[str, Connection] = field(default_factory=dict)

    def endpoints(self) -> Dict[str, np.ndarray]:
        offset = self.orientation * (ROD_LENGTH / 2.0)
        return {
            "A": self.center + offset,
            "B": self.center - offset,
        }

    def has_free_end(self, end: str) -> bool:
        return end not in self.connections

    def is_connected_to(self, other_id: int) -> bool:
        return any(conn.other_id == other_id for conn in self.connections.values())

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

    def enforce_connection(self, end: str, anchor: np.ndarray) -> None:
        direction = 1.0 if end == "A" else -1.0
        self.center = anchor - direction * self.orientation * (ROD_LENGTH / 2.0)


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


def attempt_connections(rods: List[Rod]) -> None:
    free_ends: List[Tuple[int, str, np.ndarray]] = []
    for idx, rod in enumerate(rods):
        for end, pos in rod.endpoints().items():
            if rod.has_free_end(end):
                free_ends.append((idx, end, pos))

    random.shuffle(free_ends)
    for i in range(len(free_ends)):
        rod_i, end_i, pos_i = free_ends[i]
        if not rods[rod_i].has_free_end(end_i):
            continue
        for j in range(i + 1, len(free_ends)):
            rod_j, end_j, pos_j = free_ends[j]
            if rod_i == rod_j or not rods[rod_j].has_free_end(end_j):
                continue
            # Enforce that only complementary ends (A-B) can connect.
            if end_i == end_j:
                continue
            if rods[rod_i].is_connected_to(rod_j):
                continue
            if distance(pos_i, pos_j) <= CONNECTION_DISTANCE:
                rods[rod_i].connections[end_i] = Connection(rod_j, end_j)
                rods[rod_j].connections[end_j] = Connection(rod_i, end_i)
                break


def enforce_connections(rods: List[Rod]) -> None:
    processed: set[Tuple[int, str]] = set()
    for idx, rod in enumerate(rods):
        for end, conn in rod.connections.items():
            key = (idx, end)
            if key in processed:
                continue
            other = rods[conn.other_id]
            other_end = conn.other_end
            anchor = (rod.endpoints()[end] + other.endpoints()[other_end]) / 2.0
            rod.enforce_connection(end, anchor)
            other.enforce_connection(other_end, anchor)
            processed.add(key)
            processed.add((conn.other_id, other_end))


def update_rods(rods: List[Rod]) -> None:
    for rod in rods:
        delta = random_displacement(TRANSLATION_SCALE) * TIME_STEP
        rod.apply_translation(delta)
        axis, angle = random_rotation()
        rod.apply_rotation(axis, angle * TIME_STEP)
        rod.center = clamp_to_cube(rod.center)

    enforce_connections(rods)
    attempt_connections(rods)
    enforce_connections(rods)


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


def draw_rods(screen: pygame.Surface, render_data: List[RenderRod], angle_x: float, angle_y: float) -> None:
    screen.fill(BACKGROUND_COLOR)
    scale = screen.get_width() / (CUBE_SIZE * 0.8)
    for point_a_3d, point_b_3d, connected in render_data:
        color = CONNECTED_COLOR if connected else ROD_COLOR
        point_a = project_point(np.array(point_a_3d), angle_x, angle_y, scale, screen.get_size())
        point_b = project_point(np.array(point_b_3d), angle_x, angle_y, scale, screen.get_size())
        pygame.draw.line(screen, color, point_a, point_b, 2)
    pygame.display.flip()


def simulation_worker(state_queue: mp.Queue, stop_event: mp.Event) -> None:
    rods = create_rods(NUM_RODS)
    try:
        while not stop_event.is_set():
            update_rods(rods)
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

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
        try:
            while True:
                current_state = state_queue.get_nowait()
        except queue.Empty:
            pass

        draw_rods(screen, current_state, angle_x, angle_y)
        clock.tick(60)

    stop_event.set()
    worker.join(timeout=2.0)
    if worker.is_alive():
        worker.terminate()

    pygame.quit()


if __name__ == "__main__":
    run_simulation()

