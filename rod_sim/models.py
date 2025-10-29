"""Data models used across the rod simulation."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from .constants import ROD_LENGTH


@dataclass
class RenderGeometry:
    points_a: np.ndarray = field(
        default_factory=lambda: np.empty((0, 3), dtype=np.float32)
    )
    points_b: np.ndarray = field(
        default_factory=lambda: np.empty((0, 3), dtype=np.float32)
    )
    connected: np.ndarray = field(default_factory=lambda: np.empty((0,), dtype=bool))

    def __len__(self) -> int:
        return int(self.points_a.shape[0])

    def is_empty(self) -> bool:
        return self.points_a.size == 0


@dataclass
class Rod:
    """Represent a single rod with endpoints A and B."""

    center: np.ndarray
    orientation: np.ndarray
    connections: Dict[str, int] = field(default_factory=dict)
    half_vector: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.center = self.center.astype(float, copy=False)
        if np.linalg.norm(self.orientation) < 1e-9:
            self.orientation = np.array([1.0, 0.0, 0.0], dtype=float)
        else:
            self.orientation = (
                self.orientation.astype(float, copy=False)
                / np.linalg.norm(self.orientation)
            )
        self._recompute_half_vector()

    def endpoints(self) -> Dict[str, np.ndarray]:
        return {
            "A": self.center + self.half_vector,
            "B": self.center - self.half_vector,
        }

    def has_free_end(self, end: str) -> bool:
        return end not in self.connections

    def apply_translation(self, delta: np.ndarray) -> None:
        self.center += delta

    def apply_rotation(self, axis: np.ndarray, angle: float) -> None:
        angle = float(angle)
        if abs(angle) < 1e-6:
            return
        norm_axis = np.linalg.norm(axis)
        if norm_axis < 1e-6:
            return
        axis = axis / norm_axis
        delta = np.cross(axis, self.orientation) * angle
        self.orientation += delta
        norm = np.linalg.norm(self.orientation)
        if norm < 1e-9:
            self.orientation = axis
            norm = np.linalg.norm(self.orientation)
        self.orientation /= norm
        self._recompute_half_vector()

    def _recompute_half_vector(self) -> None:
        self.half_vector = self.orientation * (ROD_LENGTH / 2.0)

    def enforce_single_anchor(self, end: str, anchor: np.ndarray) -> None:
        direction = 1.0 if end == "A" else -1.0
        self.center = anchor - direction * self.half_vector

    def enforce_dual_anchors(self, anchor_a: np.ndarray, anchor_b: np.ndarray) -> None:
        direction = anchor_a - anchor_b
        norm = np.linalg.norm(direction)
        if norm < 1e-6:
            return
        direction /= norm
        self.orientation = direction
        midpoint = (anchor_a + anchor_b) / 2.0
        self.center = midpoint
        self._recompute_half_vector()


@dataclass
class ConnectionGroup:
    members: List[Tuple[int, str]] = field(default_factory=list)


@dataclass
class SimulationStatePayload:
    centers: List[Tuple[float, float, float]]
    orientations: List[Tuple[float, float, float]]
    connections: List[Dict[str, int]]
    groups: Dict[int, List[Tuple[int, str]]]
    next_group_id: int


@dataclass
class SimulationSnapshot:
    render_data: RenderGeometry
    free_end_count: int
    largest_cluster_percent: float
    cluster_sizes: List[int] = field(default_factory=list)
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
    status_log: deque[str] = field(default_factory=lambda: deque(maxlen=8))

    # Metric histories for plotting convenience
    history_indices: deque[int] = field(default_factory=lambda: deque(maxlen=600))
    history_free: deque[float] = field(default_factory=lambda: deque(maxlen=600))
    history_cluster: deque[float] = field(default_factory=lambda: deque(maxlen=600))


@dataclass
class SavedStateInfo:
    rod_count: int
    timestamp: float
    metadata: Dict[str, float]


__all__ = [
    "HeadlessUIState",
    "RenderGeometry",
    "Rod",
    "ConnectionGroup",
    "SimulationSnapshot",
    "SimulationStatePayload",
    "RollingBallResult",
    "SavedStateInfo",
]
