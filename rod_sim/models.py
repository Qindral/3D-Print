"""Data models used across the rod simulation."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import math
import numpy as np

from .constants import ROD_LENGTH

RenderRod = Tuple[Tuple[float, float, float], Tuple[float, float, float], bool]


@dataclass
class Rod:
    """Represent a single rod with endpoints A and B."""

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
    "RenderRod",
    "Rod",
    "ConnectionGroup",
    "SimulationSnapshot",
    "SimulationStatePayload",
    "RollingBallResult",
    "SavedStateInfo",
]
