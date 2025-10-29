"""Physics update routines for the rod simulation."""

from __future__ import annotations

import random
from typing import List

import numpy as np

from .constants import CUBE_SIZE, NUM_RODS, TIME_STEP, TRANSLATION_SCALE
from .connections import ConnectionManager, attempt_connections, enforce_connections
from .models import Rod
from .utils import (
    clamp_to_cube,
    random_rotation_batch,
    random_unit_vector,
    translation_displacements,
)


def create_rods(num_rods: int = NUM_RODS) -> List[Rod]:
    """Create the initial rod population."""

    rods: List[Rod] = []
    half = CUBE_SIZE / 2.0
    for _ in range(num_rods):
        center = np.array(
            [
                random.uniform(-half, half),
                random.uniform(-half, half),
                random.uniform(-half, half),
            ],
            dtype=float,
        )
        orientation = random_unit_vector()
        rods.append(Rod(center=center, orientation=orientation))
    return rods


def update_rods(rods: List[Rod], manager: ConnectionManager) -> None:
    """Advance the physics simulation for all rods."""

    rod_count = len(rods)
    if rod_count == 0:
        return

    translations = translation_displacements(rod_count, TRANSLATION_SCALE * TIME_STEP)
    axes, angles = random_rotation_batch(rod_count, TIME_STEP)

    for idx, rod in enumerate(rods):
        rod.apply_translation(translations[idx])
        rod.apply_rotation(axes[idx], angles[idx])
        rod.center = clamp_to_cube(rod.center)

    enforce_connections(rods, manager)
    attempt_connections(rods, manager)
    enforce_connections(rods, manager)


__all__ = ["create_rods", "update_rods", "ConnectionManager"]
