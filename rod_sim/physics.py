"""Physics update routines for the rod simulation."""

from __future__ import annotations

import random
from typing import List

import numpy as np

from .constants import CUBE_SIZE, NUM_RODS, TIME_STEP, TRANSLATION_SCALE
from .connections import ConnectionManager, attempt_connections, enforce_connections
from .models import Rod
from .utils import clamp_to_cube, random_displacement, random_rotation, random_unit_vector


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

    for rod in rods:
        delta = random_displacement(TRANSLATION_SCALE) * TIME_STEP
        rod.apply_translation(delta)
        axis, angle = random_rotation()
        rod.apply_rotation(axis, angle * TIME_STEP)
        rod.center = clamp_to_cube(rod.center)

    enforce_connections(rods, manager)
    attempt_connections(rods, manager)
    enforce_connections(rods, manager)


__all__ = ["create_rods", "update_rods", "ConnectionManager"]
