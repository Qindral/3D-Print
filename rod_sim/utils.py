"""Utility helpers for the rod simulation."""

from __future__ import annotations

import math
import random
from typing import Tuple

import numpy as np

from .constants import CUBE_SIZE, ROTATION_SCALE


def random_unit_vector() -> np.ndarray:
    """Return a random unit vector uniformly distributed on the sphere."""

    phi = random.uniform(0.0, 2.0 * math.pi)
    costheta = random.uniform(-1.0, 1.0)
    sintheta = math.sqrt(1.0 - costheta * costheta)
    return np.array([math.cos(phi) * sintheta, math.sin(phi) * sintheta, costheta])


def random_displacement(scale: float) -> np.ndarray:
    """Return a normally-distributed displacement vector."""

    return np.random.normal(scale=scale, size=3)


def random_rotation() -> Tuple[np.ndarray, float]:
    """Return a random rotation axis and angle."""

    axis = random_unit_vector()
    angle = np.random.normal(scale=ROTATION_SCALE)
    return axis, angle


def clamp_to_cube(center: np.ndarray) -> np.ndarray:
    """Clamp a point so it stays inside the simulation cube."""

    half = CUBE_SIZE / 2.0
    return np.clip(center, -half, half)


def distance(a: np.ndarray, b: np.ndarray) -> float:
    """Return the Euclidean distance between two vectors."""

    return float(np.linalg.norm(a - b))
