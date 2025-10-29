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


def translation_displacements(count: int, scale: float) -> np.ndarray:
    """Return `count` displacement vectors scaled by `scale`."""

    if count <= 0:
        return np.empty((0, 3), dtype=float)
    return np.random.normal(scale=scale, size=(count, 3))


def random_rotation_batch(count: int, time_scale: float) -> Tuple[np.ndarray, np.ndarray]:
    """Return arrays of rotation axes and scaled angles."""

    if count <= 0:
        return np.empty((0, 3), dtype=float), np.empty((0,), dtype=float)

    phi = np.random.uniform(0.0, 2.0 * math.pi, size=count)
    costheta = np.random.uniform(-1.0, 1.0, size=count)
    sintheta = np.sqrt(np.maximum(0.0, 1.0 - costheta * costheta))
    axes = np.stack((np.cos(phi) * sintheta, np.sin(phi) * sintheta, costheta), axis=1)
    angles = np.random.normal(scale=ROTATION_SCALE, size=count) * time_scale
    return axes, angles


def clamp_to_cube(center: np.ndarray) -> np.ndarray:
    """Clamp a point so it stays inside the simulation cube."""

    half = CUBE_SIZE / 2.0
    return np.clip(center, -half, half)


def distance(a: np.ndarray, b: np.ndarray) -> float:
    """Return the Euclidean distance between two vectors."""

    return float(np.linalg.norm(a - b))
