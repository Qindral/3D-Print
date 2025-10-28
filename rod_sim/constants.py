"""Core constants for the rod simulation project."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

# Simulation constants
CUBE_SIZE = 200.0
ROD_LENGTH = 20.0
NUM_RODS = 500
TIME_STEP = 0.05
TRANSLATION_SCALE = 2.5
ROTATION_SCALE = 0.2
CONNECTION_DISTANCE = 2.5
MAX_GROUP_SIZE = 10
BACKGROUND_COLOR = (15, 15, 25)
ROD_COLOR = (180, 220, 255)
CONNECTED_COLOR = (255, 150, 80)
TEXT_COLOR = (240, 240, 240)
BASE_ROD_THICKNESS = 5.0
CUBE_HALF = CUBE_SIZE / 2.0
CUBE_BOUNDING_RADIUS = CUBE_HALF * math.sqrt(3.0)
DEFAULT_CAMERA_DISTANCE = CUBE_SIZE * 2.2
FOV = max(
    math.radians(30.0),
    2.0 * math.atan(CUBE_BOUNDING_RADIUS / DEFAULT_CAMERA_DISTANCE),
)
NEAR_PLANE = 5.0
FAR_PLANE = 1200.0

STATE_FILE_PATH = Path("rod_state.json")
ROLLING_EXPORT_PATH = Path("rolling_ball_surface.obj")

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
