"""Rendering helpers for the rod simulation."""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import numpy as np
import pygame

from .constants import (
    BACKGROUND_COLOR,
    BASE_ROD_THICKNESS,
    CONNECTED_COLOR,
    CUBE_EDGES,
    CUBE_VERTICES,
    FAR_PLANE,
    FOV,
    NEAR_PLANE,
    ROD_COLOR,
    TEXT_COLOR,
)
from .models import RenderRod


class CameraState:
    """Minimal free-fly camera for orbiting the cube."""

    def __init__(
        self,
        position: Optional[np.ndarray] = None,
        yaw: float = 0.0,
        pitch: float = 0.0,
        move_speed: float = 120.0,
        mouse_sensitivity: float = 0.005,
    ) -> None:
        self.position = np.zeros(3) if position is None else position.astype(float)
        self.yaw = yaw
        self.pitch = pitch
        self.move_speed = move_speed
        self.mouse_sensitivity = mouse_sensitivity

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


def project_point(
    point: np.ndarray,
    view_matrix: np.ndarray,
    screen_size: Tuple[int, int],
    camera_position: np.ndarray,
) -> Tuple[Optional[Tuple[int, int]], float]:
    """Project a 3D point into screen coordinates."""

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


def collect_cube_segments(
    view_matrix: np.ndarray,
    screen_size: Tuple[int, int],
    camera_position: np.ndarray,
) -> List[Tuple[Tuple[int, int], Tuple[int, int], float]]:
    """Return projected cube edges sorted back-to-front."""

    segments: List[Tuple[Tuple[int, int], Tuple[int, int], float]] = []
    for start_idx, end_idx in CUBE_EDGES:
        start_proj = project_point(CUBE_VERTICES[start_idx], view_matrix, screen_size, camera_position)
        end_proj = project_point(CUBE_VERTICES[end_idx], view_matrix, screen_size, camera_position)
        if start_proj[0] is None or end_proj[0] is None:
            continue
        depth = max(start_proj[1], end_proj[1])
        segments.append((start_proj[0], end_proj[0], depth))
    segments.sort(key=lambda item: item[2], reverse=True)
    return segments


def draw_scene(
    screen: pygame.Surface,
    render_data: List[RenderRod],
    camera: CameraState,
    overlay_text: Optional[str] = None,
    font: Optional[pygame.font.Font] = None,
) -> None:
    """Render rod geometry."""

    screen.fill(BACKGROUND_COLOR)
    screen_size = screen.get_size()
    view_matrix = camera.rotation_matrix()

    for start, end, depth in collect_cube_segments(view_matrix, screen_size, camera.position):
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
    overlay_text: Optional[str] = None,
    font: Optional[pygame.font.Font] = None,
) -> None:
    """Render voxel-based surface points."""

    screen.fill(BACKGROUND_COLOR)
    screen_size = screen.get_size()
    view_matrix = camera.rotation_matrix()

    for start, end, depth in collect_cube_segments(view_matrix, screen_size, camera.position):
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


__all__ = [
    "CameraState",
    "draw_scene",
    "draw_voxel_scene",
    "project_point",
]
