"""Interactive Pygame viewers for rods and organic surfaces."""

from __future__ import annotations

import math
import queue
from typing import Optional

import numpy as np
import pygame

from .constants import DEFAULT_CAMERA_DISTANCE
from .models import RollingBallResult, SimulationSnapshot
from .persistence import wait_for_state_snapshot
from .render import CameraState, draw_scene, draw_voxel_scene


def _update_camera_from_input(camera: CameraState, dt_ms: int) -> None:
    keys = pygame.key.get_pressed()
    move_direction = np.zeros(3)
    if keys[pygame.K_w]:
        move_direction += camera.forward()
    if keys[pygame.K_s]:
        move_direction -= camera.forward()
    if keys[pygame.K_d]:
        move_direction += camera.right()
    if keys[pygame.K_a]:
        move_direction -= camera.right()
    if keys[pygame.K_SPACE] or keys[pygame.K_r]:
        move_direction += camera.up()
    if keys[pygame.K_LCTRL] or keys[pygame.K_f]:
        move_direction -= camera.up()

    if np.linalg.norm(move_direction) > 1e-6:
        move_direction = move_direction / np.linalg.norm(move_direction)
        speed_multiplier = 2.5 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 1.0
        camera.position += move_direction * camera.move_speed * speed_multiplier * (dt_ms / 1000.0)


def run_interactive_viewer(
    state_queue,
    stop_event,
    pause_event,
    worker,
    current_snapshot: SimulationSnapshot,
) -> SimulationSnapshot:
    screen = pygame.display.set_mode((960, 720))
    pygame.display.set_caption("Rod Brownian Motion Simulation")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("consolas", 20)

    camera = CameraState(position=np.zeros(3))
    camera.position = np.array([0.0, 0.0, -DEFAULT_CAMERA_DISTANCE])
    mouse_rotating = False
    last_caption_update = 0
    update_fps_display = 0.0
    last_update_tick: Optional[int] = None

    running = True
    while running:
        dt_ms = clock.tick(60)
        render_fps = 1000.0 / dt_ms if dt_ms > 0 else 0.0
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_p:
                    if pause_event.is_set():
                        pause_event.clear()
                    else:
                        pause_event.set()
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                mouse_rotating = True
                pygame.mouse.get_rel()
                pygame.event.set_grab(True)
                pygame.mouse.set_visible(False)
            elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                mouse_rotating = False
                pygame.event.set_grab(False)
                pygame.mouse.set_visible(True)

        received_update = False
        try:
            while True:
                current_snapshot = state_queue.get_nowait()
                received_update = True
        except queue.Empty:
            pass

        if mouse_rotating:
            delta_x, delta_y = pygame.mouse.get_rel()
            camera.yaw += delta_x * camera.mouse_sensitivity
            camera.pitch -= delta_y * camera.mouse_sensitivity
            camera.pitch = max(min(camera.pitch, math.radians(89.0)), math.radians(-89.0))
            camera.yaw = (camera.yaw + math.pi) % (2 * math.pi) - math.pi
        else:
            pygame.mouse.get_rel()

        _update_camera_from_input(camera, dt_ms)

        if received_update:
            now_ticks = pygame.time.get_ticks()
            if last_update_tick is not None:
                delta = now_ticks - last_update_tick
                if delta > 0:
                    update_fps_display = 1000.0 / delta
            last_update_tick = now_ticks

        overlay_status = "PAUSIERT" if pause_event.is_set() else "Aktiv"
        rod_count = max(1, len(current_snapshot.render_data))
        total_endpoints = rod_count * 2
        free_percent = (current_snapshot.free_end_count / total_endpoints) * 100.0

        draw_scene(
            screen,
            current_snapshot.render_data,
            camera,
            overlay_text=(
                "Render FPS: "
                f"{render_fps:5.1f} | Sim updates/s: {update_fps_display:5.1f} | "
                f"Status: {overlay_status} | "
                f"Freie Enden: {free_percent:5.1f}% | "
                f"Größter Cluster: {current_snapshot.largest_cluster_percent:5.1f}% der Rods"
            ),
            font=font,
        )

        now_ticks = pygame.time.get_ticks()
        if now_ticks - last_caption_update >= 250:
            pygame.display.set_caption(
                "Rod Brownian Motion Simulation - "
                f"Render FPS: {render_fps:5.1f} | "
                f"Sim updates/s: {update_fps_display:5.1f} | "
                f"Status: {overlay_status}"
            )
            last_caption_update = now_ticks

    stop_event.set()
    current_snapshot = wait_for_state_snapshot(state_queue, current_snapshot)
    worker.join(timeout=2.0)
    if worker.is_alive():
        worker.terminate()

    return current_snapshot


def run_rolling_ball_viewer(result: RollingBallResult) -> None:
    screen = pygame.display.set_mode((960, 720))
    pygame.display.set_caption("Distanzfüllung-Organik")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("consolas", 20)

    camera = CameraState(position=np.zeros(3))
    camera.position = np.array([0.0, 0.0, -DEFAULT_CAMERA_DISTANCE])
    mouse_rotating = False

    running = True
    while running:
        dt_ms = clock.tick(60)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                mouse_rotating = True
                pygame.mouse.get_rel()
                pygame.event.set_grab(True)
                pygame.mouse.set_visible(False)
            elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                mouse_rotating = False
                pygame.event.set_grab(False)
                pygame.mouse.set_visible(True)

        if mouse_rotating:
            delta_x, delta_y = pygame.mouse.get_rel()
            camera.yaw += delta_x * camera.mouse_sensitivity
            camera.pitch -= delta_y * camera.mouse_sensitivity
            camera.pitch = max(min(camera.pitch, math.radians(89.0)), math.radians(-89.0))
            camera.yaw = (camera.yaw + math.pi) % (2 * math.pi) - math.pi
        else:
            pygame.mouse.get_rel()

        _update_camera_from_input(camera, dt_ms)

        overlay_text = (
            f"Schwelle: {result.threshold:.2f} | Füllung: {result.fill_fraction * 100.0:5.2f}% | "
            f"Auflösung: {result.grid_resolution}³ (Voxel {result.voxel_size:.2f}) | "
            f"Oberflächenpunkte: {len(result.surface_points)}"
        )

        draw_voxel_scene(screen, result.surface_points, camera, overlay_text=overlay_text, font=font)

    pygame.event.set_grab(False)
    pygame.mouse.set_visible(True)


__all__ = ["run_interactive_viewer", "run_rolling_ball_viewer"]
