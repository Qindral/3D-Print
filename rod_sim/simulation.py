"""High-level orchestration for running the rod simulation."""

from __future__ import annotations

import multiprocessing as mp
import queue

import pygame

from .headless import run_headless_phase
from .models import RenderGeometry, SimulationSnapshot
from .persistence import save_simulation_state, wait_for_state_snapshot
from .startup import prompt_start_choice
from .viewers import run_interactive_viewer, run_rolling_ball_viewer
from .worker import simulation_worker


def _get_initial_snapshot(state_queue: mp.Queue) -> SimulationSnapshot:
    try:
        return state_queue.get(timeout=5.0)
    except queue.Empty:
        return SimulationSnapshot(
            render_data=RenderGeometry(),
            free_end_count=0,
            largest_cluster_percent=0.0,
            cluster_sizes=[],
        )


def run_simulation() -> None:
    resume_saved_state = prompt_start_choice()

    ctx = mp.get_context("spawn")
    state_queue: mp.Queue = ctx.Queue(maxsize=2)
    stop_event = ctx.Event()
    pause_event = ctx.Event()
    worker = ctx.Process(
        target=simulation_worker,
        args=(state_queue, stop_event, pause_event, resume_saved_state),
    )
    worker.start()

    current_snapshot = _get_initial_snapshot(state_queue)
    current_snapshot, ui_state = run_headless_phase(
        state_queue, stop_event, pause_event, current_snapshot
    )

    if ui_state.abort_requested:
        stop_event.set()
        current_snapshot = wait_for_state_snapshot(state_queue, current_snapshot)
        worker.join(timeout=2.0)
        if worker.is_alive():
            worker.terminate()
        saved_path = save_simulation_state(current_snapshot)
        if saved_path is not None:
            print(f"Simulationzustand in {saved_path} gespeichert", flush=True)
        return

    pygame.init()

    if pause_event.is_set():
        pause_event.clear()

    if ui_state.launch_rolling_viewer and ui_state.rolling_result is not None:
        run_rolling_ball_viewer(ui_state.rolling_result)

    current_snapshot = run_interactive_viewer(
        state_queue, stop_event, pause_event, worker, current_snapshot
    )
    saved_path = save_simulation_state(current_snapshot)
    if saved_path is not None:
        print(f"Simulationzustand in {saved_path} gespeichert", flush=True)

    pygame.mouse.set_visible(True)
    pygame.quit()


__all__ = ["run_simulation"]
