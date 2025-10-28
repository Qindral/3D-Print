"""Background worker that advances the rod simulation."""

from __future__ import annotations

import multiprocessing as mp
import queue

from .connections import ConnectionManager, build_snapshot
from .constants import NUM_RODS, STATE_FILE_PATH
from .physics import create_rods, update_rods
from .persistence import load_simulation_state


def simulation_worker(state_queue: mp.Queue, stop_event: mp.Event) -> None:
    loaded_state = load_simulation_state()
    if loaded_state is not None:
        rods, manager = loaded_state
        print(
            f"Lade gespeicherten Zustand mit {len(rods)} Rods aus {STATE_FILE_PATH}",
            flush=True,
        )
        if len(rods) != NUM_RODS:
            print(
                f"Hinweis: gespeicherter Zustand enthält {len(rods)} Rods (Standard {NUM_RODS})",
                flush=True,
            )
    else:
        rods = create_rods(NUM_RODS)
        manager = ConnectionManager()
    try:
        while not stop_event.is_set():
            update_rods(rods, manager)
            try:
                state_queue.put_nowait(build_snapshot(rods, manager))
            except queue.Full:
                pass
    finally:
        final_snapshot = build_snapshot(rods, manager, include_state=True)
        state_queue.put(final_snapshot)


__all__ = ["simulation_worker"]
