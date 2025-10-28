"""Persistence helpers for saving and loading simulation state."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from .constants import CUBE_SIZE, MAX_GROUP_SIZE, ROD_LENGTH, STATE_FILE_PATH
from .connections import ConnectionManager
from .models import ConnectionGroup, Rod, SavedStateInfo, SimulationSnapshot
from .utils import clamp_to_cube, random_unit_vector


def save_simulation_state(
    snapshot: SimulationSnapshot, path: Path = STATE_FILE_PATH
) -> Optional[Path]:
    payload = snapshot.state_payload
    if payload is None:
        return None
    groups_serializable = {
        str(group_id): [[int(idx), end] for idx, end in members]
        for group_id, members in payload.groups.items()
    }
    data = {
        "centers": payload.centers,
        "orientations": payload.orientations,
        "connections": payload.connections,
        "groups": groups_serializable,
        "next_group_id": payload.next_group_id,
        "timestamp": time.time(),
        "metadata": {
            "rod_length": ROD_LENGTH,
            "cube_size": CUBE_SIZE,
            "max_group_size": MAX_GROUP_SIZE,
        },
    }
    tmp_path = path.parent / f"{path.name}.tmp"
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)
    tmp_path.replace(path)
    return path


def load_simulation_state(
    path: Path = STATE_FILE_PATH,
) -> Optional[Tuple[List[Rod], ConnectionManager]]:
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None

    centers = data.get("centers")
    orientations = data.get("orientations")
    connections = data.get("connections")
    groups_data = data.get("groups", {})
    next_group_id = int(data.get("next_group_id", 0))

    if not isinstance(centers, list) or not isinstance(orientations, list):
        return None
    if len(centers) != len(orientations):
        return None
    if connections is None:
        connections = [{} for _ in centers]

    rods: List[Rod] = []
    for idx, (center_values, orientation_values) in enumerate(zip(centers, orientations)):
        center_array = np.array(center_values, dtype=float)
        center_array = clamp_to_cube(center_array)
        orientation_array = np.array(orientation_values, dtype=float)
        if np.linalg.norm(orientation_array) < 1e-6:
            orientation_array = random_unit_vector()
        else:
            orientation_array = orientation_array / np.linalg.norm(orientation_array)
        rod = Rod(center=center_array, orientation=orientation_array)
        conn_dict = connections[idx] if idx < len(connections) else {}
        for end, group_id in conn_dict.items():
            try:
                rod.connections[end] = int(group_id)
            except (TypeError, ValueError):
                continue
        rods.append(rod)

    manager = ConnectionManager()
    manager.groups = {}
    for key, members in groups_data.items():
        try:
            group_id = int(key)
        except (TypeError, ValueError):
            continue
        group_members: List[Tuple[int, str]] = []
        if isinstance(members, list):
            for entry in members:
                if (
                    isinstance(entry, (list, tuple))
                    and len(entry) == 2
                    and isinstance(entry[0], (int, float))
                    and isinstance(entry[1], str)
                ):
                    rod_idx = int(entry[0])
                    if 0 <= rod_idx < len(rods):
                        group_members.append((rod_idx, entry[1]))
        manager.groups[group_id] = ConnectionGroup(members=group_members)

    if manager.groups:
        max_existing = max(manager.groups.keys())
        next_group_id = max(next_group_id, max_existing + 1)
    manager._next_id = next_group_id

    return rods, manager


def inspect_saved_state(path: Path = STATE_FILE_PATH) -> Optional[SavedStateInfo]:
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None

    centers = data.get("centers")
    if not isinstance(centers, list):
        return None
    metadata = data.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}

    info_metadata: Dict[str, float] = {}
    for key, value in metadata.items():
        try:
            info_metadata[str(key)] = float(value)
        except (TypeError, ValueError):
            continue

    timestamp = data.get("timestamp")
    try:
        timestamp_value = float(timestamp)
    except (TypeError, ValueError):
        timestamp_value = 0.0

    return SavedStateInfo(rod_count=len(centers), timestamp=timestamp_value, metadata=info_metadata)


def wait_for_state_snapshot(
    state_queue, current_snapshot: SimulationSnapshot, timeout: float = 5.0
) -> SimulationSnapshot:
    import queue
    import time

    deadline = time.perf_counter() + timeout
    snapshot = current_snapshot
    while time.perf_counter() < deadline:
        if snapshot.state_payload is not None:
            break
        try:
            snapshot = state_queue.get(timeout=0.2)
        except queue.Empty:
            continue
    return snapshot


__all__ = [
    "save_simulation_state",
    "load_simulation_state",
    "inspect_saved_state",
    "wait_for_state_snapshot",
]
