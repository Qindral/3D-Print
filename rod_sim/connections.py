"""Connection management for rods."""

from __future__ import annotations

import random
from typing import Dict, List, Optional, Tuple

import numpy as np

from .constants import CONNECTION_DISTANCE, CUBE_SIZE, MAX_GROUP_SIZE
from .models import ConnectionGroup, RenderRod, Rod, SimulationSnapshot, SimulationStatePayload


class ConnectionManager:
    """Track connection hubs that bind rod endpoints."""

    def __init__(self) -> None:
        self.groups: Dict[int, ConnectionGroup] = {}
        self._next_id = 0

    def group_size(self, group_id: int) -> int:
        group = self.groups.get(group_id)
        if group is None:
            return 0
        return len(group.members)

    def is_full(self, group_id: Optional[int]) -> bool:
        if group_id is None:
            return False
        return self.group_size(group_id) >= MAX_GROUP_SIZE

    def create_group(self, member_a: Tuple[int, str], member_b: Tuple[int, str]) -> int:
        group_id = self._next_id
        self._next_id += 1
        self.groups[group_id] = ConnectionGroup(members=[member_a, member_b])
        return group_id

    def add_member(self, group_id: int, member: Tuple[int, str]) -> None:
        group = self.groups.get(group_id)
        if group is None:
            return
        if len(group.members) >= MAX_GROUP_SIZE:
            return
        if member not in group.members:
            group.members.append(member)

    def merge_groups(self, target_id: int, source_id: int, rods: List[Rod]) -> None:
        if target_id == source_id:
            return
        target = self.groups.get(target_id)
        if target is None:
            return
        source = self.groups.pop(source_id, None)
        if source is None:
            return
        if len(target.members) + len(source.members) > MAX_GROUP_SIZE:
            self.groups[source_id] = source
            return
        for member in source.members:
            rod_idx, end = member
            rods[rod_idx].connections[end] = target_id
            if member not in target.members:
                target.members.append(member)

    def anchors(self, rods: List[Rod]) -> Dict[int, np.ndarray]:
        anchor_map: Dict[int, np.ndarray] = {}
        for group_id, group in self.groups.items():
            if not group.members:
                continue
            accumulator = np.zeros(3)
            for rod_idx, end in group.members:
                accumulator += rods[rod_idx].endpoints()[end]
            anchor_map[group_id] = accumulator / len(group.members)
        return anchor_map

    def next_group_id(self) -> int:
        return self._next_id


def attempt_connections(rods: List[Rod], manager: ConnectionManager) -> None:
    """Attempt to connect nearby rod endpoints."""

    cell_size = CONNECTION_DISTANCE * 2.0
    inv_cell = 1.0 / cell_size
    half = CUBE_SIZE / 2.0

    endpoints: List[Tuple[int, str, Tuple[float, float, float], Tuple[int, int, int]]] = []
    grid: Dict[Tuple[int, int, int], List[int]] = {}

    for idx, rod in enumerate(rods):
        rod_endpoints = rod.endpoints()
        for end, pos in rod_endpoints.items():
            pos_tuple = (float(pos[0]), float(pos[1]), float(pos[2]))
            cell = (
                int((pos_tuple[0] + half) * inv_cell),
                int((pos_tuple[1] + half) * inv_cell),
                int((pos_tuple[2] + half) * inv_cell),
            )
            entry_index = len(endpoints)
            endpoints.append((idx, end, pos_tuple, cell))
            grid.setdefault(cell, []).append(entry_index)

    order = list(range(len(endpoints)))
    random.shuffle(order)
    max_dist_sq = CONNECTION_DISTANCE * CONNECTION_DISTANCE

    for idx_i in order:
        rod_i, end_i, pos_i, cell_i = endpoints[idx_i]
        group_i = rods[rod_i].connections.get(end_i)
        if manager.is_full(group_i):
            continue

        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    neighbor_cell = (
                        cell_i[0] + dx,
                        cell_i[1] + dy,
                        cell_i[2] + dz,
                    )
                    candidates = grid.get(neighbor_cell)
                    if not candidates:
                        continue
                    for idx_j in candidates:
                        if idx_j == idx_i or idx_j < idx_i:
                            continue
                        rod_j, end_j, pos_j, _ = endpoints[idx_j]
                        if rod_i == rod_j or end_i == end_j:
                            continue

                        group_j = rods[rod_j].connections.get(end_j)
                        if manager.is_full(group_j):
                            continue
                        if group_i is not None and group_j is not None and group_i == group_j:
                            continue

                        dxp = pos_i[0] - pos_j[0]
                        dyp = pos_i[1] - pos_j[1]
                        dzp = pos_i[2] - pos_j[2]
                        dist_sq = dxp * dxp + dyp * dyp + dzp * dzp
                        if dist_sq > max_dist_sq:
                            continue

                        if group_i is None and group_j is None:
                            group_id = manager.create_group((rod_i, end_i), (rod_j, end_j))
                            rods[rod_i].connections[end_i] = group_id
                            rods[rod_j].connections[end_j] = group_id
                            break
                        if group_i is not None and group_j is None:
                            if manager.is_full(group_i):
                                continue
                            manager.add_member(group_i, (rod_j, end_j))
                            rods[rod_j].connections[end_j] = group_i
                            break
                        if group_i is None and group_j is not None:
                            if manager.is_full(group_j):
                                continue
                            manager.add_member(group_j, (rod_i, end_i))
                            rods[rod_i].connections[end_i] = group_j
                            break
                        if group_i is not None and group_j is not None and group_i != group_j:
                            combined = (
                                manager.group_size(group_i)
                                + manager.group_size(group_j)
                            )
                            if combined > MAX_GROUP_SIZE:
                                continue
                            target, source = (group_i, group_j)
                            if source < target:
                                target, source = source, target
                            manager.merge_groups(target, source, rods)
                            break
                    else:
                        continue
                    break
                else:
                    continue
                break


def enforce_connections(rods: List[Rod], manager: ConnectionManager) -> None:
    """Snap rods to their connection anchors."""

    anchor_map = manager.anchors(rods)
    for rod in rods:
        group_a = rod.connections.get("A")
        group_b = rod.connections.get("B")
        anchor_a = anchor_map.get(group_a) if group_a is not None else None
        anchor_b = anchor_map.get(group_b) if group_b is not None else None

        if anchor_a is not None and anchor_b is not None and group_a != group_b:
            rod.enforce_dual_anchors(anchor_a, anchor_b)
        elif anchor_a is not None:
            rod.enforce_single_anchor("A", anchor_a)
        elif anchor_b is not None:
            rod.enforce_single_anchor("B", anchor_b)


def compute_metrics(rods: List[Rod], manager: ConnectionManager) -> Tuple[int, float, List[int]]:
    """Compute free-end counts, largest component percentage, and component sizes."""

    free_ends = 0
    adjacency: List[set[int]] = [set() for _ in rods]

    for rod in rods:
        if "A" not in rod.connections:
            free_ends += 1
        if "B" not in rod.connections:
            free_ends += 1

    for group in manager.groups.values():
        if not group.members:
            continue
        rod_indices = {member[0] for member in group.members}
        if len(rod_indices) <= 1:
            continue
        for rod_idx in rod_indices:
            adjacency[rod_idx].update(rod_indices - {rod_idx})

    visited = set()
    largest_component = 0
    component_sizes: List[int] = []
    for idx in range(len(rods)):
        if idx in visited:
            continue
        stack = [idx]
        component_size = 0
        while stack:
            node = stack.pop()
            if node in visited:
                continue
            visited.add(node)
            component_size += 1
            stack.extend(neigh for neigh in adjacency[node] if neigh not in visited)
        component_sizes.append(component_size)
        largest_component = max(largest_component, component_size)

    percent = (largest_component / len(rods) * 100.0) if rods else 0.0
    return free_ends, percent, component_sizes


def build_snapshot(
    rods: List[Rod], manager: ConnectionManager, include_state: bool = False
) -> SimulationSnapshot:
    """Create a snapshot payload for rendering and persistence."""

    render_data: List[RenderRod] = []
    for rod in rods:
        endpoints = rod.endpoints()
        render_data.append(
            (
                tuple(float(v) for v in endpoints["A"]),
                tuple(float(v) for v in endpoints["B"]),
                bool(rod.connections),
            )
        )
    free_ends, largest_percent, component_sizes = compute_metrics(rods, manager)
    state_payload: Optional[SimulationStatePayload] = None
    if include_state:
        centers = [tuple(float(value) for value in rod.center) for rod in rods]
        orientations = [tuple(float(value) for value in rod.orientation) for rod in rods]
        connections: List[Dict[str, int]] = []
        for rod in rods:
            conn_dict = {end: int(group) for end, group in rod.connections.items()}
            connections.append(conn_dict)
        groups: Dict[int, List[Tuple[int, str]]] = {}
        for group_id, group in manager.groups.items():
            groups[int(group_id)] = [(int(idx), end) for idx, end in group.members]
        state_payload = SimulationStatePayload(
            centers=centers,
            orientations=orientations,
            connections=connections,
            groups=groups,
            next_group_id=manager.next_group_id(),
        )
    return SimulationSnapshot(
        render_data=render_data,
        free_end_count=free_ends,
        largest_cluster_percent=largest_percent,
        cluster_sizes=component_sizes,
        state_payload=state_payload,
    )


__all__ = [
    "ConnectionManager",
    "attempt_connections",
    "enforce_connections",
    "compute_metrics",
    "build_snapshot",
]
