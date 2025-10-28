"""Headless analytics phase with Matplotlib controls."""

from __future__ import annotations

import math
import queue
import time
from collections import Counter
from typing import Tuple

import matplotlib.pyplot as plt
from matplotlib.widgets import Button, TextBox

from .distance_fill import export_distance_fill, run_distance_fill
from .models import HeadlessUIState, SimulationSnapshot


def _compute_cluster_ticks(min_value: int, max_value: int) -> list[float]:
    """Return between 5 and 50 sensible tick marks for the histogram axis."""

    if min_value >= max_value:
        base = max(1, min_value)
        start = max(1, base - 2)
        return [float(start + offset) for offset in range(5)]

    span = max_value - min_value
    target_ticks = min(50, max(5, span + 1))
    step = max(1, math.ceil(span / (target_ticks - 1)))
    ticks = list(range(min_value, max_value + 1, step))
    if ticks[-1] != max_value:
        ticks.append(max_value)

    if len(ticks) < 5:
        step = max(1, math.ceil(span / 4))
        ticks = list(range(min_value, max_value + 1, step))
        if ticks[-1] != max_value:
            ticks.append(max_value)
        while len(ticks) < 5:
            ticks.append(ticks[-1] + step)

    if len(ticks) > 50:
        stride = math.ceil(len(ticks) / 50)
        ticks = ticks[::stride]
        if ticks[-1] != max_value:
            ticks.append(max_value)

    return [float(value) for value in ticks]


def _format_status_log(log: "list[str]") -> str:
    if not log:
        return "Statusbericht:"
    lines = ["Statusbericht:"]
    for entry in log:
        lines.append(f"• {entry}")
    return "\n".join(lines)


def _update_status(fig, ui_state: HeadlessUIState, message: str) -> None:
    ui_state.status_message = message
    ui_state.status_log.appendleft(message)
    fig._status_text.set_text(_format_status_log(list(ui_state.status_log)))
    fig.canvas.draw_idle()


def _install_controls(
    fig,
    ui_state: HeadlessUIState,
    state_queue,
    stop_event,
    pause_event,
    current_snapshot: SimulationSnapshot,
) -> None:
    def update_status(message: str) -> None:
        _update_status(fig, ui_state, message)

    threshold_ax = fig.add_axes([0.1, 0.18, 0.2, 0.05])
    threshold_box = TextBox(threshold_ax, "Schwelle", initial=f"{ui_state.distance_threshold:.1f}")

    def apply_threshold(text: str) -> bool:
        try:
            value = float(text)
            if value < 0:
                raise ValueError
        except ValueError:
            update_status("Ungültige Schwelle – bitte eine nichtnegative Zahl eingeben")
            threshold_box.set_val(f"{ui_state.distance_threshold:.1f}")
            return False
        ui_state.distance_threshold = value
        update_status(f"Schwellenwert gesetzt auf {value:.2f}")
        return True

    def handle_threshold_submit(text: str) -> None:
        apply_threshold(text)

    threshold_box.on_submit(handle_threshold_submit)

    resolution_ax = fig.add_axes([0.1, 0.12, 0.2, 0.05])
    resolution_box = TextBox(resolution_ax, "Voxel-Kante", initial=str(ui_state.grid_resolution))

    def apply_resolution(text: str) -> bool:
        try:
            value = int(text)
            if value <= 0:
                raise ValueError
        except ValueError:
            update_status("Ungültige Voxelzahl – bitte eine positive ganze Zahl eingeben")
            resolution_box.set_val(str(ui_state.grid_resolution))
            return False

        ui_state.grid_resolution = value
        voxels = value**3
        estimated_bytes = voxels  # bool array ~1 byte pro Voxel
        estimated_gib = estimated_bytes / (1024**3)
        if estimated_gib >= 1.0:
            update_status(f"Auflösung gesetzt auf {value}³ (~{estimated_gib:.1f} GiB Speicher)")
        else:
            update_status(
                f"Auflösung gesetzt auf {value}³ (~{estimated_bytes / (1024**2):.1f} MiB Speicher)"
            )
        return True

    def handle_resolution_submit(text: str) -> None:
        apply_resolution(text)

    resolution_box.on_submit(handle_resolution_submit)

    def perform_distance_fill() -> bool:
        if not apply_threshold(threshold_box.text):
            return False
        if not apply_resolution(resolution_box.text):
            return False
        snapshot = ui_state.latest_snapshot or current_snapshot
        if snapshot is None:
            update_status("Keine Snapshot-Daten verfügbar")
            return False
        if not snapshot.render_data:
            update_status("Snapshot enthält keine Rod-Geometrie")
            return False
        update_status(
            "Fülle basierend auf Distanzkarte … "
            f"(Auflösung {ui_state.grid_resolution}³)"
        )

        def progress_callback(progress: float, message: str) -> None:
            update_status(f"{message} – {progress * 100:.0f}%")
            fig.canvas.draw_idle()
            fig.canvas.flush_events()

        fig.canvas.draw()
        result = run_distance_fill(
            snapshot.render_data,
            ui_state.distance_threshold,
            grid_resolution=ui_state.grid_resolution,
            progress_cb=progress_callback,
        )
        ui_state.rolling_result = result
        filled_percent = result.fill_fraction * 100.0
        update_status(
            "Distanzfüllung fertig – "
            f"{result.filled_voxel_count} von {result.total_voxel_count} Voxeln "
            f"({filled_percent:.2f}%) bei {result.grid_resolution}³"
        )
        return True

    roll_ax = fig.add_axes([0.35, 0.17, 0.25, 0.06])
    roll_button = Button(roll_ax, "Distanzfüllung ausführen")
    roll_button.on_clicked(lambda _: perform_distance_fill())

    view_ax = fig.add_axes([0.64, 0.17, 0.26, 0.06])
    view_button = Button(view_ax, "3D-Ansicht anzeigen")

    def handle_view_click(_: object) -> None:
        if not apply_threshold(threshold_box.text):
            return
        if not apply_resolution(resolution_box.text):
            return
        if ui_state.rolling_result is None:
            if not perform_distance_fill():
                update_status("Distanzfüllung konnte nicht erzeugt werden")
                return
        ui_state.launch_rolling_viewer = True
        update_status("3D-Ansicht wird geöffnet")
        plt.close(fig)

    view_button.on_clicked(handle_view_click)

    export_ax = fig.add_axes([0.64, 0.08, 0.26, 0.06])
    export_button = Button(export_ax, "Distanzfüllung exportieren")

    def handle_export_click(_: object) -> None:
        if not apply_threshold(threshold_box.text):
            return
        if not apply_resolution(resolution_box.text):
            return
        if ui_state.rolling_result is None:
            update_status("Bitte zuerst die Distanzfüllung ausführen")
            return
        export_path = export_distance_fill(ui_state.rolling_result)
        update_status(f"Export gespeichert in {export_path}")

    export_button.on_clicked(handle_export_click)

    abort_ax = fig.add_axes([0.1, 0.08, 0.2, 0.06])
    abort_button = Button(abort_ax, "Simulation abbrechen", color="#c94c4c", hovercolor="#d96c6c")

    def handle_abort_click(_: object) -> None:
        ui_state.abort_requested = True
        update_status("Abbruch angefordert – schließe Simulation")
        stop_event.set()
        plt.close(fig)

    abort_button.on_clicked(handle_abort_click)

    pause_ax = fig.add_axes([0.35, 0.08, 0.25, 0.06])
    pause_button = Button(pause_ax, "Simulation pausieren", color="#4c78c9", hovercolor="#6c98e0")

    def handle_pause_click(_: object) -> None:
        if pause_event.is_set():
            pause_event.clear()
            pause_button.label.set_text("Simulation pausieren")
            update_status("Simulation fortgesetzt")
        else:
            pause_event.set()
            pause_button.label.set_text("Simulation fortsetzen")
            update_status("Simulation angehalten")

    pause_button.on_clicked(handle_pause_click)

    fig._perform_distance_fill = perform_distance_fill  # type: ignore[attr-defined]


def run_headless_phase(
    state_queue,
    stop_event,
    pause_event,
    current_snapshot: SimulationSnapshot,
) -> Tuple[SimulationSnapshot, HeadlessUIState]:
    plt.ion()
    fig, (ax_free, ax_cluster, ax_hist) = plt.subplots(3, 1, figsize=(7, 9))
    fig.subplots_adjust(bottom=0.28)
    ax_free.set_ylabel("Freie Enden (%)")
    ax_cluster.set_ylabel("Größter Cluster (% der Rods)")
    ax_cluster.set_xlabel("Simulation Samples")
    ax_free.grid(True, alpha=0.3)
    ax_cluster.grid(True, alpha=0.3)
    line_free, = ax_free.plot([], [], color="tab:blue", label="Freie Enden (%)")
    line_cluster, = ax_cluster.plot(
        [], [], color="tab:orange", label="Größter Cluster (% der Rods)"
    )
    ax_free.legend(loc="upper right")
    ax_cluster.legend(loc="upper right")
    ax_hist.set_xlabel("Clustergröße")
    ax_hist.set_ylabel("Anzahl")
    ax_hist.set_title("Verteilung der Clustergrößen")
    ax_hist.grid(True, alpha=0.3)
    fig.tight_layout(rect=(0.02, 0.28, 0.98, 0.98))

    ui_state = HeadlessUIState(latest_snapshot=current_snapshot)
    fig._status_text = fig.text(0.02, 0.02, "Statusbericht:")  # type: ignore[attr-defined]

    initial_voxels = ui_state.grid_resolution**3
    initial_gib = initial_voxels / (1024**3)
    memory_hint = (
        f"~{initial_gib:.1f} GiB" if initial_gib >= 1.0 else f"~{initial_voxels / (1024**2):.1f} MiB"
    )
    _update_status(
        fig,
        ui_state,
        "Bereit – "
        f"Schwelle {ui_state.distance_threshold:.2f}, Auflösung {ui_state.grid_resolution}³ ({memory_hint})",
    )

    _install_controls(fig, ui_state, state_queue, stop_event, pause_event, current_snapshot)

    sample_index = 0
    last_chart_update = 0.0

    while plt.fignum_exists(fig.number) and not stop_event.is_set():
        updated = False
        try:
            while True:
                current_snapshot = state_queue.get_nowait()
                ui_state.latest_snapshot = current_snapshot
                updated = True
        except queue.Empty:
            pass

        if updated:
            ui_state.history_indices.append(sample_index)
            rod_count = max(1, len(current_snapshot.render_data))
            total_endpoints = rod_count * 2
            free_percent = (current_snapshot.free_end_count / total_endpoints) * 100.0
            ui_state.history_free.append(free_percent)
            ui_state.history_cluster.append(current_snapshot.largest_cluster_percent)
            sample_index += 1

        now = time.perf_counter()
        if updated and now - last_chart_update >= 0.05:
            x_data = list(ui_state.history_indices)
            line_free.set_data(x_data, list(ui_state.history_free))
            line_cluster.set_data(x_data, list(ui_state.history_cluster))
            if x_data:
                ax_free.set_xlim(x_data[0], x_data[-1] if x_data[-1] > x_data[0] else x_data[0] + 1)
            ax_free.relim()
            ax_free.autoscale_view()
            ax_cluster.relim()
            ax_cluster.autoscale_view()
            ax_hist.cla()
            sizes = current_snapshot.cluster_sizes
            if sizes:
                counts = Counter(sizes)
                bins = sorted(counts.keys())
                heights = [counts[b] for b in bins]
                ax_hist.bar(bins, heights, color="tab:green", alpha=0.7)
                if bins:
                    min_bin = bins[0]
                    max_bin = bins[-1]
                    ticks = _compute_cluster_ticks(min_bin, max_bin)
                    ax_hist.set_xlim(min(min_bin, ticks[0]) - 0.5, max(max_bin, ticks[-1]) + 0.5)
                    ax_hist.set_xticks(ticks)
            else:
                ax_hist.text(
                    0.5,
                    0.5,
                    "Keine Cluster-Daten",
                    ha="center",
                    va="center",
                    transform=ax_hist.transAxes,
                )
            ax_hist.set_xlabel("Clustergröße")
            ax_hist.set_ylabel("Anzahl")
            ax_hist.set_title("Verteilung der Clustergrößen")
            ax_hist.grid(True, alpha=0.3)
            fig.canvas.draw_idle()
            fig.canvas.flush_events()
            last_chart_update = now

        plt.pause(0.001)

    plt.ioff()
    if plt.fignum_exists(fig.number):
        plt.close(fig)

    return current_snapshot, ui_state


__all__ = ["run_headless_phase"]
