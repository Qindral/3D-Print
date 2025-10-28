"""Startup prompt for choosing the initial simulation state."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import matplotlib.pyplot as plt
from matplotlib.widgets import Button

from .constants import BASE_ROD_THICKNESS, CUBE_SIZE, NUM_RODS, ROD_LENGTH, STATE_FILE_PATH
from .persistence import inspect_saved_state


@dataclass
class _StartupSelection:
    choice: str | None = None  # "saved" or "new"
    confirmed: bool = False


def _format_volume_summary() -> str:
    cube_volume = CUBE_SIZE ** 3
    rod_radius = BASE_ROD_THICKNESS * 0.5
    rod_volume = math.pi * (rod_radius ** 2) * ROD_LENGTH
    total_volume = rod_volume * NUM_RODS
    if cube_volume <= 0:
        return "Volumenangaben nicht verfügbar"
    fraction = max(0.0, min(1.0, total_volume / cube_volume)) * 100.0
    return (
        "Startparameter für neuen Zustand:\n"
        f"• Anzahl Rods: {NUM_RODS}\n"
        f"• Rod-Länge: {ROD_LENGTH:.1f}\n"
        f"• Rod-Durchmesser: {BASE_ROD_THICKNESS:.1f}\n"
        f"• Würfelkante: {CUBE_SIZE:.1f}\n"
        f"• Volumenkonzentration: {fraction:.2f}%"
    )


def _format_saved_summary() -> str:
    info = inspect_saved_state()
    if info is None:
        return (
            "Kein gespeicherter Zustand gefunden.\n"
            f"Es wird ein neuer Zustand in {STATE_FILE_PATH} angelegt."
        )

    timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(info.timestamp))
    rod_count = info.rod_count
    rod_length = info.metadata.get("rod_length")
    cube_size = info.metadata.get("cube_size")
    max_group = info.metadata.get("max_group_size")

    lines = [
        f"Gespeicherter Zustand vom {timestamp}",
        f"• Rods: {rod_count}",
    ]
    if rod_length is not None:
        lines.append(f"• Rod-Länge: {rod_length:.1f}")
    if cube_size is not None:
        lines.append(f"• Würfelkante: {cube_size:.1f}")
    if max_group is not None:
        lines.append(f"• Maximale Verbindungen pro Knoten: {int(max_group)}")
    return "\n".join(lines)


def prompt_start_choice() -> bool:
    """Return True to resume from saved state, False to start fresh."""

    selection = _StartupSelection()

    plt.ion()
    fig = plt.figure(figsize=(6, 4))
    fig.subplots_adjust(bottom=0.32)
    fig.suptitle("Startzustand wählen")
    info_text = fig.text(0.05, 0.6, "Bitte Auswahl treffen …", va="top")

    saved_available = STATE_FILE_PATH.exists()

    resume_ax = fig.add_axes([0.1, 0.18, 0.35, 0.1])
    resume_button = Button(resume_ax, "Gespeicherten Zustand nutzen")

    new_ax = fig.add_axes([0.55, 0.18, 0.35, 0.1])
    new_button = Button(new_ax, "Neu starten")

    continue_ax = fig.add_axes([0.3, 0.05, 0.4, 0.1])
    continue_button = Button(continue_ax, "Weiter")

    def update_info(text: str) -> None:
        info_text.set_text(text)
        fig.canvas.draw_idle()

    def handle_resume(_: object) -> None:
        if not saved_available:
            update_info(
                "Es ist kein gespeicherter Zustand vorhanden.\n"
                "Bitte 'Neu starten' wählen."
            )
            return
        selection.choice = "saved"
        update_info(_format_saved_summary())

    def handle_new(_: object) -> None:
        selection.choice = "new"
        update_info(_format_volume_summary())

    def handle_continue(_: object) -> None:
        if selection.choice is None:
            update_info("Bitte zuerst eine Option wählen.")
            return
        selection.confirmed = True
        plt.close(fig)

    resume_button.on_clicked(handle_resume)
    new_button.on_clicked(handle_new)
    continue_button.on_clicked(handle_continue)

    if not saved_available:
        update_info(
            "Es wurde kein gespeicherter Zustand gefunden.\n"
            "Bitte wählen Sie 'Neu starten', um zu beginnen."
        )

    plt.show(block=False)
    while not selection.confirmed and plt.fignum_exists(fig.number):
        plt.pause(0.05)

    plt.ioff()
    if plt.fignum_exists(fig.number):
        plt.close(fig)

    return selection.choice == "saved" and saved_available


__all__ = ["prompt_start_choice"]
