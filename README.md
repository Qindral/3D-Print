# 3D-Print

## Rod Brownian Motion Simulation

Install the required dependencies and run the simulation to watch 500 rods moving in a cube while forming connections between complementary ends. The program now starts with a fast headless phase that streams analytics into a Matplotlib chart and exposes a small control panel. From there you can abort the physics run, compute a distance-map-based organic fill (complete with a live progress readout and a configurable distance threshold), and open a dedicated 3D viewer for the resulting organic volume before proceeding to the main rod renderer.

```bash
pip install -r requirements.txt
python rod_simulation.py
```

### Controls

* **Headless analytics + UI** – While the Matplotlib window is open the simulation runs headlessly as fast as possible. Use the buttons to abort, trigger the distance-map fill (enter a non-negative threshold in simulation units), export the organic volume as an OBJ file, or open the generated organic shape in a 3D viewer. Before you start the distance fill, set the voxel resolution via the `Voxel-Kante` field (the status line shows the estimated memory footprint – values in the thousands quickly require dozens of GiB). The status bar reports fill progress percentages while the distance map is computed. Close the chart window to move on to the real-time rod renderer.
* **Distance-fill viewer** – After running the distance-map pass (or by opening the 3D view, which now triggers the fill automatically with the current parameters if no result exists yet), click the "3D-Ansicht anzeigen" button to inspect the smoothed voxel cloud. Rods are voxelised as filled volumes based on their physical thickness, so the viewer shows a dense medium instead of thin lines. Use the same controls as the rod viewer (mouse drag to orbit, `WASD` + `Space`/`Ctrl` to move, `Shift` to sprint, `Esc` to close).
* **Main rod viewer** – When the Matplotlib window closes (either manually or after launching the rolling-ball viewer) the Pygame window opens and renders the live rod simulation. The default field of view is calibrated so the simulation cube fills the camera when it opens. Camera controls match the rolling-ball view.

### Daten speichern und weiterverarbeiten

* **Automatische Zustandsdatei** – Sobald du die Simulation verlässt oder im Headless-Fenster auf „Simulation abbrechen" klickst, speichert das Programm die aktuellen Rod-Positionen, Orientierungen und Verbindungen nach `rod_state.json`. Beim nächsten Start wird dieser Zustand automatisch geladen, sodass du ohne erneuten Headless-Lauf weiterarbeiten kannst. Lösche die Datei, wenn du wieder mit einem zufälligen Startzustand beginnen möchtest.
* **Distance-Fill-Export** – Nach einem Distanzfüllungs-Lauf kannst du über den Button „Distanzfüllung exportieren" eine OBJ-Datei (`rolling_ball_surface.obj`) schreiben lassen. Die Datei enthält die gefüllte Oberfläche als Punktwolke und dient als Ausgangsbasis für weitere Modellierungsschritte.

### Running the simulation online

The simulation relies on Pygame to open a desktop window, so it runs most reliably on your own computer where a graphical display is available. If you prefer to work in the cloud, consider one of the following options:

* **GitHub Codespaces or a remote Linux VM with desktop access** – start a Codespace or VM, install the dependencies above, and launch a virtual desktop session (for example, with VS Code's Remote Desktop or a VNC server) to view the Pygame window.
* **Replit or similar sandbox services** – create a new Python Repl, upload the project files, install the requirements, and enable a graphical desktop extension (e.g. Replit's "Replit Desktop" or X11/VNC add-on) before running `python rod_simulation.py`.
* **WSL on Windows** – enable the Windows Subsystem for Linux, install an X server such as X410 or VcXsrv, and run the simulation from the WSL terminal to display the window on your desktop.

These approaches emulate a local graphical environment so the Pygame window can render. In headless shells without a display server, the program will start but no window will appear.
