# 3D-Print

## Rod Brownian Motion Simulation

Install the required dependencies and run the simulation to watch 500 rods moving in a cube while forming connections between complementary ends. Beim Programmstart erscheint zunächst ein kleines Auswahlfenster, in dem du festlegst, ob der zuletzt gespeicherte Zustand geladen oder eine frische Simulation mit den Standardparametern erzeugt werden soll (inklusive Anzeige der wichtigsten Kennzahlen wie Volumenkonzentration). Danach folgt wie gewohnt der schnelle Headless-Lauf mit Matplotlib-Dashboard, über den du Statistiken sammelst und weitere Schritte vorbereitest.

```bash
pip install -r requirements.txt
python rod_simulation.py
```

To experiment with larger systems, set the `ROD_SIM_NUM_RODS` environment variable before launching (for example `ROD_SIM_NUM_RODS=20000 python rod_simulation.py`). The optimised worker and rendering pipeline can handle rod counts well into the tens of thousands, especially when you pause the simulation to inspect specific snapshots.

### Controls

* **Headless analytics + UI** – While the Matplotlib window is open the simulation runs headlessly as fast as possible. Use the buttons to abort, pause or resume the Brownian worker, trigger the distance-map fill (enter a non-negative threshold in simulation units), export the organic volume as an OBJ file, or open the generated organic shape in a 3D viewer. The chart now contains a normalised 30-bin histogram that highlights the relative cluster-size distribution alongside the time-series plots for free ends and the largest cluster percentage. Hitting the pause button (or `P` in the live viewer later) now halts the worker and freezes the charts so you can inspect the exact snapshot even when tens of thousands of rods are simulated. Before you start the distance fill, set the voxel resolution via the `Voxel-Kante` field (the status log shows the estimated memory footprint – values in the thousands quickly require dozens of GiB). The multi-line status report at the bottom keeps the last few messages, including detailed progress while the distance map is voxelised and thresholded. Close the chart window to move on to the real-time rod renderer.
* **Distance-fill viewer** – After running the distance-map pass (or by opening the 3D view, which now triggers the fill automatically with the current parameters if no result exists yet), click the "3D-Ansicht anzeigen" button to inspect the smoothed voxel cloud. Rods are voxelised as filled volumes based on their physical thickness, so the viewer shows a dense medium instead of thin lines. Both 3D viewers start with the neutral default camera setup; use the mouse and movement keys (`WASD`, `Space`/`Ctrl`, `Shift`) to position the view as needed and press `Esc` to close.
* **Main rod viewer** – When the Matplotlib window closes (either manually or after launching the rolling-ball viewer) the Pygame window opens and renders the live rod simulation. Camera controls match the rolling-ball view, and pressing `P` toggles the pause state. The overlay and window caption include the pause indicator so you can confirm the worker is stopped before exporting or changing parameters.

### Daten speichern und weiterverarbeiten

* **Startauswahl & automatische Zustandsdatei** – Beim Start fragt dich das Programm, ob du mit dem gespeicherten Zustand aus `rod_state.json` weiterarbeiten möchtest. Entscheidest du dich für einen Neustart, zeigt dir das Fenster noch einmal die wichtigsten Parameter und die aktuelle Volumenkonzentration der Rods im Würfel an, bevor es weitergeht. Sobald du die Simulation verlässt oder im Headless-Fenster auf „Simulation abbrechen" klickst, speichert das Programm die aktuellen Rod-Positionen, Orientierungen und Verbindungen nach `rod_state.json`. Lösche die Datei, wenn du später wieder mit einem zufälligen Startzustand beginnen möchtest.
* **Distance-Fill-Export** – Nach einem Distanzfüllungs-Lauf kannst du über den Button „Distanzfüllung exportieren" eine OBJ-Datei (`rolling_ball_surface.obj`) schreiben lassen. Die Datei enthält die gefüllte Oberfläche als Punktwolke und dient als Ausgangsbasis für weitere Modellierungsschritte.

### Running the simulation online

The simulation relies on Pygame to open a desktop window, so it runs most reliably on your own computer where a graphical display is available. If you prefer to work in the cloud, consider one of the following options:

* **GitHub Codespaces or a remote Linux VM with desktop access** – start a Codespace or VM, install the dependencies above, and launch a virtual desktop session (for example, with VS Code's Remote Desktop or a VNC server) to view the Pygame window.
* **Replit or similar sandbox services** – create a new Python Repl, upload the project files, install the requirements, and enable a graphical desktop extension (e.g. Replit's "Replit Desktop" or X11/VNC add-on) before running `python rod_simulation.py`.
* **WSL on Windows** – enable the Windows Subsystem for Linux, install an X server such as X410 or VcXsrv, and run the simulation from the WSL terminal to display the window on your desktop.

These approaches emulate a local graphical environment so the Pygame window can render. In headless shells without a display server, the program will start but no window will appear.
