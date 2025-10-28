# 3D-Print

## Rod Brownian Motion Simulation

Install the required dependencies and run the simulation to watch 500 rods moving in a cube while forming connections between complementary ends. The program now starts with a fast headless phase that streams analytics into a Matplotlib chart and exposes a small control panel. From there you can abort the physics run, launch a "rolling ball" post-processing step, and open a dedicated 3D viewer for the resulting organic volume before proceeding to the main rod renderer.

```bash
pip install -r requirements.txt
python rod_simulation.py
```

### Controls

* **Headless analytics + UI** – While the Matplotlib window is open the simulation runs headlessly as fast as possible. Use the buttons to abort, trigger the rolling-ball pass (enter a ball radius in simulation units), or open the generated organic shape in a 3D viewer. Close the chart window to move on to the real-time rod renderer.
* **Rolling-ball viewer** – After running the rolling-ball pass, click the "3D-Ansicht anzeigen" button to inspect the smoothed voxel cloud. Use the same controls as the rod viewer (mouse drag to orbit, `WASD` + `Space`/`Ctrl` to move, `Shift` to sprint, `Esc` to close).
* **Main rod viewer** – When the Matplotlib window closes (either manually or after launching the rolling-ball viewer) the Pygame window opens and renders the live rod simulation. Camera controls match the rolling-ball view.

### Running the simulation online

The simulation relies on Pygame to open a desktop window, so it runs most reliably on your own computer where a graphical display is available. If you prefer to work in the cloud, consider one of the following options:

* **GitHub Codespaces or a remote Linux VM with desktop access** – start a Codespace or VM, install the dependencies above, and launch a virtual desktop session (for example, with VS Code's Remote Desktop or a VNC server) to view the Pygame window.
* **Replit or similar sandbox services** – create a new Python Repl, upload the project files, install the requirements, and enable a graphical desktop extension (e.g. Replit's "Replit Desktop" or X11/VNC add-on) before running `python rod_simulation.py`.
* **WSL on Windows** – enable the Windows Subsystem for Linux, install an X server such as X410 or VcXsrv, and run the simulation from the WSL terminal to display the window on your desktop.

These approaches emulate a local graphical environment so the Pygame window can render. In headless shells without a display server, the program will start but no window will appear.
