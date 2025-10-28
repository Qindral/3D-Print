# 3D-Print

## Rod Brownian Motion Simulation

Install the required dependencies and run the simulation to watch 500 rods moving in a cube while forming connections between complementary ends. The main window renders the rods in 3D while a second analytics window charts the number of free endpoints and the size of the largest connected structure over time.

```bash
pip install -r requirements.txt
python rod_simulation.py
```

### Controls

* **Rotate view** – Hold the left mouse button and drag to orbit the camera around the cube.
* **Move camera** – Use `W`/`S` to move forward/backward, `A`/`D` to strafe, and `Space`/`Ctrl` (or `R`/`F`) to rise/lower. Hold `Shift` to move faster.
* **Analytics window** – A Matplotlib window opens alongside the renderer and plots the count of free ends and the size of the largest connected cluster.

### Running the simulation online

The simulation relies on Pygame to open a desktop window, so it runs most reliably on your own computer where a graphical display is available. If you prefer to work in the cloud, consider one of the following options:

* **GitHub Codespaces or a remote Linux VM with desktop access** – start a Codespace or VM, install the dependencies above, and launch a virtual desktop session (for example, with VS Code's Remote Desktop or a VNC server) to view the Pygame window.
* **Replit or similar sandbox services** – create a new Python Repl, upload the project files, install the requirements, and enable a graphical desktop extension (e.g. Replit's "Replit Desktop" or X11/VNC add-on) before running `python rod_simulation.py`.
* **WSL on Windows** – enable the Windows Subsystem for Linux, install an X server such as X410 or VcXsrv, and run the simulation from the WSL terminal to display the window on your desktop.

These approaches emulate a local graphical environment so the Pygame window can render. In headless shells without a display server, the program will start but no window will appear.
