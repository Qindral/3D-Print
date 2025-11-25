import numpy as np
import matplotlib.pyplot as plt

def gyroid_slice(L=100.0,
                 n=600,
                 z_slice=0.0,
                 k=2*np.pi/50.0,
                 threshold=0.0,
                 cmap='gray'):
    """
    2D-Schnitt einer Gyroid-TPMS-Struktur darstellen.

    Parameters
    ----------
    L : float
        Kantenlänge des Quadrats (z.B. in µm).
    n : int
        Anzahl der Gitterpunkte pro Achse (Auflösung).
    z_slice : float
        z-Koordinate, bei der der 2D-Schnitt genommen wird.
    k : float
        Wellenzahl; bestimmt Periodenlänge ~ 2*pi/k.
    threshold : float
        Offset t in der Gyroid-Gleichung; steuert das Volumenverhältnis
        von Gel-Phase zu Poren-Phase.
    cmap : str
        Matplotlib-Colormap für die Darstellung.
    """
    # 1) Koordinatenraster
    x = np.linspace(0, L, n)
    y = np.linspace(0, L, n)
    X, Y = np.meshgrid(x, y)

    # 2) In "Periodenraum" umrechnen
    #    (so kannst du L ändern, ohne die Perioden zu verlieren)
    Xk = k * X
    Yk = k * Y
    Zk = k * z_slice

    # 3) Gyroid-Feld
    phi = (
        np.sin(Xk) * np.cos(Yk) +
        np.sin(Yk) * np.cos(Zk) +
        np.sin(Zk) * np.cos(Xk)
        - threshold
    )

    # 4) Binäre Phasenmaske: phi > 0 = "Gel", phi < 0 = "Pore"
    gel = phi > 0

    # Plot
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_aspect('equal')
    ax.set_axis_off()

    # imshow: True = hell, False = dunkel (je nach cmap)
    im = ax.imshow(gel,
                   origin='lower',
                   extent=[0, L, 0, L],
                   cmap=cmap)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Beispiel: Gyroid-Schnitt in der Mitte, moderater threshold
    gyroid_slice(
        L=200.0,      # „µm“
        n=800,        # Auflösung (größer = glatter)
        z_slice=20.0, # Schnittebene
        k=3*np.pi/40.0,  # Periodenlänge ~ 40 µm
        threshold=1    # >0 -> Gel-Phase wird schlanker
    )
