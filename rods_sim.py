import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi

def make_voronoi_foam(n_points=60,
                      box_size=100.0,
                      line_width=2.0,
                      seed=0):
    """
    2D-Voronoi-"Foam" erzeugen und plotten.

    Parameters
    ----------
    n_points : int
        Anzahl der Zellzentren (steuert mittlere Porengröße).
    box_size : float
        Kantenlänge des quadratischen Ausschnitts (z.B. µm).
    line_width : float
        Dicke der Struts in Plot-Pixeln (nur Visualisierung).
    seed : int
        Zufallsseed für Reproduzierbarkeit.
    """
    rng = np.random.default_rng(seed)

    # 1) Zufällige Zellzentren im Quadrat [0, box_size]^2
    pts = rng.random((n_points, 2)) * box_size

    # 2) Voronoi-Tessellation
    vor = Voronoi(pts)

    # 3) Plot vorbereiten
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_aspect('equal')
    ax.set_xlim(0, box_size)
    ax.set_ylim(0, box_size)
    ax.axis('off')
    fig.patch.set_facecolor('black')
    ax.set_facecolor('black')

    # 4) Voronoi-Kanten zeichnen (als "Struts")
    #    Wir ignorieren unendliche Kanten (vertex = -1)
    for ridge in vor.ridge_vertices:
        if -1 in ridge:
            # unendliche Kante -> hier der Einfachheit halber überspringen
            continue
        v0, v1 = ridge
        x0, y0 = vor.vertices[v0]
        x1, y1 = vor.vertices[v1]

        # nur Segmente plotten, die im Betrachtungsfenster liegen
        if ((0 <= x0 <= box_size and 0 <= y0 <= box_size) or
            (0 <= x1 <= box_size and 0 <= y1 <= box_size)):
            ax.plot([x0, x1], [y0, y1],
                    linewidth=line_width,
                    color='white')

    # optional: die Zellzentren als kleine Punkte anzeigen
    # ax.scatter(pts[:, 0], pts[:, 1], s=5, c='red')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Beispielaufruf:
    make_voronoi_foam(
        n_points=120,     # mehr Punkte -> kleinere Poren
        box_size=100.0,   # „µm“; reine Skala
        line_width=2.5,   # erscheint „massiver“
        seed=42
    )
