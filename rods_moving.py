import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (needed for 3D projection)

# -----------------------------
# Globale Parameter
# -----------------------------

BOX_SIZE_CM = 10.0       # Würfelgröße
CHAIN_LENGTH_CM = 5   # Konturlänge pro WLC
SEGMENT_LENGTH_CM = 0.15
N_CHAINS_INIT = 80*1   # Anzahl initialer Chains (anpassen je nach Speed)
KAPPA_WLC = 0.7        # Steifigkeit für initiale WLC-Geometrie

NETWORK_DIAMETER_MM = 2.0
NETWORK_RADIUS_CM = (NETWORK_DIAMETER_MM / 10.0) / 2.0  # 2 mm -> 0.2 cm -> r = 0.1 cm

# Dynamik-Parameter der Endpunkte
ENDPOINT_STEP_CM = 0.02       # Schrittweite pro Zeitschritt
KAPPA_MOVE = 0.9              # Persistenz der Bewegungsrichtung (WLC-like)
CAPTURE_RADIUS_CM = 2.0 * NETWORK_RADIUS_CM * 5.2  # "Bindungs-Radius"

# Simulation
MAX_STEPS = 500_000           # Max. Anzahl Schritte (Sicherheitslimit)
CHECK_INTERVAL = 1000       # Perkolation test alle 10k Schritte

# Tube-Mesh / STL
N_THETA_TUBE = 24            # hexagonaler Querschnitt
SCALE_TO_MM = True
SCALE_FACTOR = 10.0 if SCALE_TO_MM else 1.0

SAVE_PATH = r"C:\Users\Jonas\Downloads\3D_Print"
# Für Keyboard-Stop
STOP_SIMULATION = False  # wird von on_key() verändert


# -----------------------------
# Hilfsfunktionen
# -----------------------------

def random_unit_vector():
    v = np.random.normal(size=3)
    return v / np.linalg.norm(v)


def new_tangent(prev_t, kappa):
    """Persistent random walk für WLC-ähnliche Richtungsaktualisierung."""
    if prev_t is None or kappa <= 0.0:
        return random_unit_vector()
    r = random_unit_vector()
    t = (1.0 - kappa) * r + kappa * prev_t
    return t / np.linalg.norm(t)


def reflect_into_box(pos, box_size):
    """Reflektierende Randbedingungen im Würfel [0, box_size]^3."""
    p = pos.copy()
    for ax in range(3):
        if p[ax] < 0.0:
            p[ax] = -p[ax]
        if p[ax] > box_size:
            p[ax] = 2.0 * box_size - p[ax]
        p[ax] = np.clip(p[ax], 0.0, box_size)
    return p


def make_chain(points, radius_cm, kind="network", origin_id=None):
    return {
        "points": np.asarray(points, dtype=float),
        "radius": float(radius_cm),
        "kind":   kind,
        "origin_id": origin_id,
    }


# -----------------------------
# WLC-Initialisierung
# -----------------------------

def generate_wlc_chain(chain_id, chain_length, segment_length, kappa, radius_cm, box_size):
    pos = np.random.rand(3) * box_size
    n_segments = int(round(chain_length / segment_length))
    positions = [pos.copy()]

    t = random_unit_vector()
    for i in range(n_segments):
        if i > 0:
            t = new_tangent(t, kappa)
        pos = pos + segment_length * t
        pos = reflect_into_box(pos, box_size)
        positions.append(pos.copy())

    pts = np.array(positions)
    return make_chain(pts, radius_cm=radius_cm, kind="network", origin_id=chain_id)


def generate_wlc_system(n_chains, chain_length, segment_length, kappa, radius_cm, box_size):
    chains = []
    for cid in range(n_chains):
        chains.append(
            generate_wlc_chain(cid, chain_length, segment_length, kappa, radius_cm, box_size)
        )
    return chains


# -----------------------------
# Union-Find (für Perkolation)
# -----------------------------


class UnionFind:
    def __init__(self, n):
        # Use Python lists so we can grow the structure dynamically if new chains/links are added
        self.parent = list(range(n))
        self.rank = [0] * n
        self.components_count = n

    def _ensure(self, x):
        """Ensure internal arrays are large enough to include index x."""
        if x < len(self.parent):
            return
        old = len(self.parent)
        for i in range(old, x + 1):
            self.parent.append(i)
            self.rank.append(0)
            self.components_count += 1

    def find(self, x):
        self._ensure(x)
        # find root
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        # path compression
        while self.parent[x] != x:
            nxt = self.parent[x]
            self.parent[x] = root
            x = nxt
        return root

    def union(self, x, y):
        # allow unions with indices beyond initial size by expanding arrays
        self._ensure(x)
        self._ensure(y)
        rx = self.find(x)
        ry = self.find(y)
        if rx == ry:
            return False
        if self.rank[rx] < self.rank[ry]:
            self.parent[rx] = ry
        elif self.rank[rx] > self.rank[ry]:
            self.parent[ry] = rx
        else:
            self.parent[ry] = rx
            self.rank[rx] += 1
        self.components_count -= 1
        return True

    def n_components(self):
        # Komponenten-Zahl wird inkrementell gepflegt
        return self.components_count

    def components(self):
        comp = {}
        for i in range(len(self.parent)):
            r = self.find(i)
            comp.setdefault(r, []).append(i)
        return list(comp.values())


# -----------------------------
# Endpunkt-Dynamik
# -----------------------------

def init_endpoints_from_chains(chains):
    """
    Wir erzeugen pro Chain zwei Endpunkte (Start & Ende).
    Jeder Endpunkt bewegt sich unabhängig als persistent random walker.
    """
    endpoints = []
    for ci, ch in enumerate(chains):
        pts = ch["points"]
        if pts.shape[0] < 2:
            continue

        # Start-Ende
        p_start = pts[0].copy()
        t_start = pts[1] - pts[0]
        t_start /= np.linalg.norm(t_start)

        # Ende-Ende
        p_end = pts[-1].copy()
        t_end = pts[-1] - pts[-2]
        t_end /= np.linalg.norm(t_end)

        endpoints.append({
            "chain_index": ci,
            "pos": p_start,
            "t": t_start,
            "active": True,
            "which": "start",
        })
        endpoints.append({
            "chain_index": ci,
            "pos": p_end,
            "t": t_end,
            "active": True,
            "which": "end",
        })

    return endpoints


def step_endpoints(endpoints, box_size, step_len, kappa_move):
    """Ein Zeitschritt: alle aktiven Endpunkte bewegen sich WLC-artig."""
    for ep in endpoints:
        if not ep["active"]:
            continue
        t = ep["t"]
        t = new_tangent(t, kappa_move)
        ep["t"] = t
        pos = ep["pos"] + step_len * t
        ep["pos"] = reflect_into_box(pos, box_size)


# -----------------------------
# Broadphase: Gitter für Capture-Radius
# -----------------------------

def build_spatial_grid(endpoints, cell_size):
    """
    Einfaches uniform grid:
    Map von Zelle (ix,iy,iz) -> Liste von Endpoint-Indizes.
    Nur aktive Endpunkte werden eingetragen.
    """
    grid = {}
    for idx, ep in enumerate(endpoints):
        if not ep["active"]:
            continue
        pos = ep["pos"]
        cell = tuple((pos // cell_size).astype(int))
        grid.setdefault(cell, []).append(idx)
    return grid


def neighbor_cells(cell):
    cx, cy, cz = cell
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                yield (cx + dx, cy + dy, cz + dz)


# -----------------------------
# Bindungen durch Capture-Radius
# -----------------------------

def process_captures(endpoints, chains, uf, capture_radius):
    """
    Überprüft Endpunkte innerhalb des Capture-Radius.
    Wenn zwei aktive Endpunkte unterschiedlicher Komponenten sich "treffen",
    wird:
      - eine neue Link-Chain erzeugt (gerade Verbindung)
      - Union-Find auf die zugehörigen Chains angewendet
      - beide Endpunkte deaktiviert (jeder Endpunkt bindet nur einmal)
    """
    capture_radius2 = capture_radius ** 2
    grid = build_spatial_grid(endpoints, cell_size=capture_radius)

    new_links = []

    for cell, indices in grid.items():
        # Lokale Nachbarschafts-Suche
        candidate_indices = set(indices)
        for nc in neighbor_cells(cell):
            if nc in grid:
                candidate_indices.update(grid[nc])

        candidate_indices = list(candidate_indices)
        n = len(candidate_indices)
        for i in range(n):
            ei = candidate_indices[i]
            epi = endpoints[ei]
            if not epi["active"]:
                continue
            ci = epi["chain_index"]
            pi = epi["pos"]
            for j in range(i + 1, n):
                ej = candidate_indices[j]
                epj = endpoints[ej]
                if not epj["active"]:
                    continue
                cj = epj["chain_index"]
                if uf.find(ci) == uf.find(cj):
                    continue  # schon in gleicher Komponente

                pj = epj["pos"]
                d2 = np.sum((pi - pj) ** 2)
                if d2 <= capture_radius2:
                    # Bindung!
                    uf.union(ci, cj)
                    endpoints[ei]["active"] = False
                    endpoints[ej]["active"] = False

                    link_pts = np.vstack([pi, pj])
                    link_chain = make_chain(
                        link_pts,
                        radius_cm=NETWORK_RADIUS_CM,
                        kind="link",
                        origin_id=f"link_{ci}_{cj}",
                    )
                    new_links.append(link_chain)

    return new_links


# -----------------------------
# Plot-Funktionen
# -----------------------------

def plot_network(chains, link_chains, box_size, title="Final Network"):
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")

    for ch in chains:
        pts = ch["points"]
        ax.plot(pts[:, 0], pts[:, 1], pts[:, 2],
                lw=0.6, alpha=0.6, color="tab:blue")

    for ch in link_chains:
        pts = ch["points"]
        ax.plot(pts[:, 0], pts[:, 1], pts[:, 2],
                lw=1.5, alpha=0.9, color="tab:red")

    ax.set_xlim(0, box_size)
    ax.set_ylim(0, box_size)
    ax.set_zlim(0, box_size)
    ax.set_xlabel("x [cm]")
    ax.set_ylabel("y [cm]")
    ax.set_zlabel("z [cm]")
    ax.set_title(title)
    plt.tight_layout()
    plt.show()


def plot_percolation_history(history):
    if not history:
        return
    steps, comps = zip(*history)
    plt.figure()
    plt.plot(steps, comps, "-o")
    plt.xlabel("Steps")
    plt.ylabel("Number of components")
    plt.title("Percolation evolution")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# -----------------------------
# Keyboard-Handling
# -----------------------------

def on_key(event=None):
    global STOP_SIMULATION
    if event is None:
        print("[Key] Stop requested (no event).")
    else:
        print(f"[Key] Taste '{event.key}' gedrückt -> Simulation wird beendet.")
    STOP_SIMULATION = True


# -----------------------------
# Tube-Mesh / STL
# -----------------------------

def make_local_frame(direction):
    ez = direction / np.linalg.norm(direction)
    ref = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(ref, ez)) > 0.9:
        ref = np.array([0.0, 1.0, 0.0])
    ex = np.cross(ez, ref)
    ex /= np.linalg.norm(ex)
    ey = np.cross(ez, ex)
    ey /= np.linalg.norm(ey)
    return ex, ey, ez


def build_tube_for_chain(points, radius, n_theta=6):
    pts = np.asarray(points)
    M = pts.shape[0]
    if M < 2:
        return np.zeros((0, 3, 3))

    theta = np.linspace(0.0, 2.0 * np.pi, n_theta, endpoint=False)
    rings = []

    for i in range(M):
        if i == 0:
            direction = pts[1] - pts[0]
        elif i == M - 1:
            direction = pts[-1] - pts[-2]
        else:
            direction = pts[i+1] - pts[i-1]

        ex, ey, ez = make_local_frame(direction)
        c = pts[i]
        ring = []
        for th in theta:
            ring.append(c + radius * (np.cos(th) * ex + np.sin(th) * ey))
        rings.append(np.array(ring))

    rings = np.array(rings)
    triangles = []

    # Mantel
    for i in range(M - 1):
        r0 = rings[i]
        r1 = rings[i + 1]
        for j in range(n_theta):
            jn = (j + 1) % n_theta
            p00 = r0[j]
            p01 = r0[jn]
            p10 = r1[j]
            p11 = r1[jn]
            triangles.append([p00, p01, p11])
            triangles.append([p00, p11, p10])

    # Endkappen
    center0 = pts[0]
    r0 = rings[0]
    for j in range(1, n_theta - 1):
        triangles.append([center0, r0[j], r0[j+1]])

    center1 = pts[-1]
    r1 = rings[-1]
    for j in range(1, n_theta - 1):
        triangles.append([center1, r1[j+1], r1[j]])

    return np.array(triangles)


def build_tube_mesh_for_chains(chains, n_theta, label=""):
    all_tris = []
    tubes_total = 0

    for ch in chains:
        pts = ch["points"]
        r = ch["radius"]
        if pts.shape[0] < 2:
            continue

        tris = build_tube_for_chain(pts, r, n_theta=n_theta)
        if tris.size == 0:
            continue

        tubes_total += 1
        all_tris.append(tris)

    if all_tris:
        tris_arr = np.vstack(all_tris)
    else:
        tris_arr = np.zeros((0, 3, 3))

    print(f"[{label}] Chains: {len(chains)}, Tubes: {tubes_total}, Dreiecke: {tris_arr.shape[0]}")
    return tris_arr


def compute_normal(tri):
    v1 = tri[1] - tri[0]
    v2 = tri[2] - tri[0]
    n = np.cross(v1, v2)
    norm = np.linalg.norm(n)
    if norm == 0:
        return np.array([0.0, 0.0, 0.0])
    return n / norm


def write_ascii_stl(filename, triangles, name="mesh"):
    with open(filename, "w") as f:
        f.write(f"solid {name}\n")
        for tri in triangles:
            tri_scaled = tri * SCALE_FACTOR
            n = compute_normal(tri_scaled)
            f.write(f"  facet normal {n[0]:.6e} {n[1]:.6e} {n[2]:.6e}\n")
            f.write("    outer loop\n")
            for v in tri_scaled:
                f.write(f"      vertex {v[0]:.6e} {v[1]:.6e} {v[2]:.6e}\n")
            f.write("    endloop\n")
            f.write("  endfacet\n")
        f.write(f"endsolid {name}\n")


# -----------------------------
# Main
# -----------------------------

if __name__ == "__main__":
    np.random.seed(42)

    # 1) Initiales WLC-Netzwerk
    chains = generate_wlc_system(
        n_chains=N_CHAINS_INIT,
        chain_length=CHAIN_LENGTH_CM,
        segment_length=SEGMENT_LENGTH_CM,
        kappa=KAPPA_WLC,
        radius_cm=NETWORK_RADIUS_CM,
        box_size=BOX_SIZE_CM,
    )
    print(f"Initial network chains: {len(chains)}")

    # 2) Union-Find über Chains
    uf = UnionFind(len(chains))

    # 3) Endpunkte initialisieren
    endpoints = init_endpoints_from_chains(chains)
    print(f"Mobile endpoints (start+end): {len(endpoints)}")

    # 4) Link-Chains-Sammlung
    link_chains = []

    # 5) Perkolations-History
    percolation_history = []

    # 6) Matplotlib-Setup für Tastendruck
    plt.ion()
    fig_hint = plt.figure(figsize=(4, 3))
    ax_hint = fig_hint.add_subplot(111)
    ax_hint.text(0.5, 0.5,
                 "Simulation läuft...\n\nTaste drücken zum Stoppen",
                 ha="center", va="center")
    ax_hint.set_axis_off()
    fig_hint.canvas.mpl_connect("key_press_event", on_key)
        # Capture / Bindung
    new_links = process_captures(endpoints, chains, uf, CAPTURE_RADIUS_CM)
    if new_links:
        # collect link chains for mesh/export and account for connectivity (uf already updated)
        link_chains.extend(new_links)
    if not STOP_SIMULATION:

        STOP_SIMULATION = False  # kein global-statement nötig hier

    for step in range(1, MAX_STEPS + 1):
        # ensure matplotlib processes GUI events so on_key() is called
        try:
            fig_hint.canvas.flush_events()
            plt.pause(0.001)
        except Exception:
            pass

        if STOP_SIMULATION:
            print(f">> Simulation durch Tastendruck bei step {step} gestoppt.")
            break
        # Bewegung der Endpunkte
        step_endpoints(endpoints, BOX_SIZE_CM, ENDPOINT_STEP_CM, KAPPA_MOVE)

        # Capture / Bindung
        #new_links = process_captures(endpoints, chains, uf, CAPTURE_RADIUS_CM)
        #if new_links:
          #  link_chains.extend(new_links)

        # Perkolation alle CHECK_INTERVAL Schritte testen
        if step % CHECK_INTERVAL == 0:
            # Periodic capture check: use the top-level process_captures and register any new links
            new_links = process_captures(endpoints, chains, uf, CAPTURE_RADIUS_CM)
            if new_links:
                # collect link chains for mesh/export and account for connectivity
                link_chains.extend(new_links)
                for link in new_links:
                    # register link as a real chain so it's part of the system
                    new_idx = len(chains)
                    chains.append(link)
                    uf._ensure(new_idx)

                    # try to union the new link with its origin chains if origin_id encodes them
                    origin = link.get("origin_id")
                    if isinstance(origin, str) and origin.startswith("link_"):
                        try:
                            parts = origin.split("_")
                            ci = int(parts[1]); cj = int(parts[2])
                            uf.union(new_idx, ci)
                            uf.union(new_idx, cj)
                        except Exception:
                            pass

                    # create endpoints for the new link so it can bind further
                    pts = link["points"]
                    if pts.shape[0] >= 2:
                        pi = pts[0]; pj = pts[-1]
                        dir_vec = pj - pi
                        norm = np.linalg.norm(dir_vec) or 1.0
                        t_link = dir_vec / norm
                        endpoints.append({
                            "chain_index": new_idx,
                            "pos": pi.copy(),
                            "t": t_link.copy(),
                            "active": True,
                            "which": "link_a",
                        })
                        endpoints.append({
                            "chain_index": new_idx,
                            "pos": pj.copy(),
                            "t": (-t_link).copy(),
                            "active": True,
                            "which": "link_b",
                        })

            n_comp = uf.n_components()
            percolation_history.append((step, n_comp))
            print(f"[Step {step}] Components = {n_comp}")
            if n_comp == 1:
                print(">> Voll perkoliert! (Eine einzige Komponente)")
                break



            
    # 8) Finale Auswertung & Plots
    plt.ioff()
    plt.close(fig_hint)

    print("Simulation beendet.")
    print(f"Final components: {uf.n_components()}")
    print(f"Total link chains formed: {len(link_chains)}")

    # Netzwerk und Links 3D anzeigen
    plot_network(chains, link_chains, BOX_SIZE_CM,
                 title="Final network with dynamic endpoint links")

    # Perkolations-History (Komponenten vs. Schritte)
    plot_percolation_history(percolation_history)

    # 9) STL-Export des finalen Netzwerks (Initial + Links)
    all_chains_for_stl = chains + link_chains
    tris_all = build_tube_mesh_for_chains(
        all_chains_for_stl,
        n_theta=N_THETA_TUBE,
        label="dynamic_network"
    )
    write_ascii_stl(f"{SAVE_PATH}\\dynamic_wlc_network.stl", tris_all, name="dynamic_wlc_network")
    print(f"STL geschrieben nach: {SAVE_PATH}\\dynamic_wlc_network.stl")
    print(f"Einheiten im STL: {'mm' if SCALE_TO_MM else 'cm'}")
