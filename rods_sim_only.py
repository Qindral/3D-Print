import numpy as np

# -----------------------------------
# Globale Parameter
# -----------------------------------
BOX_SIZE_CM = 12          # "Simulationswürfel"
CHAIN_LENGTH_CM = 4.2       # Konturlänge pro Kette
SEGMENT_LENGTH_CM = 0.1     # 1 mm
N_CHAINS_INIT = 140          # Startanzahl WLC-Chains
KAPPA = 0.82                 # Steifigkeit
CHAIN_DIAMETER_MM = 3.0     # Durchmesser der Rods
CHAIN_RADIUS_CM = (CHAIN_DIAMETER_MM / 10.0) / 2.0  # 4 mm -> 0.4 cm -> r=0.2 cm

# Kontakt-Kriterium für "verbundene" Chains (leichte Überlappung)
CONTACT_FACTOR = 4
CONTACT_DIST_CM = .10 * CHAIN_RADIUS_CM * CONTACT_FACTOR

# Auflösung der Kugeln
N_THETA = 4
N_PHI = 4

# cm -> mm für STL
SCALE_TO_MM = True
SCALE_FACTOR = 10.0 if SCALE_TO_MM else 1.0


# -----------------------------------
# WLC-Generierung (unwrapped)
# -----------------------------------

def random_unit_vector() -> np.ndarray:
    v = np.random.normal(size=3)
    return v / np.linalg.norm(v)


def new_tangent(prev_t: np.ndarray, kappa: float) -> np.ndarray:
    if prev_t is None or kappa <= 0.0:
        return random_unit_vector()
    r = random_unit_vector()
    t = (1.0 - kappa) * r + kappa * prev_t
    return t / np.linalg.norm(t)


def generate_chain_unwrapped(chain_length: float,
                             segment_length: float,
                             kappa: float,
                             box_size: float) -> np.ndarray:
    n_segments = int(round(chain_length / segment_length))

    pos0 = np.random.rand(3) * box_size
    pos = pos0.copy()
    positions = [pos.copy()]

    t = random_unit_vector()

    for i in range(n_segments):
        if i > 0:
            t = new_tangent(t, kappa)
        pos = pos + segment_length * t
        positions.append(pos.copy())

    return np.array(positions)


def generate_system_unwrapped(n_chains: int,
                              box_size: float,
                              chain_length: float,
                              segment_length: float,
                              kappa: float):
    chains = []
    for _ in range(n_chains):
        chains.append(generate_chain_unwrapped(chain_length, segment_length, kappa, box_size))
    return chains


# -----------------------------------
# Union-Find für Chain-Konnektivität
# -----------------------------------

class UnionFind:
    def __init__(self, n):
        self.parent = np.arange(n)
        self.rank = np.zeros(n, dtype=int)

    def find(self, x):
        # Pfadkompression
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, x, y):
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return False
        if self.rank[rx] < self.rank[ry]:
            self.parent[rx] = ry
        elif self.rank[rx] > self.rank[ry]:
            self.parent[ry] = rx
        else:
            self.parent[ry] = rx
            self.rank[rx] += 1
        return True

    def num_components(self):
        roots = {self.find(i) for i in range(len(self.parent))}
        return len(roots)

    def components(self):
        comp = {}
        for i in range(len(self.parent)):
            r = self.find(i)
            comp.setdefault(r, []).append(i)
        return list(comp.values())


def chain_chain_min_distance(chain_a: np.ndarray,
                             chain_b: np.ndarray) -> tuple[float, int, int]:
    """
    Minimaler Abstand zwischen zwei Chains (brute force).
    Rückgabe: (dist, idx_a, idx_b)
    """
    # (Na,1,3) - (1,Nb,3) -> (Na,Nb,3)
    diff = chain_a[:, None, :] - chain_b[None, :, :]
    d2 = np.sum(diff**2, axis=2)
    idx_flat = np.argmin(d2)
    ia, ib = np.unravel_index(idx_flat, d2.shape)
    dist = np.sqrt(d2[ia, ib])
    return dist, ia, ib


def build_chain_connectivity(chains, contact_dist_cm: float):
    """
    Erstellt UnionFind über Chains: Chains sind verbunden,
    wenn irgendein Beadpaar näher als contact_dist_cm ist.
    """
    n = len(chains)
    uf = UnionFind(n)
    for i in range(n):
        for j in range(i + 1, n):
            dist, _, _ = chain_chain_min_distance(chains[i], chains[j])
            if dist <= contact_dist_cm:
                uf.union(i, j)
    return uf


def create_bridge_chain(chain_a: np.ndarray,
                        chain_b: np.ndarray,
                        idx_a: int,
                        idx_b: int,
                        segment_length: float) -> np.ndarray:
    """
    Erzeugt eine gerade Verbindungs-Chain zwischen zwei Beads
    (chain_a[idx_a] -> chain_b[idx_b]) mit Schrittweite segment_length.
    """
    p0 = chain_a[idx_a]
    p1 = chain_b[idx_b]
    v = p1 - p0
    L = np.linalg.norm(v)
    if L == 0:
        return np.array([p0.copy()])
    n_seg = max(1, int(np.ceil(L / segment_length)))
    t = v / n_seg
    points = [p0 + k * t for k in range(n_seg + 1)]
    return np.array(points)


def enforce_full_connectivity(chains,
                              segment_length: float,
                              contact_dist_cm: float):
    """
    Fügt so lange Verbindungs-Chains hinzu, bis alle Chains
    in einer einzigen Komponente sind.
    """
    while True:
        uf = build_chain_connectivity(chains, contact_dist_cm)
        comps = uf.components()
        n_comp = len(comps)
        print(f"Aktueller Zustand: {len(chains)} Chains, {n_comp} Komponenten")
        if n_comp <= 1:
            break

        # Suche global kürzestes Paar Chains aus verschiedenen Komponenten
        best_dist = np.inf
        best_pair = None
        best_indices = None  # (ia, ib)

        # Für jede Komponenten-Paarung
        for ci_idx in range(len(comps)):
            for cj_idx in range(ci_idx + 1, len(comps)):
                comp_i = comps[ci_idx]
                comp_j = comps[cj_idx]
                # Alle Chain-Paare zwischen diesen Komponenten
                for i in comp_i:
                    for j in comp_j:
                        dist, ia, ib = chain_chain_min_distance(chains[i], chains[j])
                        if dist < best_dist:
                            best_dist = dist
                            best_pair = (i, j)
                            best_indices = (ia, ib)

        if best_pair is None:
            print("Warnung: keine verbindbaren Komponenten gefunden.")
            break

        i, j = best_pair
        ia, ib = best_indices
        print(f"Verbinde Komponenten über Chains {i} und {j}, Abstand ~ {best_dist:.3f} cm")

        # Brückenkette bauen
        bridge = create_bridge_chain(chains[i], chains[j], ia, ib, segment_length)
        chains.append(bridge)

    return chains


# -----------------------------------
# Kugel-Mesh & STL
# -----------------------------------

def build_sphere_triangles(center: np.ndarray,
                           radius: float,
                           n_theta: int = 10,
                           n_phi: int = 10):
    cx, cy, cz = center
    theta = np.linspace(0.0, 2.0 * np.pi, n_theta, endpoint=False)
    phi = np.linspace(0.0, np.pi, n_phi)

    theta_grid, phi_grid = np.meshgrid(theta, phi)

    x = cx + radius * np.sin(phi_grid) * np.cos(theta_grid)
    y = cy + radius * np.sin(phi_grid) * np.sin(theta_grid)
    z = cz + radius * np.cos(phi_grid)

    triangles = []
    for i in range(n_phi - 1):
        for j in range(n_theta):
            jn = (j + 1) % n_theta

            p00 = np.array([x[i, j],    y[i, j],    z[i, j]])
            p01 = np.array([x[i, jn],   y[i, jn],   z[i, jn]])
            p10 = np.array([x[i+1, j],  y[i+1, j],  z[i+1, j]])
            p11 = np.array([x[i+1, jn], y[i+1, jn], z[i+1, jn]])

            triangles.append([p00, p01, p11])
            triangles.append([p00, p11, p10])

    return np.array(triangles)


def build_network_sphere_triangles(chains,
                                   radius: float,
                                   n_theta: int,
                                   n_phi: int,
                                   stride: int = 1):
    all_tris = []
    for chain in chains:
        # optional: Schwerpunkt ins Box-Intervall holen (rein kosmetisch)
        com = chain.mean(axis=0)
        shift = np.floor(com / BOX_SIZE_CM) * BOX_SIZE_CM
        chain_vis = chain - shift

        for idx in range(0, chain_vis.shape[0], stride):
            center = chain_vis[idx]
            tris = build_sphere_triangles(center, radius,
                                          n_theta=n_theta, n_phi=n_phi)
            if tris.size > 0:
                all_tris.append(tris)

    if not all_tris:
        return np.zeros((0, 3, 3), dtype=float)

    return np.vstack(all_tris)


def compute_normal(tri):
    v1 = tri[1] - tri[0]
    v2 = tri[2] - tri[0]
    n = np.cross(v1, v2)
    norm = np.linalg.norm(n)
    if norm == 0:
        return np.array([0.0, 0.0, 0.0])
    return n / norm


def write_ascii_stl(filename: str, triangles: np.ndarray, name: str = "wlc_connected_network"):
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


# -----------------------------------
# Main
# -----------------------------------

if __name__ == "__main__":
    np.random.seed(42)

    # 1) Startnetzwerk generieren
    chains = generate_system_unwrapped(
        n_chains=N_CHAINS_INIT,
        box_size=BOX_SIZE_CM,
        chain_length=CHAIN_LENGTH_CM,
        segment_length=SEGMENT_LENGTH_CM,
        kappa=KAPPA,
    )
    print(f"Initial: {len(chains)} Chains")

    # 2) Connectivity erzwingen (Perkolation / 1 Komponente)
    chains = enforce_full_connectivity(
        chains,
        segment_length=SEGMENT_LENGTH_CM,
        contact_dist_cm=CONTACT_DIST_CM,
    )
    print(f"Nach Verbindungsaufbau: {len(chains)} Chains (inkl. Bridges)")

    # 3) Sphere-Sweep-Mesh bauen
    print("Baue Sphere-Sweep-Mesh...")
    triangles = build_network_sphere_triangles(
        chains,
        radius=CHAIN_RADIUS_CM,
        n_theta=N_THETA,
        n_phi=N_PHI,
        stride=1,
    )
    print(f"Dreiecke im Mesh: {triangles.shape[0]}")

    # 4) STL schreiben
    out_file = "wlc_network_percolated_sphere_sweep.stl"
    write_ascii_stl(out_file, triangles, name="wlc_percolated_network")
    print(f"STL geschrieben nach: {out_file}")
    print(f"Einheiten: {'mm' if SCALE_TO_MM else 'cm'}")
