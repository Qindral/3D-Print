import numpy as np

# -----------------------------------
# Globale Parameter
# -----------------------------------

# Äußerer Simulationswürfel
BOX_SIZE_CM = 12.0

# Innerer Bereich (9x9x9 cm), zentriert
INNER_BOX_CM = 9.0
INNER_MIN_CM = (BOX_SIZE_CM - INNER_BOX_CM) / 2.0  # 1.5 cm
INNER_MAX_CM = INNER_MIN_CM + INNER_BOX_CM         # 10.5 cm

# WLC-Parameter
CHAIN_LENGTH_CM = 5.0         # Konturlänge pro WLC
SEGMENT_LENGTH_CM = 0.1       # 1 mm Schritte
N_CHAINS_INIT = 120           # Startanzahl WLC-Chains
KAPPA = 0.8                   # Steifigkeit [0..1]

# Dicke der verschiedenen Rod-Typen
NETWORK_DIAMETER_MM = 3.0
NETWORK_RADIUS_CM = (NETWORK_DIAMETER_MM / 10.0) / 2.0  # 3 mm -> 0.3 cm -> r=0.15 cm

FRAME_DIAMETER_MM = 1.5       # dünnere Stäbe für Rahmen
FRAME_RADIUS_CM = (FRAME_DIAMETER_MM / 10.0) / 2.0

SUPPORT_DIAMETER_MM = 3.0     # z.B. so dick wie Netzwerk
SUPPORT_RADIUS_CM = (SUPPORT_DIAMETER_MM / 10.0) / 2.0

# Kontakt-Kriterium für Konnektivität (auf Basis Netzwerkradius)
CONTACT_FACTOR = 1.05
CONTACT_DIST_CM = 2.0 * NETWORK_RADIUS_CM * CONTACT_FACTOR

# Auflösung der Kugeln
N_THETA = 10
N_PHI = 10

# cm -> mm für STL
SCALE_TO_MM = True
SCALE_FACTOR = 10.0 if SCALE_TO_MM else 1.0


# -----------------------------------
# Hilfsfunktionen / Datentyp
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


def make_chain(points: np.ndarray, radius_cm: float, kind: str = "network") -> dict:
    """
    Chain-Objekt:
      {
        "points": (N,3) array in cm,
        "radius": float in cm,
        "kind":   string ("network", "frame", "support", "bridge", ...)
      }
    """
    return {"points": np.asarray(points, dtype=float),
            "radius": float(radius_cm),
            "kind":   kind}


# -----------------------------------
# WLC-Generierung (unwrapped)
# -----------------------------------

def generate_wlc_chain(chain_length: float,
                       segment_length: float,
                       kappa: float,
                       box_size: float,
                       radius_cm: float) -> dict:
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

    pts = np.array(positions)
    return make_chain(pts, radius_cm=radius_cm, kind="network")


def generate_wlc_system(n_chains: int,
                        box_size: float,
                        chain_length: float,
                        segment_length: float,
                        kappa: float,
                        radius_cm: float):
    chains = []
    for _ in range(n_chains):
        chains.append(
            generate_wlc_chain(chain_length, segment_length, kappa, box_size, radius_cm)
        )
    return chains


# -----------------------------------
# Union-Find für Konnektivität
# -----------------------------------

class UnionFind:
    def __init__(self, n):
        self.parent = np.arange(n)
        self.rank = np.zeros(n, dtype=int)

    def find(self, x):
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

    def components(self):
        comp = {}
        for i in range(len(self.parent)):
            r = self.find(i)
            comp.setdefault(r, []).append(i)
        return list(comp.values())


def chain_chain_min_distance(chain_a: dict,
                             chain_b: dict) -> tuple[float, int, int]:
    pts_a = chain_a["points"]
    pts_b = chain_b["points"]
    diff = pts_a[:, None, :] - pts_b[None, :, :]
    d2 = np.sum(diff**2, axis=2)
    idx_flat = np.argmin(d2)
    ia, ib = np.unravel_index(idx_flat, d2.shape)
    dist = np.sqrt(d2[ia, ib])
    return dist, ia, ib


def build_chain_connectivity(chains, contact_dist_cm: float):
    n = len(chains)
    uf = UnionFind(n)
    for i in range(n):
        for j in range(i + 1, n):
            dist, _, _ = chain_chain_min_distance(chains[i], chains[j])
            if dist <= contact_dist_cm:
                uf.union(i, j)
    return uf


def create_bridge_chain(chain_a: dict,
                        chain_b: dict,
                        idx_a: int,
                        idx_b: int,
                        segment_length: float,
                        radius_cm: float) -> dict:
    pts_a = chain_a["points"]
    pts_b = chain_b["points"]
    p0 = pts_a[idx_a]
    p1 = pts_b[idx_b]
    v = p1 - p0
    L = np.linalg.norm(v)
    if L == 0:
        return make_chain(np.array([p0.copy()]), radius_cm, kind="bridge")
    n_seg = max(1, int(np.ceil(L / segment_length)))
    t = v / n_seg
    points = [p0 + k * t for k in range(n_seg + 1)]
    return make_chain(np.array(points), radius_cm, kind="bridge")


def enforce_full_connectivity(chains,
                              segment_length: float,
                              contact_dist_cm: float,
                              bridge_radius_cm: float):
    """
    Fügt Brückenchains hinzu, bis alle Chains in einer Komponente sind.
    """
    while True:
        uf = build_chain_connectivity(chains, contact_dist_cm)
        comps = uf.components()
        n_comp = len(comps)
        print(f"Aktueller Zustand: {len(chains)} Chains, {n_comp} Komponenten")
        if n_comp <= 1:
            break

        best_dist = np.inf
        best_pair = None
        best_indices = None

        for ci_idx in range(len(comps)):
            for cj_idx in range(ci_idx + 1, len(comps)):
                comp_i = comps[ci_idx]
                comp_j = comps[cj_idx]
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

        bridge = create_bridge_chain(chains[i], chains[j], ia, ib,
                                     segment_length, bridge_radius_cm)
        chains.append(bridge)

    return chains


# -----------------------------------
# Rahmen + Stütz-Chains
# -----------------------------------

def generate_frame_chains(inner_min: float,
                          inner_max: float,
                          segment_length: float,
                          radius_cm: float):
    """
    12 Kanten des inneren 9x9x9-cm-Würfels als dünne Stäbe.
    """
    x0 = y0 = z0 = inner_min
    x1 = y1 = z1 = inner_max

    corners = [
        np.array([x0, y0, z0]),
        np.array([x1, y0, z0]),
        np.array([x1, y1, z0]),
        np.array([x0, y1, z0]),
        np.array([x0, y0, z1]),
        np.array([x1, y0, z1]),
        np.array([x1, y1, z1]),
        np.array([x0, y1, z1]),
    ]
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    ]

    frame_chains = []
    for i1, i2 in edges:
        p0 = corners[i1]
        p1 = corners[i2]
        v = p1 - p0
        L = np.linalg.norm(v)
        if L == 0:
            continue
        n_seg = max(1, int(np.ceil(L / segment_length)))
        t = v / n_seg
        pts = [p0 + k * t for k in range(n_seg + 1)]
        frame_chains.append(make_chain(np.array(pts), radius_cm, kind="frame"))

    return frame_chains


def generate_support_chains(inner_min: float,
                            inner_max: float,
                            segment_length: float,
                            radius_cm: float):
    """
    Ein paar vertikale Stützen in den Ecken des inneren Würfels.
    """
    x0 = y0 = inner_min
    x1 = y1 = inner_max
    z0 = inner_min
    z1 = inner_max

    base_points = [
        np.array([x0, y0, z0]),
        np.array([x1, y0, z0]),
        np.array([x1, y1, z0]),
        np.array([x0, y1, z0]),
    ]

    support_chains = []
    for p0 in base_points:
        p1 = p0.copy()
        p1[2] = z1
        v = p1 - p0
        L = np.linalg.norm(v)
        n_seg = max(1, int(np.ceil(L / segment_length)))
        t = v / n_seg
        pts = [p0 + k * t for k in range(n_seg + 1)]
        support_chains.append(make_chain(np.array(pts), radius_cm, kind="support"))

    return support_chains


# -----------------------------------
# Segment-Clipping an 9x9x9-Box (mit Schnitt)
# -----------------------------------

def inside_box(p: np.ndarray, box_min: float, box_max: float) -> bool:
    return np.all((p >= box_min) & (p <= box_max))


def clip_segment_to_box(p0: np.ndarray,
                        p1: np.ndarray,
                        box_min: float,
                        box_max: float):
    """
    Schneidet ein Segment [p0,p1] an der Achsen-parallelen Box
    [box_min, box_max]^3.

    Rückgabe:
      []           -> Segment komplett draußen
      [q0,q1]      -> geclipptes Segment (inkl. Schnittpunkte)
    """
    p0 = np.asarray(p0, dtype=float)
    p1 = np.asarray(p1, dtype=float)
    v = p1 - p0

    points = []
    ts = []

    # Endpunkte, falls sie inside sind
    if inside_box(p0, box_min, box_max):
        points.append(p0)
        ts.append(0.0)
    if inside_box(p1, box_min, box_max):
        points.append(p1)
        ts.append(1.0)

    # Schnitte mit 6 Ebenen: x=min,max; y=min,max; z=min,max
    for axis in range(3):
        for bound in [box_min, box_max]:
            if v[axis] == 0:
                continue
            t = (bound - p0[axis]) / v[axis]
            if 0.0 <= t <= 1.0:
                p_int = p0 + t * v
                # Nur wenn der Punkt in den anderen beiden Koordinaten innerhalb der Box liegt
                other_axes = [a for a in range(3) if a != axis]
                if (box_min <= p_int[other_axes[0]] <= box_max and
                        box_min <= p_int[other_axes[1]] <= box_max):
                    # redundante Punkte vermeiden
                    if not any(np.allclose(p_int, q) for q in points):
                        points.append(p_int)
                        ts.append(t)

    if len(points) < 2:
        return []

    # nach t sortieren und nur frühesten & spätesten nehmen
    ts = np.array(ts)
    points = np.array(points)
    order = np.argsort(ts)
    q0 = points[order[0]]
    q1 = points[order[-1]]

    return [q0, q1]


def clip_chain_to_box(chain: dict,
                      box_min: float,
                      box_max: float):
    """
    Schneidet eine Chain an der Box.
    Rückgabe: Liste von neuen Chain-Objekten (mit gleichem Radius & kind),
    da eine Chain durch das Clipping in mehrere Teilstücke zerfallen kann.
    """
    pts = chain["points"]
    r = chain["radius"]
    kind = chain["kind"]

    new_segments = []   # Liste von Listen von Punkten
    current_segment = []

    for i in range(pts.shape[0] - 1):
        p0 = pts[i]
        p1 = pts[i + 1]
        clipped = clip_segment_to_box(p0, p1, box_min, box_max)
        if not clipped:
            # Segment komplett draußen -> Segment beendet ggf. aktuellen Strang
            if len(current_segment) > 1:
                new_segments.append(np.array(current_segment))
            current_segment = []
            continue

        q0, q1 = clipped
        if len(current_segment) == 0:
            current_segment.append(q0)
        else:
            # falls q0 numerisch identisch zum letzten Punkt ist, nicht doppeln
            if not np.allclose(current_segment[-1], q0):
                current_segment.append(q0)
        current_segment.append(q1)

    if len(current_segment) > 1:
        new_segments.append(np.array(current_segment))

    # in Chain-Objekte umwandeln
    new_chains = [make_chain(seg, r, kind=kind) for seg in new_segments]
    return new_chains


def clip_all_chains_to_box(chains,
                           box_min: float,
                           box_max: float):
    """
    Wendet das Clipping auf alle Chains an und sammelt alle Teilstücke.
    """
    clipped_chains = []
    for ch in chains:
        pieces = clip_chain_to_box(ch, box_min, box_max)
        clipped_chains.extend(pieces)
    return clipped_chains


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
                                   n_theta: int,
                                   n_phi: int,
                                   stride: int = 1):
    """
    Baut ein großes Dreieck-Mesh für alle Chains.
    Jede Chain nutzt ihren eigenen Radius.
    """
    all_tris = []
    for ch in chains:
        pts = ch["points"]
        r = ch["radius"]
        if pts.shape[0] == 0:
            continue

        # optional: Schwerpunkt in Box bringen (numerisch/kosmetisch)
        com = pts.mean(axis=0)
        shift = np.floor(com / BOX_SIZE_CM) * BOX_SIZE_CM
        pts_vis = pts - shift

        for idx in range(0, pts_vis.shape[0], stride):
            center = pts_vis[idx]
            tris = build_sphere_triangles(center, r,
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


def write_ascii_stl(filename: str, triangles: np.ndarray, name: str = "wlc_frame_support_network"):
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

    # 1) WLC-Netzwerk (120 Chains, 3 mm Durchmesser)
    chains = generate_wlc_system(
        n_chains=N_CHAINS_INIT,
        box_size=BOX_SIZE_CM,
        chain_length=CHAIN_LENGTH_CM,
        segment_length=SEGMENT_LENGTH_CM,
        kappa=KAPPA,
        radius_cm=NETWORK_RADIUS_CM,
    )
    print(f"Initial: {len(chains)} WLC-Chains")

    # 2) Perkolation / Konnektivität nur im Polymernetzwerk erzwingen
    chains = enforce_full_connectivity(
        chains,
        segment_length=SEGMENT_LENGTH_CM,
        contact_dist_cm=CONTACT_DIST_CM,
        bridge_radius_cm=NETWORK_RADIUS_CM,
    )

    # 3) Rahmen um inneren 9x9x9 cm Würfel (dünne Stäbe)
    frame_chains = generate_frame_chains(
        inner_min=INNER_MIN_CM,
        inner_max=INNER_MAX_CM,
        segment_length=SEGMENT_LENGTH_CM,
        radius_cm=FRAME_RADIUS_CM,
    )
    chains.extend(frame_chains)
    print(f"Nach Frame: {len(chains)} Chains (inkl. Rahmen)")

    # 4) Stütz-Chains
    support_chains = generate_support_chains(
        inner_min=INNER_MIN_CM,
        inner_max=INNER_MAX_CM,
        segment_length=SEGMENT_LENGTH_CM,
        radius_cm=SUPPORT_RADIUS_CM,
    )
    chains.extend(support_chains)
    print(f"Nach Stützen: {len(chains)} Chains (inkl. Stütz-Chains)")

    # 5) Noch einmal Konnektivität, jetzt inkl. Rahmen + Stützen
    chains = enforce_full_connectivity(
        chains,
        segment_length=SEGMENT_LENGTH_CM,
        contact_dist_cm=CONTACT_DIST_CM,
        bridge_radius_cm=NETWORK_RADIUS_CM,
    )
    print(f"Nach finaler Konnektivität: {len(chains)} Chains gesamt")

    # 6) Clipping: Objekt im inneren 9x9x9 cm Würfel sauber abschneiden
    chains_inner = clip_all_chains_to_box(chains, INNER_MIN_CM, INNER_MAX_CM)
    print(f"Nach Clipping: {len(chains_inner)} Chain-Stücke innerhalb der 9x9x9 cm Box")

    # 7) Sphere-Sweep-Mesh
    print("Baue Sphere-Sweep-Mesh...")
    triangles = build_network_sphere_triangles(
        chains_inner,
        n_theta=N_THETA,
        n_phi=N_PHI,
        stride=1,
    )
    print(f"Dreiecke im Mesh: {triangles.shape[0]}")

    # 8) STL schreiben
    out_file = "wlc_12cm_120chains_3mm_inner9cm_frame_support_clipped.stl"
    write_ascii_stl(out_file, triangles, name="wlc_frame_support_clipped")
    print(f"STL geschrieben nach: {out_file}")
    print(f"Einheiten im STL: {'mm' if SCALE_TO_MM else 'cm'}")
