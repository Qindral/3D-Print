import numpy as np

# -----------------------------------
# Globale Parameter
# -----------------------------------

# Würfelgröße (cm)
BOX_SIZE_CM = 7.0

# WLC-Parameter
CHAIN_LENGTH_CM = 3.0         # Konturlänge pro WLC
SEGMENT_LENGTH_CM = 0.02       # Schrittweite (1 mm)
N_CHAINS_INIT = 270           # Anzahl initialer WLC-Chains
KAPPA = 0.7                   # Steifigkeit (0=sehr knickrig, 1=fast gerade)

# Durchmesser (mm) -> Radius (cm)
NETWORK_DIAMETER_MM = 2.0
NETWORK_RADIUS_CM = (NETWORK_DIAMETER_MM / 10.0) / 2.0  # 3 mm -> 0.3 cm -> r=0.15 cm

Z_SUPPORT_DIAMETER_MM = 3.0
Z_SUPPORT_RADIUS_CM = (Z_SUPPORT_DIAMETER_MM / 10.0) / 2.0

# Konnektivität
CONTACT_FACTOR = 1.05
CONTACT_DIST_CM = 2.0 * NETWORK_RADIUS_CM * CONTACT_FACTOR

# Tube-Querschnitt-Auflösung
N_THETA_TUBE = 6  # hexagonaler Querschnitt

# Anzahl zusätzlicher z-orientierter Stütz-Rods
N_Z_SUPPORT_RODS = 30

# cm -> mm für STL
SCALE_TO_MM = True
SCALE_FACTOR = 10.0 if SCALE_TO_MM else 1.0


# -----------------------------------
# Hilfsfunktionen / Datentyp
# -----------------------------------

def random_unit_vector():
    v = np.random.normal(size=3)
    return v / np.linalg.norm(v)


def new_tangent(prev_t, kappa):
    if prev_t is None or kappa <= 0.0:
        return random_unit_vector()
    r = random_unit_vector()
    t = (1.0 - kappa) * r + kappa * prev_t
    return t / np.linalg.norm(t)


def reflect_into_box(pos, box_size):
    """
    Reflektierende Randbedingungen:
    wenn pos[axis] < 0 oder > box_size, wird an der Grenze gespiegelt.
    """
    p = pos.copy()
    for ax in range(3):
        if p[ax] < 0.0:
            p[ax] = -p[ax]
        if p[ax] > box_size:
            p[ax] = 2.0 * box_size - p[ax]
        # falls wir extrem übers Ziel hinausschießen, nochmal clampen:
        p[ax] = np.clip(p[ax], 0.0, box_size)
    return p


def make_chain(points, radius_cm, kind="network", origin_id=None):
    """
    Chain-Objekt:
      kind: "network", "z_support", "bridge"
      origin_id: z.B. 0..119 für initiale Chains oder 'zsupport_*', 'bridge_*'
    """
    return {
        "points": np.asarray(points, dtype=float),
        "radius": float(radius_cm),
        "kind":   kind,
        "origin_id": origin_id,
    }


# -----------------------------------
# WLC-Generierung im 12×12×12-Würfel
# -----------------------------------

def generate_wlc_chain(chain_id, chain_length, segment_length, kappa, radius_cm, box_size):
    # Startpunkt irgendwo im Würfel
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


# -----------------------------------
# Union-Find und Konnektivität
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


def chain_chain_min_distance(chain_a, chain_b):
    pts_a = chain_a["points"]
    pts_b = chain_b["points"]
    diff = pts_a[:, None, :] - pts_b[None, :, :]
    d2 = np.sum(diff ** 2, axis=2)
    idx_flat = np.argmin(d2)
    ia, ib = np.unravel_index(idx_flat, d2.shape)
    dist = np.sqrt(d2[ia, ib])
    return dist, ia, ib


def build_chain_connectivity(chains, contact_dist_cm):
    """
    Baut Connectivity-Graf über alle Chains (network, z_support, bridge).
    """
    n = len(chains)
    uf = UnionFind(n)
    for i in range(n):
        for j in range(i + 1, n):
            dist, _, _ = chain_chain_min_distance(chains[i], chains[j])
            if dist <= contact_dist_cm:
                uf.union(i, j)
    return uf


def create_bridge_chain(chain_a, chain_b, idx_a, idx_b,
                         segment_length, radius_cm, bridge_id):
    pts_a = chain_a["points"]
    pts_b = chain_b["points"]
    p0 = pts_a[idx_a]
    p1 = pts_b[idx_b]
    v = p1 - p0
    L = np.linalg.norm(v)
    if L == 0:
        return make_chain(
            np.array([p0.copy()]),
            radius_cm,
            kind="bridge",
            origin_id=f"bridge_{bridge_id}"
        )

    n_seg = max(1, int(np.ceil(L / segment_length)))
    t = v / n_seg
    points = [p0 + k * t for k in range(n_seg + 1)]
    pts = np.array(points)
    return make_chain(
        pts,
        radius_cm=radius_cm,
        kind="bridge",
        origin_id=f"bridge_{bridge_id}"
    )


def enforce_full_connectivity_all(base_chains,
                                  segment_length,
                                  contact_dist_cm,
                                  bridge_radius_cm):
    """
    Strenge Perkolation:
    - base_chains: Liste aller Chains, die sicher miteinander verbunden sein sollen
                    (z.B. network + z_support)
    - Wir erzeugen bridge-Chains, bis ALLE in einer einzigen Komponente liegen.
    - base_chains selbst werden NICHT verändert.

    Rückgabe:
      bridge_chains, all_chains_final
    """
    all_chains = list(base_chains)  # Kopie
    bridge_chains = []
    bridge_id = 0

    while True:
        uf = build_chain_connectivity(all_chains, contact_dist_cm)
        comps = uf.components()
        n_comp = len(comps)
        print(f"[Percolation] Aktueller Zustand: {len(all_chains)} Chains, {n_comp} Komponenten")
        if n_comp <= 1:
            break

        best_dist = np.inf
        best_pair = None
        best_indices = None

        for ci in range(len(comps)):
            for cj in range(ci + 1, len(comps)):
                comp_i = comps[ci]
                comp_j = comps[cj]
                for i in comp_i:
                    for j in comp_j:
                        dist, ia, ib = chain_chain_min_distance(all_chains[i], all_chains[j])
                        if dist < best_dist:
                            best_dist = dist
                            best_pair = (i, j)
                            best_indices = (ia, ib)

        if best_pair is None:
            print("Warnung: keine verbindbaren Komponenten gefunden (sollte eigentlich nicht passieren).")
            break

        i, j = best_pair
        ia, ib = best_indices
        print(f"  -> Verbinde Komponenten über Chains {i} und {j}, Abstand ~ {best_dist:.3f} cm")

        bridge = create_bridge_chain(
            all_chains[i], all_chains[j], ia, ib,
            segment_length, bridge_radius_cm, bridge_id
        )
        bridge_id += 1
        all_chains.append(bridge)
        bridge_chains.append(bridge)

    # Finaler Check
    uf_final = build_chain_connectivity(all_chains, contact_dist_cm)
    comps_final = uf_final.components()
    print(f"[Percolation] Nach Bridges: {len(all_chains)} Chains, {len(comps_final)} Komponenten")
    if len(comps_final) != 1:
        print("WARNUNG: Strenge Perkolation NICHT erreicht (mehr als 1 Komponente).")
    else:
        print("Strenge Perkolation OK: Ein einziges Netzwerk, kein Chain ist frei.")

    return bridge_chains, all_chains


# -----------------------------------
# Z-Printability-Heuristik & Z-wobbelige Supports
# -----------------------------------

def evaluate_z_printability(chains):
    total_segments = 0
    vertical_segments = 0

    for ch in chains:
        pts = ch["points"]
        if pts.shape[0] < 2:
            continue
        diffs = np.diff(pts, axis=0)
        seg_lengths = np.linalg.norm(diffs, axis=1)
        dz = np.abs(diffs[:, 2])
        with np.errstate(divide="ignore", invalid="ignore"):
            frac_z = dz / (seg_lengths + 1e-12)
        vertical_segments += np.sum(frac_z > 0.7)
        total_segments += len(seg_lengths)

    if total_segments == 0:
        return 0.0
    return vertical_segments / total_segments


def min_distance_to_chain_set(chain, chain_list):
    pts = chain["points"]
    if len(chain_list) == 0 or pts.shape[0] == 0:
        return np.inf
    pts_all = []
    for ch in chain_list:
        if ch["points"].shape[0] > 0:
            pts_all.append(ch["points"])
    if not pts_all:
        return np.inf
    pts_all = np.vstack(pts_all)
    diff = pts[:, None, :] - pts_all[None, :, :]
    d2 = np.sum(diff ** 2, axis=2)
    return float(np.sqrt(np.min(d2)))


def generate_z_wobble_supports(n_rods,
                               box_size,
                               segment_length,
                               radius_cm,
                               existing_chains=None,
                               bias_to_z=0.8,
                               wobble_kappa=0.6):
    """
    Erzeugt n_rods zusätzliche Stütz-Chains, die vor allem in Z-Richtung laufen,
    aber leicht wobbeln (biased random walk).
    Reflektierende Randbedingungen im 12-cm-Würfel.
    """
    if existing_chains is None:
        existing_chains = []

    height = box_size
    n_segments = int(np.ceil(height / segment_length))

    z_supports = []
    all_for_eval = list(existing_chains)

    for i in range(n_rods):
        # Startpunkt zufällig im Würfel
        pos = np.random.rand(3) * box_size
        positions = [pos.copy()]

        # initiale Richtung grob nach oben
        t = np.array([0.0, 0.0, 1.0])

        for _ in range(n_segments):
            # Zufallsrichtung
            r = random_unit_vector()
            # Bias in z-Richtung
            biased_dir = bias_to_z * np.array([0.0, 0.0, 1.0]) + (1.0 - bias_to_z) * r
            biased_dir /= np.linalg.norm(biased_dir)

            # WLC-artige Glättung
            t = (1.0 - wobble_kappa) * random_unit_vector() + wobble_kappa * biased_dir
            t /= np.linalg.norm(t)

            pos = pos + segment_length * t
            pos = reflect_into_box(pos, box_size)
            positions.append(pos.copy())

        pts = np.array(positions)
        z_support = make_chain(
            pts,
            radius_cm=radius_cm,
            kind="z_support",
            origin_id=f"zsupport_{i}"
        )
        z_supports.append(z_support)
        all_for_eval.append(z_support)

        # Evaluation: lokal + global + Distanz
        local_score = evaluate_z_printability([z_support])
        global_score = evaluate_z_printability(all_for_eval)
        min_dist = min_distance_to_chain_set(z_support, existing_chains)

        print(f"[z-support] Rod {i+1}/{n_rods}: "
              f"lokale Verticalität ~ {local_score:.2f}, "
              f"globale Verticalität ~ {global_score:.2f}, "
              f"min. Abstand zum Netzwerk ~ {min_dist:.3f} cm")

    return z_supports


# -----------------------------------
# Tube-Mesh
# -----------------------------------

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

    print(f"[{label}] Chains: {len(chains)}, Tubes gebaut: {tubes_total}, Dreiecke: {tris_arr.shape[0]}")
    return tris_arr


# -----------------------------------
# STL-Helfer
# -----------------------------------

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


# -----------------------------------
# Main
# -----------------------------------

if __name__ == "__main__":
    np.random.seed(42)

    # 1) Initiales Netzwerk
    network_chains = generate_wlc_system(
        n_chains=N_CHAINS_INIT,
        chain_length=CHAIN_LENGTH_CM,
        segment_length=SEGMENT_LENGTH_CM,
        kappa=KAPPA,
        radius_cm=NETWORK_RADIUS_CM,
        box_size=BOX_SIZE_CM,
    )
    print(f"Initial network chains: {len(network_chains)}")

    # 2) Zusätzliche wobbelige Z-Stützen (noch ohne Bridges)
    z_support_chains = generate_z_wobble_supports(
        n_rods=N_Z_SUPPORT_RODS,
        box_size=BOX_SIZE_CM,
        segment_length=SEGMENT_LENGTH_CM,
        radius_cm=Z_SUPPORT_RADIUS_CM,
        existing_chains=network_chains,
        bias_to_z=0.8,
        wobble_kappa=0.6,
    )
    print(f"Z-Support chains: {len(z_support_chains)}")

    # 3) Strenge Perkolation: network + z_support müssen EIN Netzwerk bilden
    base_for_percolation = network_chains + z_support_chains
    bridge_chains, all_chains_final = enforce_full_connectivity_all(
        base_chains=base_for_percolation,
        segment_length=SEGMENT_LENGTH_CM,
        contact_dist_cm=CONTACT_DIST_CM,
        bridge_radius_cm=NETWORK_RADIUS_CM,
    )
    all_chains_final = network_chains + z_support_chains + bridge_chains

    print(f"Bridges erzeugt: {len(bridge_chains)}")

    # 4) Tube-Mesh pro Kategorie
    tris_network = build_tube_mesh_for_chains(network_chains,   n_theta=N_THETA_TUBE, label="initial_network")
    tris_zsup    = build_tube_mesh_for_chains(z_support_chains, n_theta=N_THETA_TUBE, label="z_support")
    tris_bridge  = build_tube_mesh_for_chains(bridge_chains,    n_theta=N_THETA_TUBE, label="bridge")
    tris_all     = build_tube_mesh_for_chains(all_chains_final, n_theta=N_THETA_TUBE, label="all_chains_final")


    # 5) Drei getrennte STL-Files schreiben
    write_ascii_stl("initial_network_chains.stl", tris_network, name="initial_network_chains")
    write_ascii_stl("z_support_chains.stl",      tris_zsup,    name="z_support_chains")
    write_ascii_stl("bridge_chains.stl",        tris_bridge,   name="bridge_chains")
    write_ascii_stl("all_chains_final.stl",     tris_all,      name="all_chains_final")

    print("Fertig. Geschrieben wurden:")
    print("  initial_network_chains.stl")
    print("  z_support_chains.stl")
    print("  bridge_chains.stl")
    print("  all_chains_final.stl")
    print(f"Einheiten im STL: {'mm' if SCALE_TO_MM else 'cm'}")
