import numpy as np

# -----------------------------------
# Globale Parameter (deine Werte)
# -----------------------------------

BOX_SIZE_CM = 7.0

CHAIN_LENGTH_CM = 3.0
SEGMENT_LENGTH_CM = 0.02
N_CHAINS_INIT = 200
KAPPA = 0.8

NETWORK_DIAMETER_MM = 2.0
NETWORK_RADIUS_CM = (NETWORK_DIAMETER_MM / 10.0) / 2.0  # 2 mm -> 0.2 cm -> r=0.1 cm

Z_SUPPORT_DIAMETER_MM = NETWORK_DIAMETER_MM
Z_SUPPORT_RADIUS_CM = (Z_SUPPORT_DIAMETER_MM / 10.0) / 2.0

CONTACT_FACTOR = 1.05
CONTACT_DIST_CM = 2.0 * NETWORK_RADIUS_CM * CONTACT_FACTOR

N_THETA_TUBE = 6
N_Z_SUPPORT_RODS = 30

SCALE_TO_MM = True
SCALE_FACTOR = 10.0 if SCALE_TO_MM else 1.0

# Für die schnelle Distanzberechnung:
MAX_POINTS_DISTANCE = 30  # max Punkte pro Chain in der Distanz-Approximation


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


# -----------------------------------
# WLC-Generierung im Würfel
# -----------------------------------

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


# -----------------------------------
# Union-Find
# -----------------------------------

class UnionFind:
    def __init__(self, n):
        self.parent = np.arange(n)
        self.rank = np.zeros(n, dtype=int)
        self.components_count = n

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


# -----------------------------------
# Schnelle approx. Chain–Chain-Distanz
# -----------------------------------

def chain_chain_min_distance_approx(chain_a, chain_b, max_points=MAX_POINTS_DISTANCE):
    pts_a = chain_a["points"]
    pts_b = chain_b["points"]
    la = pts_a.shape[0]
    lb = pts_b.shape[0]

    # Subsampling, max max_points pro Chain
    step_a = max(1, la // max_points)
    step_b = max(1, lb // max_points)
    a = pts_a[::step_a]
    b = pts_b[::step_b]

    diff = a[:, None, :] - b[None, :, :]
    d2 = np.sum(diff ** 2, axis=2)
    idx_flat = np.argmin(d2)
    ia_sub, ib_sub = np.unravel_index(idx_flat, d2.shape)

    ia = min(ia_sub * step_a, la - 1)
    ib = min(ib_sub * step_b, lb - 1)
    dist = float(np.sqrt(d2[ia_sub, ib_sub]))

    return dist, ia, ib


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


# -----------------------------------
# Schnelle, strenge Perkolation (MST-artig)
# -----------------------------------

def enforce_full_connectivity_all_fast(base_chains,
                                       segment_length,
                                       contact_dist_cm,
                                       bridge_radius_cm):
    """
    base_chains: Liste aller Chains, die verbunden sein müssen (network + z_support).
    1) Precompute approx. Distanz und erste Kontakte
    2) Sortiere alle Paare nach Distanz
    3) Füge Bridges hinzu, bis Union-Find nur noch 1 Komponente hat
    """

    n = len(base_chains)
    uf = UnionFind(n)
    pair_records = []  # (dist, i, j, ia, ib)

    print(f"[Percolation-Fast] Precompute Paar-Distanzen für {n} Chains ...")
    for i in range(n):
        for j in range(i + 1, n):
            dist, ia, ib = chain_chain_min_distance_approx(base_chains[i], base_chains[j])
            pair_records.append((dist, i, j, ia, ib))
            if dist <= contact_dist_cm:
                uf.union(i, j)

    print(f"[Percolation-Fast] Nach initialen Kontakten: {uf.n_components()} Komponenten")

    # Wenn schon perkoliert: fertig
    if uf.n_components() <= 1:
        print("[Percolation-Fast] Bereits voll perkoliert, keine Bridges nötig.")
        return [], base_chains

    # Sortiere alle Paare nach Distanz
    pair_records.sort(key=lambda x: x[0])

    bridge_chains = []
    bridge_id = 0

    print("[Percolation-Fast] Füge Bridges hinzu (MST-ähnlich) ...")
    for dist, i, j, ia, ib in pair_records:
        if uf.n_components() <= 1:
            break
        ri, rj = uf.find(i), uf.find(j)
        if ri == rj:
            continue  # schon verbunden

        print(f"  -> Bridge {bridge_id}: Chains {i} und {j}, Abstand ~ {dist:.3f} cm")
        bridge = create_bridge_chain(
            base_chains[i], base_chains[j], ia, ib,
            segment_length, bridge_radius_cm, bridge_id
        )
        bridge_chains.append(bridge)
        bridge_id += 1

        # Für die Konnektivitätslogik reicht union(i, j)
        uf.union(ri, rj)

    print(f"[Percolation-Fast] Nach Bridges: {uf.n_components()} Komponenten")
    if uf.n_components() != 1:
        print("WARNUNG: Strenge Perkolation nicht vollständig erreicht (mehr als 1 Komponente).")
    else:
        print("Strenge Perkolation OK: Ein einziges Netzwerk.")

    return bridge_chains, base_chains


# -----------------------------------
# Z-wobbelige Supports
# -----------------------------------

def generate_z_wobble_supports(n_rods,
                               box_size,
                               segment_length,
                               radius_cm,
                               existing_chains=None,
                               bias_to_z=0.8,
                               wobble_kappa=0.6):
    if existing_chains is None:
        existing_chains = []

    height = box_size
    n_segments = int(np.ceil(height / segment_length))

    z_supports = []

    for i in range(n_rods):
        pos = np.random.rand(3) * box_size
        positions = [pos.copy()]

        t = np.array([0.0, 0.0, 1.0])

        for _ in range(n_segments):
            r = random_unit_vector()
            biased_dir = bias_to_z * np.array([0.0, 0.0, 1.0]) + (1.0 - bias_to_z) * r
            biased_dir /= np.linalg.norm(biased_dir)

            t = (1.0 - wobble_kappa) * random_unit_vector() + wobble_kappa * biased_dir
            t /= np.linalg.norm(t)

            pos = pos + segment_length * t
            pos = reflect_into_box(pos, box_size)
            positions.append(pos.copy())

        pts = np.array(positions)
        z_supports.append(
            make_chain(pts, radius_cm=radius_cm, kind="z_support", origin_id=f"zsupport_{i}")
        )

    print(f"[Z-Support] Erzeugt {len(z_supports)} wobbelige z-Stützen")
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

    print(f"[{label}] Chains: {len(chains)}, Tubes: {tubes_total}, Dreiecke: {tris_arr.shape[0]}")
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

    # 2) Z-Support-Chains
    z_support_chains = generate_z_wobble_supports(
        n_rods=N_Z_SUPPORT_RODS,
        box_size=BOX_SIZE_CM,
        segment_length=SEGMENT_LENGTH_CM,
        radius_cm=Z_SUPPORT_RADIUS_CM,
        existing_chains=network_chains,
        bias_to_z=0.8,
        wobble_kappa=0.6,
    )

    # 3) Schnelle, strenge Perkolation über network + z_support
    base_for_percolation = network_chains + z_support_chains
    bridge_chains, base_final = enforce_full_connectivity_all_fast(
        base_chains=base_for_percolation,
        segment_length=SEGMENT_LENGTH_CM,
        contact_dist_cm=CONTACT_DIST_CM,
        bridge_radius_cm=NETWORK_RADIUS_CM,
    )
    print(f"Bridges erzeugt: {len(bridge_chains)}")

    # 4) Tube-Mesh pro Kategorie
    tris_network = build_tube_mesh_for_chains(network_chains,   n_theta=N_THETA_TUBE, label="network")
    tris_zsup    = build_tube_mesh_for_chains(z_support_chains, n_theta=N_THETA_TUBE, label="z_support")
    tris_bridge  = build_tube_mesh_for_chains(bridge_chains,    n_theta=N_THETA_TUBE, label="bridge")
    tris_all     = build_tube_mesh_for_chains(
        base_final + bridge_chains,
        n_theta=N_THETA_TUBE,
        label="all_final"
    )
    
    # 5) Drei STL-Files schreiben
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
