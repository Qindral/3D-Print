import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import random
from collections import deque, defaultdict

# =========================
# Parameters
# =========================
N = 80               # grid size (N x N nodes)
ROD_LEN = 4          # number of edges per rod (straight segment)
seed = None          # e.g. 42 for reproducibility
pause_s = 0.005      # animation delay per rod
draw_every = 1       # update plot every k rods
highlight_pause = 0.01  # animation delay per green path segment

rng = random.Random(seed)

# =========================
# Union-Find (Disjoint Set)
# =========================
class UnionFind:
    def __init__(self, n):
        self.parent = np.arange(n)
        self.rank = np.zeros(n, dtype=int)
    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x
    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        if self.rank[ra] < self.rank[rb]:
            self.parent[ra] = rb
        elif self.rank[ra] > self.rank[rb]:
            self.parent[rb] = ra
        else:
            self.parent[rb] = ra
            self.rank[ra] += 1
        return True

def idx(i, j):
    if not (0 <= i < N and 0 <= j < N):
        raise ValueError(f"idx out of bounds for (i,j)=({i},{j}) with N={N}")
    return i * N + j

def in_bounds(i, j):
    return 0 <= i < N and 0 <= j < N

# =========================
# Build all candidate rods (straight segments of length ROD_LEN edges)
# Each rod = list of consecutive bonds [((i1,j1),(i2,j2)), ...]
# =========================
rods = []
directions = [(1,0), (0,1)]  # vertical, horizontal
for i in range(N):
    for j in range(N):
        for di, dj in directions:
            # IMPORTANT: check end node of last edge -> ROD_LEN
            ii = i + ROD_LEN * di
            jj = j + ROD_LEN * dj
            if in_bounds(ii, jj):
                bonds = []
                for k in range(ROD_LEN):
                    a = (i + k*di,     j + k*dj)
                    b = (i + (k+1)*di, j + (k+1)*dj)
                    bonds.append((a, b))
                rods.append(bonds)
rng.shuffle(rods)

# =========================
# UF with virtual top/bottom
# =========================
TOP = N*N
BOTTOM = N*N + 1
uf = UnionFind(N*N + 2)
for j in range(N):
    uf.union(TOP, idx(0, j))
    uf.union(BOTTOM, idx(N-1, j))

# =========================
# Plot helpers
# =========================
def bond_to_segment(bond):
    (i1, j1), (i2, j2) = bond
    # invert y for display (row 0 at top)
    return [(j1, N-1-i1), (j2, N-1-i2)]

open_tree_segments = []   # non-cycle edges
open_cycle_segments = []  # cycle edges
degree = np.zeros(N*N, dtype=int)
crosslink_nodes = set()

# also keep adjacency of OPEN EDGES for exact path reconstruction
adj = defaultdict(set)

def add_bond_and_classify(a, b, a_cell, b_cell):
    """Add edge a-b. Update UF, degree, crosslink set, and adjacency.
       Return True if cycle-edge, else False (tree-edge)."""
    is_cycle = (uf.find(a) == uf.find(b))
    if not is_cycle:
        uf.union(a, b)
    # degrees reflect actual graph regardless
    degree[a] += 1; degree[b] += 1
    if degree[a] >= 3: crosslink_nodes.add(a)
    if degree[b] >= 3: crosslink_nodes.add(b)
    # adjacency updates (store only once per bond)
    adj[a].add(b); adj[b].add(a)
    seg = bond_to_segment((a_cell, b_cell))
    if is_cycle:
        open_cycle_segments.append(seg)
    else:
        open_tree_segments.append(seg)
    return is_cycle

# =========================
# Visualization setup
# =========================
fig, ax = plt.subplots(figsize=(8, 8))
ax.set_aspect('equal')
ax.set_xlim(-0.5, N-0.5)
ax.set_ylim(-0.5, N-0.5)
ax.set_xticks([]); ax.set_yticks([])
ax.set_title(f"Percolation with Rods (len {ROD_LEN}) — blue: tree, red: cycles, squares: crosslinks")

# faint nodes
xs, ys = np.meshgrid(np.arange(N), np.arange(N))
ax.scatter(xs, N-1-ys, s=3, alpha=0.25)

tree_lc  = LineCollection([], linewidths=1.0, colors='tab:blue')
cycle_lc = LineCollection([], linewidths=1.8, colors='tab:red')
ax.add_collection(tree_lc)
ax.add_collection(cycle_lc)

# crosslink markers
cross_scat = ax.scatter([], [], marker='s', s=12, c='black', alpha=0.9)

# green highlight (initially empty)
green_lc = LineCollection([], linewidths=3.0, colors='tab:green', alpha=0.9)
ax.add_collection(green_lc)

status_txt = ax.text(0.02, 0.98, "", transform=ax.transAxes,
                     ha='left', va='top', fontsize=10)

def update_plot(step):
    tree_lc.set_segments(open_tree_segments)
    cycle_lc.set_segments(open_cycle_segments)
    if crosslink_nodes:
        xs_cl, ys_cl = [], []
        for node in crosslink_nodes:
            i, j = divmod(node, N)
            xs_cl.append(j); ys_cl.append(N-1-i)
        cross_scat.set_offsets(np.c_[xs_cl, ys_cl])
    status_txt.set_text(
        f"Rods: {step}/{len(rods)} | "
        f"Edges: {len(open_tree_segments)+len(open_cycle_segments)} "
        f"(tree {len(open_tree_segments)}, cycles {len(open_cycle_segments)}) | "
        f"Crosslinks: {len(crosslink_nodes)} | "
        f"Percolation: {'YES' if uf.find(TOP)==uf.find(BOTTOM) else 'no'}"
    )
    plt.pause(pause_s)

# =========================
# Path reconstruction (after percolation)
# =========================
def reconstruct_percolating_path():
    """Find a concrete Top→Bottom path through currently OPEN edges.
       We do BFS from a chosen bottom node toward any top node, with
       tie-breaking to favor moves up/right (visuell ~ links unten → rechts oben)."""
    # candidates on bottom row that are in same UF component as TOP (= percolating cluster)
    bottom_candidates = [idx(N-1, j) for j in range(N) if uf.find(idx(N-1, j)) == uf.find(TOP)]
    if not bottom_candidates:
        return []  # should not happen if percolation detected

    # choose bottom-leftmost candidate to "start left-bottom"
    start = min(bottom_candidates, key=lambda node: (node % N, ))  # minimal column j

    # set of top row targets within same cluster (just for early exit)
    top_targets = {idx(0, j) for j in range(N) if uf.find(idx(0, j)) == uf.find(TOP)}

    # BFS with direction preference: up, right, left, down (to bias ↑ and →)
    # We'll sort neighbors by a score that prefers decreasing i (up) and increasing j (right).
    def neighbor_sort_key(cur, nei):
        ci, cj = divmod(cur, N)
        ni, nj = divmod(nei, N)
        return (ni - ci, -(nj - cj))  # prioritize ni<ci (up), then nj>cj (right)

    prev = {start: None}
    dq = deque([start])
    reached = None

    while dq:
        cur = dq.popleft()
        if cur in top_targets:
            reached = cur
            break
        neighs = list(adj[cur])
        # keep only neighbors that are truly connected via current open graph
        # (adj already ensures that)
        neighs.sort(key=lambda n: neighbor_sort_key(cur, n))
        for n in neighs:
            if n not in prev:
                prev[n] = cur
                dq.append(n)

    if reached is None:
        # fallback: try reversed direction (from top toward bottom)
        top_candidates = [idx(0, j) for j in range(N) if uf.find(idx(0, j)) == uf.find(TOP)]
        if not top_candidates:
            return []
        start2 = min(top_candidates, key=lambda node: (N-1 - (node // N), node % N))
        prev = {start2: None}
        dq = deque([start2])
        targets = {idx(N-1, j) for j in range(N) if uf.find(idx(N-1, j)) == uf.find(TOP)}
        while dq:
            cur = dq.popleft()
            if cur in targets:
                reached = cur
                break
            neighs = list(adj[cur])
            neighs.sort(key=lambda n: neighbor_sort_key(cur, n))
            for n in neighs:
                if n not in prev:
                    prev[n] = cur
                    dq.append(n)

    if reached is None:
        return []

    # Reconstruct node path
    path_nodes = []
    x = reached
    while x is not None:
        path_nodes.append(x)
        x = prev[x]
    path_nodes.reverse()

    # Convert to list of segments for plotting
    path_segments = []
    for u, v in zip(path_nodes[:-1], path_nodes[1:]):
        (ui, uj) = divmod(u, N)
        (vi, vj) = divmod(v, N)
        path_segments.append(bond_to_segment(((ui, uj), (vi, vj))))
    return path_segments

# =========================
# Simulation loop
# =========================
steps = 0
percolates = False
try:
    for bonds in rods:
        for (a_cell, b_cell) in bonds:
            a = idx(*a_cell); b = idx(*b_cell)
            add_bond_and_classify(a, b, a_cell, b_cell)

        steps += 1
        if steps % draw_every == 0:
            update_plot(steps)

        if uf.find(TOP) == uf.find(BOTTOM):
            percolates = True
            update_plot(steps)
            break

    # Final state update
    update_plot(steps)

    if percolates:
        # Reconstruct and animate the green percolating path
        green_segments = reconstruct_percolating_path()
        segs = []
        for seg in green_segments:
            segs.append(seg)
            green_lc.set_segments(segs)
            plt.pause(highlight_pause)

        print(f"Stopped after {steps} rods. Percolation: True. "
              f"Green path length (segments): {len(green_segments)}")
    else:
        print(f"Stopped after {steps} rods. Percolation: False.")
except KeyboardInterrupt:
    print("Interrupted by user.")

plt.show()
