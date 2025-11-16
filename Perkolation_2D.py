import numpy as np
import matplotlib.pyplot as plt
import random
from matplotlib.collections import LineCollection

# =========================
# Parameters
# =========================
N = 80               # grid size (N x N nodes)
seed = None          # set e.g. 42 for reproducibility
pause_s = 0.01       # animation step delay (seconds)

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
            return False  # adding this edge would create a cycle
        if self.rank[ra] < self.rank[rb]:
            self.parent[ra] = rb
        elif self.rank[ra] > self.rank[rb]:
            self.parent[rb] = ra
        else:
            self.parent[rb] = ra
            self.rank[ra] += 1
        return True

def idx(i, j): return i * N + j

# =========================
# Build candidate bonds (4-neighborhood)
# Each bond is ((i1,j1),(i2,j2)) with i=row, j=col
# =========================
bonds = []
for i in range(N):
    for j in range(N):
        if i + 1 < N:
            bonds.append(((i, j), (i+1, j)))
        if j + 1 < N:
            bonds.append(((i, j), (i, j+1)))

rng.shuffle(bonds)

# =========================
# Union-Find with virtual top/bottom
# (virtuals are connected to all nodes in first/last row)
# =========================
TOP = N*N
BOTTOM = N*N + 1
uf = UnionFind(N*N + 2)
for j in range(N):
    uf.union(TOP, idx(0, j))
    uf.union(BOTTOM, idx(N-1, j))

# =========================
# Visualization setup
# =========================
fig, ax = plt.subplots(figsize=(7, 7))
ax.set_aspect('equal')
ax.set_xlim(-0.5, N-0.5)
ax.set_ylim(-0.5, N-0.5)
ax.set_xticks([]); ax.set_yticks([])
ax.set_title("Bond Percolation with Live Cyclization Highlighting")

# draw nodes as faint points (optional aesthetics)
xs, ys = np.meshgrid(np.arange(N), np.arange(N))
ax.scatter(xs, N-1-ys, s=5, alpha=0.4)

# Line collections for tree edges (blue) and cycle edges (red)
tree_lc = LineCollection([], linewidths=1.5, colors='tab:blue')
cycle_lc = LineCollection([], linewidths=2.5, colors='tab:red')
ax.add_collection(tree_lc)
ax.add_collection(cycle_lc)

# status text
status_txt = ax.text(0.02, 0.98, "", transform=ax.transAxes,
                     ha='left', va='top')

def bond_to_segment(bond):
    (i1, j1), (i2, j2) = bond
    # invert y for display so row 0 is top
    return [(j1, N-1-i1), (j2, N-1-i2)]

open_tree_segments = []
open_cycle_segments = []

# =========================
# Simulation loop (live)
# =========================
steps = 0
try:
    for a, b in bonds:
        a_id = idx(*a); b_id = idx(*b)
        if uf.find(a_id) == uf.find(b_id):
            # this edge closes a loop
            open_cycle_segments.append(bond_to_segment((a, b)))
        else:
            uf.union(a_id, b_id)
            open_tree_segments.append(bond_to_segment((a, b)))

        steps += 1

        # update collections
        tree_lc.set_segments(open_tree_segments)
        cycle_lc.set_segments(open_cycle_segments)

        # status
        perc = (steps / len(bonds)) * 100.0
        percolates = (uf.find(TOP) == uf.find(BOTTOM))
        status_txt.set_text(
            f"Opened bonds: {steps}/{len(bonds)}  ({perc:.1f}%)\n"
            f"Cycle edges: {len(open_cycle_segments)}\n"
            f"Percolation: {'YES' if percolates else 'no'}"
        )

        plt.pause(pause_s)

        if percolates:
            break

    # final redraw to ensure last frame is shown
    plt.pause(0.1)
    print(f"Percolation after {steps} opened bonds "
          f"({steps/len(bonds):.2%} of all possible bonds). "
          f"Cycle edges used: {len(open_cycle_segments)}.")
except KeyboardInterrupt:
    print("Interrupted by user.")

plt.show()
