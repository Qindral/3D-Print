# --- Install (nur in Colab notwendig, sonst auskommentieren) ---
# !pip install numpy matplotlib imageio networkx

import numpy as np
import matplotlib.pyplot as plt
import imageio.v2 as imageio
import networkx as nx
from pathlib import Path

rng = np.random.default_rng(42)

def save_frame(fig, outdir, name):
    outdir.mkdir(parents=True, exist_ok=True)
    p = outdir / f"{name}.png"
    fig.savefig(p, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return p

def make_gif(frames, outpath, fps=6):
    imgs = [imageio.imread(f) for f in frames]
    imageio.mimsave(outpath, imgs, fps=fps)

# ------------------ A) Bond Percolation ------------------
def union_find_percolation(L=70, p=0.48):
    open_h = rng.random((L, L-1)) < p
    open_v = rng.random((L-1, L)) < p
    G = nx.Graph()
    for i in range(L):
        for j in range(L):
            G.add_node((i, j))
    for i in range(L):
        for j in range(L-1):
            if open_h[i, j]:
                G.add_edge((i, j), (i, j+1))
    for i in range(L-1):
        for j in range(L):
            if open_v[i, j]:
                G.add_edge((i, j), (i+1, j))
    top = [(0, j) for j in range(L)]
    bottom = [(L-1, j) for j in range(L)]
    span_mask = np.zeros((L, L), dtype=bool)
    for comp in nx.connected_components(G):
        if any(n in comp for n in top) and any(n in comp for n in bottom):
            for (i, j) in comp:
                span_mask[i, j] = True
            break
    return open_h, open_v, span_mask

def plot_perc(open_h, open_v, span_mask, title="Bond percolation"):
    L = open_h.shape[0]
    fig, ax = plt.subplots(figsize=(4,4))
    ax.set_title(title)
    ax.set_axis_off()
    y, x = np.mgrid[0:L, 0:L]
    ax.plot(x, y, '.', ms=2)
    for i in range(L):
        for j in range(L-1):
            if open_h[i, j]:
                ax.plot([j, j+1], [i, i], lw=0.8)
    for i in range(L-1):
        for j in range(L):
            if open_v[i, j]:
                ax.plot([j, j], [i, i+1], lw=0.8)
    ii, jj = np.where(span_mask)
    ax.scatter(jj, ii, s=8, marker='s', alpha=0.6)
    ax.set_xlim(-1, L); ax.set_ylim(L, -1)
    return fig

# --------------- B) Fibrillar network + Cyclization ---------------
def draw_line(mask, x0, y0, x1, y1):
    n = int(max(abs(x1-x0), abs(y1-y0)))+1
    xs = np.linspace(x0, x1, n)
    ys = np.linspace(y0, y1, n)
    xs = np.clip(np.rint(xs).astype(int), 0, mask.shape[1]-1)
    ys = np.clip(np.rint(ys).astype(int), 0, mask.shape[0]-1)
    mask[ys, xs] = True

def make_fibrils(L=180, n_fibrils=260, mean_len=42):
    mask = np.zeros((L, L), dtype=bool)
    for _ in range(n_fibrils):
        x0, y0 = rng.integers(0, L, size=2)
        theta = rng.random()*2*np.pi
        length = max(4, int(rng.exponential(mean_len)))
        x1 = x0 + int(length*np.cos(theta))
        y1 = y0 + int(length*np.sin(theta))
        draw_line(mask, x0, y0, x1, y1)
    return mask

def graph_from_mask(mask):
    H, W = mask.shape
    G = nx.Graph()
    coords = np.argwhere(mask)
    for i, j in coords:
        G.add_node((i, j))
    for i, j in coords:
        for di, dj in [(1,0), (-1,0), (0,1), (0,-1)]:
            ni, nj = i+di, j+dj
            if 0 <= ni < H and 0 <= nj < W and mask[ni, nj]:
                G.add_edge((i, j), (ni, nj))
    return G

def apply_cyclization(G, f_cyc=0.3):
    G = G.copy()
    for comp_nodes in nx.connected_components(G):
        H = G.subgraph(comp_nodes).copy()
        cycles = nx.cycle_basis(H)
        cyc_edges = set()
        for c in cycles:
            for u, v in zip(c, c[1:]+[c[0]]):
                cyc_edges.add(tuple(sorted((u, v))))
        cyc_edges = list(cyc_edges)
        if not cyc_edges:
            continue
        n_remove = int(f_cyc * len(cyc_edges))
        if n_remove>0:
            idxs = np.random.default_rng(1).choice(len(cyc_edges), size=n_remove, replace=False)
            for idx in idxs:
                u, v = cyc_edges[idx]
                if G.has_edge(u, v):
                    G.remove_edge(u, v)
    return G

def spanning_fraction(G, H, W, solid=True):
    if G.number_of_nodes() == 0:
        return False, 0.0
    comps = list(nx.connected_components(G))
    largest = max(comps, key=len)
    frac = len(largest)/G.number_of_nodes()
    if solid:
        touch_top = any(n[0]==0 for n in largest)
        touch_bottom = any(n[0]==H-1 for n in largest)
        return (touch_top and touch_bottom), frac
    else:
        touch_left = any(n[1]==0 for n in largest)
        touch_right = any(n[1]==W-1 for n in largest)
        return (touch_left and touch_right), frac

def plot_fibril_scene(mask_solid, G_active, title, annotate=True):
    H, W = mask_solid.shape
    active_mask = np.zeros_like(mask_solid)
    for (i, j) in G_active.nodes:
        active_mask[i, j] = True
    pore_mask = ~mask_solid
    fig, axes = plt.subplots(1,2, figsize=(8,4))
    axes[0].imshow(active_mask, origin="upper", interpolation="nearest")
    axes[0].set_title("Mechanical network (active)")
    axes[0].set_axis_off()
    axes[1].imshow(pore_mask, origin="upper", interpolation="nearest")
    axes[1].set_title("Diffusive pore space")
    axes[1].set_axis_off()
    if annotate:
        Gmech = graph_from_mask(active_mask)
        Gpore = graph_from_mask(pore_mask)
        sp_mech, frac_mech = spanning_fraction(Gmech, H, W, solid=True)
        sp_pore,  frac_pore  = spanning_fraction(Gpore, H, W, solid=False)
        fig.suptitle(f"{title}\n"
                     f"G′-proxy (largest mech. component) ≈ {frac_mech:.2f} | "
                     f"Pore percolation: {sp_pore}",
                     fontsize=9)
    fig.tight_layout()
    return fig

# ------------------ Run & Export ------------------
outdir = Path("out_demo")
frames = []
for k, p in enumerate(np.linspace(0.3, 0.7, 9)):
    oh, ov, span = union_find_percolation(L=70, p=p)
    fig = plot_perc(oh, ov, span, title=f"Bond percolation (p={p:.2f})")
    frames.append(save_frame(fig, outdir, f"A_perc_{k:02d}"))
make_gif(frames, outdir / "A_percolation.gif", fps=3)

base = make_fibrils(L=180, n_fibrils=260, mean_len=42)
G_base = graph_from_mask(base)
frames = []
for idx, fcyc in enumerate(np.linspace(0.0, 0.6, 7)):
    G_act = apply_cyclization(G_base, f_cyc=fcyc)
    fig = plot_fibril_scene(base, G_act, title=f"Fibrillar gel with cyclization f_cyc={fcyc:.2f}")
    frames.append(save_frame(fig, outdir, f"B_fibrils_{idx:02d}"))
make_gif(frames, outdir / "B_fibrillar_cyclization.gif", fps=2)

G_act = apply_cyclization(G_base, f_cyc=0.3)
fig = plot_fibril_scene(base, G_act, title="Snapshot (f_cyc=0.30)")
save_frame(fig, outdir, "C_snapshot")

print("Done. Files in ./out_demo :")
for p in sorted(Path("out_demo").glob("*")):
    print(p)
