import numpy as np
import trimesh
from scipy import ndimage
#pip install trimesh scipy numpy

def estimate_mesh_size_from_stl(
    stl_path,
    voxel_size_um=0.2,
    pore_phase=True,
    return_dist_hist=False
):
    """
    Schätzt eine Mesh Size (charakteristische Porengröße) aus einem STL-Fasernetz.

    Parameters
    ----------
    stl_path : str
        Pfad zur STL-Datei.
    voxel_size_um : float
        Voxelgröße in µm (räumliche Auflösung der Voxelisierung).
    pore_phase : bool
        Wenn True: Distance Transform im Pore-Space (Void).
        Wenn False: im Gel-Space (selten sinnvoll für Mesh Size).
    return_dist_hist : bool
        Wenn True: gibt zusätzlich Histogramm der Distanzwerte zurück.

    Returns
    -------
    mesh_size_um : float
        Geschätzte Mesh Size in µm (z.B. 2 * Median der Distanz).
    (optional) hist_bins, hist_counts
    """
    # 1) STL laden
    mesh = trimesh.load(stl_path)
    if mesh.is_empty:
        raise ValueError("Mesh is empty.")

    # 2) Voxelisieren
    pitch = voxel_size_um  # gleicher Maßstab, falls STL in µm ist
    # Achtung: falls STL in mm/cm ist, musst du entsprechend skalieren!
    vox = mesh.voxelized(pitch=pitch)

    # vox.matrix ist ein bool-Array: True = inside mesh (Gelphase)
    gel = vox.matrix.astype(bool)

    if pore_phase:
        # Porenphase = 1, Gelphase = 0
        pore = ~gel
        phase = pore
    else:
        phase = gel

    # 3) Distance Transform: Abstand jedes Voxels zur nächsten "anderen" Phase
    # Für Porenphase: Abstand zum Gel
    # Dafür Distance Transform auf komplementärer Maske:
    dist = ndimage.distance_transform_edt(phase, sampling=voxel_size_um)

    # Wir interessieren uns nur für Voxels innerhalb der Phase
    dvals = dist[phase]

    # 4) Typische Distanz (z.B. Median)
    median_dist = np.median(dvals)

    # Mesh Size ~ 2 * median radius
    mesh_size_um = 2.0 * median_dist

    if not return_dist_hist:
        return mesh_size_um
    else:
        counts, bins = np.histogram(dvals, bins=50)
        return mesh_size_um, bins, counts
    
mesh_size_um = estimate_mesh_size_from_stl(
    r"C:\Users\Jonas\Documents\GitHub\3D-Print\wlc_network_7cm_70chains_5cm_4mm.stl",
    voxel_size_um=0.1
)
print("Estimated mesh size ≈ {:.2f} µm".format(mesh_size_um))