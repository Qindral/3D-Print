import numpy as np
from scipy import ndimage as ndi
import matplotlib.pyplot as plt


def generate_bubble_cube(
    cube_size_cm: float = 7.0,
    voxel_size_mm: float = 1.0,
    bubble_radius_range_mm=(5.0, 10.0),
    target_void_fraction: float = 0.3,
    max_bubbles: int = 200,
    rng_seed: int | None = 42,
):
    """
    Erzeuge einen 3D-Würfel mit zufälligen kugelförmigen Hohlräumen.

    Rückgabe
    --------
    material : np.ndarray (bool)
        True  = Material
        False = Hohlraum
    info : dict
        Metadaten (Voxelzahl, Void-Fraction, etc.)
    """
    rng = np.random.default_rng(rng_seed)

    cube_size_mm = cube_size_cm * 10.0
    n_vox = int(round(cube_size_mm / voxel_size_mm))
    shape = (n_vox, n_vox, n_vox)

    # Start: alles Material
    material = np.ones(shape, dtype=bool)

    # Koordinatengitter in mm
    coords = [np.arange(n_vox) * voxel_size_mm for _ in range(3)]
    Z, Y, X = np.meshgrid(*coords, indexing="ij")

    r_min_mm, r_max_mm = bubble_radius_range_mm

    bubble_count = 0
    while bubble_count < max_bubbles:
        # aktuelle Void-Fraction
        void_fraction = 1.0 - material.mean()
        if void_fraction >= target_void_fraction:
            break

        # zufälliger Radius und Mittelpunkt
        r_mm = rng.uniform(r_min_mm, r_max_mm)
        margin_mm = r_mm + voxel_size_mm  # damit die Blase komplett im Würfel liegt
        cz, cy, cx = rng.uniform(
            margin_mm, cube_size_mm - margin_mm, size=3
        )

        # sphärische Maske
        dist2 = (Z - cz) ** 2 + (Y - cy) ** 2 + (X - cx) ** 2
        mask = dist2 <= r_mm**2

        material[mask] = False
        bubble_count += 1

    info = dict(
        n_vox=n_vox,
        voxel_size_mm=voxel_size_mm,
        cube_size_mm=cube_size_mm,
        bubble_count=bubble_count,
        final_void_fraction=1.0 - material.mean(),
    )
    return material, info


def thin_thick_walls(
    material: np.ndarray,
    voxel_size_mm: float,
    max_half_thickness_mm: float = 2.0,
):
    """
    Dünnt zu dicke Wände aus, lässt dünne unverändert.

    Idee:
    - distance_transform_edt(material) liefert Distanz eines Materialvoxels
      zur nächsten Void/Boundary.
    - Voxel mit Distanz > max_half_thickness_mm werden entfernt (-> Void).

    Damit wird die lokale "Halb-Wandstärke" auf max_half_thickness_mm begrenzt.
    """
    # Abstand in Voxel
    dist_vox = ndi.distance_transform_edt(material)
    dist_mm = dist_vox * voxel_size_mm

    thinned = material.copy()
    thinned[dist_mm > max_half_thickness_mm] = False

    return thinned, dist_mm


if __name__ == "__main__":
    # Parameter
    voxel_size_mm = 1.0
    cube_size_cm = 7.0
    bubble_radii_mm = (5.0, 10.0)   # 0,5–1 cm
    max_half_thickness_mm = 2.0     # alles > 2 mm wird ausgedünnt

    # Geometrie generieren
    material, info = generate_bubble_cube(
        cube_size_cm=cube_size_cm,
        voxel_size_mm=voxel_size_mm,
        bubble_radius_range_mm=bubble_radii_mm,
        target_void_fraction=0.3,
        max_bubbles=200,
        rng_seed=42,
    )

    print("Cube-Info:")
    for k, v in info.items():
        print(f"  {k}: {v}")

    # Wände ausdünnen
    thinned, dist_mm = thin_thick_walls(
        material, voxel_size_mm=voxel_size_mm,
        max_half_thickness_mm=max_half_thickness_mm,
    )

    print(f"Materialvolumen vor dem Dünnen: {material.mean():.3f}")
    print(f"Materialvolumen nach dem Dünnen: {thinned.mean():.3f}")

    # einfache Kontrolle: mittlerer Slice
    mid = material.shape[0] // 2
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(material[mid], origin="lower")
    axes[0].set_title("Original (2D-Schnitt)")
    axes[0].axis("off")

    axes[1].imshow(thinned[mid], origin="lower")
    axes[1].set_title("Nach Wand-Dünnung")
    axes[1].axis("off")

    plt.tight_layout()
    plt.show()
