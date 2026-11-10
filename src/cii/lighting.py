"""Other lighting modes (dark field, multi-lighting) for an existing dataset.

The other lighting modes are recorded in separate passes at the same nominal
stage positions as the bright-field images. Each pass drives the stage again,
so an image can be shifted against its bright-field image by several µm (seen:
up to 18 µm). Per image pair:

1. pairing: the image of the other mode with the same stage position as the
   bright-field image (within MAX_STAGE_DIFF_UM, same magnification),
2. shift: phase correlation of the two edge images (Gaussian-smoothed grey
   image, gradient magnitude; the edges look alike in every lighting mode) on
   the whole image and on its four quadrants; the pair is accepted if every
   quadrant agrees with the whole image within QUADRANT_TOL_PX (pure shift, no
   rotation or scaling),
3. crops: the same window as the bright-field crop of each particle, moved by
   the rounded shift. A window that would leave the image gives no crop.

The particles, labels, split and folds of the dataset do not change; the
dataset gets a column crop_<name> (empty where there is no crop) and the
folder crops_<name>/ with files named as in crops/.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

LIGHTING_NAMES = {"df": "dark field", "ml": "multi-lighting"}   # short name -> name in tables and figures
MAX_STAGE_DIFF_UM = 2.0         # pairing: largest difference of the stored stage positions
QUADRANT_TOL_PX = 1.0           # shift check: quadrants must agree with the whole image within this
EDGE_SIGMA_PX = 1.5             # smoothing before the gradient (edge image)


def display_name(name: str) -> str:
    """Readable name of a lighting mode (LIGHTING_NAMES, otherwise the short name)."""
    return LIGHTING_NAMES.get(name, name)


def lighting_modes(ds: pd.DataFrame) -> list[str]:
    """Short names of the lighting modes added to a dataset (columns crop_<name>)."""
    return [c.removeprefix("crop_") for c in ds.columns if c.startswith("crop_") and c != "crop_size_px"
            and c not in ("crop_col0", "crop_row0")]


def lighting_images(folder: Path) -> pd.DataFrame:
    """Metadata of all VHX TIFFs in a folder (with subfolders): path, stage position,
    magnification, pixel size and image size."""
    from .vhx import read_vhx

    rows = []
    for f in sorted(Path(folder).rglob("*")):
        if f.suffix.lower() not in (".tif", ".tiff"):
            continue

        v = read_vhx(f)
        rows.append({"path": str(f), "stage_x_um": v.stage_x, "stage_y_um": v.stage_y,
                     "magnification": v.magnification, "um_per_px": v.um_per_px,
                     "width_px": v.width, "height_px": v.height})

    return pd.DataFrame(rows)


def pair_images(bright: pd.DataFrame, other: pd.DataFrame) -> pd.DataFrame:
    """For every bright-field image (rows of images.csv): the image of the other mode with the
    nearest stored stage position, if it is within MAX_STAGE_DIFF_UM and has the same
    magnification and image size. Columns image, path, stage_diff_um."""
    rows = []

    for b in bright.itertuples():
        same = other[(other.magnification == b.magnification) & (other.width_px == b.width_px)
                     & (other.height_px == b.height_px)]
        d = np.hypot(same.stage_x_um - b.stage_x_um, same.stage_y_um - b.stage_y_um)

        if len(d) and d.min() <= MAX_STAGE_DIFF_UM:
            k = int(np.argmin(d.to_numpy()))
            rows.append({"image": b.image, "path": same.path.iloc[k], "stage_diff_um": round(float(d.iloc[k]), 2)})

    return pd.DataFrame(rows, columns=["image", "path", "stage_diff_um"])


def edge_image(rgb: np.ndarray) -> np.ndarray:
    """Gradient magnitude of the smoothed grey image, scaled to unit standard deviation."""
    from scipy import ndimage as ndi

    grey = ndi.gaussian_filter(rgb[..., :3].astype(np.float32).mean(axis=2), EDGE_SIGMA_PX)
    g = np.hypot(ndi.sobel(grey, 0), ndi.sobel(grey, 1))
    return g / max(float(g.std()), 1e-6)


def measure_shift(bright: np.ndarray, other: np.ndarray) -> dict:
    """Shift (rows, columns) that maps the other image onto the bright-field image: the pixel
    (r, c) of the bright-field image shows the same place as (r - shift_row, c - shift_col) of
    the other image. Measured on the whole image and checked on the four quadrants."""
    from skimage.registration import phase_cross_correlation

    eb, eo = edge_image(bright), edge_image(other)
    shift, _, _ = phase_cross_correlation(eb, eo, upsample_factor=10)

    h, w = eb.shape
    quads = []
    for r0 in (0, h // 2):
        for c0 in (0, w // 2):
            q, _, _ = phase_cross_correlation(eb[r0:r0 + h // 2, c0:c0 + w // 2], eo[r0:r0 + h // 2, c0:c0 + w // 2],
                                              upsample_factor=10)
            quads.append(q)

    dev = float(np.max(np.hypot(*(np.array(quads) - shift).T)))
    return {"shift_row": round(float(shift[0]), 2), "shift_col": round(float(shift[1]), 2),
            "quadrant_dev_px": round(dev, 2), "ok": dev <= QUADRANT_TOL_PX}


def add_lighting(ds: pd.DataFrame, root: Path, name: str, bright_paths: dict, bright_images: pd.DataFrame,
                 folder: Path, log=print) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Cuts the crops of lighting mode `name` (images in folder) for every particle of the
    dataset (see module docstring) into root/crops_<name>/. bright_paths: bright-field image
    name -> file; bright_images: images.csv of the detection. Returns the dataset with the
    column crop_<name> and the table of image pairs (shift, check, crops)."""
    import tifffile
    from PIL import Image

    root = Path(root)
    (root / f"crops_{name}").mkdir(exist_ok=True)

    log(f"reading the metadata of the {name} images in {folder}")
    other = lighting_images(folder)
    needed = bright_images[bright_images.image.isin(ds.vhx_image)]
    pairs = pair_images(needed, other).set_index("image")
    log(f"{len(pairs)} of {len(needed)} bright-field images with particles have a {name} image")

    ds = ds.copy()
    ds[f"crop_{name}"] = ""
    rows = []

    for img_name, g in ds.groupby("vhx_image"):
        row = {"image": img_name, "n_particles": len(g), "n_crops": 0}

        if img_name not in pairs.index or img_name not in bright_paths:
            row["note"] = "no bright-field file" if img_name in pairs.index else f"no {name} image"
            rows.append(row)
            continue

        bright = tifffile.imread(bright_paths[img_name], key=0)[..., :3]
        light = tifffile.imread(pairs.loc[img_name, "path"], key=0)[..., :3]
        row.update({"path": pairs.loc[img_name, "path"], "stage_diff_um": pairs.loc[img_name, "stage_diff_um"]})
        row.update(measure_shift(bright, light))

        if not row["ok"]:
            row["note"] = f"shift not uniform (quadrants differ by {row['quadrant_dev_px']} px)"
            rows.append(row)
            continue

        dr, dc = int(round(row["shift_row"])), int(round(row["shift_col"]))
        h, w = light.shape[:2]

        for i, r in g.iterrows():
            size = int(r.crop_size_px)
            r0, c0 = int(r.crop_row0) - dr, int(r.crop_col0) - dc

            if r0 < 0 or c0 < 0 or r0 + size > h or c0 + size > w:
                continue                                             # window leaves the image: no crop

            crop_file = f"crops_{name}/{Path(r.crop).name}"
            Image.fromarray(np.ascontiguousarray(light[r0:r0 + size, c0:c0 + size])).save(root / crop_file)
            ds.at[i, f"crop_{name}"] = crop_file
            row["n_crops"] += 1

        rows.append(row)
        log(f"{img_name}: shift ({row['shift_row']:+.1f}, {row['shift_col']:+.1f}) px, "
            f"{row['n_crops']} of {len(g)} crops")

    return ds, pd.DataFrame(rows)


def overview_figure(ds: pd.DataFrame, root: Path, names: list[str], path: Path, n: int = 8, seed: int = 0):
    """Sample of particles per class (n each, random) with the bright-field crop and the
    crops of the given lighting modes side by side, mask outline in every crop; images shown
    with their original values (no normalisation)."""
    import matplotlib.pyplot as plt
    from .report import class_colours, class_order, crop_tile, save_figure, short_class, tile_grid, tile_header, use_style

    use_style()
    has = ds[[f"crop_{m}" for m in names]].fillna("").ne("").all(axis=1)
    d = ds[has]
    classes = class_order(d)
    colours = class_colours(classes)
    rows = pd.concat([g.sample(min(n, len(g)), random_state=seed) for _, g in d.groupby("training_class")])

    per = 1 + len(names)
    cols = 4 * per
    nrow = int(np.ceil(len(rows) / 4))
    fig, axes = tile_grid(nrow, cols, tile_in=1.7, caption_lines=2, fontsize=9, group=per, group_gap_in=0.3)

    for k, r in enumerate(rows.itertuples()):
        row, col = divmod(k, 4)

        for j, mode in enumerate([None] + names):
            ax = axes[row, col * per + j]
            ax.axis("on")
            view = r if mode is None else r._replace(crop=getattr(r, f"crop_{mode}"))
            crop_tile(ax, root, view, colours[r.training_class])
            label = "bright field" if mode is None else display_name(mode)
            caption = f"#{r.sem_id} {short_class(r.training_class)}\n{label}" if j == 0 else f"\n{label}"
            ax.set_title(caption, fontsize=9, pad=3)

    tile_header(fig, f"Lighting modes per particle ({n} random per class; original grey values; line = bright-field mask)")
    save_figure(fig, path, dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Hand-made features of a lighting mode, for the dataset report

LIGHT_FEATURES = {              # suffix -> label; columns are lt_<name>_<suffix>
    "in_diff": "inside minus background (grey levels)",
    "in_p95_diff": "brightest 5 % inside minus background (grey levels)",
    "in_std_ratio": "spread inside / spread of the background",
    "ring_diff": "ring around minus background (grey levels)",
    "edge_ratio": "edge strength on the outline / in the background",
    "asym": "light / shadow asymmetry in the ring (grey levels)",
    "red_share": "red share of the excess light inside",
    "blue_share": "blue share of the excess light inside",
}


def light_features(bright: np.ndarray, mask: np.ndarray, light: np.ndarray, um_per_px: float) -> dict:
    """Features of one lighting-mode crop, measured on the particle object found in the
    bright-field crop (cii.crop_features.object_and_region, the same for every particle) and
    on the matrix ring of 1 µm around it. Grey = mean of R, G, B (0-255); background = median
    of the pixels farther than 2 µm from the object. Original grey values, no normalisation."""
    from scipy import ndimage as ndi
    from .crop_features import RING_UM, SECTORS, crop_contrast, disk, object_and_region

    out = {k: np.nan for k in LIGHT_FEATURES}
    if not mask.any():
        return out

    _, c_max, _, _ = crop_contrast(bright, mask, um_per_px)
    obj, _ = object_and_region(c_max, mask, um_per_px)

    if obj.sum() < 6:
        return out

    rgb = light[..., :3].astype(np.float32)
    grey = rgb.mean(axis=2)
    away = ~ndi.binary_dilation(obj, disk(2.0 / um_per_px))
    ring = ndi.binary_dilation(obj, disk(RING_UM / um_per_px)) & ~obj

    if away.sum() < 50:
        return out

    bg = float(np.median(grey[away]))
    bg_rgb = np.median(rgb[away], axis=0)
    bg_std = max(float(grey[away].std()), 1e-3)

    # brightness inside the object and in the ring, relative to the background
    out["in_diff"] = float(grey[obj].mean() - bg)
    out["in_p95_diff"] = float(np.percentile(grey[obj], 95) - bg)
    out["in_std_ratio"] = float(grey[obj].std() / bg_std)
    out["ring_diff"] = float(grey[ring].mean() - bg) if ring.any() else np.nan

    # edges: gradient on the outline against the gradient of the matrix texture
    smooth = ndi.gaussian_filter(grey, 1.0)
    grad = np.hypot(ndi.sobel(smooth, 0), ndi.sobel(smooth, 1))
    outline = ndi.binary_dilation(obj, disk(1)) & ~ndi.binary_erosion(obj, disk(1))
    out["edge_ratio"] = float(grad[outline].mean() / max(float(np.median(grad[away])), 1e-3))

    # light / shadow: first harmonic of the ring brightness over SECTORS directions
    rows, cols = np.nonzero(ring)
    r0, c0 = np.argwhere(obj).mean(axis=0)
    sector = ((np.arctan2(-(rows - r0), cols - c0) + np.pi) / (2 * np.pi) * SECTORS).astype(int) % SECTORS
    present = [k for k in range(SECTORS) if (sector == k).any()]

    if len(present) >= SECTORS - 2:
        mean = np.array([grey[rows, cols][sector == k].mean() for k in present])
        angle = (np.array(present) + 0.5) / SECTORS * 2 * np.pi - np.pi
        mean = mean - mean.mean()
        out["asym"] = float(np.hypot((mean * np.cos(angle)).sum(), (mean * np.sin(angle)).sum()) * 2 / len(present))

    # colour of the light the particle adds (or removes) compared with the matrix
    excess = np.abs(rgb[obj].mean(axis=0) - bg_rgb)
    total = float(excess.sum())
    if total > 1.0:
        out["red_share"] = float(excess[0] / total)
        out["blue_share"] = float(excess[2] / total)

    return out


def dataset_light_features(ds: pd.DataFrame, root: Path, name: str) -> pd.DataFrame:
    """Features of lighting mode `name` for every particle (NaN where it has no crop of this
    mode); columns lt_<name>_<suffix>."""
    from PIL import Image

    root = Path(root)
    rows = []

    for r in ds.itertuples():
        light_file = getattr(r, f"crop_{name}", "")

        if not isinstance(light_file, str) or not light_file:
            rows.append({})
            continue

        bright = np.asarray(Image.open(root / r.crop))[..., :3]
        mask = np.asarray(Image.open(root / r.mask)) > 0
        light = np.asarray(Image.open(root / light_file))[..., :3]
        rows.append(light_features(bright, mask, light, float(r.um_per_px)))

    cols = {k: f"lt_{name}_{k}" for k in LIGHT_FEATURES}
    return pd.DataFrame(rows, index=ds.index).reindex(columns=list(LIGHT_FEATURES)).rename(columns=cols)


def light_feature_definitions(columns) -> dict:
    """Group and label of the lighting features among `columns`: column -> (display name of
    the mode, label)."""
    out = {}

    for c in columns:
        if not c.startswith("lt_"):
            continue

        for suffix, label in LIGHT_FEATURES.items():
            if c.endswith("_" + suffix):
                name = c.removeprefix("lt_").removesuffix("_" + suffix)
                out[c] = (display_name(name), label)
                break

    return out
