"""Training dataset export: VHX crop + mask per SEM particle, with its training class.

Particles used: ``training_ok`` (matched, not cluster) and a non-empty
``training_class`` after the class file (Type table, rules, min_ecd_um).

Crops: ``crop_px`` x ``crop_px`` pixels of the VHX image at full resolution
(no rescaling, so every crop has the image's um per pixel), centred on the
matched VHX object (on the region box centre for region matches). Near the
image border the window is shifted inwards so that it holds only real pixels;
``crop_col0`` / ``crop_row0`` give its position. Masks: the matched object
(or, for region matches, its region) from the label images of
detect_inclusions.py, 255 = particle, 0 = rest.

Split: the particles are grouped in square blocks of ``block_um`` (by their
VHX stage position), so that neighbouring particles and overlapping VHX images
stay together. With stratified group k-fold (classes balanced, groups kept
whole) about ``test_fraction`` of the particles is held out as test set; the
rest is divided into ``n_folds`` folds for cross-validation (one fold
validation, the others training, in turn).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

# columns of matches.csv copied into dataset.csv: source name -> dataset name
COPY_COLUMNS = {
    "sem_id": "sem_id", "field": "field", "vhx_image": "vhx_image", "vhx_id": "vhx_id",
    "training_class": "training_class", "class_source": "class_source", "class_label": "class_label",
    "Type": "Type", "label": "label", "match_type": "match_type", "distance_um": "distance_um",
    "n_candidates": "n_candidates", "sem_neighbour": "sem_neighbour",
    "ecd_um": "sem_ecd_um", "length_um": "sem_length_um", "breadth_um": "sem_breadth_um",
    "aspect_ratio": "sem_aspect_ratio", "vhx_ecd_um": "vhx_ecd_um", "vhx_area_um2": "vhx_area_um2",
    "vhx_aspect_ratio": "vhx_aspect_ratio", "vhx_contrast_max": "vhx_contrast_max",
    "vhx_contrast_R_max": "vhx_contrast_R_max", "vhx_contrast_G_max": "vhx_contrast_G_max",
    "vhx_contrast_B_max": "vhx_contrast_B_max", "vhx_mean_R": "vhx_mean_R", "vhx_mean_G": "vhx_mean_G",
    "vhx_mean_B": "vhx_mean_B", "vhx_x_um": "vhx_x_um", "vhx_y_um": "vhx_y_um",
}


@dataclass
class ExportParams:
    crop_px: int = 96               # crop size in pixels (96 px = 19.6 um at 500x)
    test_fraction: float = 0.2      # share held out as test set
    n_folds: int = 5                # cross-validation folds of the remaining particles
    block_um: float = 1000.0        # side of the spatial blocks used as groups for the split
    seed: int = 0


def crop_window(col: float, row: float, size: int, width: int, height: int) -> tuple[int, int]:
    """Top-left corner (col0, row0) of a size x size window centred on (col, row), shifted
    inwards where it would leave the image (so the crop holds only real pixels)."""
    c0 = int(round(col - size / 2))
    r0 = int(round(row - size / 2))
    c0 = min(max(c0, 0), max(width - size, 0))
    r0 = min(max(r0, 0), max(height - size, 0))
    return c0, r0


def crop_centre(row: pd.Series) -> tuple[float, float]:
    """Pixel position (col, row) the crop is centred on: the matched object's centroid, or the
    centre of its region box for region matches (the object is then only a dark spot)."""
    if row.get("match_type") == "region" and np.isfinite(row.get("vhx_region_bbox_col0", np.nan)):
        return ((row.vhx_region_bbox_col0 + row.vhx_region_bbox_col1 - 1) / 2,
                (row.vhx_region_bbox_row0 + row.vhx_region_bbox_row1 - 1) / 2)
    return float(row.vhx_col_px), float(row.vhx_row_px)


def spatial_groups(x_um: np.ndarray, y_um: np.ndarray, block_um: float) -> np.ndarray:
    """Group number per particle: the square block of side block_um its position falls into."""
    bx = np.floor((x_um - np.nanmin(x_um)) / block_um).astype(int)
    by = np.floor((y_um - np.nanmin(y_um)) / block_um).astype(int)
    return by * (bx.max() + 1) + bx


def assign_splits(classes: np.ndarray, groups: np.ndarray, p: ExportParams) -> tuple[np.ndarray, np.ndarray]:
    """Split per particle ('test' / 'trainval') and fold (0 .. n_folds-1, -1 for test), both with
    stratified group k-fold: classes balanced, blocks never divided. The test set is the first
    fold of a k-fold with k = round(1 / test_fraction)."""
    from sklearn.model_selection import StratifiedGroupKFold
    n = len(classes)
    split = np.full(n, "trainval", dtype=object)
    fold = np.full(n, -1)
    k_test = max(2, int(round(1 / p.test_fraction)))
    outer = StratifiedGroupKFold(n_splits=k_test, shuffle=True, random_state=p.seed)
    _, test_idx = next(outer.split(np.zeros(n), classes, groups))
    split[test_idx] = "test"
    tv = np.flatnonzero(split == "trainval")
    inner = StratifiedGroupKFold(n_splits=p.n_folds, shuffle=True, random_state=p.seed)
    for k, (_, val_idx) in enumerate(inner.split(np.zeros(len(tv)), classes[tv], groups[tv])):
        fold[tv[val_idx]] = k
    return split, fold


def find_object(obj: pd.DataFrame, img_objects: pd.DataFrame | None, img_name: str, r: pd.Series,
                max_px: float = 0.5):
    """(label, region) of the matched VHX object in the label images, and whether it had to be
    found by position: the id from matches.csv is used if its centroid agrees with the one in
    matches.csv (within max_px), otherwise the object at that centroid (ids of equally large
    objects may differ between detection runs). None if no object is there."""
    key = (img_name, int(r.vhx_id))
    if key in obj.index:
        o = obj.loc[key]
        if abs(o.col_px - r.vhx_col_px) <= max_px and abs(o.row_px - r.vhx_row_px) <= max_px:
            return (int(o.label), int(o.region)), False
    if img_objects is None or not len(img_objects):
        return None
    d = np.hypot(img_objects.col_px.to_numpy() - r.vhx_col_px, img_objects.row_px.to_numpy() - r.vhx_row_px)
    k = int(np.argmin(d))
    if d[k] > max_px:
        return None
    o = img_objects.iloc[k]
    return (int(o.label), int(o.region)), True


def select_particles(matches: pd.DataFrame, assigned: pd.DataFrame) -> pd.DataFrame:
    """Particles for the dataset: training_ok and a non-empty training_class (assigned =
    output of cii.classes.apply_classes for the same rows)."""
    m = matches.join(assigned)
    return m[m.training_ok.astype(bool) & (m.training_class != "")].copy()


def export_dataset(particles: pd.DataFrame, vhx_obj: pd.DataFrame, vhx_img: pd.DataFrame, vhx_paths: dict,
                   labels_dir: Path, out_dir: Path, sample: str, p: ExportParams, log=print) -> pd.DataFrame:
    """Writes crops/ and masks/ for the selected particles and returns the dataset table
    (one row per crop, see module docstring); particles whose VHX image or label images are
    missing are skipped and reported."""
    from PIL import Image
    import tifffile
    from .detection import read_label_images
    out_dir = Path(out_dir)
    (out_dir / "crops").mkdir(parents=True, exist_ok=True)
    (out_dir / "masks").mkdir(parents=True, exist_ok=True)
    obj = vhx_obj.set_index(["image", "id"])[["label", "region", "col_px", "row_px"]]
    by_img = {n: g for n, g in vhx_obj.groupby("image")}
    geo = vhx_img.set_index("image")
    rows, skipped = [], {}
    for img_name, g in particles.groupby("vhx_image"):
        if img_name not in vhx_paths:
            skipped["VHX image not found"] = skipped.get("VHX image not found", 0) + len(g)
            continue
        objects, regions = read_label_images(labels_dir, Path(img_name).stem)
        if objects is None or regions is None:
            skipped["label images missing"] = skipped.get("label images missing", 0) + len(g)
            continue
        rgb = tifffile.imread(vhx_paths[img_name], key=0)[..., :3]
        h, w = rgb.shape[:2]
        for _, r in g.iterrows():
            found = find_object(obj, by_img.get(img_name), img_name, r)
            if found is None:
                skipped["matched object not in objects.csv"] = skipped.get("matched object not in objects.csv", 0) + 1
                continue
            (lab_id, reg_id), moved = found
            if moved:
                skipped["object found by position (id differs from matches.csv)"] = \
                    skipped.get("object found by position (id differs from matches.csv)", 0) + 1
            if r.match_type == "region":
                mask_full, mask_kind = regions, reg_id
            else:
                mask_full, mask_kind = objects, lab_id
            cc, rr = crop_centre(r)
            c0, r0 = crop_window(cc, rr, p.crop_px, w, h)
            crop = rgb[r0:r0 + p.crop_px, c0:c0 + p.crop_px]
            mask = (mask_full[r0:r0 + p.crop_px, c0:c0 + p.crop_px] == mask_kind).astype(np.uint8) * 255
            name = f"{sample}_{int(r.sem_id)}.png"
            Image.fromarray(np.ascontiguousarray(crop)).save(out_dir / "crops" / name)
            Image.fromarray(mask).save(out_dir / "masks" / name)
            row = {"crop": f"crops/{name}", "mask": f"masks/{name}", "sample": sample}
            row.update({dst: r[src] for src, dst in COPY_COLUMNS.items() if src in r})
            row.update({"mask_source": "region" if r.match_type == "region" else "object",
                        "mask_px": int((mask > 0).sum()), "crop_col0": c0, "crop_row0": r0,
                        "crop_size_px": p.crop_px, "um_per_px": float(geo.loc[img_name, "um_per_px"]),
                        "magnification": float(geo.loc[img_name, "magnification"])})
            row.update({c: r[c] for c in particles.columns if c.endswith("_wt")})
            rows.append(row)
    for why, n in skipped.items():
        log(f"{'note' if why.startswith('object found') else 'skipped'}: {n} particles - {why}")
    ds = pd.DataFrame(rows)
    if len(ds):
        wt = [c for c in ds.columns if c.endswith("_wt")]
        empty = [c for c in wt if not (ds[c].fillna(0) > 0).any()]
        ds = ds.drop(columns=empty)                       # elements never detected in these particles
    return ds


def params_dict(p: ExportParams) -> dict:
    """Export parameters as a plain dict (for dataset_info.json)."""
    return asdict(p)
