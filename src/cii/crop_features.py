"""Hand-made appearance features from the dataset crops and masks.

Three groups per crop, meant to separate particles that look alike in colour
(e.g. empty pits against solid oxides):

shape     outline of the particle: solidity, circularity, extent, aspect ratio
          and the direction of the elongation (as a vector, so that 0 and 180 deg
          are the same; a common direction would point to polishing).
interior  structure inside the particle on the per-pixel contrast (strongest of
          R, G, B, as in the detection): spread of the contrast, core against
          rim, number of dark cores, colour variation. Relative to the darkness of
          the particle, so they describe the structure, not how dark it is.
edge      the outline and the matrix around it on the grey contrast: edge width,
          ring just outside the particle (dark or bright halo), light / shadow
          asymmetry in that ring (relief catches the light on one side).

Contrast is (background - value) / background as in cii.detection, smoothed
with the same Gaussian, but the background is the crop's own: the median per
channel of the pixels away from the particle.

Object and region: the saved mask is the matched object or, for region
matches, its region, and that choice came from the matching (with the SEM
size). So that the features need the VHX image only and are defined the same
way for every particle, the mask is used only to find the particle's darkest
point. From there, as in the detection, the object is the patch above half
that contrast and the region the patch above the faint-edge contrast
(GROW_CONTRAST, the detection's low threshold), at most REGION_UM beyond the
object. Everything uses the object, except the region size relative to the
object and the number of dark cores, which use the region.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage as ndi

SMOOTH_PX = 1.0             # Gaussian sigma against pixel noise, as in the detection
GROW_CONTRAST = 0.04        # faint edge of a particle: low threshold of the detection (channels="max")
REGION_UM = 2.0             # the region reaches at most this far beyond the object
BACKGROUND_GAP_UM = 2.0     # pixels closer than this to the particle are not used for the background
RING_UM = 1.0               # width of the matrix ring around the particle
SECTORS = 8                 # direction sectors for the light / shadow asymmetry
MIN_AREA_PX = 6             # smaller objects get no interior and edge features

FEATURES = {                # column -> (group, label for tables and figures)
    "shape_solidity": ("shape", "solidity (area / convex area)"),
    "shape_circularity": ("shape", "circularity (4 pi area / perimeter²)"),
    "shape_extent": ("shape", "extent (area / bounding box)"),
    "shape_aspect_ratio": ("shape", "aspect ratio"),
    "shape_elong_x": ("shape", "elongation along x"),
    "shape_elong_y": ("shape", "elongation along y / diagonal"),
    "int_spread": ("interior", "contrast spread inside (relative)"),
    "int_core_rim": ("interior", "core minus rim contrast (relative)"),
    "int_region_ratio": ("interior", "region area / object area"),
    "int_n_cores": ("interior", "number of dark cores in the region"),
    "int_colour_spread": ("interior", "colour variation in the region (relative)"),
    "edge_width_um": ("edge", "edge width (µm)"),
    "edge_ring": ("edge", "ring contrast around the particle"),
    "edge_ring_bright": ("edge", "bright rim around the particle"),
    "edge_asym_x": ("edge", "shadow asymmetry along x (ring)"),
    "edge_asym_y": ("edge", "shadow asymmetry along y (ring)"),
}
INFO_COLUMNS = ["object_ecd_um", "region_ecd_um", "orientation_deg", "edge_asym",
                "background_R", "background_G", "background_B"]


def disk(radius_px: float) -> np.ndarray:
    """Round structuring element with the given radius in pixels (at least 1)."""
    r = max(1, int(round(radius_px)))
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
    return xx ** 2 + yy ** 2 <= r ** 2


def crop_contrast(rgb: np.ndarray, mask: np.ndarray, um_per_px: float):
    """Contrast images of one crop: per channel (H x W x 3), strongest channel (H x W) and
    grey (H x W, contrast of the RGB mean), plus the background per channel. The background
    is the median of the pixels farther than BACKGROUND_GAP_UM from the mask (all pixels if
    too few are left)."""
    rgb = rgb[..., :3].astype(np.float32)
    away = ~ndi.binary_dilation(mask, disk(BACKGROUND_GAP_UM / um_per_px))

    if away.sum() < 50:
        away = np.ones_like(mask)

    bg = np.maximum(np.median(rgb[away], axis=0), 1.0)
    smooth = np.stack([ndi.gaussian_filter(rgb[..., k], SMOOTH_PX) for k in range(3)], axis=2)
    c_rgb = (bg - smooth) / bg

    grey_bg = float(bg.mean())
    c_grey = (grey_bg - smooth.mean(axis=2)) / grey_bg

    return c_rgb, c_rgb.max(axis=2), c_grey, bg


def object_and_region(c_max: np.ndarray, mask: np.ndarray, um_per_px: float):
    """Object and region of the particle in the saved mask (see module docstring): the patch
    above half the darkest contrast in the mask, and the patch above GROW_CONTRAST at most
    REGION_UM beyond the object, both connected to the darkest point and with holes filled."""
    masked = np.where(mask, c_max, -np.inf)
    peak = np.unravel_index(int(np.argmax(masked)), mask.shape)
    peak_contrast = float(c_max[peak])

    lab, _ = ndi.label(c_max >= 0.5 * peak_contrast)
    obj = ndi.binary_fill_holes(lab == lab[peak])

    near = ndi.binary_dilation(obj, disk(REGION_UM / um_per_px))
    lab, _ = ndi.label(((c_max >= GROW_CONTRAST) & near) | obj)
    reg = ndi.binary_fill_holes(lab == lab[peak])

    return obj, reg


def shape_features(obj: np.ndarray, um_per_px: float) -> dict:
    """Shape of the object: solidity, circularity, extent, aspect ratio and the
    elongation vector (1 - minor / major axis, at twice the angle of the major axis, so that
    opposite directions are the same; x to the right, y upwards)."""
    from skimage.measure import regionprops

    r = regionprops(obj.astype(np.uint8))[0]
    major = max(r.axis_major_length, 1e-6)
    minor = max(r.axis_minor_length, 1e-6)

    # major axis direction: regionprops gives its angle to the row axis; convert to x right, y up
    theta = np.arctan2(-np.cos(r.orientation), np.sin(r.orientation))
    elong = 1 - minor / major

    return {
        "object_ecd_um": float(np.sqrt(4 * r.area / np.pi) * um_per_px),
        "shape_solidity": float(r.solidity),
        "shape_circularity": float(min(1.0, 4 * np.pi * r.area / max(r.perimeter, 1e-6) ** 2)),
        "shape_extent": float(r.extent),
        "shape_aspect_ratio": float(major / minor),
        "shape_elong_x": float(elong * np.cos(2 * theta)),
        "shape_elong_y": float(elong * np.sin(2 * theta)),
        "orientation_deg": float(np.degrees(theta) % 180),
    }


def interior_features(c_rgb: np.ndarray, c_max: np.ndarray, obj: np.ndarray, reg: np.ndarray) -> dict:
    """Structure inside the particle, relative to the object's darkness (90th percentile of
    its contrast). In the object: spread (p90 - p10), core minus rim (median contrast of the
    inner half of the distance to the outline minus that of the outer half) and colour
    variation (spread of green - red and blue - red). In the region: area relative to the
    object and number of dark cores (patches above half the darkest contrast)."""
    keys = ["int_spread", "int_core_rim", "int_region_ratio", "int_n_cores", "int_colour_spread"]
    out = {k: np.nan for k in keys}

    if obj.sum() < MIN_AREA_PX:
        return out

    c = c_max[obj]
    p10, p90 = np.percentile(c, [10, 90])

    if p90 <= GROW_CONTRAST:
        return out

    out["int_spread"] = float((p90 - p10) / p90)

    # core against rim: split by the distance to the outline
    dist = ndi.distance_transform_edt(obj)
    half = dist.max() / 2
    core = obj & (dist > half)
    rim = obj & (dist <= half)

    if dist.max() >= 2 and core.any() and rim.any():
        out["int_core_rim"] = float((np.median(c_max[core]) - np.median(c_max[rim])) / p90)

    # colour variation: spread of the channel differences in the object
    gr = c_rgb[..., 1][obj] - c_rgb[..., 0][obj]
    br = c_rgb[..., 2][obj] - c_rgb[..., 0][obj]
    out["int_colour_spread"] = float(np.sqrt(gr.var() + br.var()) / p90)

    # region: how much fainter material surrounds the object, and separate dark cores in it
    out["int_region_ratio"] = float(reg.sum() / obj.sum())

    lab, _ = ndi.label(reg & (c_max >= 0.5 * c.max()))
    sizes = np.bincount(lab.ravel())[1:]
    out["int_n_cores"] = int((sizes >= 2).sum())

    return out


def edge_features(c_grey: np.ndarray, obj: np.ndarray, um_per_px: float) -> dict:
    """Outline of the object and its surroundings on the grey contrast: edge width (darkness /
    steepest gradient, um), median contrast of the matrix ring of RING_UM around the object (> 0:
    dark halo), bright rim (minus the 10th percentile of the ring, > 0: brighter than the
    matrix) and the light / shadow asymmetry in that ring (first harmonic of the mean contrast
    in SECTORS directions around the centre, relative to the darkness; > 0: darker on the +x /
    +y side; x right, y up)."""
    keys = ["edge_width_um", "edge_ring", "edge_ring_bright", "edge_asym_x", "edge_asym_y", "edge_asym"]
    out = {k: np.nan for k in keys}

    if obj.sum() < MIN_AREA_PX:
        return out

    p90 = float(np.percentile(c_grey[obj], 90))

    if p90 <= 0.02:
        return out

    # edge width: how far it takes to go from the matrix to the particle's darkness
    gr, gc = np.gradient(c_grey)
    grad = np.hypot(gr, gc) / um_per_px
    near_edge = ndi.binary_dilation(obj, disk(1)) & ~ndi.binary_erosion(obj, disk(1))
    steep = float(np.percentile(grad[near_edge], 90)) if near_edge.any() else 0.0
    out["edge_width_um"] = p90 / steep if steep > 0 else np.nan

    # ring of matrix around the particle
    ring = ndi.binary_dilation(obj, disk(RING_UM / um_per_px)) & ~obj

    if ring.sum() >= 10:
        out["edge_ring"] = float(np.median(c_grey[ring]))
        out["edge_ring_bright"] = float(-np.percentile(c_grey[ring], 10))

    # light / shadow asymmetry in the ring (the outline itself is at half the darkness everywhere)
    rows, cols = np.nonzero(ring)
    r0, c0 = np.argwhere(obj).mean(axis=0)
    phi = np.arctan2(-(rows - r0), cols - c0)
    sector = ((phi + np.pi) / (2 * np.pi) * SECTORS).astype(int) % SECTORS
    values = c_grey[rows, cols]

    present = [k for k in range(SECTORS) if (sector == k).any()]

    if len(present) >= SECTORS - 2:
        mean = np.array([values[sector == k].mean() for k in present])
        mean = mean - mean.mean()
        angle = (np.array(present) + 0.5) / SECTORS * 2 * np.pi - np.pi

        ax = 2 / len(present) * float((mean * np.cos(angle)).sum()) / p90
        ay = 2 / len(present) * float((mean * np.sin(angle)).sum()) / p90
        out.update({"edge_asym_x": ax, "edge_asym_y": ay, "edge_asym": float(np.hypot(ax, ay))})

    return out


def crop_features(rgb: np.ndarray, mask: np.ndarray, um_per_px: float) -> dict:
    """All features of one crop (see module docstring); NaN where a feature is undefined
    (empty mask, object too small or too faint)."""
    out = {k: np.nan for k in list(FEATURES) + INFO_COLUMNS}

    if not mask.any():
        return out

    c_rgb, c_max, c_grey, bg = crop_contrast(rgb, mask, um_per_px)
    obj, reg = object_and_region(c_max, mask, um_per_px)

    out.update({"background_R": float(bg[0]), "background_G": float(bg[1]), "background_B": float(bg[2]),
                "region_ecd_um": float(np.sqrt(4 * reg.sum() / np.pi) * um_per_px)})
    out.update(shape_features(obj, um_per_px))
    out.update(interior_features(c_rgb, c_max, obj, reg))
    out.update(edge_features(c_grey, obj, um_per_px))

    return out


def dataset_features(ds: pd.DataFrame, root: Path) -> pd.DataFrame | None:
    """Features of every crop of a dataset (rows as ds, columns FEATURES + INFO_COLUMNS);
    root is the dataset folder. None if the crop files are missing."""
    from PIL import Image

    root = Path(root)
    if not (root / ds.crop.iloc[0]).exists():
        return None

    rows = []
    for r in ds.itertuples():
        rgb = np.asarray(Image.open(root / r.crop))
        mask = np.asarray(Image.open(root / r.mask)) > 0
        rows.append(crop_features(rgb, mask, float(r.um_per_px)))

    return pd.DataFrame(rows, index=ds.index)[list(FEATURES) + INFO_COLUMNS]
