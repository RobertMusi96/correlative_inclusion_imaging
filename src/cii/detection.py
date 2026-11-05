"""Detection of dark objects (inclusions, pores, pull-outs, ...) in bright-field
light-microscope images.

Method
------
1. Grey image = mean of R, G, B.
2. Background = median filter over ``background_um`` (computed on a reduced
   image for speed), so uneven illumination does not matter.
3. Contrast = (background - grey) / background, after a slight Gaussian
   smoothing (``smooth_px``) against pixel noise. 0 = matrix, 1 = black.
4. Detection by hysteresis threshold: an object needs a core with contrast
   >= ``high`` and grows into connected pixels with contrast >= ``low``.
5. Size: within each detected object only pixels reaching ``size_fraction``
   (default 0.5, half maximum) of the object's own peak contrast are kept. The
   outline then sits at the same point regardless of optical blur, so sizes agree
   between magnifications; weakly joined neighbours are split.
6. Objects smaller than ``min_area_um2`` are dropped.

All size parameters are in um, so one parameter set works at every magnification.
No classification here: every dark object is reported; which ones are
inclusions is decided later (SEM correlation).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from skimage.filters import apply_hysteresis_threshold
from skimage.measure import regionprops

from .geometry import StageImage


@dataclass
class DetectionParams:
    background_um: float = 40.0   # median filter size for the background
    smooth_px: float = 1.0        # Gaussian sigma against pixel noise
    high: float = 0.12            # core contrast (fraction of background)
    low: float = 0.03             # edge contrast for region growing
    size_fraction: float = 0.5    # object outline at this fraction of its peak contrast (0 = off)
    min_area_um2: float = 0.3     # smallest object kept (ECD ~ 0.6 um)


@dataclass
class Detection:
    labels: np.ndarray            # int32 label image, 0 = matrix
    objects: pd.DataFrame         # one row per object, 'label' matches labels
    contrast: np.ndarray          # contrast image (float32)
    background: np.ndarray        # background grey level (float32)
    params: DetectionParams


def background_and_contrast(rgb: np.ndarray, um_per_px: float, p: DetectionParams):
    g = rgb[..., :3].astype(np.float32).mean(axis=2) if rgb.ndim == 3 else rgb.astype(np.float32)
    win_full = p.background_um / um_per_px
    k = max(1, int(np.ceil(win_full / 25)))                    # median window <= ~25 px on the reduced image
    win = max(3, int(round(win_full / k)) | 1)
    small = g[::k, ::k]
    bg = ndi.median_filter(small, size=win, mode="reflect")
    if k > 1:
        bg = ndi.zoom(bg, (g.shape[0] / small.shape[0], g.shape[1] / small.shape[1]), order=1)
        bg = bg[: g.shape[0], : g.shape[1]]
    bg = np.maximum(bg, 1.0).astype(np.float32)
    gs = ndi.gaussian_filter(g, p.smooth_px) if p.smooth_px > 0 else g
    return g, bg, ((bg - gs) / bg).astype(np.float32)


def detect(image: np.ndarray, um_per_px: float, params: DetectionParams | None = None,
           stage: StageImage | None = None) -> Detection:
    """Detect dark objects. ``stage`` (optional) adds stage coordinates."""
    p = params or DetectionParams()
    rgb = image
    g, bg, c = background_and_contrast(rgb, um_per_px, p)
    min_px = p.min_area_um2 / um_per_px ** 2
    # 1) detection: core >= high, grown into connected pixels >= low; min area applies here
    detected, n = ndi.label(apply_hysteresis_threshold(c, p.low, p.high))
    if n:
        sizes = ndi.sum(detected > 0, detected, np.arange(1, n + 1))
        keep = np.zeros(n + 1, bool); keep[1:] = sizes >= min_px
        detected, n = ndi.label(keep[detected])
    # 2) size: within each detected object keep pixels >= size_fraction * its own peak
    #    contrast (half maximum by default) - independent of blur; weakly joined
    #    particles fall apart. Parts >= min area become objects; every detected
    #    object keeps at least its largest part.
    if n and p.size_fraction > 0:
        peak = np.zeros(n + 1, np.float32)
        peak[1:] = ndi.maximum(c, detected, np.arange(1, n + 1))
        parts, m = ndi.label((detected > 0) & (c >= p.size_fraction * peak[detected]))
        part_size = np.zeros(m + 1); part_size[1:] = ndi.sum(parts > 0, parts, np.arange(1, m + 1))
        part_parent = np.zeros(m + 1, int); part_parent[1:] = ndi.maximum(detected, parts, np.arange(1, m + 1))
        keep = part_size >= min_px
        largest = pd.Series(part_size[1:], index=np.arange(1, m + 1)).groupby(part_parent[1:]).idxmax()
        keep[largest.to_numpy()] = True
        keep[0] = False
        lab, n = ndi.label(keep[parts])
    else:
        lab = detected
    h, w = lab.shape
    rows = []
    for r in regionprops(lab, intensity_image=c):
        r0, c0, r1, c1 = r.bbox
        area_um2 = r.area * um_per_px ** 2
        coords = r.coords
        px_rgb = rgb[coords[:, 0], coords[:, 1], :3].astype(float) if rgb.ndim == 3 else None
        row = {
            "label": r.label,
            "col_px": r.centroid[1], "row_px": r.centroid[0],
            "area_px": int(r.area), "area_um2": area_um2,
            "ecd_um": 2 * np.sqrt(area_um2 / np.pi),
            "length_um": r.axis_major_length * um_per_px,
            "breadth_um": r.axis_minor_length * um_per_px,
            "aspect_ratio": r.axis_major_length / max(r.axis_minor_length, 1e-9),
            "orientation_deg": float(np.degrees(r.orientation)),
            "solidity": r.solidity,
            "contrast_mean": r.intensity_mean, "contrast_max": r.intensity_max,
            "bbox_row0": r0, "bbox_col0": c0, "bbox_row1": r1, "bbox_col1": c1,
            "touches_border": bool(r0 == 0 or c0 == 0 or r1 == h or c1 == w),
        }
        if px_rgb is not None:
            row.update({"mean_R": px_rgb[:, 0].mean(), "mean_G": px_rgb[:, 1].mean(), "mean_B": px_rgb[:, 2].mean()})
        rows.append(row)
    df = pd.DataFrame(rows)
    if stage is not None and len(df):
        x, y = stage.pixel_to_stage(df.col_px.to_numpy(), df.row_px.to_numpy())
        df.insert(3, "stage_x_um", x)
        df.insert(4, "stage_y_um", y)
    if len(df):
        df = df.sort_values("area_um2", ascending=False).reset_index(drop=True)
        df.insert(0, "id", np.arange(1, len(df) + 1))   # 1 = largest object in the image
    return Detection(labels=lab.astype(np.int32), objects=df, contrast=c, background=bg, params=p)


def params_dict(p: DetectionParams) -> dict:
    return asdict(p)
