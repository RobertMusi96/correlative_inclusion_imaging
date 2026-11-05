"""Detection of dark objects (inclusions, pores, pull-outs, ...) in bright-field
light-microscope images.

Method
------
1. Background = median filter over ``background_um`` (computed on a reduced
   image for speed), so uneven illumination does not matter.
2. Contrast = (background - value) / background, after a slight Gaussian
   smoothing (``smooth_px``) against pixel noise. 0 = matrix, 1 = black.
   ``channels``:

   * ``"max"`` (default): contrast computed for R, G and B separately (each
     against its own background), the strongest of the three is used: an object
     is at least X % darker than the matrix in at least one colour. Catches
     coloured particles (e.g. pink Ti(C,N): darker mainly in green) that the
     grey mean dilutes. Slightly noisier than the mean.
   * ``"mean"``: contrast of the grey value (mean of R, G, B), the original method.

   The thresholds default per mode (``DEFAULT_THRESHOLDS``).
3. Detection by hysteresis threshold: an object needs a core with contrast
   >= ``high`` and grows into connected pixels with contrast >= ``low``.
5. Size: within each detected object only pixels reaching ``size_fraction``
   (default 0.5, half maximum) of the object's own peak contrast are kept. The
   outline then sits at the same point regardless of optical blur, so sizes agree
   between magnifications; weakly joined neighbours are split.
6. Objects smaller than ``min_area_um2`` are dropped.

Every object also carries its *region* (``region_*`` columns): the full
hysteresis-detected area it was cut from. A region can hold several objects
(e.g. two touching particles) or be much larger than its object (a faint
particle with a very dark spot: the half-maximum cut keeps only the spot).

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


# (high, low) per contrast mode. "max": noise of the strongest channel is about
# twice that of the mean (99.9 % of matrix pixels below 0.036 vs 0.019 at 500x),
# pink particles reach 0.07-0.14 in green at 500x.
DEFAULT_THRESHOLDS = {"max": (0.07, 0.04), "mean": (0.12, 0.03)}


@dataclass
class DetectionParams:
    channels: str = "max"         # "max" (strongest of R, G, B) or "mean" (grey)
    background_um: float = 40.0   # median filter size for the background
    smooth_px: float = 1.0        # Gaussian sigma against pixel noise
    high: float | None = None     # core contrast (fraction of background); None = default of the mode
    low: float | None = None      # edge contrast for region growing; None = default of the mode
    size_fraction: float = 0.5    # object outline at this fraction of its peak contrast (0 = off)
    min_area_um2: float = 0.3     # smallest object kept (ECD ~ 0.6 um)

    def __post_init__(self):
        if self.channels not in DEFAULT_THRESHOLDS:
            raise ValueError(f"channels must be one of {list(DEFAULT_THRESHOLDS)}, not {self.channels!r}")
        hi, lo = DEFAULT_THRESHOLDS[self.channels]
        if self.high is None:
            self.high = hi
        if self.low is None:
            self.low = lo


@dataclass
class Detection:
    labels: np.ndarray            # int32 label image, 0 = matrix
    objects: pd.DataFrame         # one row per object, 'label' matches labels
    contrast: np.ndarray          # contrast image used for detection (float32)
    background: np.ndarray        # background (grey: H x W, per channel: H x W x 3)
    params: DetectionParams
    channel_contrast: np.ndarray | None = None   # H x W x 3 contrast per channel (colour images)


def _background(g: np.ndarray, um_per_px: float, p: DetectionParams) -> np.ndarray:
    win_full = p.background_um / um_per_px
    k = max(1, int(np.ceil(win_full / 25)))                    # median window <= ~25 px on the reduced image
    win = max(3, int(round(win_full / k)) | 1)
    small = g[::k, ::k]
    bg = ndi.median_filter(small, size=win, mode="reflect")
    if k > 1:
        bg = ndi.zoom(bg, (g.shape[0] / small.shape[0], g.shape[1] / small.shape[1]), order=1)
        bg = bg[: g.shape[0], : g.shape[1]]
    return np.maximum(bg, 1.0).astype(np.float32)


def _contrast(g: np.ndarray, bg: np.ndarray, p: DetectionParams) -> np.ndarray:
    gs = ndi.gaussian_filter(g, p.smooth_px) if p.smooth_px > 0 else g
    return ((bg - gs) / bg).astype(np.float32)


def background_and_contrast(rgb: np.ndarray, um_per_px: float, p: DetectionParams):
    """Grey mean, background and contrast of the grey mean (``channels="mean"``)."""
    g = rgb[..., :3].astype(np.float32).mean(axis=2) if rgb.ndim == 3 else rgb.astype(np.float32)
    bg = _background(g, um_per_px, p)
    return g, bg, _contrast(g, bg, p)


def channel_contrasts(rgb: np.ndarray, um_per_px: float, p: DetectionParams):
    """Background and contrast for R, G and B separately (H x W x 3 each)."""
    bgs, cs = [], []
    for k in range(3):
        ch = rgb[..., k].astype(np.float32)
        bg = _background(ch, um_per_px, p)
        bgs.append(bg); cs.append(_contrast(ch, bg, p))
    return np.stack(bgs, axis=-1), np.stack(cs, axis=-1)


def detect(image: np.ndarray, um_per_px: float, params: DetectionParams | None = None,
           stage: StageImage | None = None) -> Detection:
    """Detect dark objects. ``stage`` (optional) adds stage coordinates."""
    p = params or DetectionParams()
    rgb = image
    colour = rgb.ndim == 3 and rgb.shape[2] >= 3
    cc = None
    if colour:
        bgc, cc = channel_contrasts(rgb, um_per_px, p)
    if p.channels == "max" and colour:
        bg, c = bgc, cc.max(axis=2)
    else:
        _, bg, c = background_and_contrast(rgb, um_per_px, p)
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
    region = {}
    if n_det := int(detected.max()):
        for r in regionprops(detected):
            r0, c0, r1, c1 = r.bbox
            region[r.label] = {"region_area_um2": r.area * um_per_px ** 2,
                               "region_ecd_um": 2 * np.sqrt(r.area * um_per_px ** 2 / np.pi),
                               "region_col_px": r.centroid[1], "region_row_px": r.centroid[0],
                               "region_bbox_row0": r0, "region_bbox_col0": c0,
                               "region_bbox_row1": r1, "region_bbox_col1": c1}
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
        reg_id = int(detected[coords[0, 0], coords[0, 1]])
        row["region"] = reg_id
        row.update(region.get(reg_id, {}))
        if cc is not None:
            pc = cc[coords[:, 0], coords[:, 1]]
            row.update({"contrast_R_mean": pc[:, 0].mean(), "contrast_G_mean": pc[:, 1].mean(),
                        "contrast_B_mean": pc[:, 2].mean(), "contrast_R_max": pc[:, 0].max(),
                        "contrast_G_max": pc[:, 1].max(), "contrast_B_max": pc[:, 2].max()})
        rows.append(row)
    df = pd.DataFrame(rows)
    if stage is not None and len(df):
        x, y = stage.pixel_to_stage(df.col_px.to_numpy(), df.row_px.to_numpy())
        df.insert(3, "stage_x_um", x)
        df.insert(4, "stage_y_um", y)
    if len(df):
        df = df.sort_values("area_um2", ascending=False).reset_index(drop=True)
        df.insert(0, "id", np.arange(1, len(df) + 1))   # 1 = largest object in the image
        df["region_n_objects"] = df.groupby("region").id.transform("size")
    return Detection(labels=lab.astype(np.int32), objects=df, contrast=c, background=bg, params=p,
                     channel_contrast=cc)


def params_dict(p: DetectionParams) -> dict:
    return asdict(p)
