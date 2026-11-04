"""Locate hardness indents (fiducials) in light-microscope images.

Workflow: mark each indent once with a red circle in a copy of the image (any
image editor), then ``locate_indent`` finds the dark indent cavity inside that
circle and returns its centroid with sub-pixel precision. Combined with
``cii.vhx.VHXImage.pixel_to_stage`` this gives the indent position in stage um.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from skimage.filters import threshold_otsu


@dataclass
class Marking:
    col: float
    row: float
    radius: float  # px


@dataclass
class IndentFit:
    col: float          # centroid, px (0-based)
    row: float
    diameter_px: float  # equivalent circle diameter of the cavity
    area_px: int
    contrast: float     # background minus cavity grey level (0-255 scale)


def find_marking(marked_path, min_pixels: int = 200) -> Marking:
    """Centre and radius of a hand-drawn red circle in an image file."""
    rgb = np.asarray(Image.open(marked_path).convert("RGB")).astype(int)
    red = (rgb[..., 0] > 180) & (rgb[..., 1] < 90) & (rgb[..., 2] < 90)
    if red.sum() < min_pixels:
        raise ValueError(f"{Path(marked_path).name}: no red marking found")
    lab, n = ndi.label(red)
    sizes = ndi.sum(red, lab, range(1, n + 1))
    keep = lab == (int(np.argmax(sizes)) + 1)   # largest red stroke
    rows, cols = np.nonzero(keep)
    c = (cols.min() + cols.max()) / 2
    r = (rows.min() + rows.max()) / 2
    radius = max(cols.max() - cols.min(), rows.max() - rows.min()) / 2
    return Marking(col=c, row=r, radius=radius)


def locate_indent(image: np.ndarray, col: float, row: float, radius: float,
                  min_area_px: int = 30) -> IndentFit:
    """Centroid of the dark indent cavity inside a circular search region.

    image: H x W (x 3) array; col, row, radius: search circle in px.
    Bright reflections inside the cavity are included (holes are filled).
    """
    g = image.astype(float).mean(axis=2) if image.ndim == 3 else image.astype(float)
    r = int(np.ceil(radius))
    r0, r1 = max(0, int(row) - r), min(g.shape[0], int(row) + r + 1)
    c0, c1 = max(0, int(col) - r), min(g.shape[1], int(col) + r + 1)
    roi = g[r0:r1, c0:c1]
    yy, xx = np.mgrid[r0:r1, c0:c1]
    inside = (xx - col) ** 2 + (yy - row) ** 2 <= radius ** 2
    background = float(np.median(roi[inside]))
    thr = threshold_otsu(roi[inside])
    dark = (roi < thr) & inside
    lab, n = ndi.label(dark)
    if n == 0:
        raise ValueError("no dark region inside the search circle")
    sizes = ndi.sum(dark, lab, range(1, n + 1))
    coms = ndi.center_of_mass(dark, lab, range(1, n + 1))
    # largest component; ties broken by distance to the circle centre
    order = sorted(range(n), key=lambda i: (-sizes[i],
                   np.hypot(coms[i][1] + c0 - col, coms[i][0] + r0 - row)))
    best = order[0]
    if sizes[best] < min_area_px:
        raise ValueError(f"dark region too small ({int(sizes[best])} px)")
    cavity = ndi.binary_fill_holes(lab == best + 1)
    cy, cx = ndi.center_of_mass(cavity)
    area = int(cavity.sum())
    return IndentFit(col=cx + c0, row=cy + r0,
                     diameter_px=float(2 * np.sqrt(area / np.pi)),
                     area_px=area,
                     contrast=background - float(roi[cavity].mean()))
