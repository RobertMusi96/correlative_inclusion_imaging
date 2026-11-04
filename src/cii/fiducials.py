"""Locate hardness indents (fiducials) in light-microscope and SEM images.

Workflow: mark each indent once with a red circle in a copy of the image (any
image editor), then ``locate_indent`` segments the indent inside that circle and
returns the centroid of the segmented region with sub-pixel precision. Combined
with ``pixel_to_stage`` of the image this gives the indent position in stage um.

Two segmentation methods:

* ``"dark"`` (light microscope, bright field): the indent cavity is dark; the
  largest dark region (Otsu threshold) is taken, bright reflections inside it
  are filled in.
* ``"edges"`` (SEM, BSE): the indent shows as strong bright/dark faces and
  edges (topography) with a grey interior; pixels far outside the matrix grey
  levels (|z| > outlier_z, robust z-score after slight smoothing) are grouped
  (pieces closer than join_gap_px) and the convex hull of the group is the indent.

Both give the outline of the whole indent, so the centroids refer to the same
physical point in both instruments.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from skimage.filters import threshold_otsu
from skimage.morphology import convex_hull_image, disk


@dataclass
class Marking:
    col: float
    row: float
    radius: float  # px


@dataclass
class IndentFit:
    col: float          # centroid, px (0-based, full image)
    row: float
    diameter_px: float  # equivalent circle diameter of the segmented region
    area_px: int
    contrast: float     # dark: background minus cavity grey level; edges: mean edge strength ratio
    method: str = "dark"
    mask: np.ndarray = field(default=None, repr=False)   # segmented region within the bbox
    bbox: tuple[int, int, int, int] = (0, 0, 0, 0)        # row0, row1, col0, col1 of mask


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


def _grey(image: np.ndarray) -> np.ndarray:
    return image.astype(float).mean(axis=2) if image.ndim == 3 else image.astype(float)


def _roi(g, col, row, radius):
    r = int(np.ceil(radius))
    r0, r1 = max(0, int(row) - r), min(g.shape[0], int(row) + r + 1)
    c0, c1 = max(0, int(col) - r), min(g.shape[1], int(col) + r + 1)
    yy, xx = np.mgrid[r0:r1, c0:c1]
    inside = (xx - col) ** 2 + (yy - row) ** 2 <= radius ** 2
    return g[r0:r1, c0:c1], inside, (r0, r1, c0, c1)


def _pick_component(binary, prefer_rc, min_area):
    lab, n = ndi.label(binary)
    if n == 0:
        raise ValueError("nothing segmented inside the search circle")
    sizes = ndi.sum(binary, lab, range(1, n + 1))
    coms = ndi.center_of_mass(binary, lab, range(1, n + 1))
    order = sorted(range(n), key=lambda i: (-sizes[i], np.hypot(coms[i][0] - prefer_rc[0], coms[i][1] - prefer_rc[1])))
    best = order[0]
    if sizes[best] < min_area:
        raise ValueError(f"segmented region too small ({int(sizes[best])} px)")
    return lab == best + 1


def locate_indent(image: np.ndarray, col: float, row: float, radius: float,
                  method: str = "dark", min_area_px: int = 30,
                  edge_sigma_px: float = 2.5, outlier_z: float = 3.5,
                  min_piece_px: int = 15, join_gap_px: int = 8) -> IndentFit:
    """Segment the indent inside a circular search region and return its centroid.

    image: H x W (x 3) array; col, row, radius: search circle in px.
    method: "dark" (light microscope) or "edges" (SEM BSE), see module docstring.
    """
    g = _grey(image)
    roi, inside, (r0, r1, c0, c1) = _roi(g, col, row, radius)
    centre_rc = (row - r0, col - c0)

    if method == "dark":
        background = float(np.median(roi[inside]))
        dark = (roi < threshold_otsu(roi[inside])) & inside
        region = ndi.binary_fill_holes(_pick_component(dark, centre_rc, min_area_px))
        contrast = background - float(roi[region].mean())
    elif method == "edges":
        # BSE: indent faces/edges are far brighter or darker than the matrix
        # (topographic contrast), the matrix itself only shows weak grain contrast.
        sm = ndi.gaussian_filter(roi, edge_sigma_px)
        med = float(np.median(sm[inside]))
        mad = float(np.median(np.abs(sm[inside] - med))) * 1.4826 + 1e-9
        z = (sm - med) / mad
        strong = (np.abs(z) > outlier_z) & inside
        strong = ndi.binary_opening(strong, structure=disk(1))
        lab, n = ndi.label(strong)
        sizes = ndi.sum(strong, lab, range(1, n + 1))
        strong = np.isin(lab, 1 + np.flatnonzero(sizes >= min_piece_px))
        lab, _ = ndi.label(strong)
        # the faces/edges of one indent break into several pieces: grow a cluster
        # from the largest piece, adding pieces closer than join_gap_px
        cluster = _pick_component(strong, centre_rc, min_area_px)
        while True:
            touching = np.unique(lab[ndi.binary_dilation(cluster, structure=disk(join_gap_px)) & strong])
            grown = np.isin(lab, touching[touching > 0])
            if grown.sum() == cluster.sum():
                break
            cluster = grown
        region = convex_hull_image(cluster)
        contrast = float(np.abs(z[cluster]).mean())
    else:
        raise ValueError(f"unknown method {method!r}")

    cy, cx = ndi.center_of_mass(region)
    area = int(region.sum())
    return IndentFit(col=cx + c0, row=cy + r0, diameter_px=float(2 * np.sqrt(area / np.pi)),
                     area_px=area, contrast=contrast, method=method,
                     mask=region, bbox=(r0, r1, c0, c1))
