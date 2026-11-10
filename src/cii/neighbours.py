"""Neighbour features: what lies around a particle in the VHX images.

Uses all objects of the VHX detection (objects.csv of detect_inclusions.py),
also the small ones below the dataset's ECD limit, never the SEM classes of
neighbouring particles (those are unknown when new images are classified).

Overlapping images show the same object twice. Each object is therefore kept
only in the image whose centre is nearest to it (its most central view); this
splits the scanned area into one cell per image without matching objects.

Per particle (its own object, centroid within its radius + SELF_MARGIN_UM of the
particle position, is left out):

nb_density_<R>     objects per 1000 µm² within R = 10, 25, 50 µm, counted over the
                   part of the circle that was scanned (coverage), so particles at
                   the edge of the scanned area are not undercounted
nb_nn_dist_um      distance to the nearest object (centroid to centroid)
nb_nn_ecd_um, nb_nn_contrast, nb_nn_pinkness
                   ECD, contrast (darkest point) and pinkness (green / red
                   contrast) of that nearest object
nb_pink_share_25   share of the objects within 25 µm with pinkness >= PINK_THRESHOLD
nb_mean_ecd_25     mean ECD of the objects within 25 µm
nb_linearity_25    how well the particle and the objects within 25 µm lie on a line:
                   1 - λ2 / λ1 of their positions (principal axes); 0 = round
                   cloud, 1 = all on one line; needs at least 2 objects
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

RADII_UM = (10, 25, 50)         # circles for the density
LOCAL_UM = 25                   # circle for pink share, mean ECD and linearity
SELF_MARGIN_UM = 0.5            # own object: centroid within its radius + this margin
PINK_THRESHOLD = 1.3            # as in the report (green / red contrast)
COVERAGE_PX_UM = 2.0            # resolution of the coverage map

FEATURES = {                    # column -> label for tables and figures
    "nb_density_10": "objects within 10 µm (per 1000 µm²)",
    "nb_density_25": "objects within 25 µm (per 1000 µm²)",
    "nb_density_50": "objects within 50 µm (per 1000 µm²)",
    "nb_nn_dist_um": "nearest object: distance (µm)",
    "nb_nn_ecd_um": "nearest object: ECD (µm)",
    "nb_nn_contrast": "nearest object: contrast",
    "nb_nn_pinkness": "nearest object: pinkness",
    "nb_pink_share_25": "pink objects within 25 µm (share)",
    "nb_mean_ecd_25": "mean ECD of objects within 25 µm (µm)",
    "nb_linearity_25": "linearity of objects within 25 µm",
}


def image_rects(images: pd.DataFrame) -> pd.DataFrame:
    """Stage rectangle and centre of every VHX image (images.csv)."""
    from .compare import image_rects as rects
    return rects(images)


def unique_objects(objects: pd.DataFrame, images: pd.DataFrame) -> pd.DataFrame:
    """Objects in their most central view: kept only if their own image is the one whose
    centre is nearest to them."""
    from .compare import nearest_image

    rects = image_rects(images)
    xy = objects[["stage_x_um", "stage_y_um"]].to_numpy()
    keep = nearest_image(xy, rects) == objects.image.to_numpy()

    return objects[keep].reset_index(drop=True)


class Coverage:
    """Map of the scanned area (union of the VHX image rectangles) at COVERAGE_PX_UM, to
    measure which part of a circle was scanned."""

    def __init__(self, images: pd.DataFrame, pad_um: float = max(RADII_UM)):
        r = image_rects(images)
        self.x0 = r.x0.min() - pad_um
        self.y0 = r.y0.min() - pad_um
        w = int(np.ceil((r.x1.max() + pad_um - self.x0) / COVERAGE_PX_UM)) + 1
        h = int(np.ceil((r.y1.max() + pad_um - self.y0) / COVERAGE_PX_UM)) + 1
        self.map = np.zeros((h, w), bool)

        for rect in r.itertuples():
            c0, c1 = self._col(rect.x0), self._col(rect.x1)
            r0, r1 = self._row(rect.y0), self._row(rect.y1)
            self.map[r0:r1 + 1, c0:c1 + 1] = True

    def _col(self, x: float) -> int:
        """Map column of a stage x position."""
        return int(round((x - self.x0) / COVERAGE_PX_UM))

    def _row(self, y: float) -> int:
        """Map row of a stage y position."""
        return int(round((y - self.y0) / COVERAGE_PX_UM))

    def area_um2(self, x: float, y: float, radius_um: float) -> float:
        """Scanned area (µm²) inside the circle of radius_um around (x, y)."""
        n = int(np.ceil(radius_um / COVERAGE_PX_UM))
        dy, dx = np.mgrid[-n:n + 1, -n:n + 1]
        inside = dx ** 2 + dy ** 2 <= (radius_um / COVERAGE_PX_UM) ** 2

        rows = np.clip(self._row(y) + dy[inside], 0, self.map.shape[0] - 1)
        cols = np.clip(self._col(x) + dx[inside], 0, self.map.shape[1] - 1)
        share = self.map[rows, cols].mean()

        return float(share * np.pi * radius_um ** 2)


def linearity(xy: np.ndarray) -> float:
    """1 - λ2 / λ1 of the positions' covariance (0 = round cloud, 1 = on one line); NaN for
    fewer than 3 points."""
    if len(xy) < 3:
        return np.nan

    ev = np.sort(np.linalg.eigvalsh(np.cov(xy.T)))[::-1]
    return float(1 - ev[1] / ev[0]) if ev[0] > 0 else np.nan


def neighbour_features(particles: pd.DataFrame, objects: pd.DataFrame, images: pd.DataFrame) -> pd.DataFrame:
    """Neighbour features (see module docstring) for every particle of a dataset (columns
    vhx_x_um, vhx_y_um, vhx_ecd_um), from the VHX detection's objects.csv and images.csv.
    Rows as particles, columns FEATURES."""
    obj = unique_objects(objects, images)
    xy = obj[["stage_x_um", "stage_y_um"]].to_numpy()
    tree = cKDTree(xy)
    cover = Coverage(images)

    pink = obj.contrast_G_max.to_numpy() / obj.contrast_R_max.clip(lower=0.01).to_numpy()
    ecd = obj.ecd_um.to_numpy()
    contrast = obj.contrast_max.to_numpy()

    rows = []
    for p in particles.itertuples():
        here = np.array([p.vhx_x_um, p.vhx_y_um])
        own = (p.vhx_ecd_um / 2 if np.isfinite(p.vhx_ecd_um) else 0) + SELF_MARGIN_UM
        row = {}

        # density in circles of growing radius, over the scanned part of each circle
        for r in RADII_UM:
            idx = [k for k in tree.query_ball_point(here, r) if np.hypot(*(xy[k] - here)) > own]
            area = cover.area_um2(here[0], here[1], r)
            row[f"nb_density_{r}"] = 1000 * len(idx) / area if area > 0 else np.nan

        # nearest other object
        dist, idx = tree.query(here, k=min(10, len(xy)))
        others = [(d, k) for d, k in zip(np.atleast_1d(dist), np.atleast_1d(idx)) if d > own]

        if others:
            d, k = others[0]
            row.update({"nb_nn_dist_um": float(d), "nb_nn_ecd_um": float(ecd[k]),
                        "nb_nn_contrast": float(contrast[k]), "nb_nn_pinkness": float(pink[k])})

        # objects within LOCAL_UM: pink share, mean size, line
        local = [k for k in tree.query_ball_point(here, LOCAL_UM) if np.hypot(*(xy[k] - here)) > own]

        if local:
            row["nb_pink_share_25"] = float((pink[local] >= PINK_THRESHOLD).mean())
            row["nb_mean_ecd_25"] = float(ecd[local].mean())
            row["nb_linearity_25"] = linearity(np.vstack([here, xy[local]]))
        else:
            row["nb_pink_share_25"] = 0.0

        rows.append(row)

    return pd.DataFrame(rows, index=particles.index).reindex(columns=list(FEATURES))


def detection_features(particles: pd.DataFrame, detection: Path) -> pd.DataFrame | None:
    """Neighbour features from a detection folder (objects.csv, images.csv); None if the folder
    or its files are missing."""
    detection = Path(detection)
    if not (detection / "objects.csv").exists() or not (detection / "images.csv").exists():
        return None

    objects = pd.read_csv(detection / "objects.csv")
    images = pd.read_csv(detection / "images.csv")
    return neighbour_features(particles, objects, images)
