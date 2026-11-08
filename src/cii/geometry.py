"""Common geometry of microscope images placed in stage coordinates.

Every image knows its pixel size, the stage position of its centre and the
directions of the stage axes relative to the image axes:

* ``x_dir = +1``: stage X increases to the right in the image (Keyence VHX);
  ``x_dir = -1``: stage X increases to the left (Oxford AZtec on the SEM used here).
* ``y_dir = +1``: stage Y increases downwards in the image (both instruments).

The stage position refers to the image centre, pixel ((width-1)/2, (height-1)/2)
in 0-based pixel indices. All stage coordinates are in um.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class StageImage:
    path: Path
    width: int
    height: int
    um_per_px: float
    stage_x: float
    stage_y: float
    x_dir: int = 1
    y_dir: int = 1

    @property
    def centre_px(self) -> tuple[float, float]:
        """Pixel coordinates (col, row) of the image centre, the point the stage position refers to."""
        return (self.width - 1) / 2, (self.height - 1) / 2

    @property
    def size_um(self) -> tuple[float, float]:
        """Image width and height in um."""
        return self.width * self.um_per_px, self.height * self.um_per_px

    def pixel_to_stage(self, col, row):
        """Pixel coordinates (0-based, sub-pixel allowed) -> stage um."""
        cx, cy = self.centre_px
        x = self.stage_x + self.x_dir * (np.asarray(col, float) - cx) * self.um_per_px
        y = self.stage_y + self.y_dir * (np.asarray(row, float) - cy) * self.um_per_px
        return x, y

    def stage_to_pixel(self, x, y):
        """Stage um -> pixel coordinates (0-based, sub-pixel)."""
        cx, cy = self.centre_px
        col = cx + self.x_dir * (np.asarray(x, float) - self.stage_x) / self.um_per_px
        row = cy + self.y_dir * (np.asarray(y, float) - self.stage_y) / self.um_per_px
        return col, row

    def contains_stage(self, x, y, margin_px: float = 0) -> np.ndarray:
        """True for stage positions (um) that lie in the image, at least margin_px from its border."""
        col, row = self.stage_to_pixel(x, y)
        return ((col >= margin_px) & (col <= self.width - 1 - margin_px)
                & (row >= margin_px) & (row <= self.height - 1 - margin_px))

    def read_image(self) -> np.ndarray:
        """Pixel data of the image; implemented by the instrument classes (VHXImage, SEMImage)."""
        raise NotImplementedError
