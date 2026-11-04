from pathlib import Path

import numpy as np

from cii.vhx import VHXImage


def make(stage_x=100.0, stage_y=-50.0, um_per_px=0.5, width=2880, height=2160):
    return VHXImage(path=Path("dummy.tif"), width=width, height=height, magnification=200.0,
                    um_per_px=um_per_px, stage_x=stage_x, stage_y=stage_y)


def test_centre_maps_to_stage_position():
    img = make()
    x, y = img.pixel_to_stage(*img.centre_px)
    assert np.isclose(x, img.stage_x) and np.isclose(y, img.stage_y)


def test_axes_directions():
    # columns increase with stage X, rows increase with stage Y
    img = make()
    cx, cy = img.centre_px
    x, y = img.pixel_to_stage(cx + 10, cy + 20)
    assert np.isclose(x - img.stage_x, 10 * img.um_per_px)
    assert np.isclose(y - img.stage_y, 20 * img.um_per_px)


def test_round_trip():
    img = make()
    cols, rows = np.array([0.0, 123.4, 2879.0]), np.array([0.0, 999.9, 2159.0])
    c2, r2 = img.stage_to_pixel(*img.pixel_to_stage(cols, rows))
    assert np.allclose(c2, cols) and np.allclose(r2, rows)


def test_contains_stage():
    img = make()
    assert img.contains_stage(img.stage_x, img.stage_y)
    assert not img.contains_stage(img.stage_x + img.size_um[0], img.stage_y)
