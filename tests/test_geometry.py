from pathlib import Path

import numpy as np

from cii.sem import SEMImage, field_id_from_name, read_field_positions
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


def test_sem_x_axis_points_left():
    img = SEMImage(path=Path("s.tif"), width=1024, height=768, um_per_px=0.3,
                   stage_x=3530.0, stage_y=507.0, x_dir=-1, y_dir=1)
    cx, cy = img.centre_px
    x, y = img.pixel_to_stage(cx + 100, cy + 50)      # right and down in the image
    assert np.isclose(x, 3530.0 - 30.0) and np.isclose(y, 507.0 + 15.0)
    c, r = img.stage_to_pixel(x, y)
    assert np.isclose(c, cx + 100) and np.isclose(r, cy + 50)


def test_field_table_and_site_number(tmp_path):
    p = tmp_path / "fields.csv"
    p.write_text("Area Name,FieldID,StageXmm,StageYmm\n30102026,25,3.53025,0.50675\n")
    f = read_field_positions(p)
    assert np.isclose(f.at[25, "stage_x_um"], 3530.25) and np.isclose(f.at[25, "stage_y_um"], 506.75)
    assert field_id_from_name("MnB__Feature_analysis__30102026__Site_25__Indent_topleft") == 25
