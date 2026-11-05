import numpy as np

from cii.detection import DetectionParams, detect


def synthetic(h=600, w=800, bg=250):
    rng = np.random.default_rng(3)
    g = np.full((h, w), float(bg)) + rng.normal(0, 0.8, (h, w))
    g += np.linspace(-8, 0, w)[None, :]                     # uneven illumination
    yy, xx = np.mgrid[0:h, 0:w]
    objects = [(300.0, 200.0, 12.0, 120.0), (500.0, 400.0, 4.0, 150.0), (100.0, 500.0, 1.0, 200.0)]
    for cx, cy, r, val in objects:
        g[(xx - cx) ** 2 + (yy - cy) ** 2 <= r ** 2] = val
    g[(xx - 795) ** 2 + (yy - 300) ** 2 <= 6 ** 2] = 120     # touches the right border
    rgb = np.repeat(g.clip(0, 255).astype(np.uint8)[..., None], 3, axis=2)
    return rgb, objects


def test_detects_objects_and_drops_tiny_ones():
    rgb, objects = synthetic()
    det = detect(rgb, um_per_px=0.2, params=DetectionParams(min_area_um2=0.3))
    o = det.objects
    # r=12 px and r=4 px objects plus the border object; r=1 px (0.12 um2) is below min area
    assert len(o) == 3
    big = o.iloc[0]
    assert abs(big.col_px - 300) < 0.5 and abs(big.row_px - 200) < 0.5
    assert abs(big.ecd_um - 2 * 12.5 * 0.2) < 0.4          # edge pixels included by hysteresis
    assert o.touches_border.sum() == 1
    assert list(o.id) == [1, 2, 3]


def test_stage_coordinates_added():
    from pathlib import Path
    from cii.vhx import VHXImage
    rgb, _ = synthetic()
    img = VHXImage(path=Path("x.tif"), width=800, height=600, um_per_px=0.2, stage_x=1000.0, stage_y=2000.0)
    det = detect(rgb, 0.2, stage=img)
    big = det.objects.iloc[0]
    assert np.isclose(big.stage_x_um, 1000 + (big.col_px - 399.5) * 0.2)
    assert np.isclose(big.stage_y_um, 2000 + (big.row_px - 299.5) * 0.2)
