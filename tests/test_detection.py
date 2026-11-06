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
    det = detect(rgb, um_per_px=0.2, params=DetectionParams(channels="mean", min_area_um2=0.3))
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


def test_channel_max_finds_pink_particle_mean_misses_it():
    rng = np.random.default_rng(5)
    h, w = 400, 500
    rgb = np.stack([np.full((h, w), v) for v in (246.0, 249.0, 252.0)], axis=-1)
    rgb += rng.normal(0, 0.8, (h, w, 3))
    yy, xx = np.mgrid[0:h, 0:w]
    pink = (xx - 150) ** 2 + (yy - 200) ** 2 <= 8 ** 2          # darker mainly in green
    rgb[pink] *= [0.96, 0.88, 0.98]                             # contrast R 0.04, G 0.12, B 0.02 -> mean 0.06
    dark = (xx - 350) ** 2 + (yy - 200) ** 2 <= 8 ** 2          # grey oxide, darker in all channels
    rgb[dark] *= 0.55
    rgb = rgb.clip(0, 255).astype(np.uint8)
    mean = detect(rgb, 0.2, DetectionParams(channels="mean")).objects
    mx = detect(rgb, 0.2, DetectionParams(channels="max")).objects
    assert len(mean) == 1 and abs(mean.iloc[0].col_px - 350) < 1
    assert len(mx) == 2
    p = mx.loc[(mx.col_px - 150).abs() < 1].iloc[0]
    assert p.contrast_G_max > 2 * p.contrast_R_max and p.contrast_G_max > 0.1
    # the dark particle keeps (nearly) its size in both modes
    d_mean, d_max = mean.iloc[0], mx.loc[(mx.col_px - 350).abs() < 1].iloc[0]
    assert abs(d_mean.ecd_um - d_max.ecd_um) < 0.15


def test_default_thresholds_per_mode():
    assert (DetectionParams().high, DetectionParams().low) == (0.07, 0.04)
    assert (DetectionParams(channels="mean").high, DetectionParams(channels="mean").low) == (0.12, 0.03)
    assert DetectionParams(channels="mean", high=0.1).high == 0.1


def test_label_images_roundtrip(tmp_path):
    from cii.detection import read_label_images, save_label_images
    rgb, _ = synthetic()
    det = detect(rgb, 0.2)
    save_label_images(det, tmp_path, "img")
    obj, reg = read_label_images(tmp_path, "img")
    assert np.array_equal(obj, det.labels) and np.array_equal(reg, det.regions)
    o = det.objects.iloc[0]
    mask = obj == o.label
    assert mask.sum() == o.area_px
    assert (reg[mask] == o.region).all()          # the object lies inside its region
