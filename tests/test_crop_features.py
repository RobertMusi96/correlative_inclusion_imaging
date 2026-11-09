import numpy as np
from scipy import ndimage as ndi

from cii.crop_features import FEATURES, crop_features

UM_PER_PX = 0.2


def ellipse_crop(a_px=8.0, b_px=4.0, darkness=0.5, bg=230.0, size=96):
    """Synthetic crop: dark ellipse (half axes a along x, b along y) on a flat background,
    blurred a little like the microscope; mask = the ellipse."""
    yy, xx = np.mgrid[:size, :size] - size / 2
    inside = (xx / a_px) ** 2 + (yy / b_px) ** 2 <= 1
    grey = np.where(inside, bg * (1 - darkness), bg)
    grey = ndi.gaussian_filter(grey, 1.0)
    rgb = np.repeat(grey[..., None], 3, axis=2).clip(0, 255).astype(np.uint8)
    return rgb, inside


def test_shape_of_an_ellipse():
    rgb, mask = ellipse_crop()
    f = crop_features(rgb, mask, UM_PER_PX)
    assert 1.6 < f["shape_aspect_ratio"] < 2.4
    assert f["shape_elong_x"] > 0.3 and abs(f["shape_elong_y"]) < 0.05        # elongated along x
    true_ecd = 2 * np.sqrt(8 * 4) * UM_PER_PX
    assert abs(f["object_ecd_um"] - true_ecd) / true_ecd < 0.15
    assert f["int_n_cores"] == 1
    assert set(FEATURES) <= set(f)


def test_features_do_not_depend_on_the_saved_mask():
    rgb, mask = ellipse_crop()
    big = ndi.binary_dilation(mask, iterations=4)                              # like a region mask
    a, b = crop_features(rgb, mask, UM_PER_PX), crop_features(rgb, big, UM_PER_PX)
    for k in FEATURES:
        assert np.isclose(a[k], b[k], equal_nan=True), k


def test_light_and_shadow_asymmetry():
    rgb, mask = ellipse_crop(a_px=6, b_px=6)
    rgb = rgb.astype(float)
    rgb[:, 54:57] *= 0.85                                                      # shadow just right of the edge
    f = crop_features(rgb.astype(np.uint8), mask, UM_PER_PX)
    assert f["edge_asym_x"] > 0.02                                             # darker on the right (+x)
    assert abs(f["edge_asym_y"]) < 0.5 * f["edge_asym_x"]


def test_empty_mask_gives_nan():
    rgb, mask = ellipse_crop()
    f = crop_features(rgb, np.zeros_like(mask), UM_PER_PX)
    assert all(np.isnan(f[k]) for k in FEATURES)
