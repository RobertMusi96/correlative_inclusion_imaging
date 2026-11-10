import numpy as np
import pandas as pd

from cii.neighbours import Coverage, linearity, neighbour_features, unique_objects


def two_images():
    """Two 100 x 100 µm images overlapping by 20 µm along x (centres at 50 and 130 µm)."""
    return pd.DataFrame({"image": ["a", "b"], "um_per_px": 1.0, "width_px": 100, "height_px": 100,
                         "stage_x_um": [50.0, 130.0], "stage_y_um": [50.0, 50.0]})


def objects(xy, image, ecd=1.0, pink=False):
    """Objects at the given stage positions in one image."""
    xy = np.atleast_2d(xy)
    return pd.DataFrame({"image": image, "stage_x_um": xy[:, 0], "stage_y_um": xy[:, 1], "ecd_um": ecd,
                         "contrast_max": 0.3, "contrast_R_max": 0.2, "contrast_G_max": 0.4 if pink else 0.2})


def test_overlap_keeps_the_most_central_view():
    img = two_images()
    obj = pd.concat([objects([[85, 50]], "a"), objects([[85, 50]], "b"),       # x = 85: nearer to centre a
                     objects([[95, 50]], "a"), objects([[95, 50]], "b")])      # x = 95: nearer to centre b
    kept = unique_objects(obj, img)
    assert len(kept) == 2
    assert set(zip(kept.stage_x_um, kept.image)) == {(85, "a"), (95, "b")}


def test_coverage_at_the_edge_of_the_scanned_area():
    cov = Coverage(two_images())
    full = np.pi * 10 ** 2
    assert abs(cov.area_um2(90, 50, 10) / full - 1) < 0.05                    # inside
    assert abs(cov.area_um2(90, 0, 10) / full - 0.5) < 0.08                   # on the lower edge: half


def test_linearity():
    assert linearity(np.c_[np.arange(5.0), 2 * np.arange(5.0)]) > 0.99        # on a line
    rng = np.random.default_rng(0)
    assert linearity(rng.normal(size=(200, 2))) < 0.2                         # round cloud
    assert np.isnan(linearity(np.zeros((2, 2))))


def test_neighbour_features():
    img = two_images()
    obj = pd.concat([objects([[40, 50]], "a", ecd=2.0),                       # the particle itself
                     objects([[44, 50], [46, 50], [48, 50]], "a", pink=True),  # a pink chain to the right
                     objects([[70, 80]], "a")])
    part = pd.DataFrame({"vhx_x_um": [40.0], "vhx_y_um": [50.0], "vhx_ecd_um": [2.0]})
    f = neighbour_features(part, obj, img).iloc[0]

    assert np.isclose(f.nb_nn_dist_um, 4.0)                                   # own object left out
    assert np.isclose(f.nb_density_10, 1000 * 3 / (np.pi * 100), rtol=0.05)
    assert f.nb_pink_share_25 == 1.0
    assert f.nb_linearity_25 > 0.99
