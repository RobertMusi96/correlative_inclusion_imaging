import numpy as np
import pandas as pd
from scipy import ndimage as ndi

from cii.lighting import dataset_light_features, light_feature_definitions, light_features, measure_shift, pair_images


def scene(size=(400, 500), seed=0):
    """Synthetic bright-field image: dark blobs and a scratch on a bright background."""
    rng = np.random.default_rng(seed)
    grey = np.full(size, 230.0)
    yy, xx = np.mgrid[:size[0], :size[1]]
    for _ in range(40):
        r, c, rad = rng.uniform(20, size[0] - 20), rng.uniform(20, size[1] - 20), rng.uniform(2, 6)
        grey[(yy - r) ** 2 + (xx - c) ** 2 <= rad ** 2] = 90
    grey[np.abs(yy - 0.5 * xx - 30) < 1.2] = 180
    grey = ndi.gaussian_filter(grey, 1.0) + rng.normal(0, 2, size)
    return np.repeat(grey[..., None], 3, axis=2).clip(0, 255).astype(np.uint8)


def test_measure_shift_of_another_lighting_mode():
    bright = scene()
    other = 255 - np.roll(bright, (-21, 72), axis=(0, 1))          # inverted (like dark field) and shifted
    s = measure_shift(bright, other)
    assert abs(s["shift_row"] - 21) < 0.6 and abs(s["shift_col"] + 72) < 0.6
    assert s["quadrants_ok"] and s["n_agree"] == 5

    # the documented convention: bright (r, c) shows the same place as other (r - shift_row, c - shift_col)
    r, c = 200, 250
    assert np.array_equal(bright[r, c], 255 - other[r - 21, c + 72])


def test_pair_images_by_stage_position():
    bright = pd.DataFrame({"image": ["a", "b"], "stage_x_um": [0.0, 500.0], "stage_y_um": [0.0, 0.0],
                           "magnification": 500.0, "width_px": 2880, "height_px": 2160})
    other = pd.DataFrame({"path": ["x.tif", "y.tif", "z.tif"], "stage_x_um": [1.0, 500.0, 0.0],
                          "stage_y_um": [0.0, 0.0, 0.0], "magnification": [500.0, 500.0, 1000.0],
                          "width_px": 2880, "height_px": 2160})
    pairs = pair_images(bright, other).set_index("image")
    assert pairs.loc["a", "path"] == "x.tif"                        # z.tif has another magnification
    assert pairs.loc["b", "path"] == "y.tif"


def test_light_features_dark_field_glow():
    yy, xx = np.mgrid[:96, :96] - 48
    inside = xx ** 2 + yy ** 2 <= 25
    bright = np.full((96, 96, 3), 230, np.uint8)
    bright[inside] = 100
    dark = np.full((96, 96, 3), 8, np.uint8)
    dark[inside] = (200, 150, 100)                                  # particle lights up, reddish
    f = light_features(bright, inside, dark, 0.2)
    assert f["in_diff"] > 100 and f["in_p95_diff"] >= f["in_diff"]
    assert f["red_share"] > f["blue_share"]


def test_light_feature_columns_and_definitions(tmp_path):
    from PIL import Image
    yy, xx = np.mgrid[:96, :96] - 48
    inside = xx ** 2 + yy ** 2 <= 25
    bright = np.full((96, 96, 3), 230, np.uint8)
    bright[inside] = 100
    for d in ("crops", "masks", "crops_df"):
        (tmp_path / d).mkdir()
    Image.fromarray(bright).save(tmp_path / "crops" / "0.png")
    Image.fromarray((inside * 255).astype(np.uint8)).save(tmp_path / "masks" / "0.png")
    Image.fromarray(255 - bright).save(tmp_path / "crops_df" / "0.png")

    ds = pd.DataFrame({"crop": ["crops/0.png", "crops/0.png"], "mask": ["masks/0.png", "masks/0.png"],
                       "crop_df": ["crops_df/0.png", ""], "um_per_px": 0.2})
    f = dataset_light_features(ds, tmp_path, "df")
    assert f.columns[0] == "lt_df_in_diff" and f.iloc[0].notna().all() and f.iloc[1].isna().all()
    assert light_feature_definitions(f.columns)["lt_df_in_diff"][0] == "dark field"


def test_neighbour_check_in_recording_order():
    from cii.lighting import neighbour_check
    names = [f"img_{k:03d}.tif" for k in range(12)]
    rows = -10 + 0.1 * np.arange(12)                                     # slow drift
    t = pd.DataFrame({"image": names, "shift_row": rows, "shift_col": 80.0})
    t.loc[6, "shift_col"] = 84.0                                         # one outlier
    dev = neighbour_check(t.sample(frac=1, random_state=0)).sort_index()  # order of the rows does not matter
    assert dev[6] > 3.5
    assert (dev.drop(6) < 0.5).all()
