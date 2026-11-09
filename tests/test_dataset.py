import numpy as np
import pandas as pd

from cii.dataset import ExportParams, assign_splits, crop_window, find_object, spatial_groups


def test_crop_window_stays_inside():
    assert crop_window(500, 400, 96, 2880, 2160) == (452, 352)
    assert crop_window(10, 5, 96, 2880, 2160) == (0, 0)                  # shifted inwards at the border
    assert crop_window(2878, 2159, 96, 2880, 2160) == (2784, 2064)


def test_split_stratified_and_blocks_kept_whole():
    rng = np.random.default_rng(0)
    n = 1200
    x, y = rng.uniform(0, 9000, n), rng.uniform(0, 4500, n)
    cls = rng.choice(["A", "B", "C"], n, p=[0.6, 0.3, 0.1])
    g = spatial_groups(x, y, 1000)
    split, fold = assign_splits(cls, g, ExportParams())
    assert 0.12 < (split == "test").mean() < 0.28
    assert set(fold[split == "test"]) == {-1} and set(fold[split == "trainval"]) == {0, 1, 2, 3, 4}
    for b in np.unique(g):                                              # a block never spans two parts
        assert len(set(zip(split[g == b], fold[g == b]))) == 1
    for c in "ABC":                                                     # every class in test and in every fold
        assert ((cls == c) & (split == "test")).any()
        assert all(((cls == c) & (fold == k)).any() for k in range(5))


def test_touches_border():
    from cii.dataset import touches_border
    lab = np.zeros((20, 30), int)
    lab[5:8, 5:8] = 1                                   # inside
    lab[17:20, 10:14] = 2                               # cut by the bottom border
    lab[3:6, 27:30] = 3                                 # cut by the right border
    assert not touches_border(lab, 1)
    assert touches_border(lab, 2) and touches_border(lab, 3)


def test_find_object_by_id_or_position():
    objs = pd.DataFrame({"image": "a.tif", "id": [1, 2], "label": [10, 20], "region": [5, 6],
                         "col_px": [100.0, 300.0], "row_px": [50.0, 80.0]})
    idx = objs.set_index(["image", "id"])[["label", "region", "col_px", "row_px"]]
    r = pd.Series({"vhx_id": 1, "vhx_col_px": 100.0, "vhx_row_px": 50.0})
    assert find_object(idx, objs, "a.tif", r) == ((10, 5), False)
    r = pd.Series({"vhx_id": 1, "vhx_col_px": 300.2, "vhx_row_px": 80.1})  # id points elsewhere: found by position
    assert find_object(idx, objs, "a.tif", r) == ((20, 6), True)
    r = pd.Series({"vhx_id": 1, "vhx_col_px": 900.0, "vhx_row_px": 80.0})
    assert find_object(idx, objs, "a.tif", r) is None
