import numpy as np
import pandas as pd

from cii.compare import compare, summary


def tiles(prefix, mag, um_per_px, w, h, nx, ny, step, rng, err_um=1.5):
    rows = []
    for k in range(nx * ny):
        ix, iy = k % nx, k // nx
        rows.append({"image": f"{prefix}_{k:03d}.tif", "path": "", "magnification": mag, "um_per_px": um_per_px,
                     "width_px": w, "height_px": h, "stage_x_um": ix * step[0], "stage_y_um": iy * step[1],
                     "err_x": rng.normal(0, err_um), "err_y": rng.normal(0, err_um)})
    return pd.DataFrame(rows)


def observe(truth, imgs, offset, rng, detect_ecd=0.0, size_factor=1.0):
    """Objects as seen in every image that contains them, shifted by the image's stage error."""
    rows = []
    for im in imgs.itertuples():
        hw, hh = im.width_px * im.um_per_px / 2, im.height_px * im.um_per_px / 2
        x = truth.x - offset[0] - im.err_x
        y = truth.y - offset[1] - im.err_y
        inside = (abs(x - im.stage_x_um) < hw - 1) & (abs(y - im.stage_y_um) < hh - 1) & (truth.ecd >= detect_ecd)
        t = truth[inside]
        for n, (k, r) in enumerate(t.iterrows()):
            rows.append({"image": im.image, "id": n + 1, "stage_x_um": x[k] + rng.normal(0, 0.1),
                         "stage_y_um": y[k] + rng.normal(0, 0.1), "ecd_um": r.ecd * size_factor,
                         "area_um2": np.pi * (r.ecd * size_factor / 2) ** 2, "contrast_max": 0.5,
                         "touches_border": False, "truth": k})
    return pd.DataFrame(rows)


def test_whole_area_comparison():
    rng = np.random.default_rng(1)
    # real tile sizes: 1000x 288 x 216 um (0.1 um/px), 500x 586 x 440 um (0.2035 um/px), ~7 % overlap
    ref_img = tiles("hi", 1000, 0.1, 2880, 2160, 4, 3, (268, 201), rng)
    oth_img = tiles("lo", 500, 0.2035, 2880, 2160, 2, 2, (545, 410), rng)
    oth_img[["stage_x_um", "stage_y_um"]] += [130.0, 50.0]          # covers the reference area with margin
    n = 1300                                                         # ~2000 objects / mm2 as in the MnB images
    truth = pd.DataFrame({"x": rng.uniform(-150, 960, n), "y": rng.uniform(-115, 520, n),
                          "ecd": rng.choice([0.5, 1.2, 2.5], n)})
    ref_obj = observe(truth, ref_img, offset=(0, 0), rng=rng)
    oth_obj = observe(truth, oth_img, offset=(5.5, 0.0), rng=rng, detect_ecd=1.0)   # objective offset 5.5 um
    res = compare(ref_obj, ref_img.drop(columns=["err_x", "err_y"]), oth_obj, oth_img.drop(columns=["err_x", "err_y"]))
    # offset per image pair = objective offset + difference of the two stage errors
    er, eo = ref_img.set_index("image"), oth_img.set_index("image")
    m = res.offsets[res.offsets.n_pairs >= 10]
    true_dx = -5.5 + er.loc[m.ref_image, "err_x"].to_numpy() - eo.loc[m.other_image, "err_x"].to_numpy()
    true_dy = er.loc[m.ref_image, "err_y"].to_numpy() - eo.loc[m.other_image, "err_y"].to_numpy()
    assert len(m) >= 10
    assert np.abs(m.dx_um - true_dx).max() < 0.3 and np.abs(m.dy_um - true_dy).max() < 0.3
    o = res.objects
    counted = o[o.status.isin(["found", "missed"])]
    # every true object counted at most once
    assert not counted.truth.duplicated().any()
    assert (o.status == "duplicate").sum() > 0
    s = summary(o, [0, 1, 2, 5]).set_index("ECD_ref_um")
    assert s.loc["0-1", "found_pct"] < 5
    assert s.loc["1-2", "found_pct"] > 97 and s.loc["2-5", "found_pct"] > 97
    # matched to the right object
    f = o[(o.status == "found") & (o.ecd_um >= 1)]
    truth_other = oth_obj.set_index(["image", "id"]).truth
    assert (truth_other.loc[list(zip(f.other_image, f.other_id))].to_numpy() == f.truth.to_numpy()).mean() > 0.995
