import numpy as np
import pandas as pd

from cii.report import composition_tables, section_composition


def toy_dataset(n=300, seed=0):
    rng = np.random.default_rng(seed)
    split = np.where(rng.random(n) < 0.2, "test", "trainval")
    fold = np.where(split == "test", -1, rng.integers(0, 5, n))
    return pd.DataFrame({"training_class": rng.choice(["A", "B", "C"], n, p=[0.6, 0.3, 0.1]),
                         "class_label": rng.choice(["N", "O", "rejected"], n),
                         "class_source": rng.choice(["type table", "rule R1"], n, p=[0.9, 0.1]),
                         "split": split, "fold": fold})


def test_composition_tables_consistent():
    ds = toy_dataset()
    t = composition_tables(ds)
    c = t["counts"]
    assert c.total.sum() == len(ds)
    assert (c.trainval + c.test == c.total).all()
    assert np.allclose(t["shares_pct"].sum(axis=0), 100, atol=0.3)
    w = t["weights"]                                    # balanced weights: weighted counts equal per class
    assert np.allclose(w.n_trainval * w.weight, (w.n_trainval * w.weight).iloc[0], rtol=0.01)


def test_section_writes_files(tmp_path):
    section_composition(toy_dataset(), tmp_path, log=lambda *a: None)
    assert {f.name for f in tmp_path.iterdir()} == {"1_composition.xlsx", "1_composition.png", "1_origin.png"}


def toy_sized(n=400, seed=1):
    """Toy dataset with sizes: class C is larger, so the size baseline beats chance."""
    rng = np.random.default_rng(seed)
    ds = toy_dataset(n, seed)
    ds["sem_ecd_um"] = np.where(ds.training_class == "C", 5.0, 2.0) * rng.lognormal(0, 0.1, n)
    ds["vhx_ecd_um"] = ds.sem_ecd_um * rng.normal(1.0, 0.05, n)
    ds["sem_length_um"] = ds.sem_ecd_um * 1.3
    ds["sem_aspect_ratio"] = 1.3
    ds["crop"] = "crops/x.png"
    ds["crop_size_px"], ds["um_per_px"] = 96, 0.2037
    return ds


def test_size_baseline_and_section(tmp_path):
    from cii.report import section_size, size_baseline
    ds = toy_sized()
    b = size_baseline(ds)
    assert b["balanced_accuracy"] > b["chance"] + 0.1
    assert b["confusion_pct"].loc["C", "C"] > 80
    section_size(ds, tmp_path, log=lambda *a: None)
    assert {f.name for f in tmp_path.iterdir()} == {"2_size.xlsx", "2_size.png"}


def test_appearance_section_with_crops(tmp_path):
    from PIL import Image
    from cii.report import colour_baseline, section_appearance
    rng = np.random.default_rng(2)
    ds = toy_sized()
    pink = ds.training_class == "B"                     # class B: darker in green -> pink
    ds["vhx_contrast_R_max"] = rng.normal(0.4, 0.03, len(ds))
    ds["vhx_contrast_G_max"] = np.where(pink, 0.7, 0.4) + rng.normal(0, 0.03, len(ds))
    ds["vhx_contrast_B_max"] = ds.vhx_contrast_R_max
    ds["vhx_contrast_max"] = ds[["vhx_contrast_R_max", "vhx_contrast_G_max"]].max(axis=1)
    for c in "RGB":
        ds[f"vhx_mean_{c}"] = 150.0
    b = colour_baseline(ds)
    assert b["confusion_pct"].loc["B", "B"] > 90

    # crops: dark particle on a bright background, round for A and C, elongated for B
    (tmp_path / "crops").mkdir()
    (tmp_path / "masks").mkdir()
    ds["crop"] = [f"crops/{i}.png" for i in range(len(ds))]
    ds["mask"] = [f"masks/{i}.png" for i in range(len(ds))]
    yy, xx = np.mgrid[:96, :96] - 48

    for i, cls in enumerate(ds.training_class):
        a = 9 if cls == "B" else 5
        b = 3 if cls == "B" else 5
        inside = (xx / (a * rng.uniform(0.9, 1.1))) ** 2 + (yy / b) ** 2 <= 1
        img = np.full((96, 96, 3), 250, np.uint8)
        img[:10] = 255
        img[inside] = 120
        Image.fromarray(img).save(tmp_path / ds.crop[i])
        Image.fromarray((inside * 255).astype(np.uint8)).save(tmp_path / ds["mask"][i])

    out = tmp_path / "report"
    out.mkdir()
    r = section_appearance(ds, out, log=lambda *a: None)
    assert np.isclose(r["exposure"].saturated_pct.iloc[0], 100 * 10 / 96, atol=0.1)
    assert {f.name for f in out.iterdir()} == {"3_appearance.xlsx", "3_appearance.png", "3_exposure.png",
                                               "3_crop_features.csv", "3_crop_features.xlsx",
                                               "3_crop_features.png", "3_crop_features_top.png"}

    cf = r["crop_features"]
    shape = cf["steps"].set_index("step")
    assert shape.loc["+ shape", "balanced_accuracy"] >= shape.loc["+ size", "balanced_accuracy"]
    assert cf["ranking"].set_index("feature").loc["shape_aspect_ratio", "auc"] > 0.95 or "B" not in cf["pair"]


def test_label_quality_section(tmp_path):
    from cii.report import label_flags, section_label_quality
    rng = np.random.default_rng(3)
    ds = toy_sized()
    n = len(ds)

    # colour: class B pink, the rest grey; EDS: O for A and C, N + Ti for B, Fe as matrix
    pink = ds.training_class == "B"
    ds["vhx_contrast_R_max"] = rng.normal(0.4, 0.03, n)
    ds["vhx_contrast_G_max"] = np.where(pink, 0.7, 0.4) + rng.normal(0, 0.03, n)
    ds["vhx_contrast_B_max"] = ds.vhx_contrast_R_max
    ds["vhx_contrast_max"] = ds[["vhx_contrast_R_max", "vhx_contrast_G_max"]].max(axis=1)
    for c in "RGB":
        ds[f"vhx_mean_{c}"] = 150.0
    ds["O_wt"] = np.where(pink, 0.0, 5.0)
    ds["N_wt"] = np.where(pink, 8.0, 0.0)
    ds["Ti_wt"] = np.where(pink, 30.0, 0.0)
    ds["Mn_wt"] = 1.4
    ds["Fe_wt"] = 100 - ds[["O_wt", "N_wt", "Ti_wt", "Mn_wt"]].sum(axis=1)

    # match info: a few warnings
    ds["match_type"] = rng.choice(["object", "region", "relaxed"], n, p=[0.8, 0.15, 0.05])
    ds["n_candidates"] = rng.choice([1, 2], n, p=[0.95, 0.05])
    ds["sem_neighbour"] = rng.random(n) < 0.05
    ds["distance_um"] = rng.gamma(2, 0.2, n)
    for c in ["sem_id", "field", "vhx_image", "Type"]:
        ds[c] = np.arange(n)

    flags = label_flags(ds)
    assert flags.uncertain.equals(flags[["flag_relaxed", "flag_candidates", "flag_neighbour", "flag_size"]].any(axis=1))

    r = section_label_quality(ds, tmp_path, log=lambda *a: None)
    assert {f.name for f in tmp_path.iterdir()} == {"4_label_quality.xlsx", "4_suspicious.csv", "4_label_quality.png",
                                                    "4_chemistry.png", "4_model_by_origin.png"}

    share = r["chemistry"]["present_pct"]
    assert "Fe" not in share.columns and "Mn" not in share.columns       # matrix elements left out
    assert (share.loc[share.index.str.startswith("B ("), "Ti"] == 100).all()

    by_origin = r["model"]["by_origin"]
    assert (by_origin.loc[by_origin.index.str.startswith("B ("), "correct_pct"] > 90).all()
    assert r["suspicious"].reasons.str.len().gt(0).all()
