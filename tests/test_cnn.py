import json

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")


def test_orientations_are_the_8_symmetries():
    from cii.cnn.data import orient
    x = torch.arange(16.0).reshape(1, 4, 4)                     # asymmetric pattern
    views = [orient(x, k) for k in range(8)]
    assert len({tuple(v.flatten().tolist()) for v in views}) == 8
    assert torch.equal(orient(x, 0), x)


def test_model_input_layer_and_output():
    from cii.cnn.model import build_model
    model = build_model(n_classes=3, in_channels=4, pretrained=False)
    assert torch.count_nonzero(model.conv1.weight[:, 3]) == 0   # mask filters start at zero
    assert model(torch.zeros(2, 4, 96, 96)).shape == (2, 3)


def test_class_weights_balanced():
    from cii.cnn.train import class_weights
    w = class_weights(np.array([0, 0, 0, 1]), 3)
    assert np.allclose(w, [4 / 9, 4 / 3, 0])


def test_cross_validation_run_folder(tmp_path):
    from PIL import Image
    from cii.cnn.train import TrainParams, cross_validate

    # toy dataset: class A dark round particles, class B pink elongated ones; 2 folds + test
    (tmp_path / "crops").mkdir()
    (tmp_path / "masks").mkdir()
    yy, xx = np.mgrid[:96, :96] - 48
    rows = []

    for i in range(40):
        cls = "A" if i % 2 == 0 else "B"
        a, b = (5, 5) if cls == "A" else (10, 3)
        inside = (xx / a) ** 2 + (yy / b) ** 2 <= 1
        img = np.full((96, 96, 3), 230, np.uint8)
        img[inside] = (90, 90, 90) if cls == "A" else (200, 120, 170)
        Image.fromarray(img).save(tmp_path / "crops" / f"{i}.png")
        Image.fromarray((inside * 255).astype(np.uint8)).save(tmp_path / "masks" / f"{i}.png")
        split = "test" if i >= 36 else "trainval"
        rows.append({"crop": f"crops/{i}.png", "mask": f"masks/{i}.png", "sem_id": i, "training_class": cls,
                     "class_label": "x", "split": split, "fold": -1 if split == "test" else (i // 2) % 2})

    ds = pd.DataFrame(rows)
    out = tmp_path / "runs" / "toy"
    p = TrainParams(epochs=2, batch_size=8, pretrained=False, threads=1)
    metrics = cross_validate(ds, tmp_path, out, p, log=lambda *a: None)

    names = {f.name for f in out.iterdir()}
    assert {"config.json", "history.csv", "predictions.csv", "metrics.json", "curves.png", "confusion.png",
            "model_fold0.pt", "model_fold1.pt"} <= names

    pred = pd.read_csv(out / "predictions.csv")
    assert len(pred) == 36 and not pred.crop.isin(ds.crop[ds.split == "test"]).any()   # test set never used
    assert np.allclose(pred[["p_A", "p_B"]].sum(axis=1), 1, atol=1e-3)
    assert set(metrics["balanced_accuracy_per_fold"]) == {0, 1}
    assert json.loads((out / "config.json").read_text())["n_particles"] == 36


def test_model_init_with_lighting_groups():
    from cii.cnn.data import channel_names, rgb_groups
    from cii.cnn.model import build_model
    channels = ("rgb", "mask", "df")
    assert channel_names(channels) == ["R", "G", "B", "mask", "df_R", "df_G", "df_B"]
    assert rgb_groups(channels) == [[0, 1, 2], [4, 5, 6]]

    model = build_model(3, in_channels=7, pretrained=False, groups=rgb_groups(channels))
    w = model.conv1.weight
    assert torch.equal(w[:, 0:3], w[:, 4:7])                     # same filters for both RGB groups
    assert torch.count_nonzero(w[:, 3]) == 0                     # mask starts at zero


def test_cross_validation_with_a_lighting_mode(tmp_path):
    from PIL import Image
    from cii.cnn.train import TrainParams, cross_validate

    for d in ("crops", "masks", "crops_df"):
        (tmp_path / d).mkdir()
    yy, xx = np.mgrid[:96, :96] - 48
    rows = []

    for i in range(24):
        cls = "A" if i % 2 == 0 else "B"
        inside = (xx / (5 if cls == "A" else 10)) ** 2 + (yy / (5 if cls == "A" else 3)) ** 2 <= 1
        img = np.full((96, 96, 3), 230, np.uint8)
        img[inside] = 90
        Image.fromarray(img).save(tmp_path / "crops" / f"{i}.png")
        Image.fromarray((inside * 255).astype(np.uint8)).save(tmp_path / "masks" / f"{i}.png")
        Image.fromarray(255 - img).save(tmp_path / "crops_df" / f"{i}.png")
        rows.append({"crop": f"crops/{i}.png", "mask": f"masks/{i}.png", "sem_id": i, "training_class": cls,
                     "class_label": "x", "split": "trainval", "fold": (i // 2) % 2,
                     "crop_df": "" if i < 4 else f"crops_df/{i}.png"})              # 4 particles without df

    ds = pd.DataFrame(rows)
    p = TrainParams(epochs=1, batch_size=8, pretrained=False, threads=1, channels=("rgb", "mask", "df"))
    out = tmp_path / "runs" / "df"
    cross_validate(ds, tmp_path, out, p, log=lambda *a: None)

    config = json.loads((out / "config.json").read_text())
    assert config["channels"][-3:] == ["df_R", "df_G", "df_B"]
    assert config["n_particles"] == 20                           # particles without df crop left out

    # the same particles without the df channel (require)
    p2 = TrainParams(epochs=1, batch_size=8, pretrained=False, threads=1, require=("df",))
    cross_validate(ds, tmp_path, tmp_path / "runs" / "rgb_same", p2, log=lambda *a: None)
    assert json.loads((tmp_path / "runs" / "rgb_same" / "config.json").read_text())["n_particles"] == 20
