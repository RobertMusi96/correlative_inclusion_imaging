"""Cross-validation of the CNN over the dataset folds.

Only training / validation particles are used (split 'trainval'); the test set
is never loaded. For every fold k: train on the other folds, validate on fold k.
Every run uses the same settings for all folds (TrainParams); the result is the
validation score after the last epoch (no choice of the best epoch on the
validation fold, which would make the score optimistic).

Loss: cross-entropy with class weights n / (n_classes x n_class) of the training
folds (as class_weight='balanced' of the baselines). Score: balanced accuracy.
Optimiser AdamW, learning rate with cosine decay to zero over the epochs.
Validation with test-time augmentation (tta): mean probability over the 8
orientations.

Run folder:
    config.json        settings, classes, channels, dataset
    history.csv        per fold and epoch: training loss, validation loss and balanced accuracy
    predictions.csv    out-of-fold prediction per particle: class probabilities, predicted class
    metrics.json       balanced accuracy per fold, mean, SD; confusion matrix (% of each true class)
    curves.png, confusion.png
    model_fold<k>.pt   weights after the last epoch
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from .data import CHANNELS, CropDataset, channel_stats, load_arrays, orient
from .model import build_model


@dataclass
class TrainParams:
    epochs: int = 30
    batch_size: int = 64
    lr: float = 1e-3                # AdamW learning rate at the start (cosine decay to 0)
    weight_decay: float = 1e-4
    brightness: float = 0.1         # augmentation: brightness factor in [1 - b, 1 + b]
    pretrained: bool = True         # ImageNet weights for ResNet-18
    tta: bool = True                # validation: mean over the 8 orientations
    seed: int = 0
    threads: int = 0                # CPU threads for torch (0 = torch default)


def class_weights(y: np.ndarray, n_classes: int) -> np.ndarray:
    """Weight per class n / (n_classes x n_class) (classes absent from y get weight 0)."""
    counts = np.bincount(y, minlength=n_classes).astype(float)
    return np.where(counts > 0, len(y) / (n_classes * np.maximum(counts, 1)), 0.0)


def predict(model: nn.Module, data: CropDataset, batch_size: int, tta: bool) -> np.ndarray:
    """Class probabilities (n x classes) of all particles of data; with tta the mean over the
    8 orientations."""
    loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=False)
    model.eval()
    out = []

    with torch.no_grad():
        for x, _ in loader:
            views = range(8) if tta else [0]
            p = torch.stack([torch.softmax(model(orient(x, k)), dim=1) for k in views]).mean(dim=0)
            out.append(p.numpy())

    return np.concatenate(out)


def train_fold(x_tr, y_tr, x_va, y_va, n_classes: int, p: TrainParams, fold: int, log=print):
    """Trains one model on the training particles and validates after every epoch. Returns
    the model, the history (one dict per epoch) and the validation probabilities after the
    last epoch."""
    from sklearn.metrics import balanced_accuracy_score

    torch.manual_seed(p.seed + fold)
    mean, std = channel_stats(x_tr)
    train = CropDataset(x_tr, y_tr, mean, std, augment=True, brightness=p.brightness, seed=p.seed + fold)
    val = CropDataset(x_va, y_va, mean, std)
    loader = torch.utils.data.DataLoader(train, batch_size=p.batch_size, shuffle=True,
                                         generator=torch.Generator().manual_seed(p.seed + fold))

    model = build_model(n_classes, in_channels=x_tr.shape[1], pretrained=p.pretrained)
    weights = torch.tensor(class_weights(y_tr, n_classes), dtype=torch.float32)
    loss_fn = nn.CrossEntropyLoss(weight=weights)
    opt = torch.optim.AdamW(model.parameters(), lr=p.lr, weight_decay=p.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=p.epochs)

    history = []
    for epoch in range(1, p.epochs + 1):
        t0 = time.time()
        model.train()
        losses = []

        for xb, yb in loader:
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            losses.append(loss.item() * len(yb))

        sched.step()

        # validation after every epoch (for the curves; the score is the last epoch's)
        last = epoch == p.epochs
        prob = predict(model, val, p.batch_size, tta=p.tta and last)
        val_loss = float(nn.functional.nll_loss(torch.log(torch.from_numpy(prob).clamp_min(1e-8)),
                                                torch.from_numpy(y_va.astype(np.int64)), weight=weights))
        acc = float(balanced_accuracy_score(y_va, prob.argmax(axis=1)))

        history.append({"fold": fold, "epoch": epoch, "train_loss": round(sum(losses) / len(y_tr), 4),
                        "val_loss": round(val_loss, 4), "val_balanced_accuracy": round(acc, 4),
                        "seconds": round(time.time() - t0, 1)})
        log(f"fold {fold} epoch {epoch:3d}/{p.epochs}: train loss {history[-1]['train_loss']:.3f}, "
            f"val loss {val_loss:.3f}, val balanced accuracy {100 * acc:.1f} %  ({history[-1]['seconds']:.0f} s)")

    return model, history, prob


def curves_figure(history: pd.DataFrame, path: Path):
    """Training and validation loss and validation balanced accuracy per epoch, one line per
    fold."""
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    from ..report import CLASS_COLOURS, save_figure, use_style

    use_style()
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.2))
    panels = [("train_loss", "training loss"), ("val_loss", "validation loss"),
              ("val_balanced_accuracy", "validation balanced accuracy")]

    for ax, (col, label) in zip(axes, panels):
        for k, (fold, h) in enumerate(history.groupby("fold")):
            v = 100 * h[col] if col == "val_balanced_accuracy" else h[col]
            ax.plot(h.epoch, v, color=CLASS_COLOURS[k % len(CLASS_COLOURS)], label=f"fold {fold}")
        ax.set_xlabel("epoch")
        ax.set_ylabel(label + (" (%)" if col == "val_balanced_accuracy" else ""))
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    axes[-1].legend(loc="lower right")
    fig.suptitle("Validation after every epoch; the reported score is the last epoch's (with test-time augmentation)",
                 x=0.01, ha="left", color="#444444")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    save_figure(fig, path, dpi=150)
    plt.close(fig)


def confusion_figure(cm: pd.DataFrame, title: str, path: Path):
    """Out-of-fold confusion matrix (% of each true class)."""
    import matplotlib.pyplot as plt
    from ..report import TEXT, save_figure, short_class, use_style

    use_style()
    fig, ax = plt.subplots(figsize=(6.5, 5.2))
    ax.imshow(cm.to_numpy(), cmap="Blues", vmin=0, vmax=100)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            v = cm.iat[i, j]
            ax.text(j, i, f"{v:.0f}", ha="center", va="center", color="white" if v >= 60 else TEXT)

    ax.set_xticks(range(cm.shape[1]), [short_class(c) for c in cm.columns], rotation=20, ha="right")
    ax.set_yticks(range(cm.shape[0]), [short_class(c) for c in cm.index])
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.grid(False)
    ax.tick_params(length=0)
    ax.set_title(title, loc="left")

    fig.tight_layout()
    save_figure(fig, path, dpi=150)
    plt.close(fig)


def cross_validate(ds: pd.DataFrame, root: Path, out: Path, p: TrainParams, folds: list[int] | None = None,
                   log=print) -> dict:
    """Cross-validation over the folds of a dataset (dataset.csv rows, root = dataset folder),
    writing the run folder out (see module docstring). folds: subset of folds to run (default
    all). Returns the metrics."""
    from sklearn.metrics import balanced_accuracy_score

    if p.threads:
        torch.set_num_threads(p.threads)

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)

    tv = ds[ds.split == "trainval"].reset_index(drop=True)          # the test set is never loaded
    classes = sorted(tv.training_class.unique())
    y = tv.training_class.map({c: i for i, c in enumerate(classes)}).to_numpy()
    folds = sorted(tv.fold.unique()) if folds is None else list(folds)

    config = {"params": asdict(p), "classes": classes, "channels": CHANNELS, "dataset": str(root),
              "n_particles": len(tv), "folds": [int(k) for k in folds], "torch": torch.__version__}
    (out / "config.json").write_text(json.dumps(config, indent=2))

    log(f"loading {len(tv)} crops")
    x = load_arrays(tv, root)

    histories, scores, probs = [], {}, np.full((len(tv), len(classes)), np.nan)

    for k in folds:
        va = (tv.fold == k).to_numpy()
        model, hist, prob = train_fold(x[~va], y[~va], x[va], y[va], len(classes), p, int(k), log)

        histories += hist
        probs[va] = prob
        scores[int(k)] = round(float(balanced_accuracy_score(y[va], prob.argmax(axis=1))), 4)
        torch.save(model.state_dict(), out / f"model_fold{k}.pt")
        pd.DataFrame(histories).to_csv(out / "history.csv", index=False)
        log(f"fold {k}: balanced accuracy {100 * scores[int(k)]:.1f} %")

    # out-of-fold predictions and metrics
    done = np.isfinite(probs[:, 0])
    pred = pd.DataFrame(probs, columns=[f"p_{c}" for c in classes]).round(4)
    pred.insert(0, "predicted", [classes[i] if ok else "" for i, ok in zip(probs.argmax(axis=1), done)])
    pred = pd.concat([tv[["crop", "sem_id", "training_class", "class_label", "fold"]], pred], axis=1)
    pred[done].to_csv(out / "predictions.csv", index=False)

    cm = pd.crosstab(pred.training_class[done], pred.predicted[done], normalize="index")
    cm = (100 * cm.reindex(index=classes, columns=classes, fill_value=0)).round(1)

    values = np.array(list(scores.values()))
    metrics = {"balanced_accuracy_per_fold": scores, "balanced_accuracy_mean": round(float(values.mean()), 4),
               "balanced_accuracy_sd": round(float(values.std(ddof=1)), 4) if len(values) > 1 else None,
               "confusion_pct": cm.to_dict(orient="index")}
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))

    curves_figure(pd.DataFrame(histories), out / "curves.png")
    sd = f" ± {100 * metrics['balanced_accuracy_sd']:.1f}" if metrics["balanced_accuracy_sd"] is not None else ""
    confusion_figure(cm, f"CNN, out-of-fold (folds {', '.join(map(str, folds))}):\nbalanced accuracy "
                         f"{100 * metrics['balanced_accuracy_mean']:.1f}{sd} %", out / "confusion.png")

    log(f"balanced accuracy {100 * metrics['balanced_accuracy_mean']:.1f}{sd} % (folds: "
        f"{', '.join(f'{100 * v:.1f}' for v in values)})")
    return metrics
