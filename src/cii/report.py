"""Statistical overview of a training dataset (dataset_vN/dataset.csv).

Built section by section; each section writes its tables and figures into
``dataset_vN/report/`` with the section number as prefix.

1. Composition and balance: class counts overall and per split / fold, origin
   of the training classes (original AZtec class, Type table or rule), class
   weights for the imbalance.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# categorical colours in fixed order (validated: CVD and normal-vision separation, light surface)
CLASS_COLOURS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
TEXT = "#0b0b0b"
TEXT_2 = "#52514e"
GRID = "#e4e3df"
SURFACE = "#fcfcfb"


def class_order(ds: pd.DataFrame) -> list[str]:
    """Training classes in a fixed order (alphabetical), so every figure uses the same
    colour for the same class."""
    return sorted(ds.training_class.unique())


def class_colours(classes: list[str]) -> dict[str, str]:
    """Colour per class, assigned in the fixed categorical order (never cycled)."""
    if len(classes) > len(CLASS_COLOURS):
        raise ValueError(f"{len(classes)} classes, only {len(CLASS_COLOURS)} colours - merge classes or facet")
    return dict(zip(classes, CLASS_COLOURS))


def part_label(ds: pd.DataFrame) -> pd.Series:
    """Part of the split per particle: 'test' or 'fold k'."""
    fold = ds.fold.astype(int).astype(str)
    return pd.Series(np.where(ds.split == "test", "test", "fold " + fold), index=ds.index)


def _style(ax):
    """Recessive axes: no top/right spines, light grid behind the marks, muted tick labels."""
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(GRID)
    ax.tick_params(colors=TEXT_2, labelsize=8)
    ax.set_axisbelow(True)


# ---------------------------------------------------------------------------
# 1. Composition and balance
# ---------------------------------------------------------------------------

def composition_tables(ds: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Tables of section 1: counts per class and part, shares, origin, class source, weights."""
    classes = class_order(ds)
    part = part_label(ds)
    parts = ["test"] + sorted(p for p in part.unique() if p != "test")

    counts = pd.crosstab(ds.training_class, part).reindex(index=classes, columns=parts, fill_value=0)
    counts.columns.name = None
    counts["trainval"] = counts[[p for p in parts if p != "test"]].sum(axis=1)
    counts["total"] = counts["trainval"] + counts["test"]
    counts["test_pct"] = (100 * counts["test"] / counts["total"]).round(1)
    counts["share_of_dataset_pct"] = (100 * counts["total"] / counts["total"].sum()).round(1)

    # share of each class within each part: should be about the same in every part
    shares = (100 * pd.crosstab(ds.training_class, part, normalize="columns")).round(1)
    shares = shares.reindex(index=classes, columns=parts, fill_value=0)
    shares.columns.name = None

    origin = pd.crosstab(ds.class_label, ds.training_class).reindex(columns=classes, fill_value=0)
    origin["total"] = origin.sum(axis=1)
    origin = origin.sort_values("total", ascending=False)

    source = pd.crosstab(ds.training_class, ds.class_source).reindex(index=classes, fill_value=0)

    # inverse-frequency weights on the training data (sklearn 'balanced': n / (k * n_class))
    tv = counts["trainval"]
    weights = pd.DataFrame({"n_trainval": tv, "weight": (tv.sum() / (len(tv) * tv)).round(3)})

    return {"counts": counts, "shares_pct": shares, "origin": origin, "class_source": source, "weights": weights}


def composition_figure(ds: pd.DataFrame, t: dict, path: Path):
    """Figure of section 1: particles per class (left) and class shares per part of the
    split (right, one bar per test set / fold)."""
    classes = class_order(ds)
    colours = class_colours(classes)
    counts, shares = t["counts"], t["shares_pct"]

    fig, (a, b) = plt.subplots(1, 2, figsize=(13, 4.2), gridspec_kw={"width_ratios": [1, 1.4]})
    fig.patch.set_facecolor(SURFACE)

    # left: total per class, labelled with count and share
    y = np.arange(len(classes))[::-1]
    a.barh(y, counts.total, height=0.55, color=[colours[c] for c in classes], edgecolor=SURFACE, linewidth=2)
    for yi, c in zip(y, classes):
        n, pct = counts.at[c, "total"], counts.at[c, "share_of_dataset_pct"]
        a.text(n + counts.total.max() * 0.01, yi, f"{n}  ({pct:.0f} %)", va="center", fontsize=8, color=TEXT)
    a.set_yticks(y, classes, fontsize=8, color=TEXT)
    a.set_xlim(0, counts.total.max() * 1.25)
    a.set_xlabel("particles", fontsize=8, color=TEXT_2)
    a.set_title(f"Particles per class (total {int(counts.total.sum())})", fontsize=10, color=TEXT, loc="left")
    a.grid(axis="x", color=GRID, linewidth=0.6)
    _style(a)

    # right: 100 % stacked bar per part of the split
    parts = list(shares.columns)
    x = np.arange(len(parts))
    bottom = np.zeros(len(parts))
    for c in classes:
        v = shares.loc[c].to_numpy()
        b.bar(x, v, bottom=bottom, width=0.6, color=colours[c], edgecolor=SURFACE, linewidth=2, label=c)
        for xi, (vi, bi) in enumerate(zip(v, bottom)):
            if vi >= 6:                                   # label only segments large enough to hold it
                b.text(xi, bi + vi / 2, f"{vi:.0f}", ha="center", va="center", fontsize=7, color=TEXT)
        bottom += v
    n_part = pd.crosstab(part_label(ds), 1)[1].reindex(parts)
    b.set_xticks(x, [f"{p}\nn={n}" for p, n in zip(parts, n_part)], fontsize=8, color=TEXT)
    b.set_ylim(0, 100)
    b.set_ylabel("share of the part (%)", fontsize=8, color=TEXT_2)
    b.set_title("Class shares in the test set and the folds", fontsize=10, color=TEXT, loc="left")
    b.legend(fontsize=8, frameon=False, bbox_to_anchor=(1.01, 1), loc="upper left", labelcolor=TEXT)
    _style(b)

    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def origin_figure(t: dict, path: Path):
    """Figure of section 1: original AZtec class (rows) -> training class (columns), counts
    as a single-hue heatmap with the numbers written in the cells."""
    origin = t["origin"].drop(columns="total")
    v = origin.to_numpy()

    fig, ax = plt.subplots(figsize=(1.6 + 1.35 * v.shape[1], 0.9 + 0.38 * v.shape[0]))
    fig.patch.set_facecolor(SURFACE)
    ax.imshow(np.log1p(v), cmap="Blues", aspect="auto", vmin=0, vmax=np.log1p(v.max()) * 1.15)

    for i in range(v.shape[0]):
        for j in range(v.shape[1]):
            if v[i, j]:
                dark = np.log1p(v[i, j]) > 0.6 * np.log1p(v.max())
                ax.text(j, i, str(v[i, j]), ha="center", va="center", fontsize=8,
                        color=SURFACE if dark else TEXT)

    ax.set_xticks(range(v.shape[1]), origin.columns, fontsize=8, color=TEXT, rotation=20, ha="right")
    ax.set_yticks(range(v.shape[0]), origin.index, fontsize=8, color=TEXT)
    ax.set_xlabel("training class", fontsize=8, color=TEXT_2)
    ax.set_ylabel("original class (AZtec)", fontsize=8, color=TEXT_2)
    ax.set_title("Origin of the training classes (particles)", fontsize=10, color=TEXT, loc="left")
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)

    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def section_composition(ds: pd.DataFrame, out: Path, log=print) -> dict:
    """Section 1: writes 1_composition.xlsx, 1_composition.png and 1_origin.png and prints
    the key numbers (class counts per part, imbalance, class weights)."""
    t = composition_tables(ds)

    with pd.ExcelWriter(out / "1_composition.xlsx") as xw:
        for name, tab in t.items():
            tab.to_excel(xw, sheet_name=name)
    composition_figure(ds, t, out / "1_composition.png")
    origin_figure(t, out / "1_origin.png")

    counts = t["counts"]
    ratio = counts.total.max() / counts.total.min()
    spread = (t["shares_pct"].max(axis=1) - t["shares_pct"].min(axis=1)).max()
    log("1. Composition and balance")
    log(counts[[c for c in counts.columns if c != "share_of_dataset_pct"]].to_string())
    log(f"largest / smallest class: {ratio:.1f}x; largest difference of a class share between parts: "
        f"{spread:.1f} percentage points")
    log("class weights (trainval, inverse frequency): "
        + ", ".join(f"{c} {w:.2f}" for c, w in t["weights"].weight.items()))
    return t
