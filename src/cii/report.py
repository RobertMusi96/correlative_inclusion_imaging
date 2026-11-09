"""Statistical overview of a training dataset (dataset_vN/dataset.csv).

Built section by section; each section writes its tables and figures into
``dataset_vN/report/`` with the section number as prefix.

1. Composition and balance: class counts overall and per split / fold, origin
   of the training classes (original AZtec class, Type table or rule), class
   weights for the imbalance.
2. Size: SEM and VHX ECD per class (and per class source), VHX / SEM size
   ratio, particles longer than the crop, and a size-only baseline: how well
   the class can be predicted from the VHX size alone (a shortcut a model
   could learn instead of the appearance).
3. Optical appearance: VHX contrast, contrast per colour channel, pinkness
   (green / red contrast) and particle colour per class, a colour-only baseline,
   and from the crops the background brightness and overexposure per class
   (a class recorded in brighter areas would be a confounder). Also from the
   crops: shape, interior and edge features (cii.crop_features), baselines
   adding one feature group after the other, and the single features that best
   separate the two classes colour confuses most.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .plot_style import setup_matplotlib_parameter

# categorical colours in fixed order (validated: CVD and normal-vision separation on white)
CLASS_COLOURS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
TEXT = "#222222"         # values written into the plots (same as the axes colour of the style)
MUTED = "#444444"        # secondary text (same as the legend text of the style)


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


def use_style():
    """Applies the project's matplotlib style (cii.plot_style) with the class colours as cycle."""
    setup_matplotlib_parameter(CLASS_COLOURS)


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
    use_style()
    classes = class_order(ds)
    colours = class_colours(classes)
    counts, shares = t["counts"], t["shares_pct"]

    fig, (a, b) = plt.subplots(1, 2, figsize=(15, 5), gridspec_kw={"width_ratios": [1, 1.5]})

    # left: total per class, labelled with count and share
    y = np.arange(len(classes))[::-1]
    a.barh(y, counts.total, height=0.55, color=[colours[c] for c in classes], edgecolor="white", linewidth=2)

    for yi, c in zip(y, classes):
        n, pct = counts.at[c, "total"], counts.at[c, "share_of_dataset_pct"]
        a.text(n + counts.total.max() * 0.01, yi, f"{n}  ({pct:.0f} %)", va="center", color=TEXT)

    a.set_yticks(y, classes)
    a.set_xlim(0, counts.total.max() * 1.3)
    a.set_xlabel("particles")
    a.set_title(f"Particles per class (total {int(counts.total.sum())})", loc="left")
    a.grid(axis="y", visible=False)
    a.set_axisbelow(True)

    # right: 100 % stacked bar per part of the split
    parts = list(shares.columns)
    x = np.arange(len(parts))
    bottom = np.zeros(len(parts))

    for c in classes:
        v = shares.loc[c].to_numpy()
        b.bar(x, v, bottom=bottom, width=0.6, color=colours[c], edgecolor="white", linewidth=2, label=c)

        for xi, (vi, bi) in enumerate(zip(v, bottom)):
            if vi >= 6:                                   # label only segments large enough to hold it
                b.text(xi, bi + vi / 2, f"{vi:.0f}", ha="center", va="center", color=TEXT)

        bottom += v

    n_part = pd.crosstab(part_label(ds), 1)[1].reindex(parts)
    b.set_xticks(x, [f"{p}\nn={n}" for p, n in zip(parts, n_part)])
    b.set_ylim(0, 100)
    b.set_ylabel("share of the part (%)")
    b.set_title("Class shares in the test set and the folds", loc="left")
    b.legend(bbox_to_anchor=(1.01, 1), loc="upper left")
    b.grid(axis="x", visible=False)
    b.set_axisbelow(True)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def origin_figure(t: dict, path: Path):
    """Figure of section 1: original AZtec class (rows) -> training class (columns), counts
    as a single-hue heatmap with the numbers written in the cells."""
    use_style()
    origin = t["origin"].drop(columns="total")
    v = origin.to_numpy()

    fig, ax = plt.subplots(figsize=(2.5 + 1.9 * v.shape[1], 1.4 + 0.45 * v.shape[0]))
    ax.imshow(np.log1p(v), cmap="Blues", aspect="auto", vmin=0, vmax=np.log1p(v.max()) * 1.15)
    ax.grid(False)

    for i in range(v.shape[0]):
        for j in range(v.shape[1]):
            if v[i, j]:
                dark = np.log1p(v[i, j]) > 0.6 * np.log1p(v.max())
                ax.text(j, i, str(v[i, j]), ha="center", va="center", color="white" if dark else TEXT)

    ax.set_xticks(range(v.shape[1]), origin.columns, rotation=15, ha="right")
    ax.set_yticks(range(v.shape[0]), origin.index)
    ax.set_xlabel("training class")
    ax.set_ylabel("original class (AZtec)")
    ax.set_title("Origin of the training classes (particles)", loc="left")
    ax.tick_params(length=0)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
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


# ---------------------------------------------------------------------------
# 2. Size
# ---------------------------------------------------------------------------

SIZE_BINS = [0, 1.5, 2, 3, 5, 10, np.inf]


def source_groups(ds: pd.DataFrame) -> pd.Series:
    """Group per particle for the comparisons: the training class, split into
    '<class> (<source>)' where a class is fed by more than one source (Type table and rules)."""
    multi = ds.groupby("training_class").class_source.nunique() > 1
    split_it = ds.training_class.map(multi)

    return pd.Series(np.where(split_it, ds.training_class + " (" + ds.class_source + ")", ds.training_class),
                     index=ds.index)


def size_tables(ds: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Tables of section 2: size quantiles per class and group, counts per SEM ECD class,
    particles longer than the crop."""
    d = ds.assign(group=source_groups(ds), size_ratio=ds.vhx_ecd_um / ds.sem_ecd_um)
    q = [0.05, 0.25, 0.5, 0.75, 0.95]

    rows = []
    for key, col in [("training_class", "training_class"), ("group", "group")]:
        for name, g in d.groupby(col):
            if key == "group" and name in set(d.training_class):
                continue                                  # unsplit classes are already listed
            row = {"group": name, "n": len(g)}
            for v in ["sem_ecd_um", "vhx_ecd_um", "size_ratio", "sem_length_um", "sem_aspect_ratio"]:
                for qi, val in zip(q, g[v].quantile(q)):
                    row[f"{v}_p{int(qi * 100)}"] = round(float(val), 2)
            rows.append(row)
    quantiles = pd.DataFrame(rows).set_index("group")

    labels = [f"{lo:g}-{hi:g}" if np.isfinite(hi) else f">{lo:g}" for lo, hi in zip(SIZE_BINS[:-1], SIZE_BINS[1:])]
    sem_bin = pd.cut(d.sem_ecd_um, SIZE_BINS, labels=labels, right=False)
    per_bin = pd.crosstab(d.training_class, sem_bin).reindex(columns=labels, fill_value=0)
    per_bin.columns.name = None

    crop_um = d.crop_size_px * d.um_per_px
    longer = d[d.sem_length_um >= crop_um]
    too_long = longer.groupby("training_class").size().reindex(class_order(ds), fill_value=0).rename("n_longer_than_crop")

    return {"quantiles": quantiles, "per_sem_ecd_class": per_bin, "longer_than_crop": too_long.to_frame(),
            "longer_than_crop_list": longer[["crop", "training_class", "sem_ecd_um", "sem_length_um"]]}


def size_baseline(ds: pd.DataFrame, feature: str = "vhx_ecd_um") -> dict:
    """Size-only baseline: logistic regression on log(size) and its square, class weights
    balanced, cross-validated over the folds (train on four, predict the fifth). Returns the
    balanced accuracy per fold and the row-normalised confusion matrix; chance level is
    1 / number of classes."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score, confusion_matrix

    tv = ds[ds.split == "trainval"]
    x = np.log(tv[feature].clip(lower=0.1).to_numpy())
    X = np.c_[x, x ** 2]
    y = tv.training_class.to_numpy()
    classes = class_order(ds)

    pred = np.empty(len(tv), dtype=object)
    scores = []
    for k in sorted(tv.fold.unique()):
        val = (tv.fold == k).to_numpy()
        model = LogisticRegression(class_weight="balanced", max_iter=1000)
        model.fit(X[~val], y[~val])
        pred[val] = model.predict(X[val])
        scores.append(balanced_accuracy_score(y[val], pred[val]))

    cm = pd.DataFrame(confusion_matrix(y, pred, labels=classes), index=classes, columns=classes)
    cm_pct = (100 * cm.div(cm.sum(axis=1), axis=0)).round(1)
    cm_pct.index.name, cm_pct.columns.name = "true", "predicted"

    return {"feature": feature, "balanced_accuracy_per_fold": np.round(scores, 3),
            "balanced_accuracy": float(np.mean(scores)), "chance": 1 / len(classes), "confusion_pct": cm_pct}


def _ecdf(ax, values, colour, label, dashed=False):
    """Cumulative distribution of values as a step line (share of particles below a size)."""
    v = np.sort(values)
    y = np.arange(1, len(v) + 1) / len(v) * 100
    ax.step(v, y, where="post", color=colour, linewidth=2, linestyle="--" if dashed else "-", label=label)


def size_figure(ds: pd.DataFrame, base: dict, path: Path):
    """Figure of section 2: cumulative SEM ECD (left) and VHX ECD (middle) per class / group,
    VHX / SEM size ratio per class (right, box = quartiles, whiskers = 5-95 %)."""
    use_style()
    classes = class_order(ds)
    colours = class_colours(classes)
    groups = source_groups(ds)

    fig, axes = plt.subplots(1, 3, figsize=(17, 5.4), gridspec_kw={"width_ratios": [1, 1, 0.85]})

    # left / middle: cumulative size distributions, one line per group (class colour, rule groups dashed)
    for ax, col, title in [(axes[0], "sem_ecd_um", "SEM ECD"), (axes[1], "vhx_ecd_um", "VHX ECD")]:

        for name in sorted(groups.unique()):
            sel = groups == name
            cls = ds.training_class[sel].iloc[0]
            dashed = "(rule" in name
            _ecdf(ax, ds.loc[sel, col].to_numpy(), colours[cls], f"{name}  (n={int(sel.sum())})", dashed)

        ax.set_xscale("log")
        ax.set_xticks([1, 1.5, 2, 3, 5, 10, 20], ["1", "1.5", "2", "3", "5", "10", "20"])
        ax.minorticks_off()
        ax.set_xlim(0.7, 25)
        ax.set_ylim(0, 100)
        ax.set_xlabel(f"{title} (µm)")
        ax.set_ylabel("particles smaller than this (%)")
        ax.set_title(f"{title} per class", loc="left")

    axes[0].legend(loc="lower right", fontsize=9)

    # right: size ratio VHX / SEM per class
    ax = axes[2]
    ratio = ds.vhx_ecd_um / ds.sem_ecd_um
    data = [ratio[ds.training_class == c].to_numpy() for c in classes]
    bp = ax.boxplot(data, orientation="horizontal", whis=(5, 95), showfliers=False, widths=0.5, patch_artist=True,
                    medianprops={"color": TEXT, "linewidth": 1.5})

    for patch, c in zip(bp["boxes"], classes):
        patch.set_facecolor(colours[c])
        patch.set_edgecolor("white")

    for k, d in enumerate(data):
        ax.text(np.percentile(d, 95) + 0.04, k + 1, f"median {np.median(d):.2f}", va="center", color=MUTED)

    ax.axvline(1, color=MUTED, linewidth=0.8, linestyle=":")
    ax.set_yticks(range(1, len(classes) + 1), classes)
    ax.invert_yaxis()
    ax.set_xlim(right=max(np.percentile(d, 95) for d in data) + 0.45)
    ax.set_xlabel("VHX ECD / SEM ECD")
    ax.set_title("Size in the light microscope vs SEM", loc="left")
    ax.grid(axis="y", visible=False)

    fig.suptitle(f"Size-only baseline (VHX ECD, cross-validated): balanced accuracy "
                 f"{100 * base['balanced_accuracy']:.0f} %  (chance {100 * base['chance']:.0f} %)",
                 x=0.01, ha="left", color=MUTED)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def section_size(ds: pd.DataFrame, out: Path, log=print) -> dict:
    """Section 2: writes 2_size.xlsx and 2_size.png and prints the size medians, the
    particles longer than the crop and the size-only baseline."""
    t = size_tables(ds)
    base_vhx = size_baseline(ds, "vhx_ecd_um")
    base_sem = size_baseline(ds, "sem_ecd_um")

    summary = pd.DataFrame({
        "feature": [b["feature"] for b in (base_vhx, base_sem)],
        "balanced_accuracy": [round(b["balanced_accuracy"], 3) for b in (base_vhx, base_sem)],
        "per_fold": [", ".join(map(str, b["balanced_accuracy_per_fold"])) for b in (base_vhx, base_sem)],
        "chance": [round(base_vhx["chance"], 3)] * 2,
    })
    with pd.ExcelWriter(out / "2_size.xlsx") as xw:
        for name, tab in t.items():
            tab.to_excel(xw, sheet_name=name[:31])
        summary.to_excel(xw, sheet_name="baseline", index=False)
        base_vhx["confusion_pct"].to_excel(xw, sheet_name="baseline_confusion_vhx")
    size_figure(ds, base_vhx, out / "2_size.png")

    q = t["quantiles"]
    log("2. Size")
    log(q[["n", "sem_ecd_um_p25", "sem_ecd_um_p50", "sem_ecd_um_p75", "vhx_ecd_um_p50", "size_ratio_p50"]].to_string())
    log("particles per SEM ECD class (um):")
    log(t["per_sem_ecd_class"].to_string())
    log(f"longer than the crop: {int(t['longer_than_crop'].n_longer_than_crop.sum())} particles")
    log(f"size-only baseline (balanced accuracy, chance {100 * base_vhx['chance']:.0f} %): "
        f"VHX ECD {100 * base_vhx['balanced_accuracy']:.0f} %, SEM ECD {100 * base_sem['balanced_accuracy']:.0f} %")
    log("confusion of the VHX size baseline (% of each true class):")
    log(base_vhx["confusion_pct"].to_string())

    return {"tables": t, "baseline_vhx": base_vhx, "baseline_sem": base_sem}


# ---------------------------------------------------------------------------
# 3. Optical appearance
# ---------------------------------------------------------------------------

APPEARANCE = {                                   # column -> axis label
    "vhx_contrast_max": "contrast (darkest point)",
    "pinkness": "pinkness (green / red contrast)",
    "vhx_contrast_R_max": "red contrast",
    "vhx_contrast_G_max": "green contrast",
    "vhx_contrast_B_max": "blue contrast",
}
PINK_THRESHOLD = 1.3                             # green / red contrast above this: pink (Ti(C,N)-like)


def add_pinkness(ds: pd.DataFrame) -> pd.DataFrame:
    """Copy of ds with 'pinkness' = green contrast / red contrast of the particle (> 1: darker
    in green than in red, i.e. pink / violet)."""
    return ds.assign(pinkness=ds.vhx_contrast_G_max / ds.vhx_contrast_R_max.clip(lower=1e-3))


def appearance_tables(ds: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Tables of section 3: quantiles of contrast, channel contrasts, pinkness and particle
    colour per class and source group; share of pink particles per group."""
    d = add_pinkness(ds).assign(group=source_groups(ds))
    q = [0.05, 0.25, 0.5, 0.75, 0.95]
    cols = list(APPEARANCE) + ["vhx_mean_R", "vhx_mean_G", "vhx_mean_B"]

    rows = []
    for key in ["training_class", "group"]:
        for name, g in d.groupby(key):
            if key == "group" and name in set(d.training_class):
                continue                                  # unsplit classes are already listed
            row = {"group": name, "n": len(g), "pink_pct": round(100 * float((g.pinkness >= PINK_THRESHOLD).mean()), 1)}
            for c in cols:
                for qi, val in zip(q, g[c].quantile(q)):
                    row[f"{c}_p{int(qi * 100)}"] = round(float(val), 3)
            rows.append(row)

    return {"quantiles": pd.DataFrame(rows).set_index("group")}


def colour_baseline(ds: pd.DataFrame) -> dict:
    """Colour-only baseline: logistic regression on contrast, channel contrasts, log pinkness
    and particle colour (no size), standardised, class weights balanced, cross-validated
    over the folds. Returns balanced accuracy per fold and the row-normalised confusion."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score, confusion_matrix
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    d = add_pinkness(ds)
    tv = d[d.split == "trainval"]
    feats = ["vhx_contrast_max", "vhx_contrast_R_max", "vhx_contrast_G_max", "vhx_contrast_B_max",
             "vhx_mean_R", "vhx_mean_G", "vhx_mean_B"]
    X = np.c_[tv[feats].fillna(0).to_numpy(), np.log(tv.pinkness.clip(lower=0.05).to_numpy())]
    y = tv.training_class.to_numpy()
    classes = class_order(ds)

    pred = np.empty(len(tv), dtype=object)
    scores = []
    for k in sorted(tv.fold.unique()):
        val = (tv.fold == k).to_numpy()
        model = make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=2000))
        model.fit(X[~val], y[~val])
        pred[val] = model.predict(X[val])
        scores.append(balanced_accuracy_score(y[val], pred[val]))

    cm = pd.DataFrame(confusion_matrix(y, pred, labels=classes), index=classes, columns=classes)
    cm_pct = (100 * cm.div(cm.sum(axis=1), axis=0)).round(1)
    cm_pct.index.name, cm_pct.columns.name = "true", "predicted"

    return {"balanced_accuracy_per_fold": np.round(scores, 3), "balanced_accuracy": float(np.mean(scores)),
            "chance": 1 / len(classes), "confusion_pct": cm_pct}


def crop_exposure(ds: pd.DataFrame, root: Path) -> pd.DataFrame | None:
    """Per crop (if the crops are present): background brightness (median grey of the pixels
    outside the particle mask, 0-255) and share of saturated pixels (255 in R, G and B).
    None if the crop files are missing."""
    from PIL import Image

    first = root / ds.crop.iloc[0]
    if not first.exists():
        return None

    rows = []
    for r in ds.itertuples():
        rgb = np.asarray(Image.open(root / r.crop))[..., :3]
        mask = np.asarray(Image.open(root / r.mask)) > 0 if (root / r.mask).exists() else np.zeros(rgb.shape[:2], bool)
        grey = rgb.mean(axis=2)

        rows.append({"background_grey": float(np.median(grey[~mask])) if (~mask).any() else np.nan,
                     "saturated_pct": 100 * float((rgb == 255).all(axis=2).mean())})

    return pd.DataFrame(rows, index=ds.index)


def _boxes(ax, data: list, labels: list, colours: list, hatches: list, xlabel: str, title: str):
    """Horizontal box plot (box = quartiles, whiskers = 5-95 %) with one box per group;
    hatched boxes mark groups that come from a rule."""
    bp = ax.boxplot(data, orientation="horizontal", whis=(5, 95), showfliers=False, widths=0.55, patch_artist=True,
                    medianprops={"color": TEXT, "linewidth": 1.5})

    for patch, col, h in zip(bp["boxes"], colours, hatches):
        patch.set_facecolor(col)
        patch.set_edgecolor("white")
        patch.set_hatch(h)

    ax.set_yticks(range(1, len(labels) + 1), labels)
    ax.invert_yaxis()
    ax.set_xlabel(xlabel)
    ax.set_title(title, loc="left")
    ax.grid(axis="y", visible=False)


def appearance_figure(ds: pd.DataFrame, base: dict, path: Path):
    """Figure of section 3: contrast and pinkness per class / source group (box plots) and
    pinkness against contrast for every particle (scatter, coloured by class)."""
    use_style()
    d = add_pinkness(ds)
    classes = class_order(ds)
    colours = class_colours(classes)
    groups = source_groups(ds)
    names = sorted(groups.unique())
    group_class = [d.training_class[groups == n].iloc[0] for n in names]
    group_colour = [colours[c] for c in group_class]
    group_hatch = ["//" if "(rule" in n else "" for n in names]
    labels = [f"{n}\n(n={int((groups == n).sum())})" for n in names]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.6), gridspec_kw={"width_ratios": [1, 1, 1.15]})

    # box plots: contrast and pinkness per group
    _boxes(axes[0], [d.loc[groups == n, "vhx_contrast_max"].to_numpy() for n in names], labels,
           group_colour, group_hatch, APPEARANCE["vhx_contrast_max"], "Contrast per class")
    _boxes(axes[1], [d.loc[groups == n, "pinkness"].to_numpy() for n in names], labels,
           group_colour, group_hatch, APPEARANCE["pinkness"], "Pinkness per class")
    axes[1].axvline(PINK_THRESHOLD, color=MUTED, linewidth=0.8, linestyle=":")
    axes[1].set_yticklabels([])

    p95 = [float(d.loc[groups == n, "pinkness"].quantile(0.95)) for n in names]
    for k, n in enumerate(names):
        pct = 100 * float((d.loc[groups == n, "pinkness"] >= PINK_THRESHOLD).mean())
        axes[1].text(p95[k] + 0.04, k + 1, f"{pct:.0f} % pink", va="center", color=MUTED)
    axes[1].set_xlim(right=max(p95) + 0.6)                     # room for the labels right of the whiskers

    # scatter: pinkness against contrast, every particle
    ax = axes[2]
    order = np.random.default_rng(0).permutation(len(d))       # random drawing order: no class hides the others
    sub = d.iloc[order]
    ax.scatter(sub.vhx_contrast_max, sub.pinkness, s=9, c=[colours[c] for c in sub.training_class],
               alpha=0.55, linewidths=0)

    for c in classes:                                          # legend entries with full opacity
        ax.scatter([], [], s=30, color=colours[c], label=c)

    ax.axhline(PINK_THRESHOLD, color=MUTED, linewidth=0.8, linestyle=":")
    ax.set_yscale("log")
    ax.set_yticks([0.5, 0.7, 1, 1.3, 2, 3], ["0.5", "0.7", "1", "1.3", "2", "3"])
    ax.minorticks_off()
    ax.set_ylim(0.4, 4)
    ax.set_xlim(0, min(1, float(d.vhx_contrast_max.quantile(0.995)) * 1.05))
    ax.set_xlabel(APPEARANCE["vhx_contrast_max"])
    ax.set_ylabel(APPEARANCE["pinkness"])
    ax.set_title("Pinkness against contrast (every particle)", loc="left")
    ax.legend(loc="upper right")

    fig.suptitle(f"Colour-only baseline (contrast + colour, no size, cross-validated): balanced accuracy "
                 f"{100 * base['balanced_accuracy']:.0f} %  (chance {100 * base['chance']:.0f} %)",
                 x=0.01, ha="left", color=MUTED)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def exposure_figure(ds: pd.DataFrame, exp: pd.DataFrame, path: Path):
    """Figure of section 3: background brightness (left) and share of saturated pixels (right)
    of the crops per class."""
    use_style()
    classes = class_order(ds)
    colours = class_colours(classes)
    labels = [f"{c}\n(n={int((ds.training_class == c).sum())})" for c in classes]

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
    _boxes(axes[0], [exp.loc[ds.training_class == c, "background_grey"].dropna().to_numpy() for c in classes],
           labels, [colours[c] for c in classes], [""] * len(classes),
           "background grey level of the crop (0-255)", "Background brightness per class")
    _boxes(axes[1], [exp.loc[ds.training_class == c, "saturated_pct"].to_numpy() for c in classes],
           labels, [colours[c] for c in classes], [""] * len(classes),
           "saturated pixels in the crop (%)", "Overexposure per class")
    axes[1].set_yticklabels([])

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# crop features (shape, interior, edge; cii.crop_features): do they separate what colour cannot?

FEATURE_GROUPS = ["colour", "size", "shape", "interior", "edge"]


def colour_columns(ds: pd.DataFrame) -> pd.DataFrame:
    """Colour features as in the colour-only baseline: contrast, channel contrasts, particle
    colour and log pinkness."""
    d = add_pinkness(ds)
    cols = ["vhx_contrast_max", "vhx_contrast_R_max", "vhx_contrast_G_max", "vhx_contrast_B_max",
            "vhx_mean_R", "vhx_mean_G", "vhx_mean_B"]
    out = d[cols].copy()
    out["log_pinkness"] = np.log(d.pinkness.clip(lower=0.05))
    return out


def feature_table(ds: pd.DataFrame, feats: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """All features side by side (colour, log VHX ECD, crop features) and the group of each
    column."""
    from .crop_features import FEATURES

    colour = colour_columns(ds)
    size = pd.DataFrame({"log_vhx_ecd": np.log(ds.vhx_ecd_um.clip(lower=0.1))}, index=ds.index)
    crop = feats[list(FEATURES)]

    group = {c: "colour" for c in colour} | {"log_vhx_ecd": "size"} | {c: FEATURES[c][0] for c in crop}
    return pd.concat([colour, size, crop], axis=1), group


def cv_balanced_accuracy(X: np.ndarray, y: np.ndarray, folds: np.ndarray, classes: list) -> tuple[np.ndarray, np.ndarray]:
    """Logistic regression (missing values -> median, standardised, class weights balanced),
    cross-validated over the folds. Returns the balanced accuracy per fold and the
    out-of-fold prediction of every particle."""
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    pred = np.empty(len(y), dtype=object)
    scores = []

    for k in sorted(np.unique(folds)):
        val = folds == k

        if len(np.unique(y[~val])) < 2:
            continue                                      # tiny datasets: fold without a second class

        model = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                              LogisticRegression(class_weight="balanced", max_iter=3000))
        model.fit(X[~val], y[~val])
        pred[val] = model.predict(X[val])
        scores.append(balanced_accuracy_score(y[val], pred[val]))

    return np.round(scores, 3), pred


def confused_pair(confusion_pct: pd.DataFrame) -> tuple[str, str]:
    """The two classes most often mistaken for each other (sum of both directions)."""
    cm = confusion_pct.to_numpy()
    both = cm + cm.T
    np.fill_diagonal(both, -1)
    i, j = np.unravel_index(int(np.argmax(both)), both.shape)
    return confusion_pct.index[i], confusion_pct.index[j]


def crop_feature_steps(ds: pd.DataFrame, X: pd.DataFrame, group: dict, pair: tuple) -> pd.DataFrame:
    """Baselines adding one feature group after the other (colour, + size, + shape,
    + interior, + edge) and the crop features alone: balanced accuracy over all classes and
    for the confused pair alone (a model trained only on these two classes)."""
    tv = (ds.split == "trainval").to_numpy()
    y = ds.training_class.to_numpy()[tv]
    folds = ds.fold.to_numpy()[tv]
    in_pair = np.isin(y, pair)

    steps = []
    for k in range(1, len(FEATURE_GROUPS) + 1):
        used = FEATURE_GROUPS[:k]
        name = "colour" if k == 1 else "+ " + FEATURE_GROUPS[k - 1]
        steps.append((name, used))
    steps.append(("crop features only", ["shape", "interior", "edge"]))

    rows = []
    for name, used in steps:
        cols = [c for c in X.columns if group[c] in used]
        Xs = X[cols].to_numpy(dtype=float)[tv]

        all_scores, _ = cv_balanced_accuracy(Xs, y, folds, class_order(ds))
        pair_scores, _ = cv_balanced_accuracy(Xs[in_pair], y[in_pair], folds[in_pair], list(pair))

        rows.append({"step": name, "groups": ", ".join(used), "n_features": len(cols),
                     "balanced_accuracy": round(float(all_scores.mean()), 3),
                     "per_fold": ", ".join(map(str, all_scores)),
                     "pair_balanced_accuracy": round(float(pair_scores.mean()), 3),
                     "pair_per_fold": ", ".join(map(str, pair_scores))})

    return pd.DataFrame(rows)


def feature_ranking(ds: pd.DataFrame, X: pd.DataFrame, group: dict, pair: tuple) -> pd.DataFrame:
    """How well each single feature separates the confused pair (training data): area under
    the ROC curve, turned so that 0.5 = no separation and 1 = perfect, and the class with the
    higher values."""
    from sklearn.metrics import roc_auc_score
    from .crop_features import FEATURES

    labels = {c: FEATURES[c][1] for c in FEATURES} | {"log_vhx_ecd": "VHX ECD (log)", "log_pinkness": "pinkness (log)"}
    labels |= {f"vhx_mean_{ch}": f"particle colour {ch} (0-255)" for ch in "RGB"}
    labels |= {c: APPEARANCE.get(c, c) for c in X.columns if c not in labels}

    sel = ((ds.split == "trainval") & ds.training_class.isin(pair)).to_numpy()
    is_second = (ds.training_class.to_numpy()[sel] == pair[1]).astype(int)

    rows = []
    for c in X.columns:
        v = X[c].to_numpy(dtype=float)[sel]
        ok = np.isfinite(v)

        if ok.sum() < 10 or len(np.unique(is_second[ok])) < 2 or np.nanstd(v) == 0:
            continue

        auc = roc_auc_score(is_second[ok], v[ok])
        rows.append({"feature": c, "label": labels[c], "group": group[c], "auc": round(max(auc, 1 - auc), 3),
                     "higher_in": pair[1] if auc >= 0.5 else pair[0],
                     f"median_{pair[0]}": round(float(np.nanmedian(v[is_second == 0])), 3),
                     f"median_{pair[1]}": round(float(np.nanmedian(v[is_second == 1])), 3)})

    return pd.DataFrame(rows).sort_values("auc", ascending=False).reset_index(drop=True)


def crop_feature_figure(steps: pd.DataFrame, ranking: pd.DataFrame, pair: tuple, chance: float, path: Path):
    """Figure of the crop features: balanced accuracy when adding one feature group after
    the other, for all classes and for the confused pair (left), and the features that best
    separate the pair on their own (right)."""
    use_style()
    fig, (a, b) = plt.subplots(1, 2, figsize=(16, 5.6), gridspec_kw={"width_ratios": [1, 1.1]})

    # left: accuracy per step, two series (all classes, the pair)
    x = np.arange(len(steps))
    series = [("balanced_accuracy", "all classes", CLASS_COLOURS[0], chance),
              ("pair_balanced_accuracy", f"{pair[0]} vs {pair[1]}", CLASS_COLOURS[1], 0.5)]

    for col, label, colour, ch in series:
        v = 100 * steps[col].to_numpy()
        a.plot(x, v, marker="o", color=colour, linewidth=2, label=label)
        a.axhline(100 * ch, color=colour, linewidth=0.8, linestyle=":")

        for xi, vi in zip(x, v):
            a.text(xi, vi + 1.5, f"{vi:.0f}", ha="center", va="bottom", color=TEXT)

    a.set_xticks(x, steps.step, rotation=20, ha="right")
    a.set_ylim(min(30, 100 * chance - 5), 100)
    a.set_ylabel("balanced accuracy (%)")
    a.set_title("Adding one feature group after the other (cross-validated)", loc="left")
    a.legend(loc="lower right")
    a.grid(axis="x", visible=False)

    # right: best single features for the pair
    top = ranking.head(15).iloc[::-1]
    y = np.arange(len(top))
    b.barh(y, top.auc - 0.5, left=0.5, height=0.6, color=MUTED, edgecolor="white")

    for yi, (_, r) in zip(y, top.iterrows()):
        b.text(r.auc + 0.005, yi, f"{r.auc:.2f}  higher in {r.higher_in}", va="center", color=MUTED, fontsize=8)

    b.set_yticks(y, [f"{r.group}: {r.label}" for _, r in top.iterrows()])
    b.set_xlim(0.5, 1.05)
    b.set_xlabel("separation of the pair by this feature alone (AUC, 0.5 = none)")
    b.set_title(f"Best single features: {pair[0]} vs {pair[1]}", loc="left")
    b.grid(axis="y", visible=False)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def crop_feature_boxes(ds: pd.DataFrame, X: pd.DataFrame, ranking: pd.DataFrame, path: Path, n: int = 6):
    """Box plots per class of the n crop features (shape, interior, edge) that best separate
    the confused pair."""
    use_style()
    classes = class_order(ds)
    colours = class_colours(classes)
    top = ranking[ranking.group.isin(["shape", "interior", "edge"])].head(n)
    labels = [f"{c}\n(n={int((ds.training_class == c).sum())})" for c in classes]

    cols = 3
    rows = int(np.ceil(len(top) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(17, 3.6 * rows), squeeze=False)

    for ax, (_, r) in zip(axes.ravel(), top.iterrows()):
        data = [X.loc[ds.training_class == c, r.feature].dropna().to_numpy() for c in classes]
        _boxes(ax, data, labels, [colours[c] for c in classes], [""] * len(classes), r.label,
               f"{r.group}: {r.label}  (AUC {r.auc:.2f})")

        if ax not in axes[:, 0]:
            ax.set_yticklabels([])

    for ax in axes.ravel()[len(top):]:
        ax.set_visible(False)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def crop_feature_part(ds: pd.DataFrame, feats: pd.DataFrame, base: dict, out: Path, log=print) -> dict:
    """Crop features of section 3: writes 3_crop_features.csv (per particle),
    3_crop_features.xlsx, 3_crop_features.png and 3_crop_features_top.png, and prints the
    baselines per step and the best single features for the most confused pair."""
    from .crop_features import FEATURES

    X, group = feature_table(ds, feats)
    pair = confused_pair(base["confusion_pct"])
    steps = crop_feature_steps(ds, X, group, pair)
    ranking = feature_ranking(ds, X, group, pair)

    q = feats[list(FEATURES)].assign(training_class=ds.training_class)
    quant = q.groupby("training_class").quantile([0.25, 0.5, 0.75]).unstack().round(3)
    quant.columns = [f"{c}_p{int(qq * 100)}" for c, qq in quant.columns]

    pd.concat([ds[["crop", "training_class", "split", "fold"]], feats], axis=1).to_csv(
        out / "3_crop_features.csv", index=False, float_format="%.4f")

    with pd.ExcelWriter(out / "3_crop_features.xlsx") as xw:
        steps.to_excel(xw, sheet_name="baselines", index=False)
        ranking.to_excel(xw, sheet_name="ranking_pair", index=False)
        quant.to_excel(xw, sheet_name="quantiles")
        pd.DataFrame([{"feature": k, "group": g, "description": d} for k, (g, d) in FEATURES.items()]
                     ).to_excel(xw, sheet_name="definitions", index=False)

    crop_feature_figure(steps, ranking, pair, base["chance"], out / "3_crop_features.png")
    crop_feature_boxes(ds, X, ranking, out / "3_crop_features_top.png")

    log(f"crop features: most confused pair {pair[0]} / {pair[1]}")
    log(steps[["step", "n_features", "balanced_accuracy", "pair_balanced_accuracy"]].to_string(index=False))
    log("best single features for the pair (AUC):")
    log(ranking.head(10)[["feature", "group", "auc", "higher_in"]].to_string(index=False))

    return {"features": feats, "steps": steps, "ranking": ranking, "pair": pair}


def section_appearance(ds: pd.DataFrame, out: Path, log=print) -> dict:
    """Section 3: writes 3_appearance.xlsx, 3_appearance.png and, if the crops are present,
    3_exposure.png and the crop features (3_crop_features.*); prints medians, pink shares,
    the colour-only baseline, the exposure and the crop feature baselines."""
    from .crop_features import dataset_features

    t = appearance_tables(ds)
    base = colour_baseline(ds)
    exp = crop_exposure(ds, out.parent)

    summary = pd.DataFrame({"balanced_accuracy": [round(base["balanced_accuracy"], 3)],
                            "per_fold": [", ".join(map(str, base["balanced_accuracy_per_fold"]))],
                            "chance": [round(base["chance"], 3)]})
    with pd.ExcelWriter(out / "3_appearance.xlsx") as xw:
        t["quantiles"].to_excel(xw, sheet_name="quantiles")
        summary.to_excel(xw, sheet_name="colour_baseline", index=False)
        base["confusion_pct"].to_excel(xw, sheet_name="colour_baseline_confusion")

        if exp is not None:
            e = exp.assign(training_class=ds.training_class)
            exp_q = e.groupby("training_class")[["background_grey", "saturated_pct"]].quantile([0.25, 0.5, 0.75]).unstack()
            exp_q.columns = [f"{c}_p{int(qq * 100)}" for c, qq in exp_q.columns]
            exp_q.round(2).to_excel(xw, sheet_name="exposure")

    appearance_figure(ds, base, out / "3_appearance.png")

    q = t["quantiles"]
    log("3. Optical appearance")
    log(q[["n", "pink_pct", "vhx_contrast_max_p25", "vhx_contrast_max_p50", "vhx_contrast_max_p75",
           "pinkness_p50"]].to_string())
    log(f"colour-only baseline (balanced accuracy, chance {100 * base['chance']:.0f} %): "
        f"{100 * base['balanced_accuracy']:.0f} %  (folds: {', '.join(f'{100 * v:.0f}' for v in base['balanced_accuracy_per_fold'])})")
    log("confusion of the colour baseline (% of each true class):")
    log(base["confusion_pct"].to_string())

    if exp is None:
        log("crops not found - background brightness, overexposure and crop features skipped")
        return {"tables": t, "baseline": base, "exposure": None, "crop_features": None}

    exposure_figure(ds, exp, out / "3_exposure.png")
    e = exp.assign(training_class=ds.training_class).groupby("training_class").median().round(1)
    log("crops, median per class: background grey level and saturated pixels (%)")
    log(e.to_string())

    feats = dataset_features(ds, out.parent)
    cf = crop_feature_part(ds, feats, base, out, log)

    return {"tables": t, "baseline": base, "exposure": exp, "crop_features": cf}
