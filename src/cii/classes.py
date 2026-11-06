"""Training classes from a class table (one row per SEM Type).

The table (``classes.xlsx``) lists every SEM Type of the typified list plus
one row each for not typified and rejected particles, with counts and a typical
composition. The user fills in the column ``training_class``; rows left empty
are not used for training. ``assign_training_class`` adds the column to a
particle table (matches.csv) for the dataset export.

Key of a row: ``class_label`` (Class for typified particles, otherwise
'not_typified' / 'rejected') and ``Type`` (empty for the latter two).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

KEY = ["class_label", "Type"]
USER_COLUMNS = ["training_class", "comment"]


def _key(df: pd.DataFrame) -> pd.Series:
    t = df["Type"].where(df["Type"].notna() & (df["Type"].astype(str) != "nan"), "")
    return df["class_label"].astype(str) + "|" + t.astype(str)


def typical_composition(g: pd.DataFrame, n: int = 5) -> str:
    """Median wt% of the main elements (Fe left out: mostly matrix), e.g. 'Ti 61, N 19, Nb 6'."""
    cols = [c for c in g.columns if c.endswith("_wt") and c != "Fe_wt"]
    med = g[cols].median().sort_values(ascending=False)
    med = med[med >= 1.0].head(n)
    return ", ".join(f"{c[:-3]} {v:.0f}" for c, v in med.items())


def make_class_table(particles: pd.DataFrame, previous: pd.DataFrame | None = None) -> pd.DataFrame:
    """One row per (class_label, Type) with counts and composition; user columns carried
    over from ``previous`` (an earlier class table) by key."""
    p = particles.copy()
    p["Type"] = p["Type"].where(p["Type"].notna(), "")
    inside = p.status.isin(["matched", "ambiguous", "not_found"])
    found = p.status.isin(["matched", "ambiguous"])
    rows = []
    for (cl, ty), g in p.groupby(KEY, sort=False):
        gi = inside[g.index]
        rows.append({
            "class_label": cl, "Type": ty,
            "n_particles": len(g),
            "n_in_vhx_area": int(gi.sum()),
            "found_pct": round(100 * found[g.index][gi].mean(), 1) if gi.any() else np.nan,
            "n_training_ok": int(g.training_ok.sum()) if "training_ok" in g else np.nan,
            "median_ecd_um": round(float(g.ecd_um.median()), 2),
            "median_Fe_wt": round(float(g.Fe_wt.median()), 1) if "Fe_wt" in g else np.nan,
            "composition_median_wt": typical_composition(g),
        })
    t = pd.DataFrame(rows)
    order = {c: k for k, c in enumerate(["N", "NS", "O", "ONS", "OS", "S", "not_typified", "rejected"])}
    t["_o"] = t.class_label.map(order).fillna(len(order))
    t = t.sort_values(["_o", "class_label", "n_particles"], ascending=[True, True, False]).drop(columns="_o")
    for c in USER_COLUMNS:
        t[c] = ""
    if previous is not None and len(previous):
        prev = previous.copy()
        prev["Type"] = prev["Type"].where(prev["Type"].notna(), "")
        prev = prev.set_index(_key(prev))
        k = _key(t)
        for c in USER_COLUMNS:
            if c in prev:
                v = k.map(prev[c].where(prev[c].notna(), ""))
                t[c] = v.where(v.notna(), "").astype(str).replace("nan", "")
    return t.reset_index(drop=True)


def write_class_table(table: pd.DataFrame, path: Path):
    from openpyxl.styles import PatternFill
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(path, engine="openpyxl") as xw:
        table.to_excel(xw, sheet_name="classes", index=False)
        ws = xw.sheets["classes"]
        ws.freeze_panes = "C2"
        fill = PatternFill("solid", fgColor="FFF2CC")
        for j, name in enumerate(table.columns, start=1):
            letter = ws.cell(row=1, column=j).column_letter
            width = max(len(str(name)), *(len(str(v)) for v in table[name].head(300))) + 2 if len(table) else len(str(name)) + 2
            ws.column_dimensions[letter].width = min(max(width, 9), 45)
            if name in USER_COLUMNS:
                for i in range(1, len(table) + 2):
                    ws.cell(row=i, column=j).fill = fill
        info = xw.book.create_sheet("help")
        for line in [
            ["One row per SEM Type (Class/Type from *_pre_corr_typified.xlsx); not typified and rejected particles one row each."],
            ["Fill in 'training_class' (yellow): the class used for training. Several Types can get the same class."],
            ["Rows with an empty 'training_class' are left out of the training data."],
            ["n_training_ok: particles usable for training (matched, not cluster). found_pct: share with a VHX object."],
            ["composition_median_wt: median wt% of the main elements, Fe left out (mostly matrix)."],
            ["Rerunning make_class_table.py with --previous keeps the entered classes and adds new Types."],
        ]:
            info.append(line)
        info.column_dimensions["A"].width = 120


def read_class_table(path: Path) -> pd.DataFrame:
    t = pd.read_excel(path, sheet_name="classes", dtype={"Type": str, "training_class": str, "comment": str})
    t["Type"] = t["Type"].where(t["Type"].notna(), "")
    t["training_class"] = t["training_class"].where(t["training_class"].notna(), "").astype(str).str.strip()
    return t


def assign_training_class(particles: pd.DataFrame, table: pd.DataFrame) -> pd.Series:
    """training_class per particle from the class table ('' = not used for training).
    Types missing from the table get '' as well (and are reported by the caller)."""
    p = particles.copy()
    p["Type"] = p["Type"].where(p["Type"].notna(), "")
    m = table.set_index(_key(table)).training_class
    return _key(p).map(m).fillna("").astype(str).rename("training_class")
