"""Training classes from a class table (one row per SEM Type).

The table (``classes.xlsx``) lists every SEM Type of the typified list plus
one row each for not typified and rejected particles, with counts and a typical
composition. The user fills in the column ``training_class``; rows left empty
are not used for training. ``assign_training_class`` adds the column to a
particle table (matches.csv) for the dataset export.

Key of a row: ``class_label`` (Class for typified particles, otherwise
'not_typified' / 'rejected') and ``Type`` (empty for the latter two).

Besides the Type table the file can hold

* sheet ``rules``: corrections applied after the Type table, in order. Each rule
  selects particles by ``class_label`` / ``Type`` (empty = all) and a
  ``condition`` on the particle table, e.g. ``Ti_wt >= 1`` or
  ``Ti_wt >= 1 and ecd_um < 3`` (operators >= <= > < == !=, joined by 'and'),
  and sets their ``training_class`` (empty = not used for training).
* sheet ``settings``: ``min_ecd_um`` (SEM ECD; smaller particles not used).

``apply_classes`` returns training_class and ``class_source`` per particle
(type table / rule <id> / below min_ecd / no class); ``class_summary`` counts
original class -> training class (printed by make_class_table.py).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

KEY = ["class_label", "Type"]
USER_COLUMNS = ["training_class", "comment"]


def _key(df: pd.DataFrame) -> pd.Series:
    """Row key 'class_label|Type' (Type empty for not typified / rejected), used to look up
    a particle's row in the class table and to carry entries over between table versions."""
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


def write_class_table(table: pd.DataFrame, path: Path, rules: pd.DataFrame | None = None,
                      settings: dict | None = None):
    """Writes the class file: sheet 'classes' (Type table, user columns marked yellow),
    'rules' (empty template if no rules), 'settings' (min_ecd_um) and 'help'."""
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
        rl = rules if rules is not None and len(rules) else pd.DataFrame(columns=RULE_COLUMNS)
        rl.reindex(columns=RULE_COLUMNS).to_excel(xw, sheet_name="rules", index=False)
        ws = xw.sheets["rules"]
        for letter, w in zip("ABCDEF", [8, 14, 18, 30, 22, 70]):
            ws.column_dimensions[letter].width = w
        st = {**DEFAULT_SETTINGS, **(settings or {})}
        pd.DataFrame({"setting": list(st), "value": list(st.values()),
                      "meaning": ["SEM ECD in um; smaller particles are not used for training (0 = all)"]}
                     ).to_excel(xw, sheet_name="settings", index=False)
        xw.sheets["settings"].column_dimensions["C"].width = 70
        info = xw.book.create_sheet("help")
        for line in [
            ["One row per SEM Type (Class/Type from *_pre_corr_typified.xlsx); not typified and rejected particles one row each."],
            ["Fill in 'training_class' (yellow): the class used for training. Several Types can get the same class."],
            ["Rows with an empty 'training_class' are left out of the training data."],
            ["n_training_ok: particles usable for training (matched, not cluster). found_pct: share with a VHX object."],
            ["composition_median_wt: median wt% of the main elements, Fe left out (mostly matrix)."],
            ["Rerunning make_class_table.py with --previous keeps the entered classes, rules and settings and adds new Types."],
            ["Sheet 'rules': corrections after the Type table, applied in order (later rules win). class_label / Type"],
            ["  select the particles (empty = all), condition e.g. 'Ti_wt >= 1' or 'Ti_wt >= 1 and ecd_um < 3'"],
            ["  (columns of matches.csv; operators >= <= > < == != joined by 'and'); training_class empty = not used."],
            ["Sheet 'settings': min_ecd_um = smallest SEM ECD used for training (applied last)."],
        ]:
            info.append(line)
        info.column_dimensions["A"].width = 120


def read_class_table(path: Path) -> pd.DataFrame:
    """Reads sheet 'classes' of a class file; empty Type and training_class cells become ''."""
    t = pd.read_excel(path, sheet_name="classes", dtype={"Type": str, "training_class": str, "comment": str})
    t["Type"] = t["Type"].where(t["Type"].notna(), "")
    t["training_class"] = t["training_class"].where(t["training_class"].notna(), "").astype(str).str.strip()
    return t


RULE_COLUMNS = ["rule", "class_label", "Type", "condition", "training_class", "comment"]
DEFAULT_SETTINGS = {"min_ecd_um": 0.0}
_OPS = [">=", "<=", "!=", "==", ">", "<"]


def read_rules(path: Path) -> pd.DataFrame:
    """Sheet 'rules' (empty table if missing). Rows without a rule id are ignored."""
    try:
        r = pd.read_excel(path, sheet_name="rules", dtype=str)
    except ValueError:
        return pd.DataFrame(columns=RULE_COLUMNS)
    r = r.reindex(columns=RULE_COLUMNS).fillna("")
    r = r.apply(lambda c: c.astype(str).str.strip())
    return r[r.rule != ""].reset_index(drop=True)


def read_settings(path: Path) -> dict:
    """Reads sheet 'settings' (setting / value); missing sheet or settings keep their defaults."""
    out = dict(DEFAULT_SETTINGS)
    try:
        s = pd.read_excel(path, sheet_name="settings", dtype={"setting": str})
    except ValueError:
        return out
    for r in s.itertuples():
        if isinstance(r.setting, str) and r.setting.strip() in out and pd.notna(r.value):
            out[r.setting.strip()] = float(r.value)
    return out


def condition_mask(particles: pd.DataFrame, condition: str) -> pd.Series:
    """Mask of a condition like 'Ti_wt >= 1 and ecd_um < 3' (no arbitrary code is evaluated)."""
    mask = pd.Series(True, index=particles.index)
    if not condition.strip():
        return mask
    for term in condition.split(" and "):
        term = term.strip()
        op = next((o for o in _OPS if o in term), None)
        if op is None:
            raise ValueError(f"condition '{term}': no operator ({' '.join(_OPS)})")
        col, val = (x.strip() for x in term.split(op, 1))
        if col not in particles:
            raise ValueError(f"condition '{term}': unknown column '{col}' (e.g. Ti_wt, N_wt, ecd_um)")
        x = particles[col]
        try:
            v = float(val)
        except ValueError:
            v = val.strip("'\"")
            x = x.astype(str)
        mask &= {">=": x >= v, "<=": x <= v, ">": x > v, "<": x < v, "==": x == v, "!=": x != v}[op].fillna(False)
    return mask


def apply_classes(particles: pd.DataFrame, table: pd.DataFrame, rules: pd.DataFrame | None = None,
                  settings: dict | None = None) -> pd.DataFrame:
    """training_class and class_source per particle: Type table, then the rules in order,
    then min_ecd_um. Returns a DataFrame with the particle index."""
    p = particles.copy()
    p["Type"] = p["Type"].where(p["Type"].notna(), "")
    tc = assign_training_class(p, table)
    src = pd.Series(np.where(tc != "", "type table", "no class"), index=p.index, dtype=object)
    for r in (rules if rules is not None else pd.DataFrame(columns=RULE_COLUMNS)).itertuples():
        sel = condition_mask(p, r.condition)
        if r.class_label:
            sel &= p.class_label.astype(str) == r.class_label
        if r.Type:
            sel &= p.Type.astype(str) == r.Type
        tc[sel] = r.training_class
        src[sel] = f"rule {r.rule}"
    st = {**DEFAULT_SETTINGS, **(settings or {})}
    small = p.ecd_um < st["min_ecd_um"]
    tc[small] = ""
    src[small] = f"below min_ecd_um {st['min_ecd_um']:g}"
    return pd.DataFrame({"training_class": tc, "class_source": src}, index=p.index)


def class_summary(particles: pd.DataFrame, assigned: pd.DataFrame, only_training_ok: bool = True) -> dict:
    """Overview tables: 'overview' (original class -> training class, counts), 'steps'
    (particles per class_source and training class)."""
    p = particles.join(assigned)
    if only_training_ok and "training_ok" in p:
        p = p[p.training_ok]
    tcl = p.training_class.replace("", "(not used)")
    ov = pd.crosstab(p.class_label, tcl, margins=True, margins_name="total")
    steps = pd.crosstab(p.class_source, tcl, margins=True, margins_name="total")
    return {"overview": ov, "steps": steps}


def assign_training_class(particles: pd.DataFrame, table: pd.DataFrame) -> pd.Series:
    """training_class per particle from the class table ('' = not used for training).
    Types missing from the table get '' as well (and are reported by the caller)."""
    p = particles.copy()
    p["Type"] = p["Type"].where(p["Type"].notna(), "")
    m = table.set_index(_key(table)).training_class
    return _key(p).map(m).fillna("").astype(str).rename("training_class")
