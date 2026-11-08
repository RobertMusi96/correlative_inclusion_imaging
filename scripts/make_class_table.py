"""Class table for the training classes: one row per SEM Type, to be filled in.

Reads matches.csv (match_sem_vhx.py) and writes classes.xlsx with one row per
SEM Type (plus not typified and rejected), counts and the typical composition.
Fill in the yellow column 'training_class'; Types with the same class are
merged, empty rows are left out of the training data.

Example:
    python scripts/make_class_table.py results/match_500x/matches.csv -o classes/classes_MnB.xlsx
    # later (new sample or new matching), keeping the classes already entered:
    python scripts/make_class_table.py results/match_500x/matches.csv -o classes/classes_v2.xlsx \
        --previous classes/classes_MnB.xlsx

Several matches.csv (several samples) can be given; the table then covers all of them.
The file also holds the sheets 'rules' (corrections by composition, e.g. rejected
with Ti_wt >= 1 -> TiN), 'settings' (min_ecd_um) and 'summary' (what all of this
does to the original classes); --previous keeps rules and settings. Editing them
in Excel and rerunning with --previous refreshes the summary.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from cii.classes import (apply_classes, class_summary, make_class_table, read_class_table, read_rules,
                         read_settings, write_class_table)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("matches", type=Path, nargs="+", help="matches.csv of match_sem_vhx.py")
    ap.add_argument("-o", "--output", type=Path, default=Path("classes.xlsx"))
    ap.add_argument("--previous", type=Path, help="earlier class table: its training_class/comment are kept")
    args = ap.parse_args()

    parts = pd.concat([pd.read_csv(f, low_memory=False) for f in args.matches], ignore_index=True)
    prev = read_class_table(args.previous) if args.previous else None
    rules = read_rules(args.previous) if args.previous else None
    settings = read_settings(args.previous) if args.previous else None
    if args.output.exists() and (args.previous is None or args.output.resolve() != args.previous.resolve()):
        raise SystemExit(f"{args.output} exists - give another -o, or --previous {args.output} to update it")
    t = make_class_table(parts, prev)
    assigned = apply_classes(parts, t, rules, settings)
    summ = class_summary(parts, assigned)
    try:
        write_class_table(t, args.output, rules, settings, summ)
    except PermissionError:
        raise SystemExit(f"{args.output} is open in another program - close it and run again")
    filled = (t.training_class != "").sum()
    print(f"{len(t)} rows ({(t.class_label.isin(['not_typified', 'rejected'])).sum()} not typified / rejected), "
          f"{int(t.n_particles.sum())} particles; training_class filled: {filled}")
    if prev is not None:
        new = t[t.training_class == ""]
        print(f"carried over from {args.previous.name}: {filled}; rows still empty: {len(new)}")
    if rules is not None and len(rules):
        print(f"rules: {len(rules)}; settings: {settings}")
    print("training_ok particles, original class (rows) -> training class (columns):")
    print(summ["overview"].to_string())
    print(f"-> {args.output}")


if __name__ == "__main__":
    main()
