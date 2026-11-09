"""Statistical overview of a training dataset.

Reads dataset_vN/dataset.csv and writes tables and figures into dataset_vN/report/,
one file group per section (prefix = section number):

    1  composition and balance: 1_composition.xlsx, 1_composition.png, 1_origin.png
    2  size: 2_size.xlsx, 2_size.png (size per class, VHX/SEM ratio, size-only baseline)
    3  optical appearance: 3_appearance.xlsx, 3_appearance.png, 3_exposure.png (contrast,
       colour, pinkness, colour-only baseline; background and overexposure from the crops),
       3_crop_features.csv/.xlsx/.png, 3_crop_features_top.png (shape, interior and edge
       features from the crops, baselines adding them, best features for the confused pair)
    4  label quality: 4_label_quality.xlsx, 4_label_quality.png, 4_chemistry.png,
       4_model_by_origin.png, 4_suspicious.csv (match warnings, EDS per class and AZtec class,
       model errors per origin and match warning, particles worth a look)

Example:
    python scripts/dataset_report.py data/datasets/dataset_v1
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from cii.report import section_appearance, section_composition, section_label_quality, section_size

SECTIONS = {1: section_composition, 2: section_size, 3: section_appearance, 4: section_label_quality}


def main():
    """Command line: runs the chosen sections (default all) on a dataset folder."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset", type=Path, help="dataset folder (with dataset.csv)")
    ap.add_argument("--sections", default=",".join(map(str, SECTIONS)),
                    help=f"sections to run, e.g. 1 or 1,2 (default all: {','.join(map(str, SECTIONS))})")
    args = ap.parse_args()

    csv = args.dataset / "dataset.csv"
    if not csv.exists():
        raise SystemExit(f"{csv} not found")
    ds = pd.read_csv(csv, low_memory=False)
    out = args.dataset / "report"
    out.mkdir(exist_ok=True)

    print(f"{args.dataset.name}: {len(ds)} particles, {ds.training_class.nunique()} classes")
    for k in (int(s) for s in args.sections.split(",")):
        print()
        SECTIONS[k](ds, out)
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
