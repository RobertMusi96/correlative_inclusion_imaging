"""Export the training dataset: VHX crop + mask + training class per SEM particle.

Inputs:
    --matches   matches.csv of match_sem_vhx.py
    --classes   class file (classes_vN.xlsx: Type table, rules, settings)
    --vhx       detection folder of detect_inclusions.py (objects.csv, images.csv, labels/)
    --vhx-images  folder with the VHX images, if they moved (default: paths in images.csv)

Output folder (default: dataset_vN next to classes_vN.xlsx):
    crops/<sample>_<sem_id>.png   VHX crop (RGB, crop_px x crop_px, full resolution)
    masks/<sample>_<sem_id>.png   mask of the particle (255 = particle)
    dataset.csv                   one row per crop: files, ids, training class and its source,
                                  match info, crop position, sizes, VHX colour, EDS, split, fold
    classes_vN.xlsx               copy of the class file used
    dataset_info.json             inputs, parameters, counts

Split: 'test' (held out, ~20 %) or 'trainval'; 'fold' 0..4 for trainval particles
(cross-validation: one fold validation, the others training), -1 for test. Both by
stratified group k-fold on spatial blocks (see cii.dataset).

Example:
    python scripts/export_dataset.py --matches data/results/sem_match_500x/matches.csv ^
        --classes data/datasets/classes_v1.xlsx --vhx data/results/.../detection_500x --sample MnB
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
from datetime import datetime
from pathlib import Path

import pandas as pd

from cii.classes import apply_classes, read_class_table, read_rules, read_settings
from cii.dataset import ExportParams, assign_splits, export_dataset, params_dict, select_particles, spatial_groups
from cii.matching_figures import resolve_vhx_paths


def default_output(classes: Path) -> Path:
    """dataset_vN next to classes_vN.xlsx (dataset_<name> for other file names)."""
    m = re.match(r"classes_(.+)$", classes.stem)
    return classes.parent / f"dataset_{m.group(1) if m else classes.stem}"


def main():
    """Command line: selects the particles with the class file, writes crops, masks, the split
    and dataset.csv, copies the class file and prints the class distribution per split and fold."""
    d = ExportParams()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--matches", type=Path, required=True)
    ap.add_argument("--classes", type=Path, required=True)
    ap.add_argument("--vhx", type=Path, required=True, help="detection folder (objects.csv, images.csv, labels/)")
    ap.add_argument("--vhx-images", type=Path, help="folder with the VHX images, if they moved")
    ap.add_argument("--sample", required=True, help="sample name, used in the file names (e.g. MnB)")
    ap.add_argument("-o", "--output", type=Path, help="output folder (default: dataset_vN next to the class file)")
    ap.add_argument("--crop-px", type=int, default=d.crop_px, help=f"crop size in pixels (default {d.crop_px})")
    ap.add_argument("--test-fraction", type=float, default=d.test_fraction)
    ap.add_argument("--folds", type=int, default=d.n_folds)
    ap.add_argument("--block-um", type=float, default=d.block_um, help="block size for the grouped split, um")
    ap.add_argument("--seed", type=int, default=d.seed)
    ap.add_argument("--overwrite", action="store_true", help="replace an existing output folder")
    args = ap.parse_args()

    out = args.output or default_output(args.classes)
    if out.exists():
        if not args.overwrite:
            raise SystemExit(f"{out} exists - give another -o or --overwrite")
        shutil.rmtree(out)
    p = ExportParams(crop_px=args.crop_px, test_fraction=args.test_fraction, n_folds=args.folds,
                     block_um=args.block_um, seed=args.seed)

    matches = pd.read_csv(args.matches, low_memory=False)
    table, rules, settings = read_class_table(args.classes), read_rules(args.classes), read_settings(args.classes)
    assigned = apply_classes(matches, table, rules, settings)
    parts = select_particles(matches, assigned)
    print(f"{len(parts)} particles selected (training_ok, training class after table, {len(rules)} rule(s), "
          f"min_ecd_um {settings['min_ecd_um']:g})")

    vo, vi = pd.read_csv(args.vhx / "objects.csv"), pd.read_csv(args.vhx / "images.csv")
    if "region" not in vo:
        raise SystemExit("objects.csv has no 'region' column - rerun detect_inclusions.py (version >= 0.3.0)")
    vpaths = resolve_vhx_paths(vi, args.vhx_images)
    print(f"VHX images found: {len(vpaths)} of {len(vi)}")
    ds = export_dataset(parts, vo, vi, vpaths, args.vhx / "labels", out, args.sample, p)
    if not len(ds):
        raise SystemExit("no crops written")

    groups = spatial_groups(ds.vhx_x_um.to_numpy(), ds.vhx_y_um.to_numpy(), p.block_um)
    ds["block"] = groups
    ds["split"], ds["fold"] = assign_splits(ds.training_class.to_numpy(), groups, p)
    ds.to_csv(out / "dataset.csv", index=False, float_format="%.4f")
    shutil.copy2(args.classes, out / args.classes.name)

    dist = pd.crosstab(ds.training_class, [ds.split, ds.fold.where(ds.split == "trainval", -1)])
    dist.columns = ["test" if s == "test" else f"fold {f}" for s, f in dist.columns]
    dist["total"] = dist.sum(axis=1)
    info = {"created": datetime.now().isoformat(timespec="seconds"), "sample": args.sample,
            "matches": str(args.matches), "classes": str(args.classes), "vhx_detection": str(args.vhx),
            "params": params_dict(p), "n_crops": len(ds), "n_blocks": int(ds.block.nunique()),
            "classes_count": ds.training_class.value_counts().to_dict(),
            "split_count": ds.split.value_counts().to_dict()}
    (out / "dataset_info.json").write_text(json.dumps(info, indent=2))
    print(f"{len(ds)} crops of {p.crop_px} px ({p.crop_px * ds.um_per_px.median():.1f} um), "
          f"{ds.block.nunique()} blocks of {p.block_um:g} um")
    print(dist.to_string())
    print(f"-> {out}")


if __name__ == "__main__":
    main()
