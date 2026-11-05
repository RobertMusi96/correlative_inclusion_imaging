"""Match SEM/EDS particles (AZtec feature analysis) to VHX detections.

Inputs:
    --sem        folder with the AZtec exports: *_pre.xlsx, *_pre_corr.xlsx (optional),
                 *_pre_rejected.xlsx, *_pre_corr_typified.xlsx, *_pre_corr_not-typified.xlsx
                 and the field table (*field_information*.csv, or --fields)
    --transform  VHX -> SEM transform from the indents (fit_transform.py)
    --vhx        output folder of detect_inclusions.py (objects.csv + images.csv)
    --bse        folder with the BSE field images, for the figures; searched with subfolders,
                 site number from the path ('Site 135' folder or 'Site_135' in the name),
                 only TIFFs with 'electron' in name or path

Example:
    python scripts/match_sem_vhx.py --sem "D:\\SEM\\MnB" --transform results\\vhx_to_sem.json ^
        --vhx results\\detection_500x_max --bse "D:\\SEM\\MnB\\images" -o results\\match_500x

Outputs in the output folder:
    matches.csv            one row per SEM particle: label, Class, Type, status, matched VHX object
    summary_by_class.csv   share with VHX object per class / label
    summary_by_size.csv    the same per SEM ECD class (typified / not typified / rejected)
    model.json             position model (offset, trend, SEM field scale/rotation, residual)
    history.csv            matches per iteration
    overview.png           share found per class and size
    verify/<class>.png     BSE | VHX crops for a sample of matched, ambiguous and not found particles
Status: matched, ambiguous, not_found, outside (no VHX image), sem_border, sem_duplicate.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from cii.matching import MatchParams, find_sem_files, load_sem_particles, match
from cii.matching_figures import bse_paths, overview, resolve_vhx_paths, verification_sheet
from cii.registration import Transform
from cii.sem import read_field_positions


def summaries(part: pd.DataFrame):
    order = ["matched", "ambiguous", "not_found", "outside", "sem_border", "sem_duplicate"]
    t = pd.crosstab(part.class_label, part.status).reindex(columns=order, fill_value=0)
    inside = t[["matched", "ambiguous", "not_found"]].sum(axis=1)
    t["found_pct"] = (100 * (t.matched + t.ambiguous) / inside.where(inside > 0)).round(1)
    m = part[part.status == "matched"]
    t["median_distance_um"] = m.groupby("class_label").distance_um.median().round(2)
    t["median_size_ratio"] = m.groupby("class_label").size_ratio.median().round(2)
    c = part[part.status.isin(["matched", "ambiguous", "not_found"])]
    bins = [1, 1.5, 2, 3, 5, 10, 1000]
    s = (c.groupby(["label", pd.cut(c.ecd_um, bins)], observed=True).status
         .agg(n="size", found_pct=lambda x: round(100 * (x != "not_found").mean(), 1)).reset_index())
    s["ecd_um"] = s.ecd_um.astype(str)
    return t, s


def main():
    d = MatchParams()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sem", type=Path, required=True)
    ap.add_argument("--fields", type=Path, help="field table (default: *field_information*.csv in --sem)")
    ap.add_argument("--transform", type=Path, required=True)
    ap.add_argument("--vhx", type=Path, required=True)
    ap.add_argument("--bse", type=Path, help="BSE field images, for the verification figures")
    ap.add_argument("--vhx-images", type=Path,
                    help="folder with the VHX images (searched with subfolders), if they are no longer at the "
                         "paths stored in images.csv")
    ap.add_argument("-o", "--output", type=Path, default=Path("match"))
    ap.add_argument("--figures", type=int, default=12, help="particles per class and status in the figures (0 = none)")
    ap.add_argument("--only-available", action="store_true",
                    help="figures only for particles whose BSE and VHX images are present")
    ap.add_argument("--tolerance", type=float, default=d.tolerance_um, help=f"match radius, um (default {d.tolerance_um})")
    ap.add_argument("--size-weight", type=float, default=d.size_weight_um,
                    help=f"score per e-fold size difference, um (default {d.size_weight_um})")
    ap.add_argument("--iterations", type=int, default=d.iterations)
    ap.add_argument("--sem-px", type=float, default=0.294864, help="SEM pixel size, um (default 0.294864)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    files = find_sem_files(args.sem)
    fields_csv = args.fields or next(iter(sorted(args.sem.glob("*field_information*.csv"))), None)
    if fields_csv is None:
        raise SystemExit("field table not found - give --fields")
    sem = load_sem_particles(files)
    fields = read_field_positions(fields_csv)
    T = Transform.from_json(args.transform)
    vo, vi = pd.read_csv(args.vhx / "objects.csv"), pd.read_csv(args.vhx / "images.csv")
    p = MatchParams(tolerance_um=args.tolerance, size_weight_um=args.size_weight, iterations=args.iterations)
    print(f"SEM: {len(sem)} particles ({', '.join(f'{k} {v}' for k, v in sem.label.value_counts().items())}); "
          f"VHX: {len(vo)} objects in {len(vi)} images")
    r = match(sem, fields, T, vo, vi, sem_um_per_px=args.sem_px, params=p)
    part = r.particles
    args.output.mkdir(parents=True, exist_ok=True)
    part.to_csv(args.output / "matches.csv", index=False, float_format="%.4f")
    by_class, by_size = summaries(part)
    by_class.to_csv(args.output / "summary_by_class.csv")
    by_size.to_csv(args.output / "summary_by_size.csv", index=False)
    pd.DataFrame(r.history).to_csv(args.output / "history.csv", index=False)
    r.offsets.to_csv(args.output / "first_offsets.csv", index=False, float_format="%.3f")
    M = r.model
    model = {"coarse_offset_um": r.coarse_offset.round(3).tolist(), "coarse_votes": r.coarse_votes}
    if M:
        model.update({"offset_um": M["c"].round(3).tolist(), "centre_um": M["centre"].round(1).tolist(),
                      "trend_um_per_mm": M["G"].round(4).tolist(), "sem_field_um_per_mm": M["M"].round(4).tolist(),
                      "residual_sd_um": M["residual_sd"].round(3).tolist(), "fit_matches": M["n_used"],
                      "equation": "VHX = SEM_in_VHX + offset + trend @ (SEM_in_VHX - centre)/1000 "
                                  "+ sem_field @ position_in_field_mm + field_error - vhx_image_error"})
    (args.output / "model.json").write_text(json.dumps(model, indent=2))

    print(f"coarse offset {r.coarse_offset.round(1)} um (objective of the series vs. indents)")
    print(pd.DataFrame(r.history).to_string(index=False))
    if M:
        print(f"residual sd of the matches: {M['residual_sd'].round(2)} um; SEM field scale/rotation "
              f"(um per mm): {M['M'].round(2).tolist()}")
    print(by_class.to_string())
    print(f"VHX objects shared by several SEM particles: {int(part.vhx_shared.sum())} SEM particles")
    overview(part, args.output / "overview.png")

    if args.figures and args.bse:
        if not args.bse.is_dir():
            raise SystemExit(f"BSE folder not found: {str(args.bse)!r} (paths with spaces in quotes)")
        bse = bse_paths(args.bse)
        rng = np.random.default_rng(args.seed)
        sel = part[part.status.isin(["matched", "ambiguous", "not_found"])]
        if args.vhx_images is not None and not args.vhx_images.is_dir():
            raise SystemExit(f"VHX image folder not found: {str(args.vhx_images)!r} (paths with spaces in quotes)")
        vpaths = resolve_vhx_paths(vi, args.vhx_images)
        print(f"VHX images: {len(vpaths)} of {len(vi)} found"
              + (f" in {args.vhx_images}" if args.vhx_images else " at the paths in images.csv"))
        if len(vpaths) < len(vi):
            missing = [pth for n, pth in zip(vi.image, vi.path) if n not in vpaths]
            print(f"  missing e.g. {missing[0]}")
            if not args.vhx_images:
                print("  -> the images were moved or renamed after the detection: give their folder with --vhx-images")
        if args.only_available:
            sel = sel[sel.field.isin(bse) & sel.vhx_image.isin(vpaths)]
        for cls, g in sel.groupby("class_label"):
            rows = []
            for st in ["matched", "ambiguous", "not_found"]:
                s = g[g.status == st]
                if len(s):
                    rows.append(s.iloc[rng.permutation(len(s))[: args.figures]].sort_values("ecd_um", ascending=False))
            rows = pd.concat(rows)
            fn = args.output / "verify" / f"{cls}.png"
            if verification_sheet(rows, vi, bse, fn, f"{cls}: {len(g)} particles", sem_um_per_px=args.sem_px,
                                  vhx_paths=vpaths):
                print(f"  {fn}")
    print(f"-> {args.output}")


if __name__ == "__main__":
    main()
