"""Fit the transformation between two stage coordinate systems from matched indents.

Input: two CSV files from ``locate_indents.py`` (columns label, stage_x_um,
stage_y_um); indents are matched by label. Reports the fitted parameters, the
residual of every indent and the leave-one-out error (each indent predicted
from the others - the honest accuracy estimate), and writes the transform as
JSON for later use (``cii.registration.Transform.from_json``).

Example:
    python scripts/fit_transform.py vhx_indents.csv sem_indents.csv -o vhx_to_sem.json
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from cii.registration import fit_affine, fit_similarity, leave_one_out, residuals


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src_csv", type=Path, help="indents in the source system (e.g. VHX)")
    ap.add_argument("dst_csv", type=Path, help="indents in the target system (e.g. SEM)")
    ap.add_argument("-o", "--output", type=Path, default=Path("transform.json"))
    ap.add_argument("--model", choices=["similarity", "affine"], default="similarity")
    args = ap.parse_args()

    a, b = pd.read_csv(args.src_csv), pd.read_csv(args.dst_csv)
    m = a.merge(b, on="label", suffixes=("_src", "_dst"))
    if len(m) < 2:
        raise SystemExit(f"only {len(m)} indent(s) matched by label")
    src = m[["stage_x_um_src", "stage_y_um_src"]].to_numpy()
    dst = m[["stage_x_um_dst", "stage_y_um_dst"]].to_numpy()
    src_name = str(m["instrument_src"].iloc[0]) if "instrument_src" in m else args.src_csv.stem
    dst_name = str(m["instrument_dst"].iloc[0]) if "instrument_dst" in m else args.dst_csv.stem

    T = (fit_similarity if args.model == "similarity" else fit_affine)(src, dst)
    T.src, T.dst = src_name, dst_name
    res = residuals(T, src, dst)
    report = pd.DataFrame({"label": m["label"], "residual_x_um": res[:, 0], "residual_y_um": res[:, 1],
                           "residual_um": np.hypot(res[:, 0], res[:, 1])})
    if len(m) >= (3 if args.model == "similarity" else 4):
        loo = leave_one_out(src, dst, args.model)
        report["leave_one_out_um"] = np.hypot(loo[:, 0], loo[:, 1])

    print(f"{args.model} {src_name} -> {dst_name} from {len(m)} indents")
    print(f"  scale {T.scale:.5f}  rotation {T.rotation_deg:+.3f} deg  mirrored X: {T.mirrored}")
    print(f"  A = {np.array2string(T.A, precision=6)}\n  t = {np.array2string(T.t, precision=2)} um")
    print(report.round(2).to_string(index=False))
    print(f"  RMS residual {np.sqrt((report.residual_um ** 2).mean()):.2f} um")
    T.to_json(args.output)
    print(f"written: {args.output}")


if __name__ == "__main__":
    main()
