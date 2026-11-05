"""Compare detections of the same sample area at two magnifications.

Objects of the higher magnification (reference) are searched among the
objects of the lower magnification by stage coordinates. A constant offset
between the objectives is estimated first (vote of all displacement vectors,
then median of the matched pairs). Reports, per reference size class, the share
found at the lower magnification (detection limit) and the size ratio.

Example:
    python scripts/compare_magnifications.py results/detection_1000x/objects.csv \
        results/detection_500x/objects.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


def estimate_offset(ref: np.ndarray, other: np.ndarray, search_um: float = 40, bin_um: float = 1.0,
                    refine_um: float = 1.5, iterations: int = 3) -> np.ndarray:
    """Offset to add to ref coordinates so that they land on other coordinates.

    Vote of all displacement vectors within search_um (2D histogram peak), then
    iterative refinement: median displacement of the pairs within refine_um.
    Pass well-defined objects (e.g. ECD >= 0.8 um) for a robust estimate.
    """
    tree = cKDTree(other)
    vec = np.array([other[j] - ref[k] for k, js in enumerate(tree.query_ball_point(ref, r=search_um)) for j in js])
    if not len(vec):
        return np.zeros(2)
    nb = int(2 * search_um / bin_um)
    H, xe, ye = np.histogram2d(vec[:, 0], vec[:, 1], bins=nb, range=[[-search_um, search_um]] * 2)
    ix, iy = np.unravel_index(H.argmax(), H.shape)
    off = np.array([(xe[ix] + xe[ix + 1]) / 2, (ye[iy] + ye[iy + 1]) / 2])
    for _ in range(iterations):
        d, i = tree.query(ref + off, distance_upper_bound=refine_um if _ else 2 * bin_um)
        ok = np.isfinite(d)
        if ok.sum() < 3:
            break
        off = np.median(other[i[ok]] - ref[ok], axis=0)
    return off


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("reference_csv", type=Path, help="objects.csv of the higher magnification")
    ap.add_argument("other_csv", type=Path, help="objects.csv of the lower magnification")
    ap.add_argument("--tolerance", type=float, default=2.5, help="match radius after offset, um")
    ap.add_argument("--offset-min-ecd", type=float, default=0.8, help="objects used to estimate the offset, um")
    ap.add_argument("--bins", default="0,0.4,0.6,0.8,1.0,1.5,2,5,1000", help="ECD class limits of the reference, um")
    ap.add_argument("-o", "--output", type=Path, help="CSV of matched pairs")
    args = ap.parse_args()

    ref_all, oth = pd.read_csv(args.reference_csv), pd.read_csv(args.other_csv)
    ref = ref_all[~ref_all.touches_border]
    R = ref[["stage_x_um", "stage_y_um"]].to_numpy()
    O = oth[["stage_x_um", "stage_y_um"]].to_numpy()
    # offset per pair of images: every image carries its own stage error (~ +-2 um),
    # and a reference image can overlap several images of the other magnification
    d = np.full(len(ref), np.inf); i = np.zeros(len(ref), int)
    r_img, o_img = ref.image.to_numpy(), oth.image.to_numpy()
    r_big = ref.ecd_um.to_numpy() >= args.offset_min_ecd
    o_big = oth.ecd_um.to_numpy() >= 0.8 * args.offset_min_ecd
    for rn in pd.unique(r_img):
        rs = r_img == rn
        for on in pd.unique(o_img):
            os_ = np.flatnonzero(o_img == on)
            ob = os_[o_big[os_]]
            if len(ob) < 3 or (rs & r_big).sum() < 3:
                continue
            off = estimate_offset(R[rs & r_big], O[ob])
            dd, ii = cKDTree(O[os_]).query(R[rs] + off, distance_upper_bound=args.tolerance)
            if np.isfinite(dd).sum() < 3:
                continue                                    # images do not overlap
            print(f"{rn} -> {on}: offset dx {off[0]:+.2f} um, dy {off[1]:+.2f} um, {np.isfinite(dd).sum()} pairs")
            idx = np.flatnonzero(rs)
            better = dd < d[idx]
            d[idx[better]] = dd[better]
            i[idx[better]] = os_[np.minimum(ii[better], len(os_) - 1)]
    ok = np.isfinite(d)
    pairs = ref[ok].assign(other_image=oth.image.to_numpy()[i[ok]], other_id=oth.id.to_numpy()[i[ok]],
                           other_ecd_um=oth.ecd_um.to_numpy()[i[ok]], distance_um=d[ok])
    print(f"reference objects (not touching border): {len(ref)}, found in other: {ok.sum()}, "
          f"median distance after offset {np.median(d[ok]):.2f} um")
    edges = [float(v) for v in args.bins.split(",")]
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (ref.ecd_um >= lo) & (ref.ecd_um < hi)
        if not sel.any():
            continue
        found = ok[sel.to_numpy()]
        pr = pairs[(pairs.ecd_um >= lo) & (pairs.ecd_um < hi)]
        rows.append({"ECD_ref_um": f"{lo:g}-{hi:g}", "n_ref": int(sel.sum()), "found_%": round(100 * found.mean()),
                     "median_ECD_ref": round(pr.ecd_um.median(), 2) if len(pr) else np.nan,
                     "median_ECD_other": round(pr.other_ecd_um.median(), 2) if len(pr) else np.nan,
                     "size_ratio": round(float(np.median(pr.other_ecd_um / pr.ecd_um)), 2) if len(pr) else np.nan})
    print(pd.DataFrame(rows).to_string(index=False))
    print("Note: reference objects outside the other images count as 'not found' - use image sets covering the same area.")
    if args.output:
        pairs.to_csv(args.output, index=False, float_format="%.4f")


if __name__ == "__main__":
    main()
