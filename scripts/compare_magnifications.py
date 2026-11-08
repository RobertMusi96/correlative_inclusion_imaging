"""Compare detections of the same sample area at two magnifications.

Objects of the higher magnification (reference) are searched among the objects
of the lower magnification. Works for whole series: offsets are estimated per
overlapping image pair, objects in the overlap of neighbouring images are
counted once (see cii.compare). Input: the output folders of
detect_inclusions.py (objects.csv + images.csv).

Example:
    python scripts/compare_magnifications.py results/detection_1000x results/detection_500x \
        -o results/compare_1000x_500x --missed-crops 80

Outputs in the output folder:
    summary.csv     share found and size ratio per ECD class of the reference
    objects.csv     every reference object: status (found / missed / border /
                    duplicate / outside), matched object, expected position
    offsets.csv     offset per overlapping image pair (measured or model)
    comparison.png  found share per ECD class and map of the missed objects
    missed.png      (--missed-crops) reference and other image side by side at the
                    expected position of the largest missed objects
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from cii.compare import compare, summary


def load(folder: Path):
    """objects.csv and images.csv of a detection folder (or of the folder of a given objects.csv)."""
    folder = folder.parent if folder.suffix == ".csv" else folder
    if not folder.is_dir():
        raise SystemExit(f"folder not found: {str(folder)!r} - paths with spaces must be in quotes")
    img_csv = folder / "images.csv"
    if not img_csv.exists():
        raise SystemExit(f"{img_csv} not found - rerun detect_inclusions.py (it writes images.csv)")
    return pd.read_csv(folder / "objects.csv"), pd.read_csv(img_csv)


def overview_figure(objs, summ, ref_mag, oth_mag, path, min_ecd):
    """Figure: share found per ECD class of the reference (left) and stage map of the reference objects
    >= min_ecd, missed ones highlighted (right)."""
    fig, (a, b) = plt.subplots(1, 2, figsize=(15, 5.5), gridspec_kw={"width_ratios": [1, 2]})
    x = np.arange(len(summ))
    a.bar(x, summ.found_pct, color="#2a78d6")
    for k, r in summ.iterrows():
        a.text(k, r.found_pct + 1.5, f"{r.found_pct:.0f}%\nn={r.n_ref}", ha="center", va="bottom", fontsize=8)
    a.set_xticks(x, summ.ECD_ref_um, rotation=45, ha="right")
    a.set_ylim(0, 115); a.set_ylabel(f"found at {oth_mag:g}x (%)"); a.set_xlabel(f"ECD at {ref_mag:g}x (µm)")
    a.spines[["top", "right"]].set_visible(False)
    big = objs[objs.ecd_um >= min_ecd]
    f, m = big[big.status == "found"], big[big.status == "missed"]
    b.scatter(f.stage_x_um / 1000, f.stage_y_um / 1000, s=2, c="#bbbbbb", label=f"found ({len(f)})", lw=0)
    b.scatter(m.stage_x_um / 1000, m.stage_y_um / 1000, s=10, c="#e8590c", label=f"missed ({len(m)})", lw=0)
    b.set_aspect("equal"); b.invert_yaxis()
    b.set_xlabel("stage X (mm)"); b.set_ylabel("stage Y (mm)")
    b.set_title(f"objects with ECD ≥ {min_ecd:g} µm at {ref_mag:g}x", fontsize=10)
    b.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=2, fontsize=8, markerscale=2, frameon=False)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def missed_figure(objs, ref_img, oth_img, path, n, min_ecd, field_um=14.0, tolerance=2.5, cols=4):
    """Figure with the n largest missed reference objects: reference crop next to the other magnification's
    crop at the expected position, with the match radius drawn."""
    from cii import read_vhx
    sel = objs[(objs.status == "missed") & (objs.ecd_um >= min_ecd)].sort_values("ecd_um", ascending=False).head(n)
    if not len(sel):
        print("no missed objects for crops"); return
    paths = {**dict(zip(ref_img.image, ref_img.path)), **dict(zip(oth_img.image, oth_img.path))}
    cache = {}

    def crop(name, x, y):
        """Crop of field_um around stage position (x, y) of image name (images cached) and its extent in um
        relative to that position, for imshow."""
        if name not in cache:
            if len(cache) > 6:
                cache.pop(next(iter(cache)))
            im = read_vhx(Path(paths[name])); cache[name] = (im, im.read_image())
        im, rgb = cache[name]
        c, r = im.stage_to_pixel(x, y)
        h = field_um / 2 / im.um_per_px
        r0, r1 = int(max(0, r - h)), int(min(rgb.shape[0], r + h)); c0, c1 = int(max(0, c - h)), int(min(rgb.shape[1], c + h))
        ext = [(c0 - c) * im.um_per_px, (c1 - c) * im.um_per_px, (r1 - r) * im.um_per_px, (r0 - r) * im.um_per_px]
        return rgb[r0:r1, c0:c1, :3], ext

    rows = int(np.ceil(len(sel) / cols))
    fig, axes = plt.subplots(rows, 2 * cols, figsize=(2 * cols * 1.9, rows * 2.1), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for k, o in enumerate(sel.itertuples()):
        ax1, ax2 = axes[k // cols, 2 * (k % cols)], axes[k // cols, 2 * (k % cols) + 1]
        for ax, (img, ext), lbl in [(ax1, crop(o.image, o.stage_x_um, o.stage_y_um), "ref"),
                                    (ax2, crop(o.other_image, o.other_x_um, o.other_y_um), "other")]:
            ax.imshow(img, extent=ext, interpolation="nearest")
            ax.add_patch(plt.Circle((0, 0), tolerance, fill=False, ec="#e8590c", lw=0.8, ls="--"))
            ax.set_xlim(-field_um / 2, field_um / 2); ax.set_ylim(field_um / 2, -field_um / 2)
        ax1.set_title(f"{Path(o.image).stem[-6:]} #{o.id}  {o.ecd_um:.2f} µm  c={o.contrast_max:.2f}", fontsize=7)
        ax2.set_title(f"{Path(o.other_image).stem[-6:]}  expected position", fontsize=7)
    fig.suptitle(f"missed objects (left: reference, right: other magnification at expected position; "
                 f"circle = match radius {tolerance:g} µm, field {field_um:g} µm, c = peak contrast of the reference object)", fontsize=9)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def main():
    """Command line: compares two detection folders, prints the summary and writes summary, objects,
    offsets and figures."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("reference", type=Path, help="detection folder of the higher magnification")
    ap.add_argument("other", type=Path, help="detection folder of the lower magnification")
    ap.add_argument("-o", "--output", type=Path, default=Path("comparison"))
    ap.add_argument("--tolerance", type=float, default=2.5, help="match radius after offset, um (default 2.5)")
    ap.add_argument("--offset-min-ecd", type=float, default=0.8, help="objects used for the offsets, um (default 0.8)")
    ap.add_argument("--bins", default="0,0.6,0.8,1.0,1.5,2,3,5,10,1000", help="ECD classes of the reference, um")
    ap.add_argument("--map-min-ecd", type=float, default=1.0, help="ECD limit for the map, um (default 1.0)")
    ap.add_argument("--missed-crops", type=int, default=0, help="crops of the N largest missed objects")
    args = ap.parse_args()

    ref_obj, ref_img = load(args.reference)
    oth_obj, oth_img = load(args.other)
    res = compare(ref_obj, ref_img, oth_obj, oth_img, tolerance=args.tolerance, offset_min_ecd=args.offset_min_ecd)
    o, off = res.objects, res.offsets
    args.output.mkdir(parents=True, exist_ok=True)
    edges = [float(v) for v in args.bins.split(",")]
    summ = summary(o, edges)
    ref_mag, oth_mag = ref_img.magnification.iloc[0], oth_img.magnification.iloc[0]

    print(f"reference {ref_mag:g}x: {len(ref_img)} images, {len(o)} objects; other {oth_mag:g}x: "
          f"{len(oth_img)} images, {len(oth_obj)} objects")
    print("status: " + ", ".join(f"{k} {v}" for k, v in o.status.value_counts().items()))
    print(f"global offset dx {res.global_offset[0]:+.2f} um, dy {res.global_offset[1]:+.2f} um; "
          f"{len(off)} image pairs, {(off.source == 'measured').sum()} measured, "
          f"scatter around per-image model {res.model_rms_um:.2f} um")
    area_mm2 = float((oth_img.width_px * oth_img.height_px * oth_img.um_per_px ** 2).sum()) / 1e6
    chance = len(oth_obj) / area_mm2 * 1e-6 * np.pi * args.tolerance ** 2 * 100
    print(summ.to_string(index=False))
    print(f"chance of a random match within {args.tolerance:g} um: about {chance:.1f} % "
          f"({len(oth_obj) / area_mm2:.0f} objects/mm2 at {oth_mag:g}x)")
    print("(found / missed only; 'border', 'duplicate' and 'outside' objects are not counted)")

    summ.to_csv(args.output / "summary.csv", index=False)
    o.to_csv(args.output / "objects.csv", index=False, float_format="%.4f")
    off.to_csv(args.output / "offsets.csv", index=False, float_format="%.3f")
    overview_figure(o, summ, ref_mag, oth_mag, args.output / "comparison.png", args.map_min_ecd)
    if args.missed_crops:
        missed_figure(o, ref_img, oth_img, args.output / "missed.png", args.missed_crops, args.map_min_ecd,
                      tolerance=args.tolerance)
    print(f"-> {args.output}")


if __name__ == "__main__":
    main()
