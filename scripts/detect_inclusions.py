"""Detect dark objects (inclusions and other features) in VHX bright-field images.

For every image: a CSV row per object (pixel and stage coordinates, size,
shape, contrast, colour), a full-resolution overlay PNG with the segmented
objects highlighted and numbered, and a gallery PNG with one tile per object.

Images can be given as files, a folder (all .tif in it) or a pattern; put
paths with spaces in quotes (Windows: "D:\\My data\\500x").

Example:
    # whole series, tables only, 4 processes
    python scripts/detect_inclusions.py data/MnB/vhx_500x/*.tif -o results/detection_500x --no-figures -j 4
    # original method (grey mean) for comparison
    python scripts/detect_inclusions.py data/MnB/vhx_500x -o results/detection_500x_mean --no-figures -j 4 --channels mean
    # a few images with review figures
    python scripts/detect_inclusions.py data/MnB/vhx_500x/*_00001*.tif -o results/check_500x

Outputs in the output folder:
    objects.csv             all objects of all images (column 'image')
    images.csv              one row per image: path, stage position, size, object count
    parameters.json         detection parameters used
    labels/<image>_objects.png  16-bit label image of the object outlines (pixel value = column 'label')
    labels/<image>_regions.png  16-bit label image of the detected regions (pixel value = column 'region')
    overlay/<image>.png     image with segmented objects (orange; blue = touches border)
    gallery/<image>.png     one tile per object, largest first

Use one output folder per magnification / series.
"""
from __future__ import annotations

import argparse
import glob
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd

from cii import read_vhx
from cii.detection import DEFAULT_THRESHOLDS, DetectionParams, detect, params_dict, save_label_images
from cii.detection_figures import save_gallery, save_overlay


def process(path: Path, p: DetectionParams, output: Path, figures: bool, gallery_max: int, labels: bool = True):
    """Detects the objects of one VHX image; writes label images and figures if asked and returns the
    object table and one row for images.csv (position, size, counts, saturated share)."""
    img = read_vhx(path)
    rgb = img.read_image()
    det = detect(rgb, img.um_per_px, p, stage=img)
    obj = det.objects.copy()
    obj.insert(0, "image", path.name)
    obj.insert(1, "magnification", img.magnification)
    obj.insert(2, "um_per_px", img.um_per_px)
    n = len(obj)
    sat = float((rgb[..., :3] == 255).all(axis=2).mean() * 100)
    info = {"image": path.name, "path": str(path.absolute()), "magnification": img.magnification,
            "um_per_px": img.um_per_px, "width_px": img.width, "height_px": img.height,
            "stage_x_um": img.stage_x, "stage_y_um": img.stage_y, "n_objects": n,
            "n_ecd_ge_1um": int((obj.ecd_um >= 1).sum()) if n else 0, "saturated_pct": sat}
    if labels:
        save_label_images(det, output / "labels", path.stem)
    if figures:
        title = (f"{path.stem}  ·  {img.magnification:g}x  ·  {n} objects  ·  "
                 f"{p.channels}, high {p.high}, low {p.low}, size {p.size_fraction}, min {p.min_area_um2} µm²")
        save_overlay(rgb, det, output / "overlay" / f"{path.stem}.png", um_per_px=img.um_per_px, title=title)
        save_gallery(rgb, det, img.um_per_px, output / "gallery" / f"{path.stem}.png",
                     max_objects=gallery_max, title=title)
    return obj, info


def expand_images(items: list[str]) -> list[Path]:
    """Files, folders (all .tif inside) and wildcard patterns - Windows does not expand *.tif itself."""
    paths, missing = [], []
    for it in items:
        p = Path(it)
        if p.is_dir():
            found = sorted(f for f in p.iterdir() if f.suffix.lower() in (".tif", ".tiff") and "_marked" not in f.stem)
        elif any(ch in it for ch in "*?["):
            found = sorted(Path(f) for f in glob.glob(it))
        else:
            found = [p] if p.is_file() else []
        if not found:
            missing.append(it)
        paths += found
    if missing:
        raise SystemExit("not found: " + ", ".join(repr(m) for m in missing)
                         + "\nPaths with spaces must be in quotes, e.g. \"D:\\My data\\500x\"")
    return sorted(dict.fromkeys(paths))


def main():
    """Command line: detection of all given images (optionally in parallel), writes objects.csv, images.csv
    and parameters.json."""
    d = DetectionParams()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("images", nargs="+", help="image files, folders or patterns such as \"D:\\data\\500x\\*.tif\"")
    ap.add_argument("-o", "--output", type=Path, default=Path("detection"))
    ap.add_argument("--channels", choices=list(DEFAULT_THRESHOLDS), default=d.channels,
                    help="contrast: 'max' = strongest of R, G, B (default), 'mean' = grey mean (original method)")
    ap.add_argument("--high", type=float, default=None,
                    help="core contrast (default: " + ", ".join(f"{k} {v[0]}" for k, v in DEFAULT_THRESHOLDS.items()) + ")")
    ap.add_argument("--low", type=float, default=None,
                    help="edge contrast (default: " + ", ".join(f"{k} {v[1]}" for k, v in DEFAULT_THRESHOLDS.items()) + ")")
    ap.add_argument("--size-fraction", type=float, default=d.size_fraction,
                    help=f"outline at this fraction of the object's peak contrast, 0 = off (default {d.size_fraction})")
    ap.add_argument("--min-area", type=float, default=d.min_area_um2, help=f"um2 (default {d.min_area_um2})")
    ap.add_argument("--background", type=float, default=d.background_um, help=f"um (default {d.background_um})")
    ap.add_argument("--no-figures", action="store_true", help="tables only (recommended for whole series)")
    ap.add_argument("--gallery-max", type=int, default=300)
    ap.add_argument("--no-labels", action="store_true",
                    help="do not write the label images (labels/<image>_objects.png, _regions.png)")
    ap.add_argument("-j", "--jobs", type=int, default=1, help="parallel processes (default 1)")
    args = ap.parse_args()

    p = DetectionParams(channels=args.channels, background_um=args.background, high=args.high, low=args.low,
                        size_fraction=args.size_fraction, min_area_um2=args.min_area)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "parameters.json").write_text(json.dumps(params_dict(p), indent=2))
    paths = expand_images(args.images)
    print(f"{len(paths)} images")
    work = [(path, p, args.output, not args.no_figures, args.gallery_max, not args.no_labels) for path in paths]
    tables, infos = [], []
    if args.jobs > 1:
        with ProcessPoolExecutor(args.jobs) as ex:
            results = ex.map(process, *zip(*work), chunksize=2)
            for k, (obj, info) in enumerate(results, 1):
                tables.append(obj); infos.append(info)
                print(f"[{k}/{len(paths)}] {info['image']}: {info['n_objects']} objects "
                      f"({info['n_ecd_ge_1um']} with ECD >= 1 um), {info['saturated_pct']:.0f} % saturated")
    else:
        for k, w in enumerate(work, 1):
            obj, info = process(*w)
            tables.append(obj); infos.append(info)
            print(f"[{k}/{len(paths)}] {info['image']}: {info['n_objects']} objects "
                  f"({info['n_ecd_ge_1um']} with ECD >= 1 um), {info['saturated_pct']:.0f} % saturated")
    allobj = pd.concat([t for t in tables if len(t)], ignore_index=True) if tables else pd.DataFrame()
    allobj.to_csv(args.output / "objects.csv", index=False, float_format="%.4f")
    pd.DataFrame(infos).to_csv(args.output / "images.csv", index=False, float_format="%.4f")
    print(f"\n{len(allobj)} objects in {len(tables)} images -> {args.output}")


if __name__ == "__main__":
    main()
