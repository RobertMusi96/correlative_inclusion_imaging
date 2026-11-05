"""Detect dark objects (inclusions and other features) in VHX bright-field images.

For every image: a CSV row per object (pixel and stage coordinates, size,
shape, contrast, colour), a full-resolution overlay PNG with the segmented
objects highlighted and numbered, and a gallery PNG with one tile per object.

Example:
    python scripts/detect_inclusions.py data/MnB/vhx_500x/*.tif -o results/detection_500x
    python scripts/detect_inclusions.py data/MnB/vhx_1000x/*.tif -o results/detection_1000x --high 0.10

Outputs in the output folder:
    objects.csv             all objects of all images (column 'image')
    parameters.json         detection parameters used
    overlay/<image>.png     image with segmented objects (orange; blue = touches border)
    gallery/<image>.png     one tile per object, largest first
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from cii import read_vhx
from cii.detection import DetectionParams, detect, params_dict
from cii.detection_figures import save_gallery, save_overlay


def main():
    d = DetectionParams()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("images", type=Path, nargs="+")
    ap.add_argument("-o", "--output", type=Path, default=Path("detection"))
    ap.add_argument("--high", type=float, default=d.high, help=f"core contrast (default {d.high})")
    ap.add_argument("--low", type=float, default=d.low, help=f"edge contrast (default {d.low})")
    ap.add_argument("--size-fraction", type=float, default=d.size_fraction,
                    help=f"outline at this fraction of the object's peak contrast, 0 = off (default {d.size_fraction})")
    ap.add_argument("--min-area", type=float, default=d.min_area_um2, help=f"um2 (default {d.min_area_um2})")
    ap.add_argument("--background", type=float, default=d.background_um, help=f"um (default {d.background_um})")
    ap.add_argument("--no-figures", action="store_true")
    ap.add_argument("--gallery-max", type=int, default=300)
    args = ap.parse_args()

    p = DetectionParams(background_um=args.background, high=args.high, low=args.low,
                        size_fraction=args.size_fraction, min_area_um2=args.min_area)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "parameters.json").write_text(json.dumps(params_dict(p), indent=2))
    tables = []
    for path in sorted(args.images):
        img = read_vhx(path)
        rgb = img.read_image()
        det = detect(rgb, img.um_per_px, p, stage=img)
        obj = det.objects.copy()
        obj.insert(0, "image", path.name)
        obj.insert(1, "magnification", img.magnification)
        obj.insert(2, "um_per_px", img.um_per_px)
        tables.append(obj)
        n = len(obj)
        big = int((obj.ecd_um >= 2).sum()) if n else 0
        sat = float((rgb[..., :3] == 255).all(axis=2).mean() * 100)
        print(f"{path.name}: {n} objects ({big} with ECD >= 2 um), {sat:.0f} % of pixels saturated (255 in R, G and B)")
        if not args.no_figures:
            title = f"{path.stem}  ·  {img.magnification:g}x  ·  {n} objects  ·  high {p.high}, low {p.low}, size {p.size_fraction}, min {p.min_area_um2} µm²"
            save_overlay(rgb, det, args.output / "overlay" / f"{path.stem}.png", um_per_px=img.um_per_px, title=title)
            save_gallery(rgb, det, img.um_per_px, args.output / "gallery" / f"{path.stem}.png",
                         max_objects=args.gallery_max, title=title)
    allobj = pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()
    allobj.to_csv(args.output / "objects.csv", index=False, float_format="%.4f")
    print(f"\n{len(allobj)} objects in {len(tables)} images -> {args.output}")


if __name__ == "__main__":
    main()
