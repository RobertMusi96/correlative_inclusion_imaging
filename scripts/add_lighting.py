"""Add the crops of another lighting mode (dark field, multi-lighting) to a dataset.

The images of the other mode were recorded at the same nominal stage positions
as the bright-field images, but in a separate pass; each image is shifted
against its bright-field image by a few µm. Per image pair the shift is
measured (phase correlation of the edge images, checked on the four quadrants)
and the same window as the bright-field crop is cut (see cii.lighting).

Particles, labels, split and folds do not change. Written into the dataset:
    crops_<name>/          crops of the lighting mode, file names as in crops/
    dataset.csv            new column crop_<name> (empty if there is no crop)
    lighting_<name>.csv    per bright-field image: paired image, shift, check, number of crops
    lighting_<name>.png    sample of particles: bright field | <name>
    dataset_info.json      entry "lighting" -> <name>

Examples:
    python scripts/add_lighting.py data/datasets/dataset_v1 --name df --images "D:\\...\\500x darkfield"
    python scripts/add_lighting.py data/datasets/dataset_v1 --name ml --images "D:\\...\\500x multilighting"
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
from datetime import datetime
from pathlib import Path

import pandas as pd

from cii.lighting import add_lighting, overview_figure
from cii.matching_figures import resolve_vhx_paths


def main():
    """Command line: adds one lighting mode to a dataset folder."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset", type=Path, help="dataset folder (dataset.csv, crops/)")
    ap.add_argument("--name", required=True, help="short name of the lighting mode, e.g. df or ml (letters and digits)")
    ap.add_argument("--images", type=Path, required=True, help="folder with the images of this lighting mode")
    ap.add_argument("--vhx", type=Path, help="VHX detection folder of the bright-field images (default: dataset_info.json)")
    ap.add_argument("--vhx-images", type=Path, help="folder with the bright-field images, if they moved")
    ap.add_argument("--overwrite", action="store_true", help="replace crops of this name that already exist")
    args = ap.parse_args()

    if not re.fullmatch(r"[A-Za-z0-9]+", args.name):
        raise SystemExit("--name: letters and digits only")

    if args.name in ("rgb", "mask", "col0", "row0", "size"):
        raise SystemExit(f"--name {args.name} is reserved - choose another name")

    root = args.dataset
    ds = pd.read_csv(root / "dataset.csv", low_memory=False)
    info_file = root / "dataset_info.json"
    info = json.loads(info_file.read_text()) if info_file.exists() else {}

    if f"crop_{args.name}" in ds:
        if not args.overwrite:
            raise SystemExit(f"crop_{args.name} exists already - give another --name or --overwrite")
        ds = ds.drop(columns=f"crop_{args.name}")
        shutil.rmtree(root / f"crops_{args.name}", ignore_errors=True)

    vhx = args.vhx or (Path(info["vhx_detection"]) if "vhx_detection" in info else None)
    if vhx is None or not (vhx / "images.csv").exists():
        raise SystemExit(f"images.csv of the VHX detection not found ({vhx}) - give the folder with --vhx")

    images = pd.read_csv(vhx / "images.csv")
    paths = resolve_vhx_paths(images, args.vhx_images)
    print(f"bright-field images found: {len(paths)} of {len(images)}")

    ds, pairs = add_lighting(ds, root, args.name, paths, images, args.images)
    ds.to_csv(root / "dataset.csv", index=False, float_format="%.4f")
    pairs.to_csv(root / f"lighting_{args.name}.csv", index=False)

    has = ds[f"crop_{args.name}"].ne("")
    if has.any():
        overview_figure(ds, root, [args.name], root / f"lighting_{args.name}.png")

    info.setdefault("lighting", {})[args.name] = {"images": str(args.images), "added": datetime.now().isoformat(timespec="seconds"),
                                                  "n_crops": int(has.sum()), "n_particles": len(ds)}
    info_file.write_text(json.dumps(info, indent=2))

    bad = pairs[pairs.get("ok", pd.Series(True, index=pairs.index)).eq(False)]
    print(f"{int(has.sum())} of {len(ds)} particles have a {args.name} crop; "
          f"{len(bad)} image pair(s) rejected by the shift check")
    print(ds.assign(has=has).groupby(["split", "training_class"]).has.agg(["sum", "size"]).to_string())
    print(f"-> {root}")


if __name__ == "__main__":
    main()
