"""Stage coordinates of hardness indents from marked VHX images.

Folder convention: for every marked copy ``<name>_marked.<ext>`` (red circle
drawn around the indent) the original VHX TIFF ``<name>.tif`` must be in the
same folder. Output: one CSV row per indent with its position in stage um.

Optional check images (e.g. the same indent at another magnification) are
searched for every indent that falls inside them; the deviation is reported.

Example:
    python scripts/locate_indents.py data/2026-11-02_MnB/indents -o indents.csv \
        --check data/2026-11-02_MnB/indents/Indent_botleft_200x.tif
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from cii import find_marking, locate_indent, read_vhx


def locate_folder(folder: Path) -> pd.DataFrame:
    rows = []
    for marked in sorted(folder.glob("*_marked.*")):
        name = marked.stem[: -len("_marked")]
        original = folder / f"{name}.tif"
        if not original.exists():
            raise FileNotFoundError(f"original image for {marked.name} not found: {original.name}")
        img = read_vhx(original)
        mk = find_marking(marked)
        fit = locate_indent(img.read_image(), mk.col, mk.row, mk.radius)
        x, y = img.pixel_to_stage(fit.col, fit.row)
        rows.append({
            "name": name, "image": original.name, "magnification": img.magnification,
            "um_per_px": img.um_per_px, "captured": img.captured,
            "col_px": round(fit.col, 2), "row_px": round(fit.row, 2),
            "diameter_um": round(fit.diameter_px * img.um_per_px, 1),
            "contrast": round(fit.contrast, 1),
            "stage_x_um": round(float(x), 2), "stage_y_um": round(float(y), 2),
        })
    if not rows:
        raise FileNotFoundError(f"no *_marked.* files in {folder}")
    return pd.DataFrame(rows)


def check_images(indents: pd.DataFrame, paths: list[Path], search_radius_um: float = 60) -> pd.DataFrame:
    rows = []
    for p in paths:
        img = read_vhx(p)
        image = None
        for _, ind in indents.iterrows():
            if not img.contains_stage(ind.stage_x_um, ind.stage_y_um, margin_px=20):
                continue
            image = img.read_image() if image is None else image
            col, row = img.stage_to_pixel(ind.stage_x_um, ind.stage_y_um)
            fit = locate_indent(image, float(col), float(row), search_radius_um / img.um_per_px)
            x, y = img.pixel_to_stage(fit.col, fit.row)
            rows.append({"check_image": p.name, "magnification": img.magnification, "indent": ind["name"],
                         "stage_x_um": round(float(x), 2), "stage_y_um": round(float(y), 2),
                         "dx_um": round(float(x) - ind.stage_x_um, 2), "dy_um": round(float(y) - ind.stage_y_um, 2)})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder", type=Path, help="folder with <name>.tif and <name>_marked.* images")
    ap.add_argument("-o", "--output", type=Path, default=Path("indents.csv"))
    ap.add_argument("--check", type=Path, nargs="*", default=[], help="further VHX images showing indents")
    args = ap.parse_args()

    indents = locate_folder(args.folder)
    indents.to_csv(args.output, index=False)
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(indents.drop(columns=["image", "um_per_px"]).to_string(index=False))
        if args.check:
            chk = check_images(indents, args.check)
            print("\ncheck images:")
            print(chk.to_string(index=False) if len(chk) else "  no indent inside the check images")
    print(f"\nwritten: {args.output}")


if __name__ == "__main__":
    main()
