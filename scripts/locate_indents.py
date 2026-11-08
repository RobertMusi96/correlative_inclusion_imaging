"""Stage coordinates of hardness indents from marked VHX or SEM images.

Folder convention: for every marked copy ``<name>_marked.<ext>`` (red circle
drawn around the indent) the original image ``<name>.tif`` must be in the same
folder. Output: one CSV row per indent with its position in stage um, plus a
figure per indent showing the segmented region (``--figures``).

The indent label (used to match indents between instruments) is taken from the
file name with ``--label-regex`` (default: ``Indent_<word>``).

Examples:
    # light microscope (Keyence VHX)
    python scripts/locate_indents.py data/MnB/vhx_indents -o vhx_indents.csv \
        --figures figures/vhx_indents --check data/MnB/vhx_indents/Indent_botleft_200x.tif

    # SEM (Oxford AZtec), image centres from the field table
    python scripts/locate_indents.py data/MnB/sem_indents --instrument sem \
        --fields data/MnB/field_information.csv -o sem_indents.csv --figures figures/sem_indents
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

from cii import find_marking, locate_indent, read_vhx
from cii.figures import save_indent_figure
from cii.sem import read_aztec_tiff, read_field_positions

METHOD = {"vhx": "dark", "sem": "edges"}


def make_reader(instrument: str, fields_csv: Path | None):
    """Image reader for the instrument: read_vhx, or read_aztec_tiff with the field table for SEM."""
    if instrument == "vhx":
        return read_vhx
    fields = read_field_positions(fields_csv) if fields_csv else None
    if fields is None:
        print("warning: no --fields table given; SEM stage positions from the TIFF are rounded to 0.1 mm")
    return lambda p: read_aztec_tiff(p, fields)


def label_from(name: str, pattern: str) -> str:
    """Indent label from a file name (first group of pattern), else the whole name."""
    m = re.search(pattern, name)
    return m.group(1) if m else name


def locate_folder(folder: Path, instrument: str, reader, label_regex: str,
                  figures: Path | None) -> pd.DataFrame:
    """Locates the indent of every <name>_marked image pair in folder; returns one row per indent (pixel
    and stage position, size, contrast) and writes a figure per indent if asked."""
    rows = []
    for marked in sorted(folder.glob("*_marked.*")):
        name = marked.stem[: -len("_marked")]
        original = folder / f"{name}.tif"
        if not original.exists():
            raise FileNotFoundError(f"original image for {marked.name} not found: {original.name}")
        img = reader(original)
        image = img.read_image()
        mk = find_marking(marked)
        fit = locate_indent(image, mk.col, mk.row, mk.radius, method=METHOD[instrument])
        x, y = img.pixel_to_stage(fit.col, fit.row)
        label = label_from(name, label_regex)
        d_um = fit.diameter_px * img.um_per_px
        rows.append({
            "label": label, "instrument": instrument, "image": original.name,
            "magnification": img.magnification, "um_per_px": img.um_per_px,
            "image_centre_x_um": img.stage_x, "image_centre_y_um": img.stage_y,
            "col_px": round(fit.col, 2), "row_px": round(fit.row, 2),
            "diameter_um": round(d_um, 1), "contrast": round(fit.contrast, 1),
            "stage_x_um": round(float(x), 2), "stage_y_um": round(float(y), 2),
        })
        if figures is not None:
            mag = f"{img.magnification:g}x" if img.magnification else ""
            save_indent_figure(
                image, fit, img.um_per_px, figures / f"{label}_{instrument}.png",
                title=f"{label} – {instrument.upper()} {mag}",
                info=[original.name, f"centroid px ({fit.col:.1f}, {fit.row:.1f})",
                      f"stage X {float(x):.1f} µm, Y {float(y):.1f} µm",
                      f"eq. diameter {d_um:.1f} µm ({fit.method})"])
    if not rows:
        raise FileNotFoundError(f"no *_marked.* files in {folder}")
    return pd.DataFrame(rows)


def check_images(indents: pd.DataFrame, paths: list[Path], instrument: str, reader,
                 search_radius_um: float = 60) -> pd.DataFrame:
    """Looks for the known indents in further images (e.g. another objective) and returns their
    measured positions next to the expected ones."""
    rows = []
    for p in paths:
        img = reader(p)
        image = None
        for _, ind in indents.iterrows():
            if not img.contains_stage(ind.stage_x_um, ind.stage_y_um, margin_px=20):
                continue
            image = img.read_image() if image is None else image
            col, row = img.stage_to_pixel(ind.stage_x_um, ind.stage_y_um)
            fit = locate_indent(image, float(col), float(row), search_radius_um / img.um_per_px,
                                method=METHOD[instrument])
            x, y = img.pixel_to_stage(fit.col, fit.row)
            rows.append({"check_image": p.name, "magnification": img.magnification, "label": ind["label"],
                         "stage_x_um": round(float(x), 2), "stage_y_um": round(float(y), 2),
                         "dx_um": round(float(x) - ind.stage_x_um, 2), "dy_um": round(float(y) - ind.stage_y_um, 2)})
    return pd.DataFrame(rows)


def main():
    """Command line: locates the indents of a folder, optionally checks further images, writes the CSV."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder", type=Path, help="folder with <name>.tif and <name>_marked.* images")
    ap.add_argument("--instrument", choices=["vhx", "sem"], default="vhx")
    ap.add_argument("--fields", type=Path, help="SEM: AZtec field table (FieldID, StageXmm, StageYmm)")
    ap.add_argument("-o", "--output", type=Path, default=Path("indents.csv"))
    ap.add_argument("--figures", type=Path, help="folder for one figure per indent")
    ap.add_argument("--check", type=Path, nargs="*", default=[], help="further images showing indents")
    ap.add_argument("--label-regex", default=r"(Indent_[A-Za-z0-9]+)")
    args = ap.parse_args()

    reader = make_reader(args.instrument, args.fields)
    indents = locate_folder(args.folder, args.instrument, reader, args.label_regex, args.figures)
    indents.to_csv(args.output, index=False)
    with pd.option_context("display.width", 220, "display.max_columns", 20):
        print(indents[["label", "magnification", "col_px", "row_px", "diameter_um", "contrast",
                       "stage_x_um", "stage_y_um"]].to_string(index=False))
        if args.check:
            chk = check_images(indents, args.check, args.instrument, reader)
            print("\ncheck images:")
            print(chk.to_string(index=False) if len(chk) else "  no indent inside the check images")
    print(f"\nwritten: {args.output}" + (f", figures in {args.figures}" if args.figures else ""))


if __name__ == "__main__":
    main()
