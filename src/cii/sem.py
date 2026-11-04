"""Read SEM images exported by Oxford Instruments AZtec (Software tag 'OINA').

AZtec stores acquisition data as ``key=value`` lines in private TIFF tag 50684:
PixelExtent (m/px), Magnification, AcceleratingVoltage (kV), WorkingDistance (mm),
StagePositionX/Y/Z (mm, rounded to 0.1 mm - too coarse for correlation), Date.

Precise stage positions of the image centres come from the field table exported
with the feature analysis (columns ``FieldID, StageXmm, StageYmm``); the field ID
is the ``Site_<n>`` number in the file name.

Axis directions on this SEM: stage X increases to the LEFT in the image
(x_dir = -1), stage Y increases downwards (y_dir = +1).
"""
from __future__ import annotations

import datetime as _dt
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile

from .geometry import StageImage

AZTEC_TAG = 50684
SITE_PATTERN = re.compile(r"Site_(\d+)")


@dataclass
class SEMImage(StageImage):
    magnification: float | None = None
    kv: float | None = None
    working_distance_mm: float | None = None
    captured: _dt.datetime | None = None
    field_id: int | None = None
    label: str | None = None
    raw_fields: dict = field(default_factory=dict, repr=False)

    def read_image(self) -> np.ndarray:
        """The image as stored (BSE: H x W, uint16)."""
        return tifffile.imread(self.path)


def read_field_positions(csv_path) -> pd.DataFrame:
    """AZtec field table -> DataFrame indexed by field_id with stage_x_um, stage_y_um."""
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    df.columns = [c.strip() for c in df.columns]
    out = pd.DataFrame({"field_id": df["FieldID"].astype(int),
                        "stage_x_um": df["StageXmm"].astype(float) * 1000,
                        "stage_y_um": df["StageYmm"].astype(float) * 1000})
    if "Area Name" in df:
        out["area"] = df["Area Name"].astype(str)
    return out.set_index("field_id")


def field_id_from_name(name: str) -> int | None:
    m = SITE_PATTERN.search(name)
    return int(m.group(1)) if m else None


def _parse_aztec_tag(text: str) -> dict:
    out = {}
    for line in text.splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            out[k.strip()] = v.strip()
    return out


def read_aztec_tiff(path, fields: pd.DataFrame | None = None,
                    stage_xy_um: tuple[float, float] | None = None) -> SEMImage:
    """Read an AZtec TIFF.

    The image-centre stage position is taken (in this order) from ``stage_xy_um``,
    from ``fields`` via the ``Site_<n>`` number in the file name, or - with a
    warning-level precision of 0.1 mm - from the TIFF tag itself.
    """
    path = Path(path)
    with tifffile.TiffFile(path) as tif:
        page = tif.pages[0]
        height, width = page.shape[:2]
        tag = page.tags.get(AZTEC_TAG)
        if tag is None:
            raise ValueError(f"{path.name}: no AZtec metadata (tag {AZTEC_TAG})")
        meta = _parse_aztec_tag(tag.value)
    um_per_px = float(meta["PixelExtent"]) * 1e6
    fid = field_id_from_name(path.stem)
    if stage_xy_um is not None:
        sx, sy = stage_xy_um
    elif fields is not None and fid is not None and fid in fields.index:
        sx, sy = float(fields.at[fid, "stage_x_um"]), float(fields.at[fid, "stage_y_um"])
    else:
        sx, sy = float(meta["StagePositionX"]) * 1000, float(meta["StagePositionY"]) * 1000
    captured = None
    if "Date" in meta:
        try:
            captured = _dt.datetime.strptime(meta["Date"], "%d/%m/%Y %H:%M:%S")
        except ValueError:
            pass
    return SEMImage(path=path, width=width, height=height, um_per_px=um_per_px,
                    stage_x=sx, stage_y=sy, x_dir=-1, y_dir=1,
                    magnification=float(meta["Magnification"]) if "Magnification" in meta else None,
                    kv=float(meta["AcceleratingVoltage"]) if "AcceleratingVoltage" in meta else None,
                    working_distance_mm=float(meta["WorkingDistance"]) if "WorkingDistance" in meta else None,
                    captured=captured, field_id=fid, label=meta.get("Label"), raw_fields=meta)
