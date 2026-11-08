"""Read Keyence VHX-7000 TIFF files: image, pixel size, stage position.

The VHX writes its acquisition data into a proprietary EXIF MakerNote
("KmsFile"). The layout is undocumented; the fields used here were identified
empirically and checked against VHX area readings and scale bars:

* pixel size: ``um_per_px = A / (B * C)`` with A = tag 0x0011, B = tag 0x0144,
  C = tag 0x0148. Verified at 100x, 200x, 500x and 1000x (2880 x 2160 px) and
  at 500x in 4K mode (4000 x 3000 px) against the VHX area readout (< 0.001 %).
* stage position (um): tag 0x0613 (X) and 0x0614 (Y). This is the position of
  the image centre. Stage readout repeatability is about +-2 um.
* magnification: tag 0x011E; capture time: tag 0x0002 (OLE automation date).

Coordinate convention (see cii.geometry)
----------------------------------------
Stage coordinates are in um. Image column index increases with stage X and
image row index increases with stage Y (verified from overlapping image pairs
and stitched overviews). The stage position refers to the image centre, i.e.
pixel ((width - 1) / 2, (height - 1) / 2) in 0-based pixel indices.
"""
from __future__ import annotations

import datetime as _dt
import re
import struct
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import tifffile

from .geometry import StageImage

# MakerNote tags (empirical)
TAG_TIME = 0x0002
TAG_WIDTH = 0x0003
TAG_HEIGHT = 0x0004
TAG_FIELD_A = 0x0011
TAG_MAGNIFICATION = 0x011E
TAG_FIELD_B = 0x0144
TAG_FIELD_C = 0x0148
TAG_STAGE_X = 0x0613
TAG_STAGE_Y = 0x0614
TAG_STAGE_Z = 0x0615  # probably Z/focus position; not verified


@dataclass
class Panorama:
    """Stitching / series information stored by the VHX (if present)."""

    path_mode: str | None = None
    corner_1: tuple[float, float] | None = None  # stage um, centre of first corner tile
    corner_2: tuple[float, float] | None = None  # stage um, centre of opposite corner tile
    x_count: int | None = None
    y_count: int | None = None
    raw_xml: str = ""


@dataclass
class VHXImage(StageImage):
    """VHX image: stage X increases to the right, stage Y downwards (x_dir = y_dir = +1)."""

    magnification: float | None = None
    stage_z: float | None = None
    captured: _dt.datetime | None = None
    panorama: Panorama | None = None
    has_multilighting_layers: bool = False
    raw_fields: dict = field(default_factory=dict, repr=False)

    def read_image(self) -> np.ndarray:
        """The main image (H x W x 3, uint8)."""
        return tifffile.imread(self.path, key=0)


def _parse_makernote(data: bytes, makernote: bytes) -> dict:
    """Keyence 'KmsFile' MakerNote as {tag: value}: 12-byte entries (tag, type, count, value/offset);
    entries of 14 undefined bytes hold a double at absolute file offset + 6."""
    if not makernote.startswith(b"KmsFile"):
        raise ValueError("MakerNote is not a Keyence 'KmsFile' block")
    fields = {}
    n = struct.unpack_from("<H", makernote, 8)[0]
    for i in range(n):
        off = 10 + 12 * i
        if off + 12 > len(makernote):
            break
        tag, typ, cnt, val = struct.unpack_from("<HHII", makernote, off)
        if typ == 7 and cnt == 14:  # 4-byte id, 2-byte type code, 8-byte double
            fields[tag] = struct.unpack("<d", data[val + 6: val + 14])[0]
        elif typ == 3 and cnt == 1:
            fields[tag] = val & 0xFFFF
        elif typ == 4 and cnt == 1:
            fields[tag] = val
        elif typ == 9 and cnt == 1:
            fields[tag] = struct.unpack("<i", struct.pack("<I", val))[0]
        elif typ == 11 and cnt == 1:
            fields[tag] = struct.unpack("<f", struct.pack("<I", val))[0]
    return fields


def _ole_to_datetime(value: float) -> _dt.datetime:
    """OLE automation date (days since 30 Dec 1899) as datetime."""
    return _dt.datetime(1899, 12, 30) + _dt.timedelta(days=value)


def _parse_panorama(data: bytes) -> Panorama | None:
    """Series / stitching information from the embedded QuickReplayPanorama2DData XML (path mode, centres of
    the two corner tiles, tile counts), or None if the image has none."""
    i = data.rfind(b"<QuickReplayPanorama2DData")
    if i == -1:
        return None
    start = data.rfind(b"<?xml", 0, i)
    end = data.find(b"</QuickReplayPanorama2DData>", i)
    if end == -1:
        return None
    xml = data[start if start != -1 else i: end + len(b"</QuickReplayPanorama2DData>")]
    text = xml.decode("utf-8", "replace").lstrip("﻿")
    pano = Panorama(raw_xml=text)
    try:
        root = ET.fromstring(re.sub(r"^<\?xml[^>]*\?>", "", text))
    except ET.ParseError:
        return pano
    pano.path_mode = root.findtext(".//PathMode")

    def point(name):
        """(X, Y) of the XML element name, or None if it is missing."""
        el = root.find(f".//{name}")
        if el is None or el.findtext("X") is None:
            return None
        return float(el.findtext("X")), float(el.findtext("Y"))

    if pano.path_mode == "RectangleEdge":
        pano.corner_1, pano.corner_2 = point("Left"), point("Right")
    else:
        pano.corner_1 = point("StartPosition")
    xc, yc = root.findtext(".//XCount"), root.findtext(".//YCount")
    pano.x_count = int(xc) if xc else None
    pano.y_count = int(yc) if yc else None
    return pano


def read_vhx(path) -> VHXImage:
    """Read metadata of a VHX TIFF (the image itself is read lazily)."""
    path = Path(path)
    data = path.read_bytes()
    with tifffile.TiffFile(path) as tif:
        page = tif.pages[0]
        height, width = page.shape[:2]
        exif = page.tags.get("ExifTag")
        makernote = exif.value.get("MakerNote") if exif is not None else None
    if makernote is None:
        raise ValueError(f"{path.name}: no Keyence MakerNote found "
                         "(was the file re-saved by another program?)")
    f = _parse_makernote(data, makernote)
    for tag in (TAG_FIELD_A, TAG_FIELD_B, TAG_FIELD_C, TAG_STAGE_X, TAG_STAGE_Y):
        if tag not in f:
            raise ValueError(f"{path.name}: MakerNote field 0x{tag:04x} missing")
    um_per_px = f[TAG_FIELD_A] / (f[TAG_FIELD_B] * f[TAG_FIELD_C])
    return VHXImage(
        path=path, width=width, height=height,
        magnification=f.get(TAG_MAGNIFICATION),
        um_per_px=um_per_px,
        stage_x=f[TAG_STAGE_X], stage_y=f[TAG_STAGE_Y], stage_z=f.get(TAG_STAGE_Z),
        captured=_ole_to_datetime(f[TAG_TIME]) if TAG_TIME in f else None,
        panorama=_parse_panorama(data),
        has_multilighting_layers=b"KFTL" in data,
        raw_fields=f,
    )


def stitched_origin(img: VHXImage) -> tuple[float, float] | None:
    """Stage position of pixel (0, 0) of a VHX-stitched image (RectangleEdge mode).

    The VHX centres the first tile on the first corner point; the stitched image
    then extends half a tile beyond it. The tile size is 2880 x 2160 px at the
    image's pixel size. Approximate: the VHX stitch deviates from the nominal
    geometry (a 100x overview was 27 px wider than nominal; indent positions
    agreed within about 20 um). Use single images for precise coordinates.
    """
    p = img.panorama
    if p is None or p.corner_1 is None or p.corner_2 is None:
        return None
    x0 = min(p.corner_1[0], p.corner_2[0]) - (2880 - 1) / 2 * img.um_per_px
    y0 = min(p.corner_1[1], p.corner_2[1]) - (2160 - 1) / 2 * img.um_per_px
    return x0, y0
