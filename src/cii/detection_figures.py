"""Review images for the inclusion detection: full-resolution overlay and gallery."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from skimage.segmentation import find_boundaries

from .detection import Detection

FILL = (235, 104, 52)       # objects
EDGE = (190, 60, 20)
BORDER = (42, 120, 214)     # objects touching the image border
TEXT = (20, 20, 20)


def _font(size: int):
    for name in ("DejaVuSans-Bold.ttf", "DejaVuSans.ttf", "Arial.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _rgb(image: np.ndarray) -> np.ndarray:
    return np.repeat(image[..., None], 3, axis=2) if image.ndim == 2 else image[..., :3]


def overlay_array(image: np.ndarray, det: Detection, alpha: float = 0.45) -> np.ndarray:
    """RGB image with objects filled (orange; blue if touching the border) and outlined."""
    out = _rgb(image).astype(np.float32).copy()
    lab = det.labels
    border_labels = det.objects.loc[det.objects.touches_border, "label"].to_numpy() if len(det.objects) else []
    is_border = np.isin(lab, border_labels)
    for sel, col in ((lab > 0) & ~is_border, FILL), (is_border, BORDER):
        out[sel] = (1 - alpha) * out[sel] + alpha * np.array(col, np.float32)
    edges = find_boundaries(lab, mode="outer")
    out[edges & ~np.isin(lab, border_labels)] = EDGE
    return out.clip(0, 255).astype(np.uint8)


def save_overlay(image: np.ndarray, det: Detection, path, label_min_area_um2: float = 0.0,
                 um_per_px: float | None = None, title: str | None = None) -> Path:
    """Full-resolution overlay PNG with object IDs written next to the objects."""
    path = Path(path)
    im = Image.fromarray(overlay_array(image, det))
    d = ImageDraw.Draw(im)
    fs = max(14, im.width // 140)
    font = _font(fs)
    for _, o in det.objects.iterrows():
        if o.area_um2 < label_min_area_um2:
            continue
        x, y = o.bbox_col1 + 3, o.bbox_row0 - fs - 2
        if x + 4 * fs > im.width:
            x = o.bbox_col0 - 3 * fs
        y = max(0, y)
        d.text((x, y), str(int(o.id)), fill=TEXT, font=font, stroke_width=3, stroke_fill=(255, 255, 255))
    if title:
        tf = _font(int(fs * 1.4))
        d.text((10, 8), title, fill=TEXT, font=tf, stroke_width=4, stroke_fill=(255, 255, 255))
    if um_per_px:
        _scale_bar(d, im.width, im.height, um_per_px, font)
    path.parent.mkdir(parents=True, exist_ok=True)
    im.save(path, optimize=True)
    return path


def _scale_bar(d: ImageDraw.ImageDraw, w: int, h: int, um_per_px: float, font, frac: float = 0.15):
    target = w * um_per_px * frac
    bar = max(v for v in (1, 2, 5, 10, 20, 50, 100, 200, 500, 1000) if v <= max(target, 1))
    L = bar / um_per_px
    x0, y0 = w - L - 30, h - 40
    d.rectangle([x0 - 8, y0 - 34, x0 + L + 8, y0 + 18], fill=(255, 255, 255))
    d.rectangle([x0, y0, x0 + L, y0 + 8], fill=TEXT)
    d.text((x0 + L / 2, y0 - 4), f"{bar:g} µm", fill=TEXT, font=font, anchor="mb")


def save_gallery(image: np.ndarray, det: Detection, um_per_px: float, path, max_objects: int = 300,
                 tile_px: int = 180, cols: int = 12, min_field_um: float = 8.0, title: str | None = None) -> Path:
    """One tile per object (largest first): crop with outline, ID and ECD."""
    path = Path(path)
    rgb = _rgb(image)
    objs = det.objects.head(max_objects)
    n = len(objs)
    rows = max(1, int(np.ceil(n / cols)))
    cap = 26
    head = 44 if title else 0
    sheet = Image.new("RGB", (cols * tile_px, head + rows * (tile_px + cap)), "white")
    d = ImageDraw.Draw(sheet)
    font, tfont = _font(14), _font(20)
    if title:
        d.text((8, 10), title, fill=TEXT, font=tfont)
    H, W = det.labels.shape
    for k, (_, o) in enumerate(objs.iterrows()):
        size_px = max(min_field_um / um_per_px,
                      2.2 * max(o.bbox_row1 - o.bbox_row0, o.bbox_col1 - o.bbox_col0))
        half = int(np.ceil(size_px / 2))
        cy, cx = int(round(o.row_px)), int(round(o.col_px))
        r0, r1, c0, c1 = max(0, cy - half), min(H, cy + half), max(0, cx - half), min(W, cx + half)
        crop = rgb[r0:r1, c0:c1].astype(np.float32).copy()
        own = det.labels[r0:r1, c0:c1] == o.label
        edge = find_boundaries(own, mode="outer")
        crop[edge] = BORDER if o.touches_border else EDGE
        scale = tile_px / max(crop.shape[:2])
        tile = Image.fromarray(crop.clip(0, 255).astype(np.uint8)).resize(
            (max(1, int(crop.shape[1] * scale)), max(1, int(crop.shape[0] * scale))), Image.NEAREST)
        x, y = (k % cols) * tile_px, head + (k // cols) * (tile_px + cap)
        sheet.paste(tile, (x, y))
        d.text((x + 4, y + tile_px + 4), f"#{int(o.id)}  {o.ecd_um:.1f} µm", fill=TEXT, font=font)
        fov = (c1 - c0) * um_per_px
        d.text((x + tile_px - 4, y + tile_px + 4), f"{fov:.0f} µm", fill=(110, 110, 110), font=font, anchor="ra")
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path, optimize=True)
    return path
