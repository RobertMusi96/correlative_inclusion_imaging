"""Verification figures for the SEM <-> VHX matching: BSE crop and VHX crop side by side."""
from __future__ import annotations

import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tifffile
from matplotlib.patches import Rectangle

STATUS_COLOUR = {"matched": "#2a78d6", "ambiguous": "#d6a12a", "not_found": "#e8590c"}


def bse_paths(folder: Path, verbose: bool = True) -> dict[int, Path]:
    """Field id -> BSE image below ``folder``.

    The site number is taken from the whole path ('Site_135' in a flattened file
    name or a folder 'Site 135'); only TIFFs with 'electron' in the file name or
    path are used (not EDS maps). Several electron images for one site: the one
    with 'electron' in its file name, then the shortest name.
    """
    from .sem import field_id_from_name
    folder = Path(folder)
    tifs = [f for f in folder.rglob("*") if f.is_file() and f.suffix.lower() in (".tif", ".tiff")]
    cand: dict[int, list[Path]] = {}
    for f in tifs:
        fid = field_id_from_name(str(f))
        if fid is None or "electron" not in str(f).lower():
            continue
        cand.setdefault(fid, []).append(f)
    out = {fid: sorted(fs, key=lambda f: ("electron" not in f.name.lower(), len(f.name)))[0] for fid, fs in cand.items()}
    if verbose:
        multi = sum(len(v) > 1 for v in cand.values())
        print(f"BSE images: {len(tifs)} TIFFs below {folder}, {len(out)} fields with an electron image"
              + (f" ({multi} fields with several, first by name used)" if multi else ""))
        if not out and tifs:
            print("  no 'Site <n>' + 'electron' found; examples of TIFF paths:")
            for f in tifs[:5]:
                print("   ", f)
    return out


class ImageCache:
    def __init__(self, loader, size: int = 8):
        self.loader, self.size, self.data = loader, size, {}

    def get(self, key):
        if key not in self.data:
            if len(self.data) >= self.size:
                self.data.pop(next(iter(self.data)))
            self.data[key] = self.loader(key)
        return self.data[key]


def _load_bse(path):
    a = tifffile.imread(path).astype(np.float32)
    if a.ndim == 3:
        a = a[..., 0]
    lo, hi = np.percentile(a, [0.5, 99.8])
    return np.clip((a - lo) / max(hi - lo, 1), 0, 1)


def _load_vhx(path):
    a = tifffile.imread(path, key=0)
    return a[..., :3] if a.ndim == 3 else a


def resolve_vhx_paths(vhx_img: pd.DataFrame, folder: Path | None = None) -> dict[str, Path]:
    """Image name -> existing file: in ``folder`` (searched with subfolders) if given,
    otherwise at the path stored in images.csv. Missing images are left out."""
    out = {}
    if folder is not None:
        by_name = {}
        for f in Path(folder).rglob("*"):
            if f.suffix.lower() in (".tif", ".tiff"):
                by_name.setdefault(f.name, f)
        for n in vhx_img.image:
            if n in by_name:
                out[n] = by_name[n]
    for n, pth in zip(vhx_img.image, vhx_img.path):
        if n not in out and isinstance(pth, str) and Path(pth).exists():
            out[n] = Path(pth)
    return out


def _crop(img, cx, cy, half_px):
    """Crop around (cx, cy) in pixels, padded with white where outside; returns crop and its origin."""
    h, w = img.shape[:2]
    x0, y0 = int(round(cx - half_px)), int(round(cy - half_px))
    n = int(round(2 * half_px)) + 1
    shape = (n, n) + img.shape[2:]
    out = np.full(shape, 1.0 if img.dtype.kind == "f" else 255, dtype=img.dtype)
    xs, ys = max(0, x0), max(0, y0)
    xe, ye = min(w, x0 + n), min(h, y0 + n)
    if xe > xs and ye > ys:
        out[ys - y0:ye - y0, xs - x0:xe - x0] = img[ys:ye, xs:xe]
    return out, x0, y0


def verification_sheet(rows: pd.DataFrame, vhx_img: pd.DataFrame, bse: dict, path: Path, title: str,
                       sem_um_per_px: float = 0.294864, cols: int = 4, min_field_um: float = 10.0,
                       vhx_paths: dict | None = None):
    """One tile pair per SEM particle: BSE (SEM bbox) | VHX at the predicted position (matched object's bbox)."""
    if not len(rows):
        return False
    vi = vhx_img.set_index("image")
    vhx_paths = vhx_paths if vhx_paths is not None else resolve_vhx_paths(vhx_img)
    vpath = lambda n: vhx_paths.get(n)
    bse_cache, vhx_cache = ImageCache(_load_bse), ImageCache(_load_vhx, 4)
    nrow = int(np.ceil(len(rows) / cols))
    fig, axes = plt.subplots(nrow, 2 * cols, figsize=(2 * cols * 1.75, nrow * 2.05), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for k, r in enumerate(rows.itertuples()):
        a1, a2 = axes[k // cols, 2 * (k % cols)], axes[k // cols, 2 * (k % cols) + 1]
        half_um = max(min_field_um / 2, 1.3 * r.ecd_um)
        col = STATUS_COLOUR.get(r.status, "k")
        # BSE
        if r.field in bse:
            img = bse_cache.get(bse[r.field])
            cx, cy = r.bb_left_px + r.bb_width_px / 2, r.bb_top_px + r.bb_height_px / 2
            hp = half_um / sem_um_per_px
            c, x0, y0 = _crop(img, cx, cy, hp)
            a1.imshow(c, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
            a1.add_patch(Rectangle((r.bb_left_px - x0 - 0.5, r.bb_top_px - y0 - 0.5), r.bb_width_px, r.bb_height_px,
                                   fill=False, ec="#e8590c", lw=0.7))
        else:
            a1.text(0.5, 0.5, "no BSE image", ha="center", va="center", fontsize=6, transform=a1.transAxes)
        # VHX
        name = r.vhx_image if isinstance(r.vhx_image, str) else None
        if name and name in vi.index and np.isfinite(r.pred_vhx_x_um) and vpath(name) is not None:
            g = vi.loc[name]
            img = vhx_cache.get(vpath(name))
            px = lambda x, y: ((x - g.stage_x_um) / g.um_per_px + (g.width_px - 1) / 2,
                               (y - g.stage_y_um) / g.um_per_px + (g.height_px - 1) / 2)
            cx, cy = px(r.pred_vhx_x_um, r.pred_vhx_y_um)
            hp = half_um / g.um_per_px
            c, x0, y0 = _crop(img, cx, cy, hp)
            a2.imshow(c, interpolation="nearest")
            a2.plot(cx - x0, cy - y0, "+", color="#e8590c", ms=6, mew=0.8)
            if r.status in ("matched", "ambiguous") and np.isfinite(r.vhx_bbox_col0):
                region = getattr(r, "match_type", "") == "region" and np.isfinite(getattr(r, "vhx_region_bbox_col0", np.nan))
                pre = "vhx_region_bbox_" if region else "vhx_bbox_"
                c0, r0 = getattr(r, pre + "col0"), getattr(r, pre + "row0")
                a2.add_patch(Rectangle((c0 - x0 - 0.5, r0 - y0 - 0.5), getattr(r, pre + "col1") - c0,
                                       getattr(r, pre + "row1") - r0, fill=False, ec=col, lw=0.8,
                                       ls="--" if region else "-"))
        else:
            a2.text(0.5, 0.5, "no VHX image", ha="center", va="center", fontsize=6, transform=a2.transAxes)
        typ = r.Type if isinstance(r.Type, str) else r.class_label
        a1.set_title(f"[{k + 1}]  #{r.sem_id} {typ}\nSEM {r.ecd_um:.2f} µm", fontsize=6)
        if r.status in ("matched", "ambiguous"):
            region = getattr(r, "match_type", "") == "region"
            vtxt = (f"VHX region {r.vhx_region_ecd_um:.2f} µm" if region else f"VHX {r.vhx_ecd_um:.2f} µm") + f", d {r.distance_um:.1f}"
        else:
            vtxt = "no object"
        a2.set_title(f"{r.status}{' (region)' if getattr(r, 'match_type', '') == 'region' else ''}\n{vtxt}", fontsize=6, color=col)
    fig.suptitle(title + "  (left BSE with SEM bbox, right VHX at predicted position +, matched object box; "
                 "dashed = detected region; field 2.6 x ECD, at least 10 µm)", fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return True


def overview(part: pd.DataFrame, path: Path):
    c = part[part.status.isin(["matched", "ambiguous", "not_found"])]
    fig, (a, b) = plt.subplots(1, 2, figsize=(13, 4.6))
    g = c.groupby("class_label").status.agg(n="size", found=lambda s: 100 * (s != "not_found").mean())
    order = [k for k in ["N", "NS", "O", "ONS", "OS", "S", "not_typified", "rejected"] if k in g.index] + \
            [k for k in g.index if k not in ["N", "NS", "O", "ONS", "OS", "S", "not_typified", "rejected"]]
    g = g.loc[order]
    a.bar(range(len(g)), g.found, color=["#2a78d6"] * (len(g) - 2) + ["#9aa5b1", "#c9ced6"][: min(2, len(g))])
    for k, (n, f) in enumerate(zip(g.n, g.found)):
        a.text(k, f + 1.5, f"{f:.0f}%\nn={n}", ha="center", fontsize=8)
    a.set_xticks(range(len(g)), g.index, rotation=30, ha="right"); a.set_ylim(0, 115)
    a.set_ylabel("SEM particles with VHX object (%)"); a.spines[["top", "right"]].set_visible(False)
    bins = [1, 1.5, 2, 3, 5, 10, 1000]
    for lab, colr in [("typified", "#2a78d6"), ("not_typified", "#7f8c99"), ("rejected", "#e8590c")]:
        s = c[c.label == lab]
        h = s.groupby(pd.cut(s.ecd_um, bins), observed=False).status.apply(lambda x: 100 * (x != "not_found").mean() if len(x) else np.nan)
        b.plot(range(len(h)), h.to_numpy(), "o-", color=colr, label=lab)
    b.set_xticks(range(len(bins) - 1), [f"{lo:g}-{hi:g}" if hi < 1000 else f">{lo:g}" for lo, hi in zip(bins[:-1], bins[1:])])
    b.set_xlabel("SEM ECD (µm)"); b.set_ylabel("with VHX object (%)"); b.set_ylim(0, 105); b.legend(frameon=False)
    b.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)
