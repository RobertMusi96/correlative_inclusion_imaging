"""Figures for documentation and presentations."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
from skimage.measure import find_contours  # noqa: E402

from .fiducials import IndentFit  # noqa: E402

OUTLINE = "#eb6834"   # segmented indent
CENTRE = "#2a78d6"    # centroid
INK, INK2 = "#1f1f1d", "#5f5e58"


def _display(image: np.ndarray) -> tuple[np.ndarray, str | None]:
    """Image for display: colour images (light microscope) unchanged, grey images
    (SEM) contrast-stretched to percentiles 0.5 / 99.5."""
    if image.ndim == 3 and image.dtype == np.uint8:
        return image[..., :3].astype(float) / 255, None
    a = image.astype(float)
    lo, hi = np.percentile(a, [0.5, 99.5])
    a = np.clip((a - lo) / (hi - lo + 1e-9), 0, 1)
    return (a, None) if a.ndim == 3 else (a, "gray")


def _nice_bar(width_um: float) -> float:
    """Scale bar length (1, 2, 5, 10 ... 1000 um) of at most a quarter of the shown width."""
    for v in (1000, 500, 200, 100, 50, 20, 10, 5, 2, 1):
        if v <= width_um / 4:
            return v
    return 1


def _scale_bar(ax, x0, y0, length_px, label, colour="white"):
    """Scale bar with label above it; outlined so it reads on light and dark images."""
    import matplotlib.patheffects as pe
    halo = [pe.withStroke(linewidth=3, foreground="black", alpha=0.6)]
    bottom, top = ax.get_ylim()
    ax.plot([x0, x0 + length_px], [y0, y0], color=colour, lw=4, solid_capstyle="butt", path_effects=halo)
    ax.text(x0 + length_px / 2, y0 - 0.03 * abs(bottom - top), label, color=colour, ha="center",
            va="bottom", fontsize=10, fontweight="bold", path_effects=halo)


def save_indent_figure(image: np.ndarray, fit: IndentFit, um_per_px: float, path,
                       title: str, info: list[str], zoom_um: float | None = None) -> Path:
    """Two panels: whole image with the zoom window, and the zoom with the
    outline of the segmented indent and its centroid."""
    path = Path(path)
    disp, cmap = _display(image)
    h, w = disp.shape[:2]
    d_um = fit.diameter_px * um_per_px
    zoom_um = zoom_um or max(4 * d_um, 80.0)
    half = zoom_um / um_per_px / 2
    c0, c1 = int(max(0, fit.col - half)), int(min(w, fit.col + half))
    r0, r1 = int(max(0, fit.row - half)), int(min(h, fit.row + half))

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 5.4), gridspec_kw={"width_ratios": [1.33, 1]})
    fig.patch.set_facecolor("white")
    for a in (a1, a2):
        a.set_xticks([]); a.set_yticks([])

    a1.imshow(disp, cmap=cmap, interpolation="lanczos")
    a1.add_patch(Rectangle((c0, r0), c1 - c0, r1 - r0, fill=False, ec=OUTLINE, lw=1.8))
    bar = _nice_bar(w * um_per_px)
    a1.set_xlim(0, w); a1.set_ylim(h, 0)
    _scale_bar(a1, 0.05 * w, 0.93 * h, bar / um_per_px, f"{bar:g} µm")
    a1.set_title("Whole image", loc="left", fontsize=11, color=INK2)

    a2.imshow(disp[r0:r1, c0:c1], cmap=cmap, interpolation="nearest",
              extent=(c0 - 0.5, c1 - 0.5, r1 - 0.5, r0 - 0.5))
    m_r0, _, m_c0, _ = fit.bbox
    for cnt in find_contours(fit.mask.astype(float), 0.5):
        a2.plot(cnt[:, 1] + m_c0, cnt[:, 0] + m_r0, color=OUTLINE, lw=2)
    a2.plot(fit.col, fit.row, marker="+", ms=16, mew=2.2, color=CENTRE)
    a2.set_xlim(c0 - 0.5, c1 - 0.5); a2.set_ylim(r1 - 0.5, r0 - 0.5)
    zbar = _nice_bar((c1 - c0) * um_per_px)
    _scale_bar(a2, c0 + 0.06 * (c1 - c0), r0 + 0.92 * (r1 - r0), zbar / um_per_px, f"{zbar:g} µm")
    a2.set_title("Segmented indent and centroid", loc="left", fontsize=11, color=INK2)

    fig.suptitle(title, x=0.01, ha="left", fontsize=13, fontweight="bold", color=INK)
    fig.text(0.01, 0.015, "   ·   ".join(info), fontsize=9.5, color=INK2, ha="left")
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path
