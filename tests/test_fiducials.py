import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage as ndi

from cii.fiducials import find_marking, locate_indent


def synthetic_indent(h=600, w=800, col=412.3, row=287.6, radius=15.0):
    yy, xx = np.mgrid[0:h, 0:w]
    img = np.full((h, w, 3), 220, np.uint8)
    cavity = (xx - col) ** 2 + (yy - row) ** 2 <= radius ** 2
    img[cavity] = 60
    img[(xx - col) ** 2 + (yy - row) ** 2 <= 3 ** 2] = 250  # bright reflection in the centre
    return img


def test_locate_indent_subpixel():
    img = synthetic_indent()
    fit = locate_indent(img, 400, 300, 60)
    assert abs(fit.col - 412.3) < 0.3 and abs(fit.row - 287.6) < 0.3
    assert abs(fit.diameter_px - 30) < 2


def test_locate_indent_edges_bse_like():
    # grey noisy matrix, triangle indent outlined by a dark shadow and bright edges
    rng = np.random.default_rng(1)
    h, w = 400, 500
    img = rng.normal(23000, 600, (h, w))
    yy, xx = np.mgrid[0:h, 0:w]
    apex, left, right = (250.0, 120.0), (200.0, 230.0), (310.0, 230.0)   # (col, row)

    def side(p, q):
        return (xx - p[0]) * (q[1] - p[1]) - (yy - p[1]) * (q[0] - p[0])

    inside = (side(apex, left) <= 0) & (side(left, right) <= 0) & (side(right, apex) <= 0)
    inside |= (side(apex, left) >= 0) & (side(left, right) >= 0) & (side(right, apex) >= 0)
    edge = inside & ~ndi.binary_erosion(inside, iterations=4)
    img[edge & (yy > 220)] = 6000      # dark shadow along the bottom edge
    img[edge & (yy <= 220)] = 6000     # dark diagonal faces
    img[edge & (xx < 215) & (yy > 150)] = 32000   # bright left edge
    fit = locate_indent(img, 250, 200, 120, method="edges")
    true_c = ((apex[0] + left[0] + right[0]) / 3, (apex[1] + left[1] + right[1]) / 3)
    assert abs(fit.col - true_c[0]) < 2 and abs(fit.row - true_c[1]) < 3


def test_find_marking(tmp_path):
    img = Image.fromarray(synthetic_indent())
    ImageDraw.Draw(img).ellipse([362, 237, 462, 337], outline=(230, 20, 20), width=6)
    p = tmp_path / "x_marked.tif"
    img.save(p)
    mk = find_marking(p)
    assert abs(mk.col - 412) < 2 and abs(mk.row - 287) < 2 and 45 < mk.radius < 56
