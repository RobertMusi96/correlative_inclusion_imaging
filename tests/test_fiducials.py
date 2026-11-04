import numpy as np
from PIL import Image, ImageDraw

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


def test_find_marking(tmp_path):
    img = Image.fromarray(synthetic_indent())
    ImageDraw.Draw(img).ellipse([362, 237, 462, 337], outline=(230, 20, 20), width=6)
    p = tmp_path / "x_marked.tif"
    img.save(p)
    mk = find_marking(p)
    assert abs(mk.col - 412) < 2 and abs(mk.row - 287) < 2 and 45 < mk.radius < 56
