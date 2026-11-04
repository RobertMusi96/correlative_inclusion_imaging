# correlative_inclusion_imaging

Correlative characterisation of non-metallic inclusions: Keyence VHX-7000 digital
light microscopy and automated SEM/EDS on the same sample area. Goal: a
training dataset in which SEM/EDS supplies the inclusion class for each
inclusion seen in the light microscope.

## Install

```bash
pip install -e ".[dev]"
python -m pytest
```

Image data is not part of the repository (see `.gitignore`); keep it under `data/`
or anywhere else and pass paths to the scripts.

## Coordinate convention

All positions are VHX **stage coordinates in µm**.

* The stage position stored in an image refers to the **image centre**, pixel
  `((width-1)/2, (height-1)/2)` (0-based).
* Image **columns increase with stage X**, image **rows increase with stage Y**
  (stage Y grows "downwards" on screen).
* `VHXImage.pixel_to_stage()` / `stage_to_pixel()` convert between the two.

## What `cii.vhx` reads from a VHX TIFF

The VHX stores its acquisition data in a proprietary EXIF MakerNote. Fields used
(identified empirically, see module docstring):

| Quantity | Source | Verified |
| --- | --- | --- |
| Pixel size | `A / (B * C)`, MakerNote tags 0x0011, 0x0144, 0x0148 | 100x-1000x and 4K mode vs. VHX area readout (< 0.001 %) |
| Stage X, Y (µm) | tags 0x0613, 0x0614 | step between overlapping images vs. image matching (< 3 µm) |
| Magnification | tag 0x011E | |
| Capture time | tag 0x0002 (OLE date) | |
| Series / stitch region | embedded `QuickReplayPanorama2DData` XML | |

Measured pixel sizes at 2880 x 2160 px (VHX-7000, Nov 2026):
100x 1.0275, 200x 0.5141, 500x 0.2037, 1000x 0.1023 µm/px.

## Notes on VHX serial recording

* The two corner points of a series are the **centres** of the corner tiles. The
  covered area is therefore the corner rectangle plus half a tile on every side,
  and it shrinks with magnification. For comparisons use the corner rectangle.
* The VHX picks the number of tiles for a minimum overlap of about 6.5 % and
  spreads them evenly, so the actual overlap differs per magnification.
* Stage readout repeatability: about +-2 µm. Different objectives agree within
  about 3 µm (indent at 100x vs. 200x).
* Multi-lighting layers (four directions + coaxial) are only stored in
  VHX-stitched files (tile container "KFTL"); series images contain only the
  rendered image.

## Scripts

### `scripts/locate_indents.py`

Stage coordinates of hardness indents. For each indent, put the original VHX TIFF
`<name>.tif` and a copy with a red circle drawn around the indent,
`<name>_marked.tif` (or `.png`), into one folder:

```bash
python scripts/locate_indents.py data/2026-11-02_MnB/indents -o indents.csv \
    --check data/2026-11-02_MnB/indents/Indent_botleft_200x.tif
```

The indent centre is the centroid of the dark cavity inside the circle
(sub-pixel). `--check` searches further images for the indents and reports the
deviation, e.g. to verify the coordinate frame across objectives.

## Roadmap

1. SEM/EDS particle list import (needs stage X/Y per particle).
2. VHX -> SEM transform from the indents (similarity/affine), refined with
   inclusion centroids.
3. Inclusion detection in VHX series images and matching to SEM particles.
4. Dataset export (image crops + SEM class) for training.
