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

Every instrument has its own **stage coordinates in µm**; images know where they
sit in them (`cii.geometry.StageImage`):

* The stage position of an image refers to the **image centre**, pixel
  `((width-1)/2, (height-1)/2)` (0-based).
* Axis directions relative to the image:

  | Instrument | stage X | stage Y |
  | --- | --- | --- |
  | Keyence VHX-7000 | increases to the right (`x_dir = +1`) | increases downwards |
  | SEM, Oxford AZtec | increases to the **left** (`x_dir = -1`) | increases downwards |

* `pixel_to_stage()` / `stage_to_pixel()` convert between pixels and stage µm.
* Between instruments: `cii.registration.Transform` (fitted from indents),
  `dst = A @ [x, y] + t`.

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

## What `cii.sem` reads from an AZtec TIFF

Private TIFF tag 50684 (`key=value` text): `PixelExtent` (m/px), magnification,
kV, working distance, date. Its stage position is rounded to 0.1 mm, so the
image-centre position is taken from the field table exported with the feature
analysis (`FieldID, StageXmm, StageYmm`), matched via `Site_<n>` in the file name.

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

Stage coordinates of hardness indents. For each indent, put the original image
`<name>.tif` and a copy with a red circle drawn around the indent,
`<name>_marked.tif` (or `.png`), into one folder. The label used to match
indents between instruments is `Indent_<word>` from the file name.

```bash
# light microscope
python scripts/locate_indents.py data/MnB/vhx_indents -o vhx_indents.csv \
    --figures figures --check data/MnB/vhx_indents/Indent_botleft_200x.tif
# SEM
python scripts/locate_indents.py data/MnB/sem_indents --instrument sem \
    --fields data/MnB/field_information.csv -o sem_indents.csv --figures figures
```

Segmentation: VHX - centroid of the dark cavity; SEM (BSE) - convex hull of the
strongly bright/dark indent faces (robust z-score > 3.5 after smoothing, pieces
within 8 px grouped). `--figures` writes one PNG per indent (whole image + zoom
with the segmented outline and centroid). `--check` verifies indent positions in
further images (e.g. another objective).

### `scripts/fit_transform.py`

```bash
python scripts/fit_transform.py vhx_indents.csv sem_indents.csv -o vhx_to_sem.json
```

Similarity transform (rotation, scale, translation, mirror) from the matched
indents; prints residuals and the leave-one-out error (each indent predicted
from the others), writes the transform as JSON. `--model affine` for an affine fit.

## Results so far

MnB sample (VHX 2 Nov 2026, SEM 30 Oct 2026), 4 indents, VHX -> SEM similarity:
X mirrored, scale 1.0011, rotation +0.41 deg, RMS residual 1.5 µm,
leave-one-out error <= 4.2 µm.

## Roadmap

1. SEM/EDS particle list import with stage X/Y per particle.
2. Refine the VHX -> SEM transform with inclusion centroids (done: indent-based transform).
3. Inclusion detection in VHX series images and matching to SEM particles.
4. Dataset export (image crops + SEM class) for training.
