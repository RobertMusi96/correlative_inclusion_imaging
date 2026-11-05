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

### `scripts/detect_inclusions.py`

Detection of dark objects in VHX bright-field images (`cii.detection`):
background by median filter (40 µm), contrast = (background - value) / background,
hysteresis threshold (core >= `high`, grown into pixels >= `low`), object outline
at half of the object's own peak contrast (`size_fraction`, makes sizes
independent of blur), minimum area 0.3 µm² (applied to the detected object).

Contrast modes (`--channels`):

| mode | contrast | default high / low | use |
| --- | --- | --- | --- |
| `max` (default) | R, G, B separately, strongest of the three | 0.07 / 0.04 | also catches coloured particles that are darker in one colour only (pink Ti(C,N): mainly green) |
| `mean` | grey value (mean of R, G, B), original method | 0.12 / 0.03 | dark/grey particles; fewer faint objects |

`max` also finds many sub-µm violet dots (~1000/mm², about 0.6 µm at 1000x,
shown as ~1.05 µm at 500x because of blur) and faint polishing scratches;
they are below the SEM limit and drop out in the SEM matching. A clipped
channel (blue at 255 in 91 % of the pixels at 500x) carries no contrast:
record with the matrix at about 180-200. The per-channel contrast of every
object is stored (`contrast_R/G/B_mean`, `contrast_R/G/B_max`).

```bash
# whole series: tables only, 4 parallel processes (~2 s per image and process)
python scripts/detect_inclusions.py data/MnB/vhx_500x/*.tif -o results/detection_500x --no-figures -j 4
# a few images with review figures (same parameters -> same objects)
python scripts/detect_inclusions.py data/MnB/vhx_500x/*_00001*.tif -o results/check_500x
```

Writes `objects.csv` (pixel and stage coordinates, area, ECD, shape, contrast,
colour, border flag), `images.csv` (path, stage position, size, object count and
saturated share per image), `parameters.json`, and per image (unless
`--no-figures`) `overlay/<image>.png` (full resolution, objects orange, border
objects blue, IDs) and `gallery/<image>.png` (one tile per object, largest first).
Use one output folder per series / magnification.

### `scripts/compare_magnifications.py`

Checks the detection with two series of the same area at different
magnifications: objects of the higher magnification are searched among the
objects of the lower one (`cii.compare`). Works on whole series:

* offset per overlapping image pair (objective offset + stage error of both
  images), estimated from the objects; pairs with too few objects get it from a
  model with one stage error per image; offsets contradicting the model are
  re-estimated;
* objects seen in two overlapping images of the reference series are counted
  once (the most central view); border objects are skipped;
* the expected rate of random matches is printed (density x match circle).

```bash
python scripts/compare_magnifications.py results/detection_1000x results/detection_500x \
    -o results/compare_1000x_500x --missed-crops 80
```

Writes `summary.csv` (share found and size ratio per ECD class), `objects.csv`
(every reference object with status found / missed / border / duplicate /
outside and its match), `offsets.csv` (per image pair), `comparison.png` (found
share per class and map of the missed objects >= 1 µm) and with
`--missed-crops N` `missed.png` (the N largest missed objects, both
magnifications side by side at the expected position).

## Results so far

MnB sample (VHX 2 Nov 2026, SEM 30 Oct 2026), 4 indents, VHX -> SEM similarity:
X mirrored, scale 1.0011, rotation +0.41 deg, RMS residual 1.5 µm,
leave-one-out error <= 4.2 µm.

VHX detection, 4 images each at 500x and 1000x (MnB, 2 Nov 2026), 1000x as reference:
objects >= 1 µm found at 500x in 89-96 % (misses: faint trails), sizes agree
within 5-7 % (half-maximum outline), positions within 0.25 µm after an offset
of 4.3-6.6 µm (X) / -1.1-+0.9 µm (Y) per image pair (objective offset + stage).
Objects of 0.6-0.8 µm: 63 % found at 500x, 0.4-0.6 µm: 6 %.
Images were overexposed (14-35 % of pixels at 255 in R, G and B).
Whole area (864 images 1000x, 247 images 500x, mean mode): objects >= 1 µm found
at 500x in 94 %, dark ones (contrast >= 0.3) in 97-99 %; misses are faint
features (73 % with contrast < 0.2), a polishing scratch, and dust that moved
between the series. Offset between the series 7.2 / -2.0 µm, changing with Y
(~0.5 / -0.7 µm per mm, drift during acquisition), per-image scatter ~0.9 µm.

## Roadmap

1. SEM/EDS particle list import with stage X/Y per particle.
2. Refine the VHX -> SEM transform with inclusion centroids (done: indent-based transform).
3. Matching of VHX detections to SEM particles (done: VHX detection).
4. Dataset export (image crops + SEM class) for training.
