"""Crops and masks of a dataset as network input.

Input channels are chosen by name, in this order:
    rgb     the bright-field crop, 3 channels (R, G, B; 0-1)
    mask    the particle mask, 1 channel (1 = particle, 0 = rest): which object of
            the crop to classify
    <name>  the crop of an added lighting mode (column crop_<name>, see
            scripts/add_lighting.py), 3 channels
Default: rgb, mask. Channels are normalised with mean and standard deviation of
the training particles of the current fold only.

Augmentation (training only): one of the 8 orientations (rotation by 0, 90,
180, 270° and mirroring), the same for all channels, and per RGB group (each
lighting mode) its own brightness factor in [1 - b, 1 + b]. No colour shift:
the colour (pink) is the strongest cue.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

DEFAULT_CHANNELS = ("rgb", "mask")


def channel_names(channels) -> list[str]:
    """Names of the single input channels, e.g. rgb, mask, df -> R, G, B, mask, df_R, df_G, df_B."""
    out = []

    for c in channels:
        if c == "mask":
            out.append("mask")
        elif c == "rgb":
            out += ["R", "G", "B"]
        else:
            out += [f"{c}_R", f"{c}_G", f"{c}_B"]

    return out


def rgb_groups(channels) -> list[list[int]]:
    """Indices of the three channels of every RGB group (bright field and lighting modes)."""
    groups, i = [], 0

    for c in channels:
        if c == "mask":
            i += 1
        else:
            groups.append([i, i + 1, i + 2])
            i += 3

    return groups


def has_channels(ds: pd.DataFrame, channels) -> pd.Series:
    """True for the particles that have a crop of every lighting mode among the channels."""
    ok = pd.Series(True, index=ds.index)

    for c in channels:
        if c not in ("rgb", "mask"):
            if f"crop_{c}" not in ds:
                raise ValueError(f"channel {c}: the dataset has no column crop_{c} (see scripts/add_lighting.py)")
            ok &= ds[f"crop_{c}"].fillna("").ne("")

    return ok


def load_arrays(ds: pd.DataFrame, root: Path, channels=DEFAULT_CHANNELS) -> np.ndarray:
    """All particles as one float32 array (n x channels x H x W): RGB groups scaled to 0-1,
    mask 0 / 1. Every particle needs a crop of each lighting mode among the channels."""
    root = Path(root)
    out = []

    for r in ds.itertuples():
        parts = []

        for c in channels:
            if c == "mask":
                parts.append((np.asarray(Image.open(root / r.mask)) > 0).astype(np.float32)[None])
                continue

            file = r.crop if c == "rgb" else getattr(r, f"crop_{c}")
            rgb = np.asarray(Image.open(root / file))[..., :3].astype(np.float32) / 255
            parts.append(rgb.transpose(2, 0, 1))

        out.append(np.concatenate(parts, axis=0))

    return np.stack(out)


def channel_stats(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Mean and standard deviation per channel (to be computed on training particles only)."""
    mean = x.mean(axis=(0, 2, 3))
    std = x.std(axis=(0, 2, 3))
    return mean.astype(np.float32), np.maximum(std, 1e-6).astype(np.float32)


def orient(x: torch.Tensor, k: int) -> torch.Tensor:
    """Orientation k (0-7) of a C x H x W or N x C x H x W tensor: rotation by (k mod 4) x 90°,
    mirrored for k >= 4."""
    y = torch.rot90(x, k % 4, dims=(-2, -1))
    return torch.flip(y, dims=(-1,)) if k >= 4 else y


class CropDataset(torch.utils.data.Dataset):
    """Particles of one part of the data (training or validation) as normalised tensors with
    their class index; augmentation as in the module docstring if augment is True. groups:
    channel indices of the RGB groups (each gets its own brightness factor)."""

    def __init__(self, x: np.ndarray, y: np.ndarray, mean: np.ndarray, std: np.ndarray,
                 augment: bool = False, brightness: float = 0.1, seed: int = 0, groups=([0, 1, 2],)):
        self.x = torch.from_numpy(x)
        self.y = torch.from_numpy(y.astype(np.int64))
        self.mean = torch.from_numpy(mean)[:, None, None]
        self.std = torch.from_numpy(std)[:, None, None]
        self.augment = augment
        self.brightness = brightness
        self.groups = [list(g) for g in groups]
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        """Number of particles."""
        return len(self.y)

    def __getitem__(self, i: int):
        """(input tensor, class index) of particle i."""
        x = self.x[i].clone()

        if self.augment:
            x = orient(x, int(self.rng.integers(8)))

            for g in self.groups:
                factor = 1 + self.rng.uniform(-self.brightness, self.brightness)
                x[g] = (x[g] * factor).clamp(0, 1)

        return (x - self.mean) / self.std, self.y[i]
