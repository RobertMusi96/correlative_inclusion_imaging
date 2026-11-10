"""Crops and masks of a dataset as network input.

Each particle is one tensor with 4 channels: the RGB crop (0-1) and its mask (1 =
particle, 0 = rest). The mask tells the network which object of the crop to
classify. Channels are normalised with mean and standard deviation of the
training particles of the current fold only.

Augmentation (training only): one of the 8 orientations (rotation by 0, 90,
180, 270° and mirroring) and a brightness factor in [1 - b, 1 + b] for the RGB
channels. No colour shift: the colour (pink) is the strongest cue.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

CHANNELS = ["R", "G", "B", "mask"]


def load_arrays(ds: pd.DataFrame, root: Path) -> np.ndarray:
    """All crops with their masks as one float32 array (n x 4 x H x W): RGB scaled to 0-1,
    mask 0 / 1."""
    root = Path(root)
    out = []

    for r in ds.itertuples():
        rgb = np.asarray(Image.open(root / r.crop))[..., :3].astype(np.float32) / 255
        mask = (np.asarray(Image.open(root / r.mask)) > 0).astype(np.float32)
        out.append(np.concatenate([rgb.transpose(2, 0, 1), mask[None]], axis=0))

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
    their class index; augmentation as in the module docstring if augment is True."""

    def __init__(self, x: np.ndarray, y: np.ndarray, mean: np.ndarray, std: np.ndarray,
                 augment: bool = False, brightness: float = 0.1, seed: int = 0):
        self.x = torch.from_numpy(x)
        self.y = torch.from_numpy(y.astype(np.int64))
        self.mean = torch.from_numpy(mean)[:, None, None]
        self.std = torch.from_numpy(std)[:, None, None]
        self.augment = augment
        self.brightness = brightness
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        """Number of particles."""
        return len(self.y)

    def __getitem__(self, i: int):
        """(input tensor, class index) of particle i."""
        x = self.x[i].clone()

        if self.augment:
            x = orient(x, int(self.rng.integers(8)))
            factor = 1 + self.rng.uniform(-self.brightness, self.brightness)
            x[:3] = (x[:3] * factor).clamp(0, 1)

        return (x - self.mean) / self.std, self.y[i]
