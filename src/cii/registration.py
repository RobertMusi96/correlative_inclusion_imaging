"""Transformations between the stage coordinate systems of two instruments.

``dst = A @ src + t`` with points as columns (x, y) in um. Fitted from matched
fiducials (indents). Two models:

* similarity: rotation, uniform scale, translation and - if needed - a mirror
  (the stage X axes of the VHX and the SEM point in opposite directions);
* affine: additionally different scales per axis and shear (needs >= 3 points).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class Transform:
    A: np.ndarray        # 2 x 2
    t: np.ndarray        # 2
    kind: str = "similarity"
    src: str = "src"
    dst: str = "dst"

    def apply(self, x, y):
        """Transforms x, y (scalars or arrays) with dst = A @ [x, y] + t; returns x', y' in the input shape."""
        p = np.vstack([np.ravel(x), np.ravel(y)]).astype(float)
        q = self.A @ p + self.t[:, None]
        return q[0].reshape(np.shape(x)), q[1].reshape(np.shape(y))

    def inverse(self) -> "Transform":
        """Inverse transform (dst -> src)."""
        Ai = np.linalg.inv(self.A)
        return Transform(Ai, -Ai @ self.t, self.kind, self.dst, self.src)

    # descriptive parameters (similarity part)
    @property
    def mirrored(self) -> bool:
        """True if the transform mirrors (negative determinant), e.g. VHX -> SEM with opposite X axes."""
        return bool(np.linalg.det(self.A) < 0)

    @property
    def scale(self) -> float:
        """Mean scale factor (square root of the absolute determinant)."""
        return float(np.sqrt(abs(np.linalg.det(self.A))))

    @property
    def rotation_deg(self) -> float:
        """Rotation in degrees after undoing an X mirror."""
        A = self.A @ np.diag([-1, 1]) if self.mirrored else self.A   # undo X mirror first
        return float(np.degrees(np.arctan2(A[1, 0] - A[0, 1], A[0, 0] + A[1, 1])))

    def to_json(self, path) -> None:
        """Writes the transform with scale, rotation, mirror flag and equation as JSON."""
        Path(path).write_text(json.dumps({
            "kind": self.kind, "src": self.src, "dst": self.dst,
            "A": self.A.tolist(), "t": self.t.tolist(),
            "scale": self.scale, "rotation_deg": self.rotation_deg, "mirrored_x": self.mirrored,
            "equation": "dst = A @ [x, y] + t (um)"}, indent=2))

    @classmethod
    def from_json(cls, path) -> "Transform":
        """Reads a transform written by to_json."""
        d = json.loads(Path(path).read_text())
        return cls(np.array(d["A"], float), np.array(d["t"], float), d["kind"], d["src"], d["dst"])


def fit_similarity(src: np.ndarray, dst: np.ndarray, allow_mirror: bool = True) -> Transform:
    """Least-squares similarity (Umeyama). src, dst: N x 2. Mirror chosen if it fits better."""
    src, dst = np.asarray(src, float), np.asarray(dst, float)
    mu_s, mu_d = src.mean(0), dst.mean(0)
    S, D = src - mu_s, dst - mu_d
    best = None
    for mirror in ((False, True) if allow_mirror else (False,)):
        Sm = S * np.array([-1.0, 1.0]) if mirror else S
        U, sig, Vt = np.linalg.svd(D.T @ Sm / len(src))
        E = np.diag([1.0, np.sign(np.linalg.det(U @ Vt))])
        R = U @ E @ Vt
        c = np.trace(np.diag(sig) @ E) / (Sm ** 2).sum(1).mean()
        A = c * R @ (np.diag([-1.0, 1.0]) if mirror else np.eye(2))
        t = mu_d - A @ mu_s
        res = np.linalg.norm((A @ src.T).T + t - dst, axis=1)
        if best is None or res.mean() < best[0]:
            best = (res.mean(), A, t)
    return Transform(best[1], best[2], "similarity")


def fit_affine(src: np.ndarray, dst: np.ndarray) -> Transform:
    """Least-squares affine transform src -> dst (6 parameters, at least 3 point pairs)."""
    src, dst = np.asarray(src, float), np.asarray(dst, float)
    if len(src) < 3:
        raise ValueError("affine fit needs at least 3 points")
    M = np.hstack([src, np.ones((len(src), 1))])
    P, *_ = np.linalg.lstsq(M, dst, rcond=None)   # 3 x 2
    return Transform(P[:2].T, P[2], "affine")


def residuals(T: Transform, src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Transformed src points minus dst points (um), one row per point pair."""
    x, y = T.apply(src[:, 0], src[:, 1])
    return np.column_stack([x, y]) - np.asarray(dst, float)


def leave_one_out(src: np.ndarray, dst: np.ndarray, kind: str = "similarity") -> np.ndarray:
    """Prediction error for each point when it is left out of the fit (N x 2, um)."""
    fit = fit_similarity if kind == "similarity" else fit_affine
    out = []
    for i in range(len(src)):
        keep = np.arange(len(src)) != i
        T = fit(src[keep], dst[keep])
        out.append(residuals(T, src[i:i + 1], dst[i:i + 1])[0])
    return np.array(out)
