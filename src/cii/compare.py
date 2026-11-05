"""Compare detections of the same sample area at two magnifications.

Used to check the detection: objects found at the higher magnification
(reference) are searched among the objects of the lower magnification.

Every image carries its own stage error (about +-2 um) and the objectives are
offset against each other by a few um. The offset is therefore estimated for
each overlapping pair of images (reference image i, other image j):

1. global offset: vote of all displacement vectors (2D histogram), refined;
2. pair offset: same, searched close to the global offset, using objects
   with ECD >= ``offset_min_ecd`` of the two images;
3. pairs with too few objects for their own estimate get the offset of a
   model ``offset_ij = global + b_j - a_i`` (one stage error per image),
   fitted to the measured pairs.

Reference objects seen in two overlapping reference images are counted once
(neighbouring images matched against each other, the most central view is
kept, see ``duplicates``), objects touching the image border
are skipped, objects outside the other magnification's images are reported
as 'outside'.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

XY = ["stage_x_um", "stage_y_um"]


def image_rects(images: pd.DataFrame) -> pd.DataFrame:
    """Stage rectangle of every image (images.csv of detect_inclusions.py)."""
    hw = images.width_px * images.um_per_px / 2
    hh = images.height_px * images.um_per_px / 2
    return pd.DataFrame({"x0": images.stage_x_um - hw, "x1": images.stage_x_um + hw,
                         "y0": images.stage_y_um - hh, "y1": images.stage_y_um + hh,
                         "cx": images.stage_x_um, "cy": images.stage_y_um}).set_index(images.image.to_numpy())


def nearest_image(xy: np.ndarray, rects: pd.DataFrame) -> np.ndarray:
    """Name of the image whose centre is nearest (= most central image for equal tile sizes)."""
    _, k = cKDTree(rects[["cx", "cy"]].to_numpy()).query(xy)
    return rects.index.to_numpy()[k]


def estimate_offset(ref: np.ndarray, other: np.ndarray, prior=None, search_um: float = 40.0, bin_um: float = 1.0,
                    refine_um: float = 1.5, iterations: int = 3, min_pairs: int = 3):
    """Offset to add to ref coordinates so that they land on other coordinates.

    Vote of all displacement vectors within ``search_um`` of ``prior`` (2D
    histogram peak), then iterative refinement: median displacement of the pairs
    within ``refine_um``. Returns (offset, number of pairs) or (None, n) if fewer
    than ``min_pairs`` pairs support it.
    """
    prior = np.zeros(2) if prior is None else np.asarray(prior, float)
    if len(ref) == 0 or len(other) == 0:
        return None, 0
    tree = cKDTree(other)
    shifted = ref + prior
    vec = [other[j] - shifted[k] for k, js in enumerate(tree.query_ball_point(shifted, r=search_um)) for j in js]
    if not vec:
        return None, 0
    vec = np.asarray(vec)
    nb = max(1, int(round(2 * search_um / bin_um)))
    H, xe, ye = np.histogram2d(vec[:, 0], vec[:, 1], bins=nb, range=[[-search_um, search_um]] * 2)
    ix, iy = np.unravel_index(H.argmax(), H.shape)
    off = prior + np.array([(xe[ix] + xe[ix + 1]) / 2, (ye[iy] + ye[iy + 1]) / 2])
    n = 0
    for it in range(iterations):
        d, i = tree.query(ref + off, distance_upper_bound=refine_um if it else max(refine_um, 2 * bin_um))
        ok = np.isfinite(d)
        n = int(ok.sum())
        if n < min_pairs:
            return None, n
        off = np.median(other[i[ok]] - ref[ok], axis=0)
    return off, n


def duplicates(obj: pd.DataFrame, rects: pd.DataFrame, tolerance: float = 2.5, search_um: float = 8.0,
               min_pairs: int = 3, extra_um: float = 3.0) -> np.ndarray:
    """True for objects that are a second view of an object seen more centrally in an
    overlapping image of the same series.

    For every pair of overlapping images, the objects lying in the other image are
    matched (offset estimated from the overlap; if too few objects, no offset and a
    match radius enlarged by ``extra_um`` for the stage error). Of each matched pair
    the view farther from its image centre is flagged. Border objects are ignored.
    """
    xy = obj[XY].to_numpy()
    names = obj.image.to_numpy()
    usable = ~obj.touches_border.to_numpy(bool)
    r = rects.loc[names]
    cen = np.maximum(np.abs(xy[:, 0] - r.cx.to_numpy()) / (r.x1 - r.x0).to_numpy(),
                     np.abs(xy[:, 1] - r.cy.to_numpy()) / (r.y1 - r.y0).to_numpy())
    by = {n: np.flatnonzero((names == n) & usable) for n in rects.index}
    dup = np.zeros(len(obj), bool)
    m = extra_um + tolerance
    work = []
    for a, b in _overlapping_pairs(rects, rects, margin=0):
        if a >= b:
            continue
        ra, rb = rects.loc[a], rects.loc[b]
        ia = by[a][(xy[by[a], 0] > rb.x0 - m) & (xy[by[a], 0] < rb.x1 + m) & (xy[by[a], 1] > rb.y0 - m) & (xy[by[a], 1] < rb.y1 + m)]
        ib = by[b][(xy[by[b], 0] > ra.x0 - m) & (xy[by[b], 0] < ra.x1 + m) & (xy[by[b], 1] > ra.y0 - m) & (xy[by[b], 1] < ra.y1 + m)]
        if not len(ia) or not len(ib):
            continue
        off, n = estimate_offset(xy[ia], xy[ib], search_um=search_um, min_pairs=min_pairs)
        work.append((a, b, ia, ib, off, n))
    # pairs with too few objects (e.g. corner overlaps): offset from one stage error per image,
    # fitted to the measured pairs
    meas = [(a, b, off, n) for a, b, _, _, off, n in work if off is not None]
    err = fit_image_errors([w[0] for w in meas], [w[1] for w in meas], np.array([w[2] for w in meas]).reshape(-1, 2),
                           np.array([w[3] for w in meas], float))
    for a, b, ia, ib, off, n in work:
        if off is not None:
            tol = tolerance
        elif a in err and b in err:
            off, tol = err[b] - err[a], tolerance + 1.0
        else:
            off, tol = np.zeros(2), tolerance + extra_um
        d, j = cKDTree(xy[ib]).query(xy[ia] + off, distance_upper_bound=tol)
        ok = np.isfinite(d)
        pa, pb = ia[ok], ib[j[ok]]
        dup[np.where(cen[pa] <= cen[pb], pb, pa)] = True
    return dup


@dataclass
class Comparison:
    objects: pd.DataFrame      # reference objects with status and match
    offsets: pd.DataFrame      # one row per overlapping image pair
    global_offset: np.ndarray
    model_rms_um: float        # scatter of measured pair offsets around the model


def _overlapping_pairs(ra: pd.DataFrame, rb: pd.DataFrame, margin: float):
    a = ra[["x0", "x1", "y0", "y1"]].to_numpy()[:, None, :]
    b = rb[["x0", "x1", "y0", "y1"]].to_numpy()[None, :, :]
    ov = ((a[..., 0] < b[..., 1] + margin) & (b[..., 0] < a[..., 1] + margin)
          & (a[..., 2] < b[..., 3] + margin) & (b[..., 2] < a[..., 3] + margin))
    ia, ib = np.nonzero(ov)
    return list(zip(ra.index[ia], rb.index[ib]))


def fit_image_errors(keys_a, keys_b, offsets: np.ndarray, weights: np.ndarray, prior=None, ridge: float = 0.1) -> dict:
    """One stage error per image from measured pair offsets: offset_ab = prior + e_b - e_a.

    Weighted least squares with a weak ridge (images without measurements get 0).
    Returns {image key: e}; only images that occur in a measured pair are returned.
    """
    prior = np.zeros(2) if prior is None else np.asarray(prior, float)
    keys = list(dict.fromkeys(list(keys_a) + list(keys_b)))
    if not keys:
        return {}
    idx = {k: i for i, k in enumerate(keys)}
    A = np.zeros((len(offsets) + len(keys), len(keys))); B = np.zeros((len(offsets) + len(keys), 2))
    for r, (a, b, o, w) in enumerate(zip(keys_a, keys_b, offsets, weights)):
        w = np.sqrt(w)
        A[r, idx[a]] -= w; A[r, idx[b]] += w; B[r] = w * (o - prior)
    A[len(offsets):] = np.sqrt(ridge) * np.eye(len(keys))
    sol, *_ = np.linalg.lstsq(A, B, rcond=None)
    return {k: sol[i] for k, i in idx.items()}


def _fit_pair_model(offsets: pd.DataFrame, g: np.ndarray):
    """Model offsets for all pairs (ref image i, other image j): g + b_j - a_i."""
    meas = offsets[offsets.source == "measured"]
    err = fit_image_errors([("r", n) for n in meas.ref_image], [("o", n) for n in meas.other_image],
                           meas[["dx_um", "dy_um"]].to_numpy(), meas.n_pairs.to_numpy(float), prior=g)
    z = np.zeros(2)
    model = np.array([g + err.get(("o", o), z) - err.get(("r", r), z)
                      for r, o in zip(offsets.ref_image, offsets.other_image)]).reshape(-1, 2)
    is_m = offsets.source.to_numpy() == "measured"
    res = np.full(len(offsets), np.nan)
    res[is_m] = np.hypot(*(offsets[["dx_um", "dy_um"]].to_numpy()[is_m] - model[is_m]).T)
    rms = float(np.sqrt(np.mean(res[is_m] ** 2))) if is_m.any() else np.nan
    return model, res, rms


def compare(ref_obj: pd.DataFrame, ref_img: pd.DataFrame, oth_obj: pd.DataFrame, oth_img: pd.DataFrame,
            tolerance: float = 2.5, offset_min_ecd: float = 0.8, global_search_um: float = 40.0,
            pair_search_um: float = 15.0, min_pairs: int = 3, coverage_margin_um: float = 3.0,
            outlier_um: float = 1.5) -> Comparison:
    ref = ref_obj.reset_index(drop=True).copy()
    oth = oth_obj.reset_index(drop=True)
    rr, orr = image_rects(ref_img), image_rects(oth_img)
    R, O = ref[XY].to_numpy(), oth[XY].to_numpy()
    r_border, o_border = ref.touches_border.to_numpy(bool), oth.touches_border.to_numpy(bool)

    status = np.full(len(ref), "", dtype=object)
    status[r_border] = "border"
    status[(status == "") & duplicates(ref, rr, tolerance=tolerance)] = "duplicate"
    m = coverage_margin_um
    cov = np.zeros(len(R), bool)
    for r in orr.itertuples():
        cov |= (R[:, 0] >= r.x0 + m) & (R[:, 0] <= r.x1 - m) & (R[:, 1] >= r.y0 + m) & (R[:, 1] <= r.y1 - m)
    status[(status == "") & ~cov] = "outside"

    r_by_img = {n: np.flatnonzero(ref.image.to_numpy() == n) for n in rr.index}
    o_by_img = {n: np.flatnonzero(oth.image.to_numpy() == n) for n in orr.index}
    r_big = (ref.ecd_um.to_numpy() >= offset_min_ecd) & ~r_border
    o_big = (oth.ecd_um.to_numpy() >= 0.8 * offset_min_ecd) & ~o_border

    g, n_g = estimate_offset(R[r_big & (status != "duplicate")], O[o_big], search_um=global_search_um,
                             bin_um=2.0, refine_um=4.0, iterations=6, min_pairs=min_pairs)
    if g is None:
        raise RuntimeError("no common offset found - do the two detections cover the same area?")

    pairs = _overlapping_pairs(rr, orr, margin=pair_search_um)
    rows = []
    for rn, on in pairs:
        ri, oi = r_by_img[rn], o_by_img[on]
        off, n = estimate_offset(R[ri[r_big[ri]]], O[oi[o_big[oi]]], prior=g, search_um=pair_search_um,
                                 min_pairs=min_pairs)
        rows.append({"ref_image": rn, "other_image": on, "dx_um": off[0] if off is not None else np.nan,
                     "dy_um": off[1] if off is not None else np.nan, "n_pairs": n,
                     "source": "measured" if off is not None else "model"})
    offsets = pd.DataFrame(rows, columns=["ref_image", "other_image", "dx_um", "dy_um", "n_pairs", "source"])
    rms = np.nan
    if len(offsets):
        # pair offsets that contradict the per-image model (false peak with few objects)
        # are re-estimated close to the model; pairs without own estimate use the model
        for _ in range(2):
            model, res, rms = _fit_pair_model(offsets, g)
            bad = (offsets.source == "measured") & (res > max(outlier_um, 3 * rms))
            offsets.loc[bad, "source"] = "model"
        for k in np.flatnonzero(offsets.source.to_numpy() == "model"):
            rn, on = offsets.ref_image.iat[k], offsets.other_image.iat[k]
            ri, oi = r_by_img[rn], o_by_img[on]
            off, n = estimate_offset(R[ri[r_big[ri]]], O[oi[o_big[oi]]], prior=model[k], search_um=3.0,
                                     min_pairs=min_pairs)
            if off is not None:
                offsets.iloc[k, offsets.columns.get_indexer(["dx_um", "dy_um", "n_pairs", "source"])] = [off[0], off[1], n, "measured"]
        model, res, rms = _fit_pair_model(offsets, g)
        offsets["model_dx_um"], offsets["model_dy_um"] = model[:, 0], model[:, 1]
        use = offsets.source == "model"
        offsets.loc[use, "dx_um"] = offsets.loc[use, "model_dx_um"]
        offsets.loc[use, "dy_um"] = offsets.loc[use, "model_dy_um"]
    off_of = {(r.ref_image, r.other_image): np.array([r.dx_um, r.dy_um]) for r in offsets.itertuples()}

    # matching: best match over all overlapping other images; objects touching the
    # border of the other image only if there is no complete one
    cand = status == ""
    best = np.full(len(ref), np.inf); best_j = np.full(len(ref), -1); best_d = np.full(len(ref), np.nan)
    best_off = np.full((len(ref), 2), np.nan)
    for (rn, on), off in off_of.items():
        ri = r_by_img[rn][cand[r_by_img[rn]]]
        oi = o_by_img[on]
        if not len(ri) or not len(oi):
            continue
        k = min(4, len(oi))
        dd, jj = cKDTree(O[oi]).query(R[ri] + off, k=k, distance_upper_bound=tolerance)
        dd, jj = dd.reshape(len(ri), k), jj.reshape(len(ri), k)
        for c in range(k):
            ok = np.isfinite(dd[:, c])
            if not ok.any():
                continue
            j = oi[jj[ok, c]]
            score = dd[ok, c] + 10.0 * o_border[j]
            idx = ri[ok]
            better = score < best[idx]
            best[idx[better]] = score[better]; best_j[idx[better]] = j[better]
            best_d[idx[better]] = dd[ok, c][better]; best_off[idx[better]] = off
    found = cand & (best_j >= 0)
    status[cand & ~found] = "missed"
    status[found] = "found"

    # expected position in the most central other image (for missed objects: where to look)
    cover = nearest_image(R, orr)
    exp_off = np.array([off_of.get((rn, on), g) for rn, on in zip(ref.image, cover)]) if len(ref) else np.zeros((0, 2))
    exp_off[found] = best_off[found]
    jf = best_j[found]
    ref["status"] = status
    ref["other_image"] = np.where(found, oth.image.to_numpy()[np.maximum(best_j, 0)], cover)
    ref["other_x_um"] = R[:, 0] + exp_off[:, 0]
    ref["other_y_um"] = R[:, 1] + exp_off[:, 1]
    ref["other_id"] = pd.array(np.where(found, oth.id.to_numpy()[np.maximum(best_j, 0)], -1)).astype("Int64")
    ref.loc[~found, "other_id"] = pd.NA
    for col in ["ecd_um", "area_um2", "contrast_max", "touches_border"]:
        v = pd.Series(np.nan, index=ref.index, dtype=object if col == "touches_border" else float)
        v[found] = oth[col].to_numpy()[jf]
        ref[f"other_{col}"] = v
    ref["distance_um"] = best_d
    return Comparison(objects=ref, offsets=offsets, global_offset=g, model_rms_um=rms)


def summary(objects: pd.DataFrame, edges) -> pd.DataFrame:
    """Share found per ECD class of the reference (only 'found' and 'missed' objects)."""
    rows = []
    o = objects[objects.status.isin(["found", "missed"])]
    for lo, hi in zip(edges[:-1], edges[1:]):
        s = o[(o.ecd_um >= lo) & (o.ecd_um < hi)]
        if not len(s):
            continue
        f = s[s.status == "found"]
        rows.append({"ECD_ref_um": f"{lo:g}-{hi:g}", "n_ref": len(s), "n_found": len(f),
                     "found_pct": round(100 * len(f) / len(s), 1),
                     "median_ECD_ref": round(f.ecd_um.median(), 2) if len(f) else np.nan,
                     "median_ECD_other": round(f.other_ecd_um.median(), 2) if len(f) else np.nan,
                     "size_ratio": round(float(np.median(f.other_ecd_um / f.ecd_um)), 3) if len(f) else np.nan,
                     "median_distance_um": round(f.distance_um.median(), 2) if len(f) else np.nan})
    return pd.DataFrame(rows)
