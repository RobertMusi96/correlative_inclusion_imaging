"""Matching of SEM/EDS particles (AZtec feature analysis) to VHX detections.

Steps
-----
1. SEM particles (``_pre.xlsx``, stage X/Y per particle) get their label from the
   class files: typified (Class, Type), not typified, rejected (contamination).
2. Rough position: SEM stage coordinates -> VHX stage coordinates with the
   indent transform (``cii.registration.Transform``, VHX -> SEM, inverted).
3. Coarse offset: vote of the displacement vectors of large particles (ECD >=
   ``coarse_min_ecd``) within ``coarse_search_um``. Needed when the series was
   recorded with another objective than the indents (100x/200x vs 500x: ~130 um).
4. First alignment: offset per pair (SEM field, VHX image) from the particles
   themselves (``cii.compare``). SEM particles seen in two overlapping SEM
   fields are counted once; particles touching a SEM field border are skipped
   (their complete view is in the neighbouring field).
5. Position model, fitted to the confident matches and used to match again
   (``iterations`` times), see ``fit_position_model``:
   VHX = SEM->VHX(indent transform) + c + G * position (affine trend: transform
   error, drift) + M * position within the SEM field (SEM scan scale/rotation)
   + stage error of the SEM field - stage error of the VHX image.
6. Assignment: among the VHX objects within the match radius
   (``tolerance_um + tolerance_size_fraction * SEM length``) the one with the best score
   ``distance + size_weight_um * |ln(ECD_vhx / ECD_sem)|`` is taken; views of the
   same object in overlapping VHX images count as one candidate. If a second,
   different object scores within ``ambiguity_margin_um`` the match is
   'ambiguous'. After the model fit the match radius is
   ``max(min_tolerance_um, model_tolerance_sigma * residual sd) + fraction * length``.
   If no object is that close but the predicted position lies inside the
   detected region of a VHX object (``region_*`` columns of detect_inclusions.py:
   faint particle with a dark spot, particle merged with a neighbour at this
   magnification), that object is taken with ``match_type = 'region'``.

Status per SEM particle: matched, ambiguous, not_found, outside (not covered by
the VHX images), sem_border, sem_duplicate. ``vhx_shared``: the VHX object was
assigned to more than one SEM particle (cluster resolved by SEM only).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from .compare import compare, estimate_offset
from .registration import Transform

SEM_FILE_SUFFIXES = {               # longest first: "_pre_corr.xlsx" must not catch "_pre_corr_typified.xlsx"
    "typified": "_pre_corr_typified.xlsx",
    "not_typified": "_pre_corr_not-typified.xlsx",
    "rejected": "_pre_rejected.xlsx",
    "corr": "_pre_corr.xlsx",
    "pre": "_pre.xlsx",
}


def find_sem_files(folder: Path) -> dict[str, Path]:
    """The AZtec export files in a folder, by role (pre, corr, rejected, typified, not_typified)."""
    found: dict[str, Path] = {}
    for f in sorted(Path(folder).glob("*.xlsx")):
        for role, suffix in SEM_FILE_SUFFIXES.items():
            if f.name.endswith(suffix):
                if role in found:
                    raise ValueError(f"two files for '{role}': {found[role].name}, {f.name}")
                found[role] = f
                break
    missing = [r for r in ("pre", "rejected", "typified", "not_typified") if r not in found]
    if missing:
        raise FileNotFoundError(f"missing in {folder}: " + ", ".join(SEM_FILE_SUFFIXES[r] for r in missing))
    return found


def load_sem_particles(files: dict[str, Path], field_width_px: int = 1024, field_height_px: int = 768) -> pd.DataFrame:
    """All particles of ``_pre.xlsx`` with label, Class and Type.

    label: 'typified', 'not_typified', 'rejected' (or 'unlisted' if in none of the lists).
    class_label: Class for typified particles, otherwise the label.
    """
    pre = pd.read_excel(files["pre"])
    rej = set(pd.read_excel(files["rejected"], usecols=["ID"]).ID)
    typ = pd.read_excel(files["typified"], usecols=["ID", "Class", "Type"]).set_index("ID")
    nottyp = set(pd.read_excel(files["not_typified"], usecols=["ID"]).ID)
    df = pd.DataFrame({
        "sem_id": pre.ID.astype(int),
        "field": pre.Field.astype(int),
        "stage_x_um": pre["Stage X (mm)"] * 1000.0,
        "stage_y_um": pre["Stage Y (mm)"] * 1000.0,
        "ecd_um": pre["ECD (µm)"],
        "area_um2": pre["Area (sq. µm)"],
        "length_um": pre["Length (µm)"],
        "breadth_um": pre["Breadth (µm)"],
        "aspect_ratio": pre["Aspect Ratio"],
        "mean_grey": pre["Mean grey"],
        "bb_left_px": pre.BBLeftPix, "bb_top_px": pre.BBTopPix,
        "bb_width_px": pre.BBWidthPix, "bb_height_px": pre.BBHeightPix,
    })
    df["touches_border"] = ((pre.BBLeftPix <= 0) | (pre.BBTopPix <= 0)
                            | (pre.BBLeftPix + pre.BBWidthPix >= field_width_px)
                            | (pre.BBTopPix + pre.BBHeightPix >= field_height_px)).to_numpy()
    label = np.full(len(df), "unlisted", dtype=object)
    label[df.sem_id.isin(rej).to_numpy()] = "rejected"
    label[df.sem_id.isin(nottyp).to_numpy()] = "not_typified"
    label[df.sem_id.isin(typ.index).to_numpy()] = "typified"
    df["label"] = label
    df["Class"] = df.sem_id.map(typ.Class)
    df["Type"] = df.sem_id.map(typ.Type)
    df["class_label"] = np.where(df.label == "typified", df.Class.astype(str), df.label)
    elements = [c for c in pre.columns if c.endswith("(Wt%)")]
    df = pd.concat([df, pre[elements].rename(columns=lambda c: c.replace(" (Wt%)", "_wt"))], axis=1)
    return df


@dataclass
class MatchParams:
    tolerance_um: float = 2.5             # match radius for small particles after fine alignment
    tolerance_size_fraction: float = 0.25  # + this fraction of the SEM length (centroid vs bbox centre of large/long particles)
    size_weight_um: float = 1.0           # score penalty per e-fold size difference
    ambiguity_margin_um: float = 0.5      # second object this close in score -> ambiguous
    vhx_border_penalty_um: float = 5.0    # prefer a complete VHX view over one cut by the image border
    same_object_um: float = 1.5           # views in overlapping VHX images closer than this = same object
    offset_min_ecd: float = 1.0           # particles used for the fine alignment
    coarse_search_um: float = 500.0       # offset between indent objective and series objective
    coarse_min_ecd: float = 2.0
    global_search_um: float = 15.0        # around the coarse offset
    pair_search_um: float = 15.0
    iterations: int = 3                   # fit position model -> match again
    model_tolerance_sigma: float = 4.0    # match radius after the model fit, in residual sd
    min_tolerance_um: float = 1.0
    region_margin_um: float = 0.5         # predicted position inside a VHX region bbox (+ margin)
    model_ridge: float = 1.0              # weight pulling per-field / per-image errors to 0
    outlier_sigma: float = 3.0            # matches with larger model residual are not used for the fit


@dataclass
class MatchResult:
    particles: pd.DataFrame     # one row per SEM particle
    offsets: pd.DataFrame       # first alignment: (SEM field, VHX image) offsets in VHX coordinates
    coarse_offset: np.ndarray   # offset found by the coarse search (VHX coordinates)
    coarse_votes: int
    model: dict                 # position model of the last iteration
    history: list               # per iteration: matches used, residual sd, statuses


def sem_fields_in_vhx(fields: pd.DataFrame, to_vhx: Transform, um_per_px: float,
                      width_px: int = 1024, height_px: int = 768) -> pd.DataFrame:
    """SEM fields as 'images' in VHX coordinates (axis-aligned; rotation < 1 deg)."""
    x, y = to_vhx.apply(fields.stage_x_um.to_numpy(), fields.stage_y_um.to_numpy())
    return pd.DataFrame({"image": [f"field_{i}" for i in fields.index], "width_px": width_px, "height_px": height_px,
                         "um_per_px": um_per_px * to_vhx.scale, "stage_x_um": x, "stage_y_um": y})


def fit_position_model(R: np.ndarray, UV: np.ndarray, field: np.ndarray, image: np.ndarray, V: np.ndarray,
                       ridge: float = 1.0, outlier_sigma: float = 3.0, iterations: int = 3) -> dict:
    """Least-squares model of VHX - SEM(in VHX coordinates), separately for x and y.

    R: SEM positions in VHX coordinates (um), UV: position within the SEM field
    (mm, image axes), field / image: SEM field and VHX image of each match,
    V: matched VHX positions. Unknowns: c, G (per mm of R), M (per mm of UV), one
    error per field (+) and per VHX image (-), the latter two with a ridge.
    Robust: matches with residual > outlier_sigma * sd are dropped and the fit repeated.
    """
    from scipy.sparse import csr_matrix, vstack as svstack
    from scipy.sparse.linalg import lsqr
    fk = {f: k for k, f in enumerate(pd.unique(field))}
    ik = {i: k for k, i in enumerate(pd.unique(image))}
    n, nf, ni = len(R), len(fk), len(ik)
    centre = R.mean(axis=0)
    Rm = (R - centre) / 1000.0
    dense = np.c_[np.ones(n), Rm, UV]                       # 5 global columns
    rows = np.repeat(np.arange(n), 7)
    cols = np.c_[np.tile(np.arange(5), (n, 1)), 5 + np.array([fk[f] for f in field]),
                 5 + nf + np.array([ik[i] for i in image])].ravel()
    vals = np.c_[dense, np.ones(n), -np.ones(n)].ravel()
    A = csr_matrix((vals, (rows, cols)), shape=(n, 5 + nf + ni))
    reg = csr_matrix((np.full(nf + ni, np.sqrt(ridge)), (np.arange(nf + ni), 5 + np.arange(nf + ni))),
                     shape=(nf + ni, 5 + nf + ni))
    D = V - R
    use = np.ones(n, bool)
    sol = np.zeros((2, 5 + nf + ni))
    for _ in range(iterations):
        Au = svstack([A[use], reg]).tocsr()
        for ax in range(2):
            b = np.r_[D[use, ax], np.zeros(nf + ni)]
            sol[ax] = lsqr(Au, b, atol=1e-10, btol=1e-10, iter_lim=20000)[0]
        res = D - np.c_[A @ sol[0], A @ sol[1]]
        sd = 1.4826 * np.median(np.abs(res[use] - np.median(res[use], axis=0)), axis=0)
        use = np.all(np.abs(res) <= outlier_sigma * np.maximum(sd, 0.05), axis=1)
    return {"centre": centre, "c": sol[:, 0], "G": sol[:, 1:3], "M": sol[:, 3:5],
            "field_err": {f: sol[:, 5 + k] for f, k in fk.items()},
            "image_err": {i: sol[:, 5 + nf + k] for i, k in ik.items()},
            "residual_sd": sd, "n_used": int(use.sum()), "n": n}


def model_base(model: dict, R: np.ndarray, UV: np.ndarray, field: np.ndarray) -> np.ndarray:
    """Predicted VHX position without the VHX image error (subtract that per image)."""
    z = np.zeros(2)
    fe = np.array([model["field_err"].get(f, z) for f in field]).reshape(-1, 2)
    Rm = (R - model["centre"]) / 1000.0
    return R + model["c"] + Rm @ model["G"].T + UV @ model["M"].T + fe


def _assign(n: int, rows: np.ndarray, s_ecd: np.ndarray, s_len: np.ndarray, candidates_of, V, v_ecd, v_border,
            p: MatchParams, tolerance_um: float, region_lookup=None):
    """Best VHX object per SEM particle. candidates_of(row) -> iterable of (vhx image, predicted position,
    tree, object indices); region_lookup(image, position) -> object indices whose region contains it."""
    out = {k: np.full(n, np.nan) for k in ["distance_um", "score", "second_score", "pred_x_um", "pred_y_um"]}
    out["idx"] = np.full(n, -1)
    out["image"] = np.full(n, None, dtype=object)
    out["n_candidates"] = np.zeros(n, int)
    out["status"] = np.full(n, "", dtype=object)
    out["match_type"] = np.full(n, "", dtype=object)
    for row in rows:
        tol = tolerance_um + p.tolerance_size_fraction * s_len[row]
        cands, views = [], []
        for img, pred, tree, idx in candidates_of(row):
            views.append((img, pred))
            for j in tree.query_ball_point(pred, r=tol):
                vj = idx[j]
                d = float(np.hypot(*(V[vj] - pred)))
                score = d + p.size_weight_um * abs(np.log(v_ecd[vj] / s_ecd[row])) + p.vhx_border_penalty_um * v_border[vj]
                cands.append((score, d, vj, img, pred))
        if not views:                                      # no VHX image around this position
            out["status"][row] = "outside"
            continue
        if not cands and region_lookup is not None:
            for img, pred in views:                        # most central image first
                hits = region_lookup(img, pred)
                if len(hits):
                    vj = min(hits, key=lambda j: np.hypot(*(V[j] - pred)))
                    d = float(np.hypot(*(V[vj] - pred)))
                    out["status"][row], out["match_type"][row] = "matched", "region"
                    out["distance_um"][row], out["idx"][row], out["image"][row] = d, vj, img
                    out["pred_x_um"][row], out["pred_y_um"][row] = pred
                    out["n_candidates"][row] = len(hits)
                    break
            if out["status"][row]:
                continue
        if not cands:
            out["status"][row] = "not_found"
            out["image"][row] = views[0][0]
            out["pred_x_um"][row], out["pred_y_um"][row] = views[0][1]
            continue
        cands.sort(key=lambda t: t[0])
        best = cands[0]
        rel = lambda t: V[t[2]] - t[4]                    # object position relative to its prediction
        distinct = [t for t in cands[1:] if np.hypot(*(rel(t) - rel(best))) > p.same_object_um]
        groups = []
        for t in distinct:
            if all(np.hypot(*(rel(t) - g)) > p.same_object_um for g in groups):
                groups.append(rel(t))
        out["n_candidates"][row] = 1 + len(groups)
        out["score"][row], out["distance_um"][row] = best[0], best[1]
        out["pred_x_um"][row], out["pred_y_um"][row] = best[4]
        out["idx"][row], out["image"][row] = best[2], best[3]
        out["match_type"][row] = "object"
        if distinct:
            out["second_score"][row] = distinct[0][0]
        out["status"][row] = "ambiguous" if distinct and distinct[0][0] - best[0] < p.ambiguity_margin_um else "matched"
    return out


def match(sem: pd.DataFrame, fields: pd.DataFrame, vhx_to_sem: Transform, vhx_obj: pd.DataFrame,
          vhx_img: pd.DataFrame, sem_um_per_px: float = 0.294864, params: MatchParams | None = None,
          field_size_px: tuple[int, int] = (1024, 768)) -> MatchResult:
    """Match SEM particles to VHX objects. ``fields``: field table indexed by field id
    (``read_field_positions``); ``vhx_obj`` / ``vhx_img``: objects.csv / images.csv of detect_inclusions.py."""
    p = params or MatchParams()
    to_vhx = vhx_to_sem.inverse()
    fields = fields.loc[sorted(set(sem.field))]
    f_img = sem_fields_in_vhx(fields, to_vhx, sem_um_per_px, *field_size_px)
    x, y = to_vhx.apply(sem.stage_x_um.to_numpy(), sem.stage_y_um.to_numpy())
    fname = ("field_" + sem.field.astype(str)).to_numpy()
    ref = pd.DataFrame({"image": fname, "id": sem.sem_id.to_numpy(), "stage_x_um": x, "stage_y_um": y,
                        "ecd_um": sem.ecd_um.to_numpy(), "touches_border": sem.touches_border.to_numpy()})
    R = np.c_[x, y]
    s_ecd = sem.ecd_um.to_numpy()
    s_len = np.fmax(sem.length_um.to_numpy(), s_ecd)
    # position within the SEM field, mm, image axes (same orientation as the VHX image)
    UV = np.c_[(sem.bb_left_px + sem.bb_width_px / 2 - (field_size_px[0] - 1) / 2).to_numpy(),
               (sem.bb_top_px + sem.bb_height_px / 2 - (field_size_px[1] - 1) / 2).to_numpy()] * sem_um_per_px / 1000.0

    # 1) coarse offset (other objective than the indents)
    big_s = (s_ecd >= p.coarse_min_ecd) & ~sem.touches_border.to_numpy(bool)
    big_v = (vhx_obj.ecd_um.to_numpy() >= p.coarse_min_ecd) & ~vhx_obj.touches_border.to_numpy(bool)
    V = vhx_obj[["stage_x_um", "stage_y_um"]].to_numpy()
    coarse, n_coarse = estimate_offset(R[big_s], V[big_v], search_um=p.coarse_search_um, bin_um=4.0,
                                       refine_um=4.0, iterations=6)
    if coarse is None:
        raise RuntimeError("no common offset between SEM particles and VHX objects found")
    # 2) first alignment per (field, image) pair; SEM duplicates / border / outside
    res = compare(ref, f_img, vhx_obj, vhx_img, tolerance=p.tolerance_um, offset_min_ecd=p.offset_min_ecd,
                  global_search_um=p.global_search_um, pair_search_um=p.pair_search_um, global_prior=coarse)
    st = res.objects.status.to_numpy()
    base_status = np.select([st == "border", st == "duplicate", st == "outside"],
                            ["sem_border", "sem_duplicate", "outside"], "")
    rows = np.flatnonzero(base_status == "")

    v_ecd = vhx_obj.ecd_um.to_numpy()
    v_border = vhx_obj.touches_border.to_numpy(bool)
    v_img = vhx_obj.image.to_numpy()
    by_img = {n: np.flatnonzero(v_img == n) for n in pd.unique(v_img)}
    trees = {n: cKDTree(V[i]) for n, i in by_img.items()}
    # VHX images around each particle (most central first)
    rects = vhx_img.assign(hw=vhx_img.width_px * vhx_img.um_per_px / 2, hh=vhx_img.height_px * vhx_img.um_per_px / 2)
    ctree = cKDTree(rects[["stage_x_um", "stage_y_um"]].to_numpy())
    reach = float(np.hypot(rects.hw.max(), rects.hh.max())) + 20
    approx = R + res.global_offset
    near = []
    for row in range(len(R)):
        ks = ctree.query_ball_point(approx[row], r=reach)
        ks = [k for k in ks if abs(approx[row, 0] - rects.stage_x_um.iat[k]) < rects.hw.iat[k] + 10
              and abs(approx[row, 1] - rects.stage_y_um.iat[k]) < rects.hh.iat[k] + 10]
        ks.sort(key=lambda k: max(abs(approx[row, 0] - rects.stage_x_um.iat[k]) / rects.hw.iat[k],
                                  abs(approx[row, 1] - rects.stage_y_um.iat[k]) / rects.hh.iat[k]))
        near.append([rects.image.iat[k] for k in ks])

    off_of = {(r.ref_image, r.other_image): np.array([r.dx_um, r.dy_um]) for r in res.offsets.itertuples()}

    def cands_pairs(row):
        for img in near[row]:
            off = off_of.get((fname[row], img))
            if off is not None and img in trees:
                yield img, R[row] + off, trees[img], by_img[img]

    # detected regions (if present) in stage coordinates, per VHX image
    region_lookup = None
    if "region_bbox_col0" in vhx_obj:
        g = vhx_img.set_index("image").loc[v_img]
        ox = g.stage_x_um.to_numpy() - (g.width_px.to_numpy() - 1) / 2 * g.um_per_px.to_numpy()
        oy = g.stage_y_um.to_numpy() - (g.height_px.to_numpy() - 1) / 2 * g.um_per_px.to_numpy()
        u = g.um_per_px.to_numpy()
        m = p.region_margin_um
        rb = np.c_[ox + (vhx_obj.region_bbox_col0.to_numpy() - 0.5) * u - m, ox + (vhx_obj.region_bbox_col1.to_numpy() - 0.5) * u + m,
                   oy + (vhx_obj.region_bbox_row0.to_numpy() - 0.5) * u - m, oy + (vhx_obj.region_bbox_row1.to_numpy() - 0.5) * u + m]
        rc = np.c_[(rb[:, 0] + rb[:, 1]) / 2, (rb[:, 2] + rb[:, 3]) / 2]
        rreach = float(np.nanmax(np.hypot(rb[:, 1] - rb[:, 0], rb[:, 3] - rb[:, 2]))) / 2
        finite = np.isfinite(rc).all(axis=1)
        r_idx = {n: i[finite[i]] for n, i in by_img.items()}
        rtrees = {n: cKDTree(rc[i]) for n, i in r_idx.items() if len(i)}

        def region_lookup(img, pred):
            if img not in rtrees:
                return []
            idx = r_idx[img][rtrees[img].query_ball_point(pred, r=rreach)]
            inside = (rb[idx, 0] <= pred[0]) & (pred[0] <= rb[idx, 1]) & (rb[idx, 2] <= pred[1]) & (pred[1] <= rb[idx, 3])
            return idx[inside]

    out = _assign(len(R), rows, s_ecd, s_len, cands_pairs, V, v_ecd, v_border, p, p.tolerance_um)
    history = [{"step": "pair offsets", **{k: int(v) for k, v in pd.Series(out["status"][rows]).value_counts().items()}}]
    model = None
    for it in range(p.iterations):
        ok = (out["status"] == "matched") & (out["match_type"] == "object")
        if ok.sum() < 20:
            break
        model = fit_position_model(R[ok], UV[ok], fname[ok], out["image"][ok], V[out["idx"][ok]],
                                   ridge=p.model_ridge, outlier_sigma=p.outlier_sigma)
        base = model_base(model, R, UV, fname)
        z = np.zeros(2)

        def cands_model(row):
            for img in near[row]:
                if img in trees:
                    yield img, base[row] - model["image_err"].get(img, z), trees[img], by_img[img]

        tol = max(p.min_tolerance_um, p.model_tolerance_sigma * float(np.max(model["residual_sd"])))
        out = _assign(len(R), rows, s_ecd, s_len, cands_model, V, v_ecd, v_border, p, tol, region_lookup)
        history.append({"step": f"model {it + 1}", "fit_matches": model["n_used"],
                        "residual_sd_x": round(float(model["residual_sd"][0]), 3),
                        "residual_sd_y": round(float(model["residual_sd"][1]), 3), "tolerance_um": round(tol, 2),
                        "region_matches": int((out["match_type"] == "region").sum()),
                        **{k: int(v) for k, v in pd.Series(out["status"][rows]).value_counts().items()}})

    status = base_status.copy()
    status[rows] = out["status"][rows]
    part = sem.reset_index(drop=True).copy()
    part["status"] = status
    part["pred_vhx_x_um"], part["pred_vhx_y_um"] = out["pred_x_um"], out["pred_y_um"]
    part["vhx_image"] = out["image"]          # matched image, or for not_found the image where it should be
    idx = out["idx"]
    has = idx >= 0
    part["vhx_id"] = pd.array(np.where(has, vhx_obj.id.to_numpy()[np.maximum(idx, 0)], 0)).astype("Int64")
    part.loc[~has, "vhx_id"] = pd.NA
    for col in ["ecd_um", "area_um2", "aspect_ratio", "contrast_max", "region_ecd_um", "region_n_objects",
                "region_bbox_row0", "region_bbox_col0", "region_bbox_row1", "region_bbox_col1", "contrast_R_max", "contrast_G_max",
                "contrast_B_max", "mean_R", "mean_G", "mean_B", "touches_border", "col_px", "row_px",
                "bbox_row0", "bbox_col0", "bbox_row1", "bbox_col1"]:
        if col in vhx_obj:
            v = pd.Series(np.nan, index=part.index, dtype=object if col == "touches_border" else float)
            v[has] = vhx_obj[col].to_numpy()[idx[has]]
            part[f"vhx_{col}"] = v
    part["vhx_x_um"] = np.where(has, V[np.maximum(idx, 0), 0], np.nan)
    part["vhx_y_um"] = np.where(has, V[np.maximum(idx, 0), 1], np.nan)
    part["distance_um"], part["score"], part["second_score"] = out["distance_um"], out["score"], out["second_score"]
    part["n_candidates"] = out["n_candidates"]
    part["match_type"] = out["match_type"]                 # object (centroid within radius) or region
    part["size_ratio"] = part.vhx_ecd_um / part.ecd_um
    key = part.vhx_image.astype(str) + "#" + part.vhx_id.astype(str)
    part["vhx_shared"] = has & key.duplicated(keep=False).to_numpy()
    offsets = res.offsets.rename(columns={"ref_image": "sem_field", "other_image": "vhx_image"})
    return MatchResult(particles=part, offsets=offsets, coarse_offset=coarse, coarse_votes=n_coarse,
                       model=model or {}, history=history)
