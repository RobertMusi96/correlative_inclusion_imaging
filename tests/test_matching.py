import numpy as np
import pandas as pd

from cii.matching import MatchParams, match
from cii.registration import Transform


def scene(seed=0, n=900, visible=0.7, noise_per_mm2=1200):
    rng = np.random.default_rng(seed)
    # VHX 500x tiles 586 x 440 um, 7 % overlap, 4 x 3
    imgs, k = [], 0
    for iy in range(3):
        for ix in range(4):
            imgs.append({"image": f"v{k:03d}.tif", "width_px": 2880, "height_px": 2160, "um_per_px": 0.2035,
                         "stage_x_um": ix * 545.0, "stage_y_um": iy * 410.0, "err": rng.normal(0, 0.8, 2)})
            k += 1
    vi = pd.DataFrame(imgs)
    x0, x1, y0, y1 = -250, 3 * 545 + 250, -180, 2 * 410 + 180
    truth = pd.DataFrame({"x": rng.uniform(x0, x1, n), "y": rng.uniform(y0, y1, n),
                          "ecd": rng.choice([1.2, 2.0, 3.5, 6.0], n), "vis": rng.random(n) < visible})
    # VHX objects: visible particles in every image that contains them + random other objects
    rows = []
    for im in vi.itertuples():
        hw, hh = 293.0, 219.8
        inside = (abs(truth.x - im.stage_x_um) < hw - 2) & (abs(truth.y - im.stage_y_um) < hh - 2) & truth.vis
        for t in truth[inside].itertuples():
            rows.append({"image": im.image, "stage_x_um": t.x - im.err[0] + rng.normal(0, 0.2),
                         "stage_y_um": t.y - im.err[1] + rng.normal(0, 0.2), "ecd_um": t.ecd * rng.normal(1.05, 0.05),
                         "truth": t.Index})
        m = rng.poisson(noise_per_mm2 * 4 * hw * hh / 1e6)
        for _ in range(m):
            rows.append({"image": im.image, "stage_x_um": im.stage_x_um + rng.uniform(-hw, hw),
                         "stage_y_um": im.stage_y_um + rng.uniform(-hh, hh), "ecd_um": rng.choice([0.9, 1.1, 1.5]),
                         "truth": -1})
    vo = pd.DataFrame(rows)
    vo["id"] = vo.groupby("image").cumcount() + 1
    vo["touches_border"] = False
    vo["area_um2"] = np.pi * vo.ecd_um ** 2 / 4
    vo["contrast_max"] = 0.4
    # SEM: indent transform VHX(indent objective) -> SEM; the series objective is offset by c
    th = np.radians(0.35)
    A = 1.0011 * np.array([[-np.cos(th), -np.sin(th)], [-np.sin(th), np.cos(th)]])
    T = Transform(A, np.array([560.0, -2130.0]), "similarity", "vhx", "sem")
    c = np.array([-132.0, 11.5])
    sx, sy = T.apply(truth.x.to_numpy() - c[0], truth.y.to_numpy() - c[1])
    # SEM fields 302 x 226 um, 10 % overlap, each with its own stage error; particle in the first field containing it
    fx = np.arange(sx.min() - 100, sx.max() + 150, 272.0)
    fy = np.arange(sy.min() - 100, sy.max() + 150, 204.0)
    fields = pd.DataFrame([{"stage_x_um": a, "stage_y_um": b} for b in fy for a in fx])
    fields.index = np.arange(1, len(fields) + 1)
    ferr = {f: rng.normal(0, 0.8, 2) for f in fields.index}
    sem_rows = []
    for k, (x, y) in enumerate(zip(sx, sy)):
        inf = fields[(abs(fields.stage_x_um - x) < 151 - 3) & (abs(fields.stage_y_um - y) < 113 - 3)]
        if not len(inf):
            continue
        f = inf.index[0]
        col = (fields.stage_x_um[f] - x) / 0.294864 + 511.5      # SEM: stage X increases to the left
        row = (y - fields.stage_y_um[f]) / 0.294864 + 383.5
        e = truth.ecd[k]
        sem_rows.append({"sem_id": k, "field": f, "stage_x_um": x + ferr[f][0], "stage_y_um": y + ferr[f][1],
                         "ecd_um": e, "length_um": e * 1.2, "touches_border": False,
                         "bb_left_px": col - 2, "bb_width_px": 4, "bb_top_px": row - 2, "bb_height_px": 4})
    sem = pd.DataFrame(sem_rows)
    return sem, fields, T, vo, vi.drop(columns="err"), truth


def test_matching_finds_visible_particles_and_right_objects():
    sem, fields, T, vo, vi, truth = scene()
    r = match(sem, fields, T, vo, vi)
    assert np.allclose(r.coarse_offset, [-132.0, 11.5], atol=2.5)
    P = r.particles
    vis = truth.vis.to_numpy()[P.sem_id.to_numpy()]
    inside = P.status.isin(["matched", "ambiguous", "not_found"])
    found = P.status.isin(["matched", "ambiguous"])
    # visible particles are found, invisible ones mostly not (chance matches only)
    assert found[inside & vis].mean() > 0.97
    assert found[inside & ~vis].mean() < 0.06
    # matched object is the right particle
    vt = vo.set_index(["image", "id"]).truth
    m = P[found & vis]
    got = vt.loc[list(zip(m.vhx_image, m.vhx_id.astype(int)))].to_numpy()
    assert (got == m.sem_id.to_numpy()).mean() > 0.99
    assert r.model["residual_sd"].max() < 0.5


def test_relaxed_pass_and_cluster_flags():
    sem, fields, T, vo, vi, truth = scene(seed=2)
    # three large particles: their VHX view shifted by 3 um (leftover local error) -> only the relaxed pass finds them
    vis = truth.vis.to_numpy()[sem.sem_id.to_numpy()]
    big = sem.index[vis][:6]
    sem.loc[big, ["ecd_um", "length_um"]] = [[3.0, 3.6]] * len(big)
    for k in big:
        t = sem.at[k, "sem_id"]
        vo.loc[vo.truth == t, "stage_x_um"] += 3.0
        vo.loc[vo.truth == t, "ecd_um"] = 3.1
    r = match(sem, fields, T, vo, vi)
    P = r.particles.set_index("sem_id")
    assert (r.particles.match_type == "relaxed").sum() >= 1
    for k in big:
        row = P.loc[sem.at[k, "sem_id"]]
        if row.status in ("outside", "sem_border", "sem_duplicate"):
            continue
        assert row.status == "matched" and row.match_type in ("relaxed", "object")
    assert {"cluster", "cluster_reason", "training_ok", "sem_neighbour"} <= set(r.particles.columns)
    tr = r.particles[r.particles.training_ok]
    assert (tr.status == "matched").all() and not tr.cluster.any()
    off = MatchParams(relaxed_min_ecd=None)
    r2 = match(sem, fields, T, vo, vi, params=off)
    assert (r2.particles.match_type == "relaxed").sum() == 0
