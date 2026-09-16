"""Does the near-wall gap survive in body-fitted coordinates?

PRE-REGISTERED. This docstring is committed before the stage is run and before any
number it produces exists. Reviewer 2 has named this experiment as the one that
decides the paper, in three consecutive rounds; the two previous times an experiment
was named that way and run, it reversed the paper's headline. It is run here without
knowing which way it falls, and the rule below is fixed either way.

THE OBJECTION THIS ANSWERS
--------------------------
The paper bounds what ANY weighting of the 800 training fields can achieve near the
wall (``point_space_headtohead.py``, amendments 2 and 3), and reports that the bound
is far above the surrogate. But the mechanism it measures for WHY is a statement
about coordinates, not about representational capacity: inside ``0.005c`` a test
query lands 7.8x further from the training airfoil's wall than from its own, and 35%
of the kernel's weight mass falls INSIDE the training airfoil. The manuscript then
declines the obvious fix -- interpolate in wall-normal coordinates instead -- on the
grounds that it "would need a blend length scale and an arc-length correspondence
with no published provenance".

That refusal is a choice, and a hostile reviewer is right that it is. The family the
paper bounds is defined in physical coordinates, which is exactly the coordinate
system in which the measured mechanism exists. A body-fitted parameter interpolator
is not bounded by anything in the paper.

WHY THE STATED OBJECTIONS DISSOLVE UNDER SCOPING
------------------------------------------------
Both objections are about the FAR FIELD, and the claim is not.

* The blend length scale exists only because a wall-normal frame has to be faded back
  into a global frame somewhere out in the domain. Restricting the arm to the
  near-wall strip -- ``n <= 0.05c``, bands 0-4, which is where the entire claim lives
  -- means there is no far field to blend into and no blend to choose.
* The arc-length correspondence is parameter-free once the strip is fixed. AirfRANS
  stores each section in its own chord-aligned frame with ``x`` spanning exactly
  [0, 1]. Split the surface by the sign of the surface normal's y-component, order
  each chain by ``x``, and measure arc length from the leading edge (``argmin x``),
  positive on the upper chain and negative on the lower. Nothing is selected, tuned,
  or fitted; the same construction runs on the training section and the test section.

The frame is ``(s, n)``: ``s`` the signed arc length above, ``n`` the wall distance
already cached as ``sdf``. Both are in chord units and are NOT rescaled relative to
one another -- near the wall that map is an isometry up to curvature, which is the
standard boundary-layer coordinate and introduces no free parameter. A relative
scaling between ``s`` and ``n`` would be exactly the kind of tunable this arm exists
to avoid.

WHAT IS COMPUTED
----------------
The same least-squares projection bound as amendment 3, with one thing changed: each
training field is transferred to the test nodes through ``(s, n)`` instead of through
``(x, y)``. Everything else -- the 800 training fields, the redimensionalisation, the
bands, the test nodes, the solve, the Transolver reference -- is identical. So this
bounds the WHOLE body-fitted interpolation family at those nodes, for the same reason
the physical-coordinate version bounds the physical family: the published weights sum
to one, redimensionalisation commutes with the combination, and any weighting lies in
the span of the 800 transferred fields.

    THE RULE, fixed before the run
    ------------------------------
    B1. Body-fitted LS residual MSE inside ``0.005c`` on ``u``, divided by
        Transolver's MSE in the same band -- the same estimator, band, channel and
        thresholds as the committed physical-coordinate rule:

          > 10   -> BODY-FITTED-BOUND-CLOSED. The near-wall gap is not a coordinate
                    artifact. It survives the strongest reframing available to the
                    interpolation family, and the paper's claim widens from
                    "in physical coordinates" to the family as such.
          <= 1   -> BODY-FITTED-BOUND-OPEN. The gap IS a coordinate artifact. The
                    headline must be rescoped to physical coordinates, and the
                    paper's result becomes a statement about coordinate systems:
                    what a surrogate buys near the wall is available to parameter
                    interpolation too, once it is asked in the right frame.
          else   -> PARTIAL, reported with both sides.

    B2 (diagnostic, no threshold). The ratio of the body-fitted LS MSE to the
        physical-coordinate LS MSE in the same band on the same cases: how much the
        coordinate change buys, whichever side of B1 it lands on. Reported with the
        per-case distribution, as in amendment 3.

    B1 is read on bands with ``n_b >> 800`` free parameters. The ``wall`` band
    (~1000 nodes) is NOT read; it is reported with its n.

GATES -- the run aborts if any fails
------------------------------------
GB1  IDENTITY OF EVERYTHING BUT THE FRAME. Run with ``--frame physical`` and the
     script must reproduce ``point_space_oracle_ls_n200.json`` on bands 0-4 to 1e-9
     relative. The two arms share one code path and differ only in the coordinate
     map, so any difference in the result is attributable to the frame and to
     nothing else. This is the gate that makes the comparison mean anything.

     Read on the FIRST 20 test cases, fixed here before the run. GB1 asks whether
     two code paths are the same code path, which is a question about
     implementation and not about the world; 20 cases x 5 bands x 2 channels x 4
     statistics is 800 independent agreements at 1e-9, and a frame bug survives
     none of them. The remaining 180 cases would cost ~1.5 h to tell us nothing
     further. The scientific comparison, B1 and B2, runs on all 200.

GB2  THE FRAME IS FAITHFUL. Transferring a training field to ITSELF through its own
     ``(s, n)`` map must return its own values: the construction is only a
     relabelling of that case's nodes, so the round trip is exact up to the
     triangulation's own interpolation of a point that is a vertex. Checked on the
     strip nodes of the first three training fields, tolerance 1e-9 relative.

GB3  THE STRIP IS THE CLAIM'S DOMAIN. Every node scored lies within ``0.05c`` of the
     wall. Reported with the node count per band, which must match the physical
     arm's counts for bands 0-4 exactly -- the same nodes are being scored, only
     reached through a different frame.

Run
---
    .venv/Scripts/python.exe scripts/bodyfit_bound.py --stage surfcache
    .venv/Scripts/python.exe scripts/bodyfit_bound.py --stage run --frame physical   # GB1
    .venv/Scripts/python.exe scripts/bodyfit_bound.py --stage run --frame bodyfit
"""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
import time

import neuroforge  # noqa: F401  -- caps BLAS threads before numpy; see CLAUDE.md

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import point_space_headtohead as P  # noqa: E402

STRIP_BAND_MAX = 4          # bands 0..4 == wall .. 0.02-0.05c, i.e. n <= 0.05c
VERDICT_BAND = 1            # '0-0.005c', where the claim lives
TOL = 1e-9


def log(msg: str) -> None:
    print(f"[bodyfit] {msg}", flush=True)


# --------------------------------------------------------------------------- #
# The frame. Parameter-free by construction; see the module docstring.
# --------------------------------------------------------------------------- #
def surface_arclength(apos: np.ndarray, anrm: np.ndarray):
    """Signed arc length ``s`` for each surface point, in chord units.

    Split by the sign of the normal's y-component, order each chain by ``x``, and
    measure arc length from the leading edge (``argmin x``) along that chain.
    Positive on the upper chain, negative on the lower. No free parameter: there is
    nothing here to select, tune or fit, and the same construction runs on the
    training section and on the test section.
    """
    s = np.zeros(len(apos), np.float64)
    for side, sign in ((anrm[:, 1] >= 0.0, +1.0), (anrm[:, 1] < 0.0, -1.0)):
        idx = np.nonzero(side)[0]
        if idx.size == 0:
            continue
        order = idx[np.argsort(apos[idx, 0], kind="stable")]
        pts = apos[order]
        seg = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))])
        s[order] = sign * seg
    return s


def to_frame(pos, sdf, apos, anrm, frame, kd_workers=2):
    """Map node positions into the requested frame.

    ``physical`` returns ``(x, y)`` unchanged, so the two arms share one code path
    and differ only here -- which is what GB1 checks.
    """
    if frame == "physical":
        return np.asarray(pos, np.float64)
    from scipy.spatial import cKDTree
    sv = surface_arclength(np.asarray(apos, np.float64), np.asarray(anrm, np.float64))
    _d, i = cKDTree(np.asarray(apos, np.float64)).query(np.asarray(pos, np.float64),
                                                        k=1, workers=kd_workers)
    return np.column_stack([sv[i], np.asarray(sdf, np.float64)])


# --------------------------------------------------------------------------- #
# Test-surface cache: the test .npz holds no airfoil, and the frame needs one.
# --------------------------------------------------------------------------- #
def stage_surfcache(a) -> None:
    import airfrans.simulation as afsim

    names_tr, names_te, _cfg, _W, _base = P.setup(a)
    outdir = os.path.join(a.scratch, "testsurf")
    os.makedirs(outdir, exist_ok=True)
    todo = [n for n in names_te if not os.path.exists(os.path.join(outdir, n + ".npz"))]
    log(f"test surface cache: {len(names_te) - len(todo)} present, {len(todo)} to build")
    t0 = time.time()
    for i, nm in enumerate(todo):
        sim = afsim.Simulation(root=a.data_root, name=nm)
        np.savez(os.path.join(outdir, nm + ".npz"),
                 apos=np.asarray(sim.airfoil_position, np.float32),
                 anrm=np.asarray(sim.airfoil_normals, np.float32))
        if (i + 1) % 25 == 0:
            log(f"  {i+1}/{len(todo)} ({time.time()-t0:.0f}s)")
    log(f"test surface cache ready ({time.time()-t0:.0f}s)")


def load_test_surface(scratch, name):
    z = np.load(os.path.join(scratch, "testsurf", name + ".npz"))
    return np.asarray(z["apos"], np.float64), np.asarray(z["anrm"], np.float64)


# --------------------------------------------------------------------------- #
# Worker: the whole bound for ONE test case, in ONE frame.
# --------------------------------------------------------------------------- #
def case_worker(args: tuple) -> str:
    import neuroforge  # noqa: F401
    import numpy as np
    from scipy.interpolate import LinearNDInterpolator
    from scipy.spatial import Delaunay, cKDTree

    (scratch, tname, train_names, kd_workers, chans, outdir, Wrow, frame) = args
    os.makedirs(outdir, exist_ok=True)
    dest = os.path.join(outdir, tname + ".json")
    if os.path.exists(dest):
        return dest

    ntr = len(train_names)
    pos_t, tgt, sdf_t, _incrop = P.load_test_case(scratch, tname)
    b_all = P.band_index8(sdf_t)
    strip = b_all <= STRIP_BAND_MAX
    pos_t, tgt, sdf_t, b = pos_t[strip], tgt[strip], sdf_t[strip], b_all[strip]

    if frame == "bodyfit":
        apos_t, anrm_t = load_test_surface(scratch, tname)
        q = to_frame(pos_t, sdf_t, apos_t, anrm_t, frame, kd_workers)
    else:
        q = to_frame(pos_t, sdf_t, None, None, frame)

    U_t, al_t = P.case_U_alpha(tname)
    ci = [P.CHANS.index(c) for c in chans]
    A = np.zeros((len(chans), ntr, q.shape[0]), np.float32)

    t0 = time.time()
    for jj, jname in enumerate(train_names):
        pos_j, yhat_j, apos_j, anrm_j = P.load_train_case(scratch, jname)
        zj = np.load(P._train_cache_path(scratch, jname))
        sdf_j = np.asarray(zj["sdf"], np.float64)

        if frame == "bodyfit":
            # Only the strip is defined in this frame, and only the strip is scored.
            keep = P.band_index8(sdf_j) <= STRIP_BAND_MAX
            src = to_frame(pos_j[keep], sdf_j[keep], apos_j, anrm_j, frame, kd_workers)
            val = yhat_j[keep]
        else:
            src = np.asarray(pos_j, np.float64)
            val = yhat_j

        lin = LinearNDInterpolator(Delaunay(src), val, fill_value=np.nan)
        kt = cKDTree(src)
        vals = np.asarray(lin(q), np.float64)
        out = np.isnan(vals[:, 0])
        if out.any():
            _d, idx = kt.query(q[out], k=1, workers=kd_workers)
            vals[out] = val[idx]

        if frame == "physical":
            # P1's `nearfill` arm, exactly as the committed physical run does it
            _dj, isurf = cKDTree(apos_j).query(q, k=1, workers=kd_workers)
            inbody = np.einsum("ij,ij->i", q - apos_j[isurf], anrm_j[isurf]) > 0.0
            if inbody.any():
                _d2, idx2 = kt.query(q[inbody], k=1, workers=kd_workers)
                vals[inbody] = val[idx2]
        # In (s, n) a query cannot land inside the training body: n is the TEST
        # node's own wall distance and is non-negative by construction. That is the
        # mechanism this arm exists to remove, so there is nothing to patch here.

        phys = P.redim_nodes(vals, U_t, al_t)
        for k, c in enumerate(ci):
            A[k, jj] = phys[:, c].astype(np.float32)
        if (jj + 1) % 200 == 0:
            log(f"[{frame} {tname[:26]}] {jj+1}/{ntr} ({time.time()-t0:.0f}s)")

    row = {"name": tname, "frame": frame, "bands": P.NAMES8[:STRIP_BAND_MAX + 1],
           "n_band": [int((b == bi).sum()) for bi in range(STRIP_BAND_MAX + 1)],
           "channels": {}}
    for k, c in enumerate(chans):
        y = tgt[:, P.CHANS.index(c)]
        ent = {"ls_mse": [], "ls_mse_ridge": [], "krr_mse": [], "best_single_mse": []}
        for bi in range(STRIP_BAND_MAX + 1):
            m = b == bi
            if int(m.sum()) == 0:
                for key in ent:
                    ent[key].append(float("nan"))
                continue
            Ab = np.asarray(A[k][:, m], np.float64).T
            yb = np.asarray(y[m], np.float64)
            G = Ab.T @ Ab
            rhs = Ab.T @ yb
            tr = float(np.trace(G)) / ntr
            for lam, key in ((1e-10 * tr, "ls_mse"), (1e-6 * tr, "ls_mse_ridge")):
                w = np.linalg.solve(G + lam * np.eye(ntr), rhs)
                ent[key].append(float(np.mean((Ab @ w - yb) ** 2)))
            ent["krr_mse"].append(float(np.mean((Ab @ Wrow - yb) ** 2)))
            ent["best_single_mse"].append(
                float(np.min(np.mean((Ab - yb[:, None]) ** 2, axis=0))))
            del Ab, yb, G, rhs
        row["channels"][c] = ent
    del A

    row["wallclock_sec"] = time.time() - t0
    with io.open(dest, "w", encoding="utf-8") as fh:
        fh.write(json.dumps(row))
    log(f"[{frame}] {tname[:36]} done ({row['wallclock_sec']:.0f}s)")
    return dest


# --------------------------------------------------------------------------- #
def gate_gb2(a, names_tr) -> dict:
    """GB2: the frame is a relabelling, so a field transferred to itself is itself."""
    from scipy.interpolate import LinearNDInterpolator
    from scipy.spatial import Delaunay

    worst, at = 0.0, None
    for jname in names_tr[:3]:
        pos_j, yhat_j, apos_j, anrm_j = P.load_train_case(a.scratch, jname)
        sdf_j = np.asarray(np.load(P._train_cache_path(a.scratch, jname))["sdf"], np.float64)
        keep = P.band_index8(sdf_j) <= STRIP_BAND_MAX
        src = to_frame(pos_j[keep], sdf_j[keep], apos_j, anrm_j, "bodyfit")
        val = yhat_j[keep]
        lin = LinearNDInterpolator(Delaunay(src), val, fill_value=np.nan)
        got = np.asarray(lin(src), np.float64)
        ok = np.isfinite(got[:, 0])
        rel = np.abs(got[ok] - val[ok]) / np.maximum(np.abs(val[ok]), 1e-12)
        m = float(np.nanmax(rel))
        if m > worst:
            worst, at = m, jname[:34]
    if worst > TOL:
        raise SystemExit(f"GB2 FAILED: worst relative round-trip {worst:.3e} at {at}")
    log(f"GB2 PASS  frame round-trip exact to {worst:.3e} ({at})")
    return {"worst_rel": worst, "at": at, "threshold": TOL, "status": "PASS"}


def gate_gb1(a, rows) -> dict:
    """GB1: with the frame set to identity, reproduce the committed n=200 artifact."""
    ref_path = os.path.join(a.out_dir, "point_space_oracle_ls_n200.json")
    if not os.path.exists(ref_path):
        raise SystemExit(f"GB1 cannot run: {ref_path} does not exist yet")
    ref = {r["name"]: r for r in json.load(open(ref_path, encoding="utf-8"))["rows"]}
    worst, at = 0.0, None
    for r in rows:
        rr = ref[r["name"]]
        assert r["n_band"] == rr["n_band"][:STRIP_BAND_MAX + 1], \
            f"GB3 FAILED: strip node counts differ for {r['name']}"
        for c, ent in r["channels"].items():
            for key in ("ls_mse", "ls_mse_ridge", "krr_mse", "best_single_mse"):
                for bi, (x, y) in enumerate(zip(ent[key], rr["channels"][c][key])):
                    if not np.isfinite(x) and not np.isfinite(y):
                        continue
                    rel = abs(x - y) / max(abs(y), 1e-300)
                    if rel > worst:
                        worst, at = rel, f"{r['name'][:28]} {c} {key} band{bi}"
    if worst > TOL:
        raise SystemExit(f"GB1 FAILED: worst relative difference {worst:.3e} at {at} "
                         f"-- the physical frame does not reproduce {ref_path}")
    log(f"GB1 PASS  physical frame reproduces the committed artifact to {worst:.3e}")
    return {"worst_rel": worst, "at": at, "threshold": TOL, "status": "PASS"}


def stage_run(a) -> dict:
    names_tr, names_te, cfg, W, base = P.setup(a)
    chans = tuple(a.ls_chans)
    pte = names_te[:a.n_ls]
    outdir = os.path.join(a.scratch, "bodyfit_" + a.frame)
    os.makedirs(outdir, exist_ok=True)

    gb2 = gate_gb2(a, names_tr) if a.frame == "bodyfit" else {"status": "n/a"}

    chunks = [(a.scratch, nm, names_tr, a.kd_workers, chans, outdir,
               W[names_te.index(nm)], a.frame) for nm in pte]
    todo = [c for c in chunks if not os.path.exists(os.path.join(outdir, c[1] + ".json"))]
    log(f"{len(chunks) - len(todo)} cases already done, {len(todo)} to run ({a.frame})")
    t0 = time.time()
    if todo:
        if a.n_proc <= 1:
            for c in todo:
                case_worker(c)
        else:
            from concurrent.futures import ProcessPoolExecutor
            with ProcessPoolExecutor(max_workers=a.n_proc) as ex:
                for r in ex.map(case_worker, todo):
                    log(f"wrote {os.path.basename(r)}")
    dt = time.time() - t0

    rows = []
    for nm in pte:
        with io.open(os.path.join(outdir, nm + ".json"), encoding="utf-8") as fh:
            rows.append(json.loads(fh.read()))

    gb1 = gate_gb1(a, rows) if a.frame == "physical" else {"status": "n/a"}

    tso = P._ls_tso_band(a)
    tso_u1 = float(tso["u"][VERDICT_BAND])
    per_case = [float(r["channels"]["u"]["ls_mse"][VERDICT_BAND]) for r in rows]
    n_b1 = [int(r["n_band"][VERDICT_BAND]) for r in rows]
    ls1 = float(np.mean(per_case))
    q = ls1 / tso_u1
    verdict = ("BODY-FITTED-BOUND-CLOSED" if q > 10 else
               "BODY-FITTED-BOUND-OPEN" if q <= 1 else "PARTIAL")

    ratios = [v / tso_u1 for v in per_case]
    diag = {"note": "diagnostics only; no threshold here can change B1",
            "mean_of_ratios": float(np.mean(ratios)),
            "median_of_ratios": float(np.median(ratios)),
            "pooled_node_weighted_ratio":
                float(np.sum(np.multiply(per_case, n_b1)) / np.sum(n_b1)) / tso_u1,
            "min_ratio": float(np.min(ratios)), "max_ratio": float(np.max(ratios)),
            "n_cases_above_10": int(np.sum(np.asarray(ratios) > 10)),
            "n_cases_at_or_below_1": int(np.sum(np.asarray(ratios) <= 1)),
            "per_case_ratio": ratios, "per_case_ls_mse_u_band1": per_case,
            "per_case_n_band1": n_b1}

    # B2: what the coordinate change bought, against the physical arm on the same cases
    b2 = {"status": "n/a (this is the physical arm)"}
    if a.frame == "bodyfit":
        ref_path = os.path.join(a.out_dir, "point_space_oracle_ls_n200.json")
        if os.path.exists(ref_path):
            ref = {r["name"]: r for r in json.load(open(ref_path, encoding="utf-8"))["rows"]}
            phys = [float(ref[r["name"]]["channels"]["u"]["ls_mse"][VERDICT_BAND])
                    for r in rows]
            rat = [bf / ph for bf, ph in zip(per_case, phys)]
            b2 = {"note": "body-fitted LS MSE / physical LS MSE, same band, same cases",
                  "ratio_of_casemeans": ls1 / float(np.mean(phys)),
                  "mean_of_ratios": float(np.mean(rat)),
                  "median_of_ratios": float(np.median(rat)),
                  "n_cases_bodyfit_better": int(np.sum(np.asarray(rat) < 1.0)),
                  "n_cases": len(rat), "per_case_ratio": rat}

    out = {"meta": {"script": "bodyfit_bound.py", "frame": a.frame, "n_ls": len(pte),
                    "n_train": len(names_tr), "channels": list(chans),
                    "strip": f"bands 0-{STRIP_BAND_MAX} (n <= 0.05c)",
                    "wallclock_sec": dt, "n_proc": a.n_proc,
                    "rule": "pre-registered in the module docstring (B1/B2, GB1-GB3)"},
           "gates": {"GB1": gb1, "GB2": gb2},
           "rows": rows,
           "B1": {"band": P.NAMES8[VERDICT_BAND], "ls_mse_u_casemean": ls1,
                  "transolver_band_mse_u": tso_u1, "ratio": q, "verdict": verdict},
           "B2_vs_physical": b2,
           "diagnostics": diag}
    dest = os.path.join(a.out_dir, f"bodyfit_bound_{a.frame}.json")
    P.write_json(dest, out)
    log(f"B1 [{a.frame}] n={len(pte)}: LS {ls1:.5g} vs Transolver {tso_u1:.5g} "
        f"-> {q:.4g}  {verdict}")
    if a.frame == "bodyfit" and "ratio_of_casemeans" in b2:
        log(f"B2: body-fitted / physical = {b2['ratio_of_casemeans']:.4g} "
            f"(median per-case {b2['median_of_ratios']:.4g}, "
            f"{b2['n_cases_bodyfit_better']}/{b2['n_cases']} cases improved)")
    log(f"wrote {dest}")
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--stage", required=True, choices=("surfcache", "run"))
    p.add_argument("--frame", default="bodyfit", choices=("physical", "bodyfit"))
    p.add_argument("--task", default="full")
    p.add_argument("--rep", default="nd")
    p.add_argument("--fill", default="nearest")
    p.add_argument("--n-train", type=int, default=800)
    p.add_argument("--n-test", type=int, default=200)
    p.add_argument("--n-ls", type=int, default=200)
    p.add_argument("--ls-chans", nargs="+", default=["u", "p"])
    p.add_argument("--n-proc", type=int, default=18)
    p.add_argument("--kd-workers", type=int, default=2)
    p.add_argument("--data-root", default=os.path.join("data", "Dataset"))
    p.add_argument("--out-dir", default=os.path.join("results", "interpolation"))
    p.add_argument("--scratch", default=P.DEF_SCRATCH)
    a = p.parse_args(argv)

    t0 = time.time()
    if a.stage == "surfcache":
        stage_surfcache(a)
    else:
        stage_run(a)
    log(f"stage {a.stage} total {time.time()-t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
