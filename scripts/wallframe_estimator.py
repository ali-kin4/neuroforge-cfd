"""Does the wall-frame interpolator, re-selected for its own frame, match the surrogate?

PRE-REGISTERED. This docstring is committed before any stage of this script runs, before
any number it produces exists, and before the selection it makes is known.

Background
----------
``bodyfit_bound.py`` re-posed parameter interpolation in a parameter-free wall-following
frame ``(s, n)`` and found the least-squares bound over the 800 training fields falls to
0.0044x the surrogate's error inside 0.005c. The DEPLOYED estimator in that frame kept the
weights selected for the physical-coordinate raster (KRR, sigma=1.0, lam=1e-3, w_U=0.25)
and trails the surrogate on the case-mean, R = 1.7085 (``paired_band_stats.json``,
``krr_bodyfit_restricted``). Those weights were never selected for the frame they are now
used in. This script asks the one remaining question: selected for the wall frame, on
training data only, does the estimator match the surrogate?

The family, frozen here
-----------------------
``config_grid("nd")`` of ``parameter_interpolation_baseline.py``, verbatim: nearest
neighbour; inverse-distance kNN with k in {2,4,8,16,32,64}; local-linear with k in
{16,32,64,128}; Gaussian KRR with sigma in {0.5,1,2} (x median distance) and lam in
{1e-3,1e-1}; each with w_U in {0.25,1}; plus the uniform mean. 33 configurations. The
frame, the strip (n <= 0.05c), the interior-projection restriction (amendment B-1 of
``bodyfit_bound.py``) and the field transfer are those of ``bodyfit_bound.py``, unchanged
and imported.

Selection -- stage ``cv``, TRAINING DATA ONLY
---------------------------------------------
Leave-one-out over all 800 training cases. For query case q, the weights are
``build_weights(cfg, Xtr, Xtr[q], active = all but q)``, so q never predicts itself
(gate GR3). Every training field j is transferred into q's wall frame by the identical
transfer used for test cases. Objective: the mean over the 800 queries of the per-case
MSE of u, in physical units, on band 0-0.005c of q's restricted strip, evaluated on a
seeded random subsample of at most 1500 nodes per query (``numpy.random.default_rng
((0, q))``). The configuration with the smallest objective is the selection; ties go to
the first in grid order. The selection and the full table are written to
``results/interpolation/wallframe_cv_selection.json``, which must be COMMITTED before
stage ``test`` will run (gate GR2).

Verdict -- stage ``test``
-------------------------
The selected configuration, all 800 training cases active, the 200 test cases, channel u,
band 0-0.005c, restricted nodes (the node set of ``bodyfit_bound.py``):

    R = mean over cases of MSE_c  /  T_pool_res(u)

where ``T_pool_res`` is the surrogate's error computed exactly as ``paired_band_stats.py``
computes it (three seeds, node-weighted pooled over the restricted nodes). This is the
definition that gives R = 1.7085 for the current weights.

    R <= 1.0   ->  WALL-FRAME-ESTIMATOR-MATCHES
    R >  1.0   ->  WALL-FRAME-ESTIMATOR-TRAILS

Descriptive only, no verdict: the paired per-case ratios (median, cases below 1); v, p
and nu_t on the same nodes; the published configuration's numbers beside the selection's.

Gates -- the run aborts if any fails
------------------------------------
GR1  The published configuration, computed by this script, reproduces
     ``bodyfit_bound.json`` (frame bodyfit, channel u, band 0-0.005c, ``krr_mse``) on
     every test case to a relative 1e-4, and its R reproduces 1.7085 to a relative 1e-3.
     Asserted before the selected configuration's R is computed.
GR2  No test case appears in stage ``cv``; the selection file is committed and unmodified
     in the working tree before stage ``test`` runs.
GR3  Every leave-one-out weight row gives its own query zero weight.

Stop rule
---------
One grid, one selection, one scoring. There is no second grid and no re-run with other
settings, whatever the verdict.

Run
---
    .venv/Scripts/python.exe scripts/wallframe_estimator.py --stage smoke
    .venv/Scripts/python.exe scripts/wallframe_estimator.py --stage cv
    git add results/interpolation/wallframe_cv_selection.json && git commit
    .venv/Scripts/python.exe scripts/wallframe_estimator.py --stage test
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import subprocess
import sys
import time

import neuroforge  # noqa: F401  -- caps BLAS threads before numpy; see CLAUDE.md

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bodyfit_bound as B  # noqa: E402
import point_space_headtohead as P  # noqa: E402
from parameter_interpolation_baseline import (  # noqa: E402
    build_weights, case_features, cfg_name, config_grid)

BAND = B.VERDICT_BAND            # 1 == '0-0.005c'
STRIP = B.STRIP_BAND_MAX         # 4 == n <= 0.05c
N_SUB = 1500                     # CV nodes per query (pre-registered)
R_PUBLISHED = 1.7085             # paired_band_stats.json, krr_bodyfit_restricted
SEL_PATH = os.path.join("results", "interpolation", "wallframe_cv_selection.json")
OUT_PATH = os.path.join("results", "interpolation", "wallframe_estimator.json")


def log(msg: str) -> None:
    print(f"[wallframe] {msg}", flush=True)


# --------------------------------------------------------------------------- #
# Query sets. Pure functions of the caches; no weights, no targets of other cases.
# --------------------------------------------------------------------------- #
def train_query(args):
    """Seeded band-1 restricted nodes of TRAIN case q, in q's wall frame, with truth."""
    scratch, qi, name = args
    pos, yhat, apos, anrm = P.load_train_case(scratch, name)
    sdf = np.asarray(np.load(P._train_cache_path(scratch, name))["sdf"], np.float64)
    b = P.band_index8(sdf)
    inter, _ = B.interior_mask(pos, apos, anrm, 1)
    idx = np.nonzero((b <= STRIP) & inter & (b == BAND))[0]
    rng = np.random.default_rng((0, qi))
    if idx.size > N_SUB:
        idx = np.sort(rng.choice(idx, N_SUB, replace=False))
    qb = B.to_frame(pos[idx], sdf[idx], apos, anrm, "bodyfit", 1)
    U, al = P.case_U_alpha(name)
    truth_u = P.redim_nodes(yhat[idx], U, al)[:, 0]
    return qb, truth_u, U, al


def test_query(args):
    """All band-1 restricted nodes of TEST case t, in t's wall frame, with raw targets."""
    scratch, name = args
    pos, tgt, sdf, _ = P.load_test_case(scratch, name)
    apos, anrm = B.load_test_surface(scratch, name)
    b = P.band_index8(sdf)
    inter, _ = B.interior_mask(pos, apos, anrm, 1)
    sel = (b <= STRIP) & inter & (b == BAND)
    qb = B.to_frame(pos[sel], sdf[sel], apos, anrm, "bodyfit", 1)
    U, al = P.case_U_alpha(name)
    return qb, tgt[sel], U, al


# --------------------------------------------------------------------------- #
# The transfer: identical to bodyfit_bound.case_worker's body-fitted branch.
# --------------------------------------------------------------------------- #
def source(scratch, name):
    from scipy.interpolate import LinearNDInterpolator
    from scipy.spatial import Delaunay, cKDTree
    pos_j, yhat_j, apos_j, anrm_j = P.load_train_case(scratch, name)
    sdf_j = np.asarray(np.load(P._train_cache_path(scratch, name))["sdf"], np.float64)
    kj = P.band_index8(sdf_j) <= STRIP
    ij, _ = B.interior_mask(pos_j, apos_j, anrm_j, 1)
    kj = kj & ij
    src = B.to_frame(pos_j[kj], sdf_j[kj], apos_j, anrm_j, "bodyfit", 1)
    valj = yhat_j[kj]
    return LinearNDInterpolator(Delaunay(src), valj, fill_value=np.nan), cKDTree(src), valj


def transfer(lin, kt, valj, Q):
    v = np.asarray(lin(Q), np.float64)
    out = np.isnan(v[:, 0])
    if out.any():
        _d, idx = kt.query(Q[out], k=1)
        v[out] = valj[idx]
    return v


def redim(v, U, cosa, sina):
    """``P.redim_nodes`` vectorised over nodes that carry their own case's (U, alpha)."""
    out = np.empty_like(v)
    out[:, 0] = v[:, 0] * U + U * cosa
    out[:, 1] = v[:, 1] * U + U * sina
    out[:, 2] = v[:, 2] * U ** 2
    out[:, 3] = v[:, 3] * (U * P.CHORD)
    return out


# --------------------------------------------------------------------------- #
# Workers over training fields j: build each source ONCE, evaluate at every query.
# --------------------------------------------------------------------------- #
def cv_worker(args):
    scratch, work, js, names_tr = args
    z = np.load(os.path.join(work, "cv_queries.npz"))
    Q, qid, U, ca, sa = z["Q"], z["qid"], z["U"], z["cos"], z["sin"]
    Wl = np.load(os.path.join(work, "cv_weights.npy"), mmap_mode="r")   # (G, 800, 800)
    acc = np.zeros((Wl.shape[0], Q.shape[0]), np.float64)
    t0 = time.time()
    for n, j in enumerate(js):
        lin, kt, valj = source(scratch, names_tr[j])
        u = transfer(lin, kt, valj, Q)[:, 0] * U + U * ca
        acc += np.asarray(Wl[:, :, j], np.float64)[:, qid] * u[None, :]
        if (n + 1) % 10 == 0:
            log(f"  cv worker {js[0]}: {n+1}/{len(js)} ({time.time()-t0:.0f}s)")
    dest = os.path.join(work, f"cv_acc_{js[0]}.npy")
    np.save(dest, acc)
    return dest


def test_worker(args):
    scratch, work, js, names_tr, tag = args
    z = np.load(os.path.join(work, f"{tag}_queries.npz"))
    Q, cid, U, ca, sa = z["Q"], z["cid"], z["U"], z["cos"], z["sin"]
    Wt = np.load(os.path.join(work, f"{tag}_weights.npy"))           # (K, n_test, 800)
    acc = np.zeros((Wt.shape[0], 4, Q.shape[0]), np.float32)
    t0 = time.time()
    for n, j in enumerate(js):
        lin, kt, valj = source(scratch, names_tr[j])
        ph = redim(transfer(lin, kt, valj, Q), U, ca, sa)
        for k in range(Wt.shape[0]):
            w = Wt[k, :, j][cid]
            acc[k] += (ph * w[:, None]).T.astype(np.float32)
        if (n + 1) % 10 == 0:
            log(f"  {tag} worker {js[0]}: {n+1}/{len(js)} ({time.time()-t0:.0f}s)")
    dest = os.path.join(work, f"{tag}_acc_{js[0]}.npy")
    np.save(dest, acc)
    return dest


def run_pool(fn, jobs, n_proc):
    if n_proc <= 1:
        return [fn(j) for j in jobs]
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=n_proc) as ex:
        return list(ex.map(fn, jobs))


def split(n, k):
    return [list(range(i, n, k)) for i in range(k)]


# --------------------------------------------------------------------------- #
# The surrogate's error, exactly as paired_band_stats.py computes it.
# --------------------------------------------------------------------------- #
def surrogate(names):
    tpc = json.load(open(os.path.join("results", "interpolation",
                                      "transolver_percase_bands.json"), encoding="utf-8"))
    seeds = [f"seed{s}" for s in tpc["meta"]["seeds"]]
    T = {r["name"]: r for r in tpc["rows"]}
    out = {}
    for c in P.CHANS:
        if c not in T[names[0]]["seeds"][seeds[0]]["se_restricted"]:
            continue
        t_res = np.array([np.mean([T[nm]["seeds"][s]["se_restricted"][c][BAND]
                                   / T[nm]["n_band_restricted"][BAND] for s in seeds])
                          for nm in names])
        n_res = np.array([T[nm]["n_band_restricted"][BAND] for nm in names], float)
        out[c] = {"t_res": t_res, "pool": float(np.sum(t_res * n_res) / np.sum(n_res)),
                  "n_res": n_res}
    return out


def summary(x):
    x = np.asarray(x, np.float64)
    return {"median": float(np.median(x)), "mean": float(np.mean(x)),
            "p05": float(np.percentile(x, 5)), "p95": float(np.percentile(x, 95)),
            "max": float(np.max(x)), "n_below_1": int(np.sum(x < 1)), "n": int(x.size)}


# --------------------------------------------------------------------------- #
# Stages
# --------------------------------------------------------------------------- #
def stage_cv(a, names_tr, names_te, Xtr):
    assert not set(names_tr) & set(names_te), "GR2 FAILED: train/test overlap"
    work = os.path.join(a.scratch, "wallframe")
    os.makedirs(work, exist_ok=True)
    grid = config_grid("nd")
    ntr = len(names_tr)
    t0 = time.time()

    qs = run_pool(train_query, [(a.scratch, i, nm) for i, nm in enumerate(names_tr)], a.n_proc)
    Q = np.concatenate([q[0] for q in qs])
    truth = np.concatenate([q[1] for q in qs])
    qid = np.concatenate([np.full(len(q[1]), i, np.int64) for i, q in enumerate(qs)])
    Un = np.concatenate([np.full(len(q[1]), q[2]) for q in qs])
    an = np.radians(np.concatenate([np.full(len(q[1]), q[3]) for q in qs]))
    np.savez(os.path.join(work, "cv_queries.npz"), Q=Q, qid=qid, U=Un,
             cos=np.cos(an), sin=np.sin(an))
    log(f"cv queries: {ntr} cases, {Q.shape[0]} nodes ({time.time()-t0:.0f}s)")

    Wl = np.zeros((len(grid), ntr, ntr), np.float32)
    for g, cfg in enumerate(grid):
        for q in range(ntr):
            act = np.ones(ntr, bool)
            act[q] = False
            Wl[g, q] = build_weights(cfg, Xtr, Xtr[q:q + 1], active=act)[0]
        assert np.all(Wl[g][np.arange(ntr), np.arange(ntr)] == 0.0), "GR3 FAILED"
    np.save(os.path.join(work, "cv_weights.npy"), Wl)
    log(f"GR3 PASS: {len(grid)} configs x {ntr} leave-one-out rows ({time.time()-t0:.0f}s)")

    parts = run_pool(cv_worker, [(a.scratch, work, js, names_tr)
                                 for js in split(ntr, a.n_proc)], a.n_proc)
    pred = sum(np.load(p) for p in parts)
    for p in parts:
        os.remove(p)
    se = (pred - truth[None, :]) ** 2
    cnt = np.bincount(qid, minlength=ntr).astype(float)
    per_case = np.stack([np.bincount(qid, weights=se[g], minlength=ntr) / cnt
                         for g in range(len(grid))])
    obj = per_case.mean(1)
    best = int(np.argmin(obj))
    table = sorted(({"cfg": cfg_name(c), "objective_mse_u": float(obj[g])}
                    for g, c in enumerate(grid)), key=lambda r: r["objective_mse_u"])
    names_hash = hashlib.sha256("\n".join(names_tr).encode()).hexdigest()
    out = {"artifact": "wallframe_cv_selection", "stage": "cv (training data only)",
           "selected_cfg": grid[best], "selected_name": cfg_name(grid[best]),
           "selected_objective_mse_u": float(obj[best]),
           "published_cfg_objective_mse_u": float(obj[[cfg_name(c) for c in grid].index(
               "krr_s1.0_l0.001_wU0.25")]),
           "n_queries": ntr, "n_nodes": int(Q.shape[0]), "n_sub_per_query": N_SUB,
           "train_names_sha256": names_hash, "table": table,
           "gates": {"GR2": "PASS (no test case used)", "GR3": "PASS"},
           "wallclock_sec": time.time() - t0}
    with io.open(SEL_PATH, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(out, indent=2) + "\n")
    log(f"selected {out['selected_name']} (objective {obj[best]:.4g}; published config "
        f"{out['published_cfg_objective_mse_u']:.4g}). Commit {SEL_PATH} before --stage test.")


def test_pass(a, names_tr, names_te, Xtr, cfgs, tag):
    work = os.path.join(a.scratch, "wallframe")
    os.makedirs(work, exist_ok=True)
    t0 = time.time()
    qs = run_pool(test_query, [(a.scratch, nm) for nm in names_te], a.n_proc)
    Q = np.concatenate([q[0] for q in qs])
    tgt = np.concatenate([q[1] for q in qs])
    cid = np.concatenate([np.full(len(q[1]), i, np.int64) for i, q in enumerate(qs)])
    Un = np.concatenate([np.full(len(q[1]), q[2]) for q in qs])
    an = np.radians(np.concatenate([np.full(len(q[1]), q[3]) for q in qs]))
    np.savez(os.path.join(work, f"{tag}_queries.npz"), Q=Q, cid=cid, U=Un,
             cos=np.cos(an), sin=np.sin(an))
    Xte = np.stack([case_features(n) for n in names_te])
    Wt = np.stack([build_weights(c, Xtr, Xte) for c in cfgs]).astype(np.float32)
    np.save(os.path.join(work, f"{tag}_weights.npy"), Wt)
    log(f"{tag} queries: {len(names_te)} cases, {Q.shape[0]} nodes ({time.time()-t0:.0f}s)")

    parts = run_pool(test_worker, [(a.scratch, work, js, names_tr, tag)
                                   for js in split(len(names_tr), a.n_proc)], a.n_proc)
    pred = sum(np.load(p).astype(np.float64) for p in parts)
    for p in parts:
        os.remove(p)
    nte = len(names_te)
    cnt = np.bincount(cid, minlength=nte).astype(float)
    mse = np.zeros((len(cfgs), 4, nte))
    for k in range(len(cfgs)):
        for ci in range(4):
            mse[k, ci] = np.bincount(cid, weights=(pred[k, ci] - tgt[:, ci]) ** 2,
                                     minlength=nte) / cnt
    return mse, cnt, time.time() - t0


def gr1(names_te, mse_pub_u, cnt):
    bf = {r["name"]: r for r in json.load(open(os.path.join(
        "results", "interpolation", "bodyfit_bound.json"), encoding="utf-8"))["rows"]}
    ref = np.array([bf[n]["frames"]["bodyfit"]["u"]["krr_mse"][BAND] for n in names_te])
    nref = np.array([bf[n]["n_band"][BAND] for n in names_te])
    assert np.array_equal(nref, cnt.astype(int)), "GR1 FAILED: node counts differ"
    rel = np.abs(mse_pub_u - ref) / ref
    assert rel.max() <= 1e-4, f"GR1 FAILED: worst relative {rel.max():.3e}"
    return float(rel.max())


def stage_smoke(a, names_tr, names_te, Xtr, cfg_pub):
    sub = names_te[:a.smoke_cases]
    mse, cnt, dt = test_pass(a, names_tr, sub, Xtr, [cfg_pub], "smoke")
    worst = gr1(sub, mse[0, 0], cnt)
    log(f"GR1 (smoke, {len(sub)} cases, all {len(names_tr)} fields): worst relative "
        f"{worst:.3e}  PASS  ({dt:.0f}s)")
    # full-size timing: one training field evaluated at every test node
    qs = run_pool(test_query, [(a.scratch, nm) for nm in names_te], a.n_proc)
    Q = np.concatenate([q[0] for q in qs])
    t0 = time.time()
    lin, kt, valj = source(a.scratch, names_tr[0])
    transfer(lin, kt, valj, Q)
    per = time.time() - t0
    log(f"one field at all {Q.shape[0]} test nodes: {per:.1f}s -> full test pass "
        f"~{per * len(names_tr) / a.n_proc / 60:.0f} min on {a.n_proc} workers")


def stage_test(a, names_tr, names_te, Xtr, cfg_pub):
    tracked = subprocess.run(["git", "ls-files", "--error-unmatch", SEL_PATH],
                             capture_output=True).returncode == 0
    clean = subprocess.run(["git", "diff", "--quiet", "HEAD", "--", SEL_PATH]).returncode == 0
    assert tracked and clean, f"GR2 FAILED: commit {SEL_PATH} unmodified before --stage test"
    sel = json.load(open(SEL_PATH, encoding="utf-8"))
    assert sel["train_names_sha256"] == hashlib.sha256(
        "\n".join(names_tr).encode()).hexdigest(), "GR2 FAILED: training set differs"
    cfg_sel = sel["selected_cfg"]
    log(f"GR2 PASS: selection {sel['selected_name']} is committed and unmodified")

    mse, cnt, dt = test_pass(a, names_tr, names_te, Xtr, [cfg_pub, cfg_sel], "test")
    S = surrogate(names_te)
    worst = gr1(names_te, mse[0, 0], cnt)
    r_pub = float(mse[0, 0].mean() / S["u"]["pool"])
    assert abs(r_pub - R_PUBLISHED) / R_PUBLISHED <= 1e-3, f"GR1 FAILED: R_pub {r_pub:.5f}"
    log(f"GR1 PASS: worst per-case relative {worst:.3e}; R(published) {r_pub:.4f}")

    r_sel = float(mse[1, 0].mean() / S["u"]["pool"])
    verdict = ("WALL-FRAME-ESTIMATOR-MATCHES" if r_sel <= 1.0
               else "WALL-FRAME-ESTIMATOR-TRAILS")
    desc = {}
    for ci, c in enumerate(P.CHANS):
        if c not in S:
            continue
        desc[c] = {k: {"casemean_ratio": float(mse[i, ci].mean() / S[c]["pool"]),
                       "paired": summary(mse[i, ci] / S[c]["t_res"])}
                   for i, k in ((0, "published_cfg"), (1, "selected_cfg"))}
    out = {"artifact": "wallframe_estimator", "band": P.NAMES8[BAND], "channel": "u",
           "selected_cfg": cfg_sel, "selected_name": sel["selected_name"],
           "R_selected": r_sel, "R_published": r_pub, "threshold": 1.0,
           "verdict": verdict, "descriptive": desc,
           "gates": {"GR1": {"status": "PASS", "worst_rel": worst, "R_published": r_pub},
                     "GR2": "PASS", "GR3": "PASS (stage cv)"},
           "rows": [{"name": n, "n_band": int(cnt[i]),
                     "mse": {c: {"published_cfg": float(mse[0, ci, i]),
                                 "selected_cfg": float(mse[1, ci, i])}
                             for ci, c in enumerate(P.CHANS)}}
                    for i, n in enumerate(names_te)],
           "wallclock_sec": dt}
    with io.open(OUT_PATH, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(out, indent=2) + "\n")
    log(f"R(selected) = {r_sel:.4f}  ->  {verdict}")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--stage", required=True, choices=("smoke", "cv", "test"))
    p.add_argument("--task", default="full")
    p.add_argument("--rep", default="nd")
    p.add_argument("--fill", default="nearest")
    p.add_argument("--n-train", type=int, default=800)
    p.add_argument("--n-test", type=int, default=200)
    p.add_argument("--n-proc", type=int, default=12)
    p.add_argument("--smoke-cases", type=int, default=3)
    p.add_argument("--data-root", default=os.path.join("data", "Dataset"))
    p.add_argument("--out-dir", default=os.path.join("results", "interpolation"))
    p.add_argument("--scratch", default=P.DEF_SCRATCH)
    a = p.parse_args(argv)

    names_tr, names_te, cfg_pub, _W, _base = P.setup(a)
    Xtr = np.stack([case_features(n) for n in names_tr])
    t0 = time.time()
    if a.stage == "smoke":
        stage_smoke(a, names_tr, names_te, Xtr, cfg_pub)
    elif a.stage == "cv":
        stage_cv(a, names_tr, names_te, Xtr)
    else:
        stage_test(a, names_tr, names_te, Xtr, cfg_pub)
    log(f"stage {a.stage} total {time.time()-t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
