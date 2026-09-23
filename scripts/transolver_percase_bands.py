"""Per-case, per-band Transolver error at the native nodes, so the band statistics can be PAIRED.

Why this exists
---------------
Every per-case statistic this paper quotes for the near-wall bounds -- "161 of 200 cases
clear the threshold", "all 200 fall at or below 1", "152/200 cases beat Transolver" -- was
computed against ONE number: Transolver's band error pooled over all 200 cases
(``transolver_band_mse_u = 1.157649``). Dividing each case's interpolation residual by a
constant rescales that residual; it does not compare it with anything. Reviewer 2 (round 4,
W2) is right that such statistics say nothing about Transolver's case-to-case variation, and
that "cases beating Transolver" is not what they measure.

``point_space_transolver.json`` keeps per-case errors only as whole-case means. This script
recomputes, for every test case and every band, Transolver's per-node squared-error sum and
node count at the native nodes -- the identical checkpoints, loader, test cache, band index
and redimensionalisation the committed Transolver stage uses -- and writes them out so each
case can be compared with its OWN band error.

This is a diagnostic. It carries no threshold and it cannot change any pre-registered
verdict: B1 and the amendment-3 rule are registered against the pooled band error and stay
read that way. What it adds is the paired reading beside them.

It also records Transolver's error on the RESTRICTED node set of ``bodyfit_bound.py`` --
the near-wall nodes whose nearest-surface projection is interior to a surface chain. The
body-fitted bound is computed on those nodes only, while B1's denominator is Transolver's
error on all band nodes. The restricted sums let the paired body-fitted comparison use
exactly the same nodes on both sides.

Gate
----
T1  Pooling the per-case values back (sum of squared errors over sum of node counts, per band,
    per seed) must reproduce ``point_space_transolver.json`` ``acc`` to 1e-9 relative on every
    band, channel and seed. The per-case numbers are then the committed numbers, split out,
    and not a re-implementation.

Run
---
    PYTHONPATH=src .venv/Scripts/python.exe scripts/transolver_percase_bands.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import neuroforge  # noqa: F401  -- caps BLAS threads before numpy; see CLAUDE.md

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import point_space_headtohead as P  # noqa: E402

TOL = 1e-9


def log(msg: str) -> None:
    print(f"[tpc] {msg}", flush=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--task", default="full")
    ap.add_argument("--rep", default="nd")
    ap.add_argument("--fill", default="nearest")
    ap.add_argument("--n-train", type=int, default=800)
    ap.add_argument("--n-test", type=int, default=200)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--ckpt-dir", default=os.path.join("checkpoints", "v2_transolver"))
    ap.add_argument("--cache-dir", default=os.path.join("data", "cache"))
    ap.add_argument("--device", default="auto")
    ap.add_argument("--data-root", default=os.path.join("data", "Dataset"))
    ap.add_argument("--out-dir", default=os.path.join("results", "interpolation"))
    ap.add_argument("--scratch", default=P.DEF_SCRATCH)
    a = ap.parse_args(argv)

    import torch
    from neuroforge.data.pointcloud import load_airfrans_pointclouds
    from recompute_force_vs_official import load_backbone

    _tr, names_te, _cfg, _W, _base = P.setup(a)
    device = torch.device(a.device if a.device not in ("auto", "") else
                          ("cuda" if torch.cuda.is_available() else "cpu"))
    log(f"device={device}, {len(names_te)} test cases, seeds {a.seeds}")
    pcs = load_airfrans_pointclouds(root="data", task=a.task, train=False,
                                    limit=a.n_test, cache_dir=a.cache_dir)
    by = {pc.name: pc for pc in pcs}
    models = {}
    for s in a.seeds:
        m, pn, _gn, _nu = load_backbone(os.path.join(a.ckpt_dir, f"seed{s}.pt"), device)
        models[s] = (m, pn)

    import bodyfit_bound as BF

    NB = P.NB
    rows = []
    pool_se = {s: {c: np.zeros(NB) for c in P.CHANS} for s in a.seeds}
    pool_n = {s: np.zeros(NB) for s in a.seeds}
    t0 = time.time()
    for ci, nm in enumerate(names_te):
        pc = by[nm]
        _pos, tgt, sdf, _incrop = P.load_test_case(a.scratch, nm)
        b = P.band_index8(sdf)
        n_b = np.array([int((b == bi).sum()) for bi in range(NB)], np.float64)
        # the body-fitted arm's scored set: strip nodes with an interior projection
        apos_t, anrm_t = BF.load_test_surface(a.scratch, nm)
        inter, _isurf = BF.interior_mask(_pos, apos_t, anrm_t)
        keep = inter & (b <= BF.STRIP_BAND_MAX)
        n_r = np.array([int(((b == bi) & keep).sum()) for bi in range(BF.STRIP_BAND_MAX + 1)],
                       np.float64)
        row = {"name": nm, "n_band": n_b.astype(int).tolist(),
               "n_band_restricted": n_r.astype(int).tolist(), "seeds": {}}
        for s in a.seeds:
            m, pn = models[s]
            feats = pn.transform_in(pc.features)
            with torch.no_grad():
                y = m(torch.from_numpy(feats).to(device).unsqueeze(0)).squeeze(0)
                y = pn.inverse_out(y)
            pr = np.asarray(y.detach().cpu().numpy(), np.float64)
            se, se_r = {}, {}
            for j, c in enumerate(P.CHANS):
                d2 = (pr[:, j] - tgt[:, j]) ** 2
                v = np.array([float(d2[b == bi].sum()) for bi in range(NB)])
                se[c] = v.tolist()
                se_r[c] = [float(d2[(b == bi) & keep].sum())
                           for bi in range(BF.STRIP_BAND_MAX + 1)]
                pool_se[s][c] += v
            pool_n[s] += n_b
            row["seeds"][f"seed{s}"] = {"se": se, "se_restricted": se_r}
        rows.append(row)
        if (ci + 1) % 25 == 0:
            log(f"{ci+1}/{len(names_te)} ({time.time()-t0:.0f}s)")

    # ---- T1: pooled back, the per-case values must BE the committed numbers ----
    ref = json.load(open(os.path.join(a.out_dir, "point_space_transolver.json"),
                         encoding="utf-8"))
    worst, at = 0.0, None
    for s in a.seeds:
        acc = ref["acc"][f"seed{s}"]
        n_ref = np.asarray(acc["n"], np.float64)
        if np.max(np.abs(pool_n[s] - n_ref)) > 0:
            raise SystemExit(f"T1 FAILED seed{s}: band node counts differ from the artifact")
        for c in P.CHANS:
            se_ref = np.asarray(acc["se"][c], np.float64)
            m = n_ref > 0
            rel = np.abs(pool_se[s][c][m] - se_ref[m]) / np.maximum(np.abs(se_ref[m]), 1e-300)
            r = float(np.max(rel))
            if r > worst:
                worst, at = r, f"seed{s}/{c}"
    if worst > TOL:
        raise SystemExit(f"T1 FAILED: worst relative difference {worst:.3e} at {at}")
    log(f"T1 PASS  per-case values pool back to point_space_transolver.json "
        f"(worst rel {worst:.2e} at {at})")

    out = {"artifact": "transolver_percase_bands",
           "note": ("per-case, per-band Transolver squared-error sums and node counts at the "
                    "native nodes, so near-wall band statistics can be paired per case; a "
                    "diagnostic with no threshold -- the registered verdicts are unchanged"),
           "meta": {"seeds": list(a.seeds), "n_test": len(names_te), "device": str(device),
                    "bands": P.NAMES8, "channels": list(P.CHANS),
                    "wallclock_sec": time.time() - t0},
           "T1": {"worst_rel_vs_point_space_transolver": worst, "at": at,
                  "threshold": TOL, "status": "PASS"},
           "rows": rows}
    dest = os.path.join(a.out_dir, "transolver_percase_bands.json")
    P.write_json(dest, out)
    log(f"wrote {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
