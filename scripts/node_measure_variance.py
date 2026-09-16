"""The node-measure variance of the AirfRANS fields, and the two claims it corrects.

This is a measurement of a property of the dataset, not a hypothesis test, so it
carries no decision rule. It exists because two claims in this project compare a
NODE-measure error against an AREA-measure variance, which is the exact confusion the
paper is about.

``VAR_TRAIN`` in ``point_space_headtohead.py`` is the variance THE SURROGATES ARE
TRAINED ON. The training pipeline rasterises to r128, so the normaliser is fitted on
area-uniform samples and ``VAR_TRAIN`` is an area-uniform statistic:

    u 341.88    v 52.45    p 135590    nut 1.3199e-6

``body.tex:714`` uses it correctly and says so -- that table is area-uniform in every
cell and the divisors are common to both arms, so the ordering is unaffected.

Two other uses are not correct, and this script measures the comparator they need.

CLAIM 1 -- the surface row on pressure (``body.tex:806``). The text compares a native,
per-node, surface-band MSE (99,227 for Transolver, 149,203 for the interpolator)
against the area-uniform global variance 135,590, and concludes both arms are "poor
there in absolute terms". Those are different measures. The comparator the sentence
needs is the node-measure variance of ``p`` among the surface nodes themselves. This
script measures it. Reviewer 2's round-3 W4 inherits the same mismatch and reads
"roughly R^2 = 0.27 on the channel every force integrates from" off it.

CLAIM 2 -- is this a competent Transolver (Reviewer 2, round 3, W5)? The manuscript
declines a leaderboard row because AirfRANS results are reported in incompatible
conventions, which is right for a head-to-head but leaves the surrogate's own
competence unevidenced. The AirfRANS paper's own convention -- volume MSE on
standardized fields -- is recoverable: standardized MSE is physical MSE divided by the
train-set variance, and AirfRANS is a POINT-CLOUD benchmark, so its volume MSE is a
node quantity and the divisor must be the node-measure train variance. Both are
computed here from the cached fields, so the comparison is made in their units rather
than ours.

Everything runs on the native clouds with no raster anywhere.
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

#: Published AirfRANS `full`-task baselines, volume MSE on standardized fields (x1e-2).
#: Bonnet et al., NeurIPS 2022 D&B, Table 3; read at source and recorded in
#: docs/paper/review/published_baselines_verified.md. Same rows as tab:airfrans-sota.
PUBLISHED = {"MLP": (0.95, 0.74), "GraphSAGE": (0.83, 0.66),
             "PointNet": (3.50, 1.15), "Graph U-Net": (1.52, 0.66)}

IDX = {"u": 0, "v": 1, "p": 2, "nut": 3}


def log(msg: str) -> None:
    print(f"[nodevar] {msg}", flush=True)


class Pool:
    """Pooled mean/variance over concatenated node sets, one streaming pass."""

    def __init__(self, k: int) -> None:
        self.n = 0.0
        self.s = np.zeros(k, np.float64)
        self.ss = np.zeros(k, np.float64)

    def add(self, x: np.ndarray) -> None:
        self.n += x.shape[0]
        self.s += x.sum(axis=0)
        self.ss += (x.astype(np.float64) ** 2).sum(axis=0)

    def mean(self) -> np.ndarray:
        return self.s / max(self.n, 1.0)

    def var(self) -> np.ndarray:
        m = self.mean()
        return self.ss / max(self.n, 1.0) - m * m


def test_variance(scratch: str):
    """Per-band and pooled variance of the TEST targets, node measure."""
    names = sorted(n[:-4] for n in os.listdir(os.path.join(scratch, "test")))
    log(f"pooling {len(names)} test cases (node measure, native cloud)")
    bands = [Pool(4) for _ in range(P.NB)]
    allp = Pool(4)
    for nm in names:
        _pos, tgt, sdf, _incrop = P.load_test_case(scratch, nm)
        b = P.band_index8(sdf)
        y = np.asarray(tgt, np.float64)
        allp.add(y)
        for bi in range(P.NB):
            m = b == bi
            if m.any():
                bands[bi].add(y[m])
    return bands, allp, len(names)


def train_variance(scratch: str):
    """Pooled variance of the TRAIN fields in PHYSICAL units, node measure.

    The cache stores the published nondimensional representation, so each case is
    redimensionalised with its own (U, alpha) through the same ``redim_nodes`` the
    rest of the pipeline uses.
    """
    names = sorted(n[:-4] for n in os.listdir(os.path.join(scratch, "train")))
    log(f"pooling {len(names)} train cases (node measure, physical units)")
    pool = Pool(4)
    t0 = time.time()
    for i, nm in enumerate(names):
        z = np.load(P._train_cache_path(scratch, nm))
        U, al = P.case_U_alpha(nm)
        pool.add(P.redim_nodes(np.asarray(z["yhat"], np.float64), U, al))
        if (i + 1) % 200 == 0:
            log(f"  {i+1}/{len(names)} ({time.time()-t0:.0f}s)")
    return pool, len(names)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--scratch", default=P.DEF_SCRATCH)
    ap.add_argument("--out-dir", default=os.path.join("results", "interpolation"))
    ap.add_argument("--out", default="node_measure_variance.json")
    a = ap.parse_args(argv)

    bands, allp, n_te = test_variance(a.scratch)
    trp, n_tr = train_variance(a.scratch)

    var_b = np.stack([p.var() for p in bands])
    n_b = np.asarray([p.n for p in bands])
    var_te = allp.var()
    var_tr = trp.var()

    log("")
    log("%-12s %10s %14s %14s %14s" % ("band", "n", "var(u)", "var(v)", "var(p)"))
    for bi in range(P.NB):
        log("%-12s %10d %14.6g %14.6g %14.6g"
            % (P.NAMES8[bi], n_b[bi], var_b[bi, 0], var_b[bi, 1], var_b[bi, 2]))
    log("%-12s %10d %14.6g %14.6g %14.6g"
        % ("ALL(test)", allp.n, var_te[0], var_te[1], var_te[2]))
    log("%-12s %10d %14.6g %14.6g %14.6g"
        % ("ALL(train)", trp.n, var_tr[0], var_tr[1], var_tr[2]))
    log("")
    log("node-measure / area-uniform (VAR_TRAIN):  u %.2fx  v %.2fx  p %.2fx"
        % (var_tr[0] / P.VAR_TRAIN["u"], var_tr[1] / P.VAR_TRAIN["v"],
           var_tr[2] / P.VAR_TRAIN["p"]))

    # ---- CLAIM 1: the surface row on pressure -------------------------------
    surf_var_p = float(var_b[0, 2])
    claim1 = {"band": P.NAMES8[0], "channel": "p",
              "node_measure_var": surf_var_p,
              "area_uniform_var_used_in_text": P.VAR_TRAIN["p"],
              "ratio": surf_var_p / P.VAR_TRAIN["p"], "arms": {}}
    log("")
    log(f"CLAIM 1  surface-band node-measure Var(p) = {surf_var_p:.6g}; "
        f"the text compares against {P.VAR_TRAIN['p']:.6g} "
        f"({surf_var_p / P.VAR_TRAIN['p']:.0f}x smaller)")
    for arm, mse in (("interpolator", 149203.0), ("Transolver", 99227.0)):
        frac = mse / surf_var_p
        claim1["arms"][arm] = {"surface_mse_p": mse,
                               "frac_of_node_measure_band_var": frac,
                               "one_minus_frac": 1.0 - frac,
                               "one_minus_frac_with_area_divisor": 1.0 - mse / P.VAR_TRAIN["p"]}
        log(f"  {arm:<13} MSE {mse:9.0f} = {100*frac:6.3f}% of the band's own variance "
            f"(1-frac {1-frac:.4f}); the area divisor would say "
            f"{1 - mse / P.VAR_TRAIN['p']:.4f}")

    # ---- CLAIM 2: competence, in the AirfRANS paper's convention ------------
    hh = json.load(open(os.path.join(a.out_dir, "point_space_headtohead.json"),
                        encoding="utf-8"))
    claim2 = {"convention": ("volume MSE on standardized fields, x1e-2; divisor is the "
                            "NODE-measure physical train variance measured here"),
              "divisor_train_node_var": {c: float(var_tr[IDX[c]]) for c in ("u", "v", "p")},
              "ours": {}, "published": PUBLISHED}
    log("")
    log("CLAIM 2  our Transolver in the AirfRANS paper's convention (x1e-2)")
    for dom in ("full", "in_crop"):
        t = hh["tables"]["transolver_mean"][dom]
        rec = {}
        for c in ("u", "v", "p"):
            rec[c] = 100.0 * float(t[c]["pooled"]) / float(var_tr[IDX[c]])
        claim2["ours"][dom] = rec
        log("  ours (%-8s)  u %.4f   v %.4f   p %.4f" % (dom, rec["u"], rec["v"], rec["p"]))
    best_u = min(v[0] for v in PUBLISHED.values())
    best_p = min(v[1] for v in PUBLISHED.values())
    ou = claim2["ours"]["full"]["u"]
    op = claim2["ours"]["full"]["p"]
    claim2["vs_best_published"] = {"u_factor_better": best_u / ou, "p_factor_better": best_p / op,
                                   "best_published_u": best_u, "best_published_p": best_p}
    for k, (uu, pp) in PUBLISHED.items():
        log("    %-12s u %.2f   p %.2f" % (k, uu, pp))
    log(f"  best published: u {best_u:.2f}, p {best_p:.2f} "
        f"-> ours is {best_u/ou:.1f}x better on u and {best_p/op:.1f}x better on p")

    out = {"artifact": "node_measure_variance",
           "question": ("the node-measure variance of the AirfRANS fields, and the two "
                        "claims that used an area-uniform variance against a node-measure "
                        "error"),
           "n_test_cases": n_te, "n_train_cases": n_tr,
           "bands": P.NAMES8, "channels": list(P.CHANS),
           "n_band_test": n_b.tolist(),
           "var_per_band_test": var_b.tolist(),
           "var_all_test": var_te.tolist(),
           "var_all_train_physical": var_tr.tolist(),
           "n_nodes_test": allp.n, "n_nodes_train": trp.n,
           "var_train_area_uniform": P.VAR_TRAIN,
           "claim1_surface_pressure": claim1,
           "claim2_competence_vs_published": claim2}
    os.makedirs(a.out_dir, exist_ok=True)
    dest = os.path.join(a.out_dir, a.out)
    with open(dest, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    log(f"wrote {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
