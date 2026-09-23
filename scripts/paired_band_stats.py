"""Paired per-case statistics for the near-wall bounds, on identical nodes.

The pre-registered verdicts (amendment 3's FAMILY-BOUND-CLOSED, B1's BODY-FITTED-BOUND-OPEN)
are read against Transolver's band error POOLED over all 200 cases -- one constant. That is
how they were registered and how they stay. But every per-case figure quoted beside them
("161/200 clear 10", "all 200 at or below 1", "152/200 beat Transolver") divided each case's
interpolation residual by that same constant, which rescales the residual and compares it
with nothing. This script supplies the comparison those sentences imply: each case's
interpolation error against THAT case's Transolver error, on exactly the same nodes.

Two node sets matter and are kept apart:

* the physical-frame bound of amendment 3 is scored on ALL band nodes, so it is paired with
  Transolver's error on all band nodes;
* the body-fitted arm (and the physical arm rescored beside it) is scored on the RESTRICTED
  set -- strip nodes whose nearest-surface projection is interior to a chain -- so it is
  paired with Transolver's error on exactly those nodes.

Diagnostic only: no thresholds, and no power to change a registered verdict.

Gate
----
S1  The restricted node counts recorded with the per-case Transolver sums must equal the
    band counts ``bodyfit_bound.json`` scored, case by case and band by band, so both sides
    of every paired ratio sit on the same nodes.

Inputs: results/interpolation/{transolver_percase_bands, bodyfit_bound,
point_space_oracle_ls_n200}.json. Output: results/interpolation/paired_band_stats.json.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

R = os.path.join("results", "interpolation")
BAND = 1  # 0-0.005c, the verdict band


def load(name):
    with open(os.path.join(R, name), encoding="utf-8") as fh:
        return json.load(fh)


def summary(x):
    x = np.asarray(x, np.float64)
    return {"n": int(x.size), "median": float(np.median(x)), "mean": float(np.mean(x)),
            "p05": float(np.percentile(x, 5)), "p95": float(np.percentile(x, 95)),
            "min": float(np.min(x)), "max": float(np.max(x)),
            "n_above_10": int(np.sum(x > 10)), "n_at_or_below_1": int(np.sum(x <= 1)),
            "n_below_1": int(np.sum(x < 1))}


def main() -> int:
    tpc = load("transolver_percase_bands.json")
    bf = load("bodyfit_bound.json")
    n2 = load("point_space_oracle_ls_n200.json")
    seeds = [f"seed{s}" for s in tpc["meta"]["seeds"]]

    T = {r["name"]: r for r in tpc["rows"]}
    BFr = {r["name"]: r for r in bf["rows"]}
    N2 = {r["name"]: r for r in n2["rows"]}
    names = [r["name"] for r in bf["rows"]]
    assert set(names) == set(T) == set(N2), "case sets differ between artifacts"

    # ---- S1: identical nodes on both sides of every restricted ratio -------------
    bad = [nm for nm in names if T[nm]["n_band_restricted"] != BFr[nm]["n_band"]]
    if bad:
        raise SystemExit(f"S1 FAILED: restricted node counts differ on {len(bad)} cases, "
                         f"e.g. {bad[0]}")
    print(f"S1 PASS  restricted node counts match bodyfit_bound.json on all {len(names)} cases")

    out = {"artifact": "paired_band_stats", "band": n2["rows"][0]["bands"][BAND],
           "note": ("per-case paired ratios, interpolation over Transolver on identical nodes; "
                    "diagnostic only, registered verdicts unchanged"),
           "S1": {"status": "PASS", "n_cases": len(names)}, "channels": {}}

    for c in ("u", "p"):
        t_all, t_res = [], []
        for nm in names:
            n_all = T[nm]["n_band"][BAND]
            n_res = T[nm]["n_band_restricted"][BAND]
            t_all.append(np.mean([T[nm]["seeds"][s]["se"][c][BAND] / n_all for s in seeds]))
            t_res.append(np.mean([T[nm]["seeds"][s]["se_restricted"][c][BAND] / n_res
                                  for s in seeds]))
        t_all = np.asarray(t_all)
        t_res = np.asarray(t_res)
        n_all = np.array([T[nm]["n_band"][BAND] for nm in names], float)
        n_res = np.array([T[nm]["n_band_restricted"][BAND] for nm in names], float)

        ls_phys_all = np.array([N2[nm]["channels"][c]["ls_mse"][BAND] for nm in names])
        ls_phys_res = np.array([BFr[nm]["frames"]["physical"][c]["ls_mse"][BAND] for nm in names])
        ls_bf = np.array([BFr[nm]["frames"]["bodyfit"][c]["ls_mse"][BAND] for nm in names])
        krr_phys_res = np.array([BFr[nm]["frames"]["physical"][c]["krr_mse"][BAND] for nm in names])
        krr_bf = np.array([BFr[nm]["frames"]["bodyfit"][c]["krr_mse"][BAND] for nm in names])
        bs_phys_res = np.array([BFr[nm]["frames"]["physical"][c]["best_single_mse"][BAND]
                                for nm in names])
        bs_bf = np.array([BFr[nm]["frames"]["bodyfit"][c]["best_single_mse"][BAND] for nm in names])

        # pooled denominators, node-weighted exactly as the registered constant is
        T_pool_all = float(np.sum(t_all * n_all) / np.sum(n_all))
        T_pool_res = float(np.sum(t_res * n_res) / np.sum(n_res))

        rec = {
            "transolver_pooled_all_nodes": T_pool_all,
            "transolver_pooled_restricted_nodes": T_pool_res,
            "restricted_over_all": T_pool_res / T_pool_all,
            # the registered estimators, with the denominator on the matching node set
            "casemean_ratio": {
                "physical_bound_all_nodes": float(ls_phys_all.mean() / T_pool_all),
                "physical_bound_restricted": float(ls_phys_res.mean() / T_pool_res),
                "bodyfit_bound_restricted": float(ls_bf.mean() / T_pool_res),
                "krr_physical_restricted": float(krr_phys_res.mean() / T_pool_res),
                "krr_bodyfit_restricted": float(krr_bf.mean() / T_pool_res),
                "best_single_physical_restricted": float(bs_phys_res.mean() / T_pool_res),
                "best_single_bodyfit_restricted": float(bs_bf.mean() / T_pool_res),
            },
            # the paired per-case distributions
            "paired": {
                "physical_bound_all_nodes": summary(ls_phys_all / t_all),
                "physical_bound_restricted": summary(ls_phys_res / t_res),
                "bodyfit_bound_restricted": summary(ls_bf / t_res),
                "krr_physical_restricted": summary(krr_phys_res / t_res),
                "krr_bodyfit_restricted": summary(krr_bf / t_res),
                "best_single_physical_restricted": summary(bs_phys_res / t_res),
                "best_single_bodyfit_restricted": summary(bs_bf / t_res),
                "bodyfit_over_physical_bound": summary(ls_bf / ls_phys_res),
            },
        }
        out["channels"][c] = rec

        print(f"\n=== channel {c}, band {out['band']} ===")
        print(f"  Transolver pooled: all nodes {T_pool_all:.6g} | restricted {T_pool_res:.6g} "
              f"(ratio {T_pool_res / T_pool_all:.4f})")
        for k, v in rec["casemean_ratio"].items():
            print(f"  case-mean ratio  {k:<28} {v:.5g}")
        for k, v in rec["paired"].items():
            print(f"  paired {k:<30} median {v['median']:.4g}  mean {v['mean']:.4g}  "
                  f"p05 {v['p05']:.3g}  p95 {v['p95']:.3g}  max {v['max']:.3g}  "
                  f">10: {v['n_above_10']}  <=1: {v['n_at_or_below_1']}  <1: {v['n_below_1']}")

    dest = os.path.join(R, "paired_band_stats.json")
    with open(dest, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nwrote {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
