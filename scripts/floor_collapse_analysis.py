"""Does the floor depend on (s, h) separately, or only on the ratio s/h?

POST-HOC. This analysis was NOT pre-registered. It is run after
``scripts/floor_cloud_decimation.py`` returned its pre-registered verdict
(REPRESENTATION-PARTIAL: the floor rises with cloud spacing at fixed ``h`` in 16/16
cases on every band at every rung, exponent ``q ~ 0.5``, but far below the
``q = 1 - p_h ~ 1.6-1.8`` that the registered model predicted). It is reported as
exploratory and is labelled as such wherever it is quoted.

What it asks
------------
The registered model (theorem_targets.md Prop A4) was

    floor = ||eps|| * (s/2h)^(beta/2),   ||eps|| ~ s |grad^2 u_c|,

giving ``floor ~ s^(1+beta/2) h^(-beta/2)``, i.e. the s-exponent ``q`` and the
h-exponent ``r := -p_h`` should satisfy ``q - 1 = r``. The decimation run measured
``q ~ 0.5`` and ``r ~ 0.5-0.8``, so ``q - 1 = r`` fails but ``q = r`` may hold. If it
does, the amplitude ``||eps||`` is s-INDEPENDENT and the floor is a function of the
single dimensionless group ``s/h`` -- which is a stronger and simpler statement than
the registered one, and links the s-dependence and the h-dependence to one mechanism.

Data pooled per case and per band
---------------------------------
* ``results/certificates/floor_cloud_decimation.json`` -- 4 decimations x 2 rungs,
  with ``s`` measured per band.
* ``results/certificates/floor_resolution_decomposition.json`` -- the committed D=1
  five-rung ladder. Its extra rungs (181, 362, 512) are added at the D=1 ``s`` of the
  same case and band, which is exact: at D=1 the cloud is not touched, so ``s`` is
  the same object at every rung. This extends the ``s/h`` range from x6 to x12.

Three fits per case per band, compared on adjusted R^2 over the pooled points:
  FULL      log f = a + q log s - r log h                (2 slopes)
  COLLAPSE  log f = a + g log(s/h)                       (1 slope; tests q = r)
  REGISTERED log f = a + log s + (b/2) log(s/h)          (1 slope; tests q - 1 = r)

Usage
-----
    python scripts/floor_collapse_analysis.py
"""

from __future__ import annotations

import argparse
import json
import os
import time

import neuroforge  # noqa: F401
import numpy as np

DEC_JSON = "results/certificates/floor_cloud_decimation.json"
LAD_JSON = "results/certificates/floor_resolution_decomposition.json"
OUT_JSON = "results/certificates/floor_collapse_analysis.json"
BANDS = (0.05, 0.10, 0.25)


def adj_r2(y, yhat, k):
    n = y.size
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    if ss_tot <= 0 or n - k - 1 <= 0:
        return float("nan")
    return 1.0 - (ss_res / (n - k - 1)) / (ss_tot / (n - 1))


def fits(s, h, f):
    """FULL / COLLAPSE / REGISTERED fits on one case-band point cloud."""
    ls, lh, lf = np.log(s), np.log(h), np.log(f)
    out = {}

    A = np.stack([np.ones_like(ls), ls, lh], axis=1)
    c, *_ = np.linalg.lstsq(A, lf, rcond=None)
    out["full"] = {"q": float(c[1]), "r": float(-c[2]),
                   "adj_r2": adj_r2(lf, A @ c, 2)}

    x = ls - lh
    A2 = np.stack([np.ones_like(x), x], axis=1)
    c2, *_ = np.linalg.lstsq(A2, lf, rcond=None)
    out["collapse"] = {"gamma": float(c2[1]), "beta": float(2 * c2[1]),
                       "adj_r2": adj_r2(lf, A2 @ c2, 1)}

    y3 = lf - ls                       # registered: amplitude linear in s
    c3, *_ = np.linalg.lstsq(A2, y3, rcond=None)
    out["registered"] = {"beta_over_2": float(c3[1]),
                         "adj_r2": adj_r2(lf, A2 @ c3 + ls, 1)}

    out["n_points"] = int(f.size)
    out["s_over_h_range"] = float(np.max(s / h) / np.min(s / h))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=OUT_JSON)
    a = ap.parse_args(argv)

    dec = json.load(open(DEC_JSON, encoding="utf-8"))
    lad = json.load(open(LAD_JSON, encoding="utf-8"))
    lad_by = {c["name"]: c for c in lad["per_case"]}

    res = {"artifact": "floor_collapse_analysis",
           "status": "POST-HOC / exploratory -- not pre-registered",
           "question": ("Is the floor a function of the single group s/h (q = r), "
                        "rather than of s and h separately as the registered model "
                        "q - 1 = r assumed?"),
           "inputs": [DEC_JSON, LAD_JSON],
           "by_band": {}, "per_case": {}}

    for b in BANDS:
        key = f"band_{b:g}"
        rows_q, rows_r, rows_g, r2f, r2c, r2r, dq, dqr = [], [], [], [], [], [], [], []
        for c in dec["per_case"]:
            name = c["name"]
            s_d1 = {r["n"]: r[f"s_band_{b:g}"] for r in c["rows"] if r["D"] == 1}
            s_ref = float(np.mean(list(s_d1.values())))
            S = [r[f"s_band_{b:g}"] for r in c["rows"]]
            H = [r["h"] for r in c["rows"]]
            F = [r[key] for r in c["rows"]]
            lc = lad_by.get(name)
            if lc is not None:
                have = {int(r["n"]) for r in c["rows"]}
                for rr in lc["rungs"]:
                    if int(rr["n"]) in have:
                        continue
                    S.append(s_ref)
                    H.append(float(rr["h"]))
                    F.append(float(rr[f"truth_{key}"]))
            S, H, F = np.array(S), np.array(H), np.array(F)
            ok = np.isfinite(S) & np.isfinite(H) & np.isfinite(F) & (S > 0) & (F > 0)
            fi = fits(S[ok], H[ok], F[ok])
            res["per_case"].setdefault(name, {})[key] = fi
            rows_q.append(fi["full"]["q"])
            rows_r.append(fi["full"]["r"])
            rows_g.append(fi["collapse"]["gamma"])
            r2f.append(fi["full"]["adj_r2"])
            r2c.append(fi["collapse"]["adj_r2"])
            r2r.append(fi["registered"]["adj_r2"])
            dq.append(fi["full"]["q"] - fi["full"]["r"])
            dqr.append(fi["full"]["q"] - 1.0 - fi["full"]["r"])
        A = np.array
        res["by_band"][key] = {
            "n_cases": len(rows_q),
            "q_mean": float(A(rows_q).mean()), "q_std": float(A(rows_q).std()),
            "r_mean": float(A(rows_r).mean()), "r_std": float(A(rows_r).std()),
            "gamma_mean": float(A(rows_g).mean()), "gamma_std": float(A(rows_g).std()),
            "beta_mean": float(2 * A(rows_g).mean()),
            "q_minus_r_mean": float(A(dq).mean()), "q_minus_r_std": float(A(dq).std()),
            "q_minus_r_n_within_0.35": int(np.sum(np.abs(A(dq)) <= 0.35)),
            "q_minus_1_minus_r_mean": float(A(dqr).mean()),
            "q_minus_1_minus_r_n_within_0.35": int(np.sum(np.abs(A(dqr)) <= 0.35)),
            "adj_r2_full_mean": float(A(r2f).mean()),
            "adj_r2_collapse_mean": float(A(r2c).mean()),
            "adj_r2_registered_mean": float(A(r2r).mean()),
            "n_cases_collapse_beats_registered": int(np.sum(A(r2c) > A(r2r))),
            "n_points_per_case": int(res["per_case"][dec["per_case"][0]["name"]][key]["n_points"]),
            "s_over_h_range": float(np.mean([res["per_case"][c["name"]][key]["s_over_h_range"]
                                             for c in dec["per_case"]])),
        }

    res["reading"] = (
        "q = r within 0.35 in {c}/16 cases on the primary band and q - 1 = r in "
        "{d}/16, so the one-group s/h collapse is the supported form and the "
        "registered s-linear amplitude is not.".format(
            c=res["by_band"]["band_0.1"]["q_minus_r_n_within_0.35"],
            d=res["by_band"]["band_0.1"]["q_minus_1_minus_r_n_within_0.35"]))
    res["metadata"] = {"date": time.strftime("%Y-%m-%d %H:%M:%S"), "device": "cpu"}

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w", newline="\n", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1, allow_nan=False)
    print(f"wrote {a.out}\n")
    for b in BANDS:
        r = res["by_band"][f"band_{b:g}"]
        print(f"band {b:g}  ({r['n_points_per_case']} pts/case, s/h range "
              f"x{r['s_over_h_range']:.1f})")
        print(f"   FULL      q = {r['q_mean']:+.3f} +- {r['q_std']:.3f}   "
              f"r = {r['r_mean']:+.3f} +- {r['r_std']:.3f}   adjR2 {r['adj_r2_full_mean']:.4f}")
        print(f"   q - r          = {r['q_minus_r_mean']:+.3f} +- {r['q_minus_r_std']:.3f}   "
              f"|.| <= 0.35 in {r['q_minus_r_n_within_0.35']}/{r['n_cases']}")
        print(f"   q - 1 - r      = {r['q_minus_1_minus_r_mean']:+.3f}   "
              f"|.| <= 0.35 in {r['q_minus_1_minus_r_n_within_0.35']}/{r['n_cases']}")
        print(f"   COLLAPSE   gamma = {r['gamma_mean']:+.3f} (beta = {r['beta_mean']:.2f})  "
              f"adjR2 {r['adj_r2_collapse_mean']:.4f}")
        print(f"   REGISTERED                        adjR2 {r['adj_r2_registered_mean']:.4f}   "
              f"collapse wins in {r['n_cases_collapse_beats_registered']}/{r['n_cases']}\n")
    print(res["reading"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
