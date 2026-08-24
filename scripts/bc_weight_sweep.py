"""Does *any* weighting of the no-slip penalty rescue the residual as an objective?

The paper concedes one gap in the residual-floor argument: the monitored operator omits
the no-slip term, and "a no-slip penalty weighted heavily enough to dominate the interior
floor is out of scope and untested". That concession is unnecessary -- the question has a
closed-form answer on data already committed.

``control_bc_inclusive_residual.py`` logs, at each point of the correction sweep, both

    bc_excl = ||R_h||                       (interior residual, no-slip omitted)
    bc_incl = sqrt(||R_h||^2 + mean(bc^2))  (the same with an implicit weight of 1)

so the no-slip term itself is recoverable, ``bc2 = bc_incl^2 - bc_excl^2``, and the
weighted monitor for any lambda >= 0 is

    rho(lambda) = sqrt(bc_excl^2 + lambda * bc2).

Sweeping lambda therefore costs no forward passes at all. Two questions are asked:

  (A) Spurious minimum -- is the uniform freestream field still preferred to the truth?
  (B) Dissociation -- along the correction path, where field error *falls*, does the
      weighted residual still *rise*?

If both hold for every lambda, up-weighting no-slip cannot save the residual as a
correction objective, and the concession can be replaced by a result.

    python scripts/bc_weight_sweep.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import neuroforge  # noqa: F401  -- caps BLAS threads before numpy import
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_IN = ROOT / "results" / "control" / "bc_inclusive_sweep.json"
DEFAULT_OUT = ROOT / "results" / "control" / "bc_weight_sweep.json"


def rho(bc_excl: float, bc2: float, lam: float) -> float:
    """Weighted monitor sqrt(||R_h||^2 + lambda * mean(bc^2))."""
    return float(np.sqrt(max(bc_excl ** 2 + lam * bc2, 0.0)))


def bc2_of(bc_excl: float, bc_incl: float) -> float:
    """Recover the no-slip term from the two logged norms."""
    return max(bc_incl ** 2 - bc_excl ** 2, 0.0)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in-json", type=Path, default=DEFAULT_IN)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--lambdas", type=float, nargs="+",
                    default=[0.0, 1.0, 10.0, 100.0, 1e3, 1e4, 1e6])
    a = ap.parse_args()

    src = json.loads(a.in_json.read_text(encoding="utf-8"))
    sweep = src["check_b_sweep"]
    chk = src["check_a_truth_vs_uniform"]

    # ---- (A) is the uniform field still a spurious minimum, for every lambda? ----
    t_bc2 = bc2_of(chk["truth_bc_excl"], chk["truth_bc_incl"])
    u_bc2 = bc2_of(chk["uniform_bc_excl"], chk["uniform_bc_incl"])
    # rho_u(l)^2 - rho_t(l)^2 = (u_r2 - t_r2) + l*(u_bc2 - t_bc2); both brackets are
    # <= 0 here, so the uniform field stays below the truth for every lambda >= 0.
    a_rows = []
    for lam in a.lambdas:
        ru = rho(chk["uniform_bc_excl"], u_bc2, lam)
        rt = rho(chk["truth_bc_excl"], t_bc2, lam)
        a_rows.append({"lambda": lam, "rho_uniform": ru, "rho_truth": rt,
                       "uniform_below_truth": bool(ru < rt)})
    a_all = all(r["uniform_below_truth"] for r in a_rows)
    # closed form: holds for all lambda iff both the constant and the slope favour uniform
    a_closed = (chk["uniform_bc_excl"] ** 2 <= chk["truth_bc_excl"] ** 2) and (u_bc2 <= t_bc2)

    # ---- (B) does the weighted residual still rise as error falls? ----
    bc2s = [bc2_of(r["bc_excl"], r["bc_incl"]) for r in sweep]
    first, last = 0, len(sweep) - 1
    err_falls = sweep[last]["mse_u"] < sweep[first]["mse_u"]

    b_rows = []
    for lam in a.lambdas:
        r0 = rho(sweep[first]["bc_excl"], bc2s[first], lam)
        r1 = rho(sweep[last]["bc_excl"], bc2s[last], lam)
        b_rows.append({"lambda": lam, "rho_start": r0, "rho_end": r1,
                       "rho_rises": bool(r1 > r0)})
    b_all = all(r["rho_rises"] for r in b_rows)
    # closed form: rho_end^2 - rho_start^2 = d_r2 + lambda*d_bc2 > 0 for all lambda >= 0
    # iff both increments are >= 0 and at least one is > 0.
    d_r2 = sweep[last]["bc_excl"] ** 2 - sweep[first]["bc_excl"] ** 2
    d_bc2 = bc2s[last] - bc2s[first]
    b_closed = (d_r2 >= 0 and d_bc2 >= 0 and (d_r2 + d_bc2) > 0)

    # lambda that would be needed to flip (B), if one exists at all
    lam_flip = None if b_closed else (-d_r2 / d_bc2 if d_bc2 < 0 else None)

    result = {
        "meta": {
            "script": "scripts/bc_weight_sweep.py",
            "source": str(a.in_json.relative_to(ROOT)).replace("\\", "/"),
            "monitor": "rho(lambda) = sqrt(||R_h||^2 + lambda * mean(bc^2))",
            "note": "post-hoc reweighting of already-logged components; no forward passes",
            "n_cases": src["meta"].get("n_cases"),
            "checkpoint": src["meta"].get("checkpoint"),
        },
        "A_spurious_minimum": {
            "truth_bc2": t_bc2, "uniform_bc2": u_bc2,
            "rows": a_rows,
            "holds_for_all_sampled_lambda": a_all,
            "holds_for_all_lambda_closed_form": bool(a_closed),
        },
        "B_dissociation": {
            "mse_u_start": sweep[first]["mse_u"], "mse_u_end": sweep[last]["mse_u"],
            "error_falls": bool(err_falls),
            "delta_interior_r2": d_r2, "delta_bc2": d_bc2,
            "rows": b_rows,
            "holds_for_all_sampled_lambda": b_all,
            "holds_for_all_lambda_closed_form": bool(b_closed),
            "lambda_that_would_flip_it": lam_flip,
        },
        "verdict": ("NO-WEIGHTING-RESCUES-THE-RESIDUAL"
                    if (a_closed and b_closed and err_falls) else "WEIGHTING-MATTERS"),
    }

    a.out.parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(result, fh, indent=2)
        fh.write("\n")

    print("(A) spurious minimum -- uniform vs truth")
    for r in a_rows:
        print(f"    lambda={r['lambda']:>9.4g}  uniform={r['rho_uniform']:.4f}  "
              f"truth={r['rho_truth']:.4f}  uniform<truth: {r['uniform_below_truth']}")
    print(f"    holds for ALL lambda (closed form): {a_closed}\n")

    print(f"(B) dissociation -- error {sweep[first]['mse_u']:.3f} -> "
          f"{sweep[last]['mse_u']:.3f} (falls: {err_falls})")
    for r in b_rows:
        print(f"    lambda={r['lambda']:>9.4g}  rho {r['rho_start']:.4f} -> "
              f"{r['rho_end']:.4f}   rises: {r['rho_rises']}")
    print(f"    d(interior^2)={d_r2:+.5f}  d(bc^2)={d_bc2:+.5f}")
    print(f"    holds for ALL lambda (closed form): {b_closed}\n")
    print("verdict:", result["verdict"])
    print("wrote", a.out.relative_to(ROOT))


if __name__ == "__main__":
    main()
