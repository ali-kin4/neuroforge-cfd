"""Is the eddy-viscosity channel actually unlearned, and does nu_eff lean on nu?

Both claims in the Limitations bullet were read off the raw per-channel MSE
(``mse_nut`` ~ 5e-8), which is *dimensional*. nu_t is physically tiny -- three to four
orders of magnitude below u -- so its MSE is tiny for reasons that have nothing to do
with how well it is predicted. The meaningful quantities are

    R^2 = 1 - mse_nut / Var[nu_t]

with Var taken over the **same fluid mask** ``physics/evaluation.py::per_channel_mse``
uses, and the share of nu_eff = nu + nu_t that the turbulent term actually supplies.

Writes ``results/control/nut_learned.json``.

    python scripts/check_nut_learned.py
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import platform
from pathlib import Path

import neuroforge  # noqa: F401  -- caps BLAS threads before numpy import
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CACHE = ROOT / "data" / "cache" / "airfrans_full_test_r128_n200.pkl"
DEFAULT_OUT = ROOT / "results" / "control" / "nut_learned.json"

# Per-arm mse_nut as committed in the results tree (grid fields, fluid-masked).
ARMS = {
    "grid_backbone_seed0": ("results/baselines/seed0.json", ("metrics", "mse_nut")),
    "meshgraphnet_seed0": ("results/mgn/mgn_results.json", ("per_seed", 0, "metrics", "mse_nut")),
    "meshgraphnet_seed1": ("results/mgn/mgn_results.json", ("per_seed", 1, "metrics", "mse_nut")),
    "meshgraphnet_seed2": ("results/mgn/mgn_results.json", ("per_seed", 2, "metrics", "mse_nut")),
}


def _dig(obj, path):
    for k in path:
        obj = obj[k]
    return obj


def fluid_stats(cache: Path) -> dict:
    """Mean/std/var of nu_t over fluid cells, matching per_channel_mse's mask."""
    with open(cache, "rb") as fh:
        items = pickle.load(fh)

    vals, fluid_frac = [], []
    for case, field in items:
        nut = np.asarray(field.nut, np.float64)
        solid = getattr(field, "mask", None)
        if solid is None:
            m = np.ones_like(nut, dtype=bool)
        else:
            solid = np.asarray(solid)
            # the mask convention is whichever label covers most of the domain = fluid
            m = solid > 0.5 if solid.mean() > 0.5 else solid < 0.5
        vals.append(nut[m])
        fluid_frac.append(float(m.mean()))

    v = np.concatenate(vals)
    return {
        "n_cases": len(items),
        "fluid_fraction": float(np.mean(fluid_frac)),
        "mean": float(v.mean()),
        "std": float(v.std()),
        "var": float(v.var()),
        "min": float(v.min()),
        "max": float(v.max()),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    ap.add_argument("--nu", type=float, default=1.56e-05,
                    help="laminar kinematic viscosity of the AirfRANS cases")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    a = ap.parse_args()

    if not a.cache.exists():
        raise SystemExit(
            f"cache not found: {a.cache}\n"
            "Rasterise the AirfRANS test split first (see docs/REPRODUCE.md)."
        )

    stats = fluid_stats(a.cache)
    var = stats["var"]

    arms = {}
    for name, (rel, path) in ARMS.items():
        f = ROOT / rel
        if not f.exists():
            continue
        try:
            mse = float(_dig(json.loads(f.read_text(encoding="utf-8")), path))
        except (KeyError, IndexError, TypeError):
            continue
        arms[name] = {
            "mse_nut": mse,
            "r2": 1.0 - mse / var,
            "rmse": mse ** 0.5,
            "rmse_frac_of_std": mse ** 0.5 / stats["std"],
            "source": rel,
        }

    turb_share = stats["mean"] / (a.nu + stats["mean"])
    result = {
        "meta": {
            "script": "scripts/check_nut_learned.py",
            "cache": str(a.cache.relative_to(ROOT)).replace("\\", "/"),
            "platform": platform.platform(),
            "nu_laminar": a.nu,
        },
        "nut_fluid_stats": stats,
        "nu_eff_composition": {
            "nu_laminar": a.nu,
            "mean_nut": stats["mean"],
            "ratio_nut_over_nu": stats["mean"] / a.nu,
            "turbulent_share_of_nu_eff": turb_share,
        },
        "arms": arms,
        "verdict": {
            "channel_unlearned": bool(arms) and max(v["r2"] for v in arms.values()) < 0.5,
            "nu_eff_dominated_by_laminar": turb_share < 0.5,
        },
    }

    a.out.parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(result, fh, indent=2)
        fh.write("\n")

    print(f"nu_t over fluid cells: mean={stats['mean']:.4g} std={stats['std']:.4g} "
          f"({stats['n_cases']} cases, {100*stats['fluid_fraction']:.1f}% fluid)")
    print(f"nu_t / nu = {stats['mean']/a.nu:.1f}x  ->  turbulent share of nu_eff "
          f"= {100*turb_share:.1f}%")
    for name, d in sorted(arms.items(), key=lambda kv: -kv[1]["r2"]):
        print(f"  {name:<24} mse={d['mse_nut']:.4g}  R2={d['r2']:.4f}  "
              f"rmse={100*d['rmse_frac_of_std']:.1f}% of std")
    print(f"\nchannel unlearned: {result['verdict']['channel_unlearned']}   "
          f"nu_eff laminar-dominated: {result['verdict']['nu_eff_dominated_by_laminar']}")
    print(f"wrote {a.out.relative_to(ROOT)}")
    os.sys.exit(0)


if __name__ == "__main__":
    main()
