"""CONTROL 3 -- the fixed-step control on the monotone-residual acceptance gate.

The objection
-------------
``results/control/acceptance_gate.json`` reports that the gate admits a step on
99.8% of deployed cases and that the admitted step lowers true rel-L2 error on
89.3% of them by a median 5.8% (``backbone_*`` arms; body.tex:1111-1134).  The
described mechanism is backtracking: the full step is refused because it raises
the residual, and a halved step is admitted instead.  A hostile reviewer asks
the obvious question:

    if the mechanism is "take half a step", then a FIXED damped step with NO
    residual test at all should do the same thing -- in which case the gate's
    contribution is a step-size schedule, not a physics certificate, and
    "certified self-correction" overstates it.

What this measures
------------------
For every arm and every one of the 200 AirfRANS ``full`` test cases we replay
the SAME deployed step ``delta = corrected - raw`` that
``measure_acceptance_gate.py`` gated, but apply it at a grid of FIXED step
sizes with the gate switched off:

    y(s) = y0 + s * delta,      s in {0.25, 0.5, 0.75, 1.0}

and score ``rel_l2_speed(y(s), gt)`` with the byte-for-byte evaluate_cases
formula reused from ``measure_acceptance_gate.py``.  The gate policy is read
from the committed ``acceptance_gate.json`` (accepted -> error at the admitted
step; rejected -> no step, error unchanged), so gate and controls are scored on
one common denominator of 200 cases per arm.

Because the gate's own selling point is the CERTIFICATE and not the accuracy,
we additionally recompute the residual norm at the single decisive competitor
s = 0.5 and report what fraction of ungated fixed-half steps VIOLATE the
monotone-residual guarantee (residual rises).  That is the quantity a fixed
schedule cannot deliver and the gate can.

Zero forward passes: both sides come from committed caches
(``data/cache/w2/{ensemble,corrected/seed*}`` for the ensemble-mean path and
``data/cache/acceptance_gate/seed*`` for the deployed backbone+DEQ path).
CPU-only.

Usage
-----
    PYTHONPATH=src .venv/Scripts/python.exe scripts/control_fixed_step.py
    PYTHONPATH=src .venv/Scripts/python.exe scripts/control_fixed_step.py --limit 8   # smoke
"""

from __future__ import annotations

import argparse
import json
import os
import statistics as stats
import time

import neuroforge  # noqa: F401  -- MUST precede numpy/torch (sets BLAS thread caps)
import numpy as np

from neuroforge.core.config import Config
from neuroforge.core.types import DTYPE, FlowField
from neuroforge.data.airfrans_loader import load_airfrans
from neuroforge.geometry.encode import encode_case
from neuroforge.physics.residuals import PhysicsChecker

# The gate under test, imported from the shipped loop so this cannot drift.
from neuroforge.solver.correction_loop import _EPS  # noqa: F401


def log(msg: str) -> None:
    print(f"[fixed_step] {msg}", flush=True)


def rel_l2_speed(pred: FlowField, gt: FlowField) -> float:
    """EXACT evaluate_cases formula: fluid-masked rel-L2 of speed.

    Copied byte-for-byte from scripts/measure_acceptance_gate.py so the control
    and the gate are scored by one and the same metric.
    """
    mask = gt.mask if gt.mask is not None else pred.mask
    m = np.asarray(mask) > 0.5
    if not m.any():
        m = np.ones(gt.shape, dtype=bool)
    num = float(np.sqrt(np.sum(((pred.speed() - gt.speed())[m]) ** 2)))
    den = float(np.sqrt(np.sum((gt.speed()[m]) ** 2))) + 1e-12
    return num / den


def make_field(arr, case, sdf, mask_geo, tag):
    return FlowField.from_array(
        np.ascontiguousarray(arr, DTYPE), case.domain, mask=mask_geo, sdf=sdf,
        meta={"source": tag, "case": case.name},
    )


def _median(xs):
    xs = [x for x in xs if x is not None and np.isfinite(x)]
    return float(stats.median(xs)) if xs else None


def summarise(rows, steps, gate_key="gate"):
    """Per-arm policy comparison on a COMMON denominator of all cases."""
    n = len(rows)
    out = {"n_cases": n, "policies": {}}

    gate_ch = [r[gate_key]["err_rel_change"] for r in rows]
    out["policies"]["gate"] = {
        "n_improves": int(sum(1 for c in gate_ch if c < 0)),
        "frac_improves": float(sum(1 for c in gate_ch if c < 0) / n),
        "median_err_rel_change": _median(gate_ch),
        "mean_err_rel_change": float(np.mean(gate_ch)),
        "mean_rel_l2_after": float(np.mean([r[gate_key]["rel_l2"] for r in rows])),
    }
    for s in steps:
        k = f"fixed_{s:g}"
        ch = [r["fixed"][k]["err_rel_change"] for r in rows]
        out["policies"][k] = {
            "n_improves": int(sum(1 for c in ch if c < 0)),
            "frac_improves": float(sum(1 for c in ch if c < 0) / n),
            "median_err_rel_change": _median(ch),
            "mean_err_rel_change": float(np.mean(ch)),
            "mean_rel_l2_after": float(np.mean([r["fixed"][k]["rel_l2"] for r in rows])),
        }
    # paired: gate vs each fixed step, on the same cases
    out["paired_vs_gate"] = {}
    for s in steps:
        k = f"fixed_{s:g}"
        d = [r["fixed"][k]["err_rel_change"] - r[gate_key]["err_rel_change"] for r in rows]
        out["paired_vs_gate"][k] = {
            "n_gate_better": int(sum(1 for x in d if x > 0)),
            "n_fixed_better": int(sum(1 for x in d if x < 0)),
            "n_tie": int(sum(1 for x in d if x == 0)),
            "frac_gate_better": float(sum(1 for x in d if x > 0) / n),
            "median_gap_fixed_minus_gate": _median(d),
        }
    # certificate violation at the decisive competitor s = 0.5
    viol = [r["fixed"]["fixed_0.5"].get("residual_violates") for r in rows]
    viol = [v for v in viol if v is not None]
    if viol:
        out["certificate_at_fixed_0.5"] = {
            "n_scored": len(viol),
            "n_residual_increases": int(sum(viol)),
            "frac_residual_increases": float(sum(viol) / len(viol)),
            "note": ("fraction of cases on which an UNGATED fixed half step raises the "
                     "monotone-residual norm above y0 -- the guarantee a fixed schedule "
                     "cannot provide and the gate can."),
        }
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", default="data")
    p.add_argument("--cache-dir", default="data/cache")
    p.add_argument("--w2-cache-dir", default="data/cache/w2")
    p.add_argument("--deployed-cache-dir", default="data/cache/acceptance_gate")
    p.add_argument("--gate-json", default="results/control/acceptance_gate.json")
    p.add_argument("--corrector-seeds", nargs="+", default=["seed0", "seed1", "seed2"])
    p.add_argument("--backbone-seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--steps", type=float, nargs="+", default=[0.25, 0.5, 0.75, 1.0])
    p.add_argument("--resolution", type=int, default=128)
    p.add_argument("--n-val", type=int, default=200)
    p.add_argument("--limit", type=int, default=0, help="smoke: only first N cases")
    p.add_argument("--residual-at", type=float, default=0.5,
                   help="recompute the residual norm at this fixed step (certificate check)")
    p.add_argument("--out", default="results/review/control3_fixed_step.json")
    a = p.parse_args(argv)

    checker = PhysicsChecker(Config().physics)
    gate = json.load(open(a.gate_json))
    gate_pc = {arm: {r["name"]: r for r in rows} for arm, rows in gate["per_case"].items()}

    log(f"loading AirfRANS full/test (res {a.resolution}, limit {a.n_val}) ...")
    pairs = load_airfrans(
        root=a.root, task="full", train=False, resolution=a.resolution,
        limit=a.n_val, cache_dir=a.cache_dir, download=False, progress=False,
    )
    if a.limit:
        pairs = pairs[: a.limit]
    log(f"{len(pairs)} cases")

    # arm -> (kind, directory).  'ensemble' arms need y0 from the shared ensemble
    # cache; 'deployed' arms carry both raw and corrected in one npz.
    arms = {}
    for s in a.corrector_seeds:
        arms[s] = ("ensemble", os.path.join(a.w2_cache_dir, "corrected", s))
    for k in a.backbone_seeds:
        arms[f"backbone_seed{k}"] = ("deployed", os.path.join(a.deployed_cache_dir, f"seed{k}"))

    ens_dir = os.path.join(a.w2_cache_dir, "ensemble")
    enc = {}          # case name -> (sdf, mask_geo)  (encode_case is the slow bit)
    per_case = {arm: [] for arm in arms}
    t0 = time.time()
    n_res = 0

    for arm, (kind, cdir) in arms.items():
        ta = time.time()
        for i, (case, gt) in enumerate(pairs):
            grow = gate_pc.get(arm, {}).get(case.name)
            if grow is None:
                continue
            if kind == "ensemble":
                epath = os.path.join(ens_dir, f"{case.name}.npz")
                cpath = os.path.join(cdir, f"{case.name}.npz")
                if not (os.path.exists(epath) and os.path.exists(cpath)):
                    continue
                y0_arr = np.load(epath)["mean"].astype(DTYPE)
                y1_arr = np.load(cpath)["corrected"].astype(DTYPE)
            else:
                npz = os.path.join(cdir, f"{case.name}.npz")
                if not os.path.exists(npz):
                    continue
                d = np.load(npz)
                y0_arr, y1_arr = d["raw"].astype(DTYPE), d["corrected"].astype(DTYPE)

            if case.name not in enc:
                stack = encode_case(case)
                enc[case.name] = (stack[0].astype(DTYPE), stack[1].astype(DTYPE))
            sdf, mask_geo = enc[case.name]

            y0 = make_field(y0_arr, case, sdf, mask_geo, "y0")
            err0 = rel_l2_speed(y0, gt)
            delta = (y1_arr - y0_arr).astype(DTYPE)

            fixed = {}
            for s in a.steps:
                cand = make_field(y0_arr + DTYPE(s) * delta, case, sdf, mask_geo, "fixed")
                e = rel_l2_speed(cand, gt)
                rec = {"rel_l2": e, "err_rel_change": (e - err0) / max(err0, 1e-30)}
                if abs(s - a.residual_at) < 1e-12:
                    rn = float(checker.diagnose(cand, case).residual_norm())
                    rec["residual_norm"] = rn
                    rec["residual_violates"] = bool(
                        not np.isfinite(rn) or rn > grow["residual_before"] + _EPS)
                    n_res += 1
                fixed[f"fixed_{s:g}"] = rec

            # the gate's own policy, on the SAME 200-case denominator:
            # accepted -> error at the admitted step; rejected -> no step taken.
            if grow["accepted"]:
                g = {"step": grow["accepted_step"],
                     "rel_l2": grow["err_at_accepted_step"],
                     "err_rel_change": grow["err_rel_change_at_accepted_step"]}
            else:
                g = {"step": 0.0, "rel_l2": err0, "err_rel_change": 0.0}

            per_case[arm].append({
                "name": case.name, "err_before": err0,
                "residual_before": grow["residual_before"],
                "gate": g, "fixed": fixed,
            })
            if (i + 1) % 50 == 0:
                log(f"  {arm}: {i + 1}/{len(pairs)} ({(time.time() - ta) / (i + 1):.2f}s/case)")
        log(f"{arm}: {len(per_case[arm])} cases, {time.time() - ta:.0f}s")

    steps = list(a.steps)
    summary = {arm: summarise(rows, steps) for arm, rows in per_case.items() if rows}

    # pooled groups: the paper's headline is the backbone_* (deployed) arms
    pooled = {}
    for grp, names in (("deployed_backbone_DEQ", [f"backbone_seed{k}" for k in a.backbone_seeds]),
                       ("ensemble_mean_path", list(a.corrector_seeds))):
        rows = [r for nm in names for r in per_case.get(nm, [])]
        if rows:
            pooled[grp] = summarise(rows, steps)
            pooled[grp]["arms"] = names

    out = {
        "artifact": "control3_fixed_step",
        "question": ("Would a FIXED damped step with no residual gate match the "
                     "monotone-residual acceptance gate's error reduction? If yes, the "
                     "gate's contribution is the step-size schedule, not the physics test."),
        "meta": {
            "date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "runtime_s": time.time() - t0,
            "device": "cpu (all fields served from committed caches; zero forward passes)",
            "n_cases": len(pairs),
            "steps": steps,
            "residual_recomputed_at": a.residual_at,
            "n_residual_evals": n_res,
            "gate_source": a.gate_json,
            "gate_policy_on_reject": "no step taken; error unchanged (err_rel_change = 0)",
            "metric": "fluid-masked rel-L2 of speed, evaluate_cases formula "
                      "(rel_l2_speed copied from measure_acceptance_gate.py)",
            "arms": {k: v[0] for k, v in arms.items()},
        },
        "summary": summary,
        "pooled": pooled,
        "per_case": per_case,
    }
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w", newline="\n") as f:
        json.dump(out, f, indent=1)
    log(f"wrote {a.out}  ({time.time() - t0:.0f}s)")

    for grp, s in pooled.items():
        log(f"== {grp} (n={s['n_cases']})")
        for k, v in s["policies"].items():
            log(f"   {k:12s} frac_improves={v['frac_improves']:.4f} "
                f"median={v['median_err_rel_change']:+.4f} "
                f"mean_rel_l2={v['mean_rel_l2_after']:.6f}")
        if "certificate_at_fixed_0.5" in s:
            c = s["certificate_at_fixed_0.5"]
            log(f"   ungated s=0.5 breaks monotone residual on "
                f"{c['n_residual_increases']}/{c['n_scored']} = "
                f"{c['frac_residual_increases']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
