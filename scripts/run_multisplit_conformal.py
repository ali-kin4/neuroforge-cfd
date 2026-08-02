"""Multi-split resampling study of the split-conformal calibration numbers.

Why this exists
---------------
The paper's conformal headline (backbone: q=2.352, coverage=0.9155, ECE~0.074,
ch=2/pressure, alpha=0.1, 100 cal / 100 test) and the W2 corrected-field
numbers were computed on a SINGLE calibration/test split (seed_split=0). This
study replaces the "single split" caveat with mean +/- std over N random
splits of the same 200-case pool.

Pure CPU + IO by construction
-----------------------------
ALL model outputs are read from the disk caches produced by
``scripts/run_w2_conformal_corrected.py``:

* ``<w2-cache>/ensemble/<case>.npz``          — 5-member ensemble mean / std / mask
* ``<w2-cache>/corrected/<seed>/<case>.npz``  — DEQ-corrected ensemble-mean field

No model forwards, no training, no checkpoints touched. Ground truth comes
from the (also disk-cached) AirfRANS grid loader.

Faithfulness
------------
``_split`` and ``conformal_one`` are IMPORTED from
``run_w2_conformal_corrected`` (not copied), so every split runs bit-for-bit
the published code path: sigma is the FROZEN physical ensemble std straight
from the npz (no std_out rescale), the mask is ``gt.mask``, and q is refit per
split (per arm) with the finite-sample-corrected conformal quantile. A hard
GATE checks that split seed 0 / channel 2 / backbone reproduces the published
q and coverage to 1e-6 BEFORE any results are written; on failure the script
aborts with the discrepancy.

Arms, per split seed s in 0..N-1 and per channel:
* backbone        — conformal on ensemble-mean-field errors, ensemble sigma.
* corrected_seedK — conformal on corrector-K field errors, SAME frozen sigma.

Usage (defaults mirror the pre-registered W2 full run):
    .venv/Scripts/python.exe scripts/run_multisplit_conformal.py \
        --n-splits 20 --channels 0 1 2 --alpha 0.1 \
        --root data --cache-dir data/cache --w2-cache-dir data/cache/w2 \
        --out-dir results/uq_ensemble
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

import neuroforge  # noqa: F401  -- MUST precede numpy/torch heavy work (thread caps)

# Import the published code path (the module is __main__-guarded, so this is a
# plain import; it lives next to this file).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_w2_conformal_corrected import (  # noqa: E402
    _CHANNEL_NAMES,
    _PUB_COVERAGE,
    _PUB_Q,
    _split,
    conformal_one,
)

from neuroforge.data.airfrans_loader import load_airfrans  # noqa: E402

# Multi-split gate: identical code path -> exact reproduction expected.
_GATE_TOL_Q = 1e-6
_GATE_TOL_COV = 1e-6


def log(msg: str) -> None:
    print(f"[multisplit] {msg}", flush=True)


def _load_ensemble_cache(ens_dir: str, names: list[str]) -> dict:
    """Load the corrector-independent ensemble cache written by the W2 run.

    Mirrors the cache-hit branch of ``ensemble_forward_cached`` (provenance:
    scripts/run_w2_conformal_corrected.py) — but NEVER computes: a missing case
    is a hard error, because this study must not run model forwards.
    """
    out = {}
    for name in names:
        npz = os.path.join(ens_dir, f"{name}.npz")
        done = npz + ".done"
        if not (os.path.exists(done) and os.path.exists(npz)):
            raise FileNotFoundError(
                f"ensemble cache missing for case '{name}' ({npz}); this study is "
                "pure IO — run scripts/run_w2_conformal_corrected.py first."
            )
        d = np.load(npz)
        out[name] = {"mean": d["mean"], "std": d["std"], "mask": d["mask"]}
    return out


def _load_corrected_cache(corr_dir: str, names: list[str]) -> dict:
    """Load one corrector's corrected-field cache (cache-hit branch of
    ``corrected_forward_cached`` in run_w2_conformal_corrected.py)."""
    out = {}
    for name in names:
        npz = os.path.join(corr_dir, f"{name}.npz")
        done = npz + ".done"
        if not (os.path.exists(done) and os.path.exists(npz)):
            raise FileNotFoundError(
                f"corrected cache missing for case '{name}' ({npz}); this study is "
                "pure IO — run scripts/run_w2_conformal_corrected.py first."
            )
        out[name] = np.load(npz)["corrected"]
    return out


def _aggregate(records: list[dict]) -> dict:
    """mean / std (ddof=0 and ddof=1) over splits for q, coverage, ece."""
    agg = {"n_splits": len(records)}
    for key in ("q", "coverage", "ece"):
        vals = np.asarray([r[key] for r in records], np.float64)
        agg[key] = {
            "mean": float(vals.mean()),
            "std_ddof0": float(vals.std(ddof=0)),
            "std_ddof1": float(vals.std(ddof=1)) if vals.size > 1 else float("nan"),
            "min": float(vals.min()),
            "max": float(vals.max()),
        }
    return agg


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description="Multi-split resampling of the split-conformal calibration "
        "numbers (backbone + corrected arms) over the cached 200-case pool."
    )
    p.add_argument("--n-splits", type=int, default=20)
    p.add_argument("--channels", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--alpha", type=float, default=0.1)
    p.add_argument("--correctors", nargs="+", default=["seed0", "seed1", "seed2"])
    # data / cache args (mirror run_w2_conformal_corrected defaults)
    p.add_argument("--task", default="full")
    p.add_argument("--n-val", type=int, default=200, help="cal+test pool (split 50/50)")
    p.add_argument("--resolution", type=int, default=128)
    p.add_argument("--root", default="data")
    p.add_argument("--cache-dir", default="data/cache")
    p.add_argument("--w2-cache-dir", default="data/cache/w2")
    p.add_argument("--out-dir", default="results/uq_ensemble")
    a = p.parse_args(argv)

    t_start = time.time()
    os.makedirs(a.out_dir, exist_ok=True)

    # ---- ground truth (disk-cached AirfRANS grid pairs; same loader/order as W2) ----
    log(f"loading GT grid pairs (task={a.task}, n_val={a.n_val}, res={a.resolution}) ...")
    test_pairs = load_airfrans(
        root=a.root, task=a.task, train=False, resolution=a.resolution,
        limit=a.n_val, cache_dir=a.cache_dir, download=False, progress=False,
    )
    names = [case.name for case, _gt in test_pairs]
    log(f"loaded {len(test_pairs)} GT pairs")

    # ---- cached model outputs (pure IO) ----
    ens_dir = os.path.join(a.w2_cache_dir, "ensemble")
    ens_cache = _load_ensemble_cache(ens_dir, names)
    backbone_lookup = {n: ens_cache[n]["mean"] for n in ens_cache}
    log(f"loaded ensemble cache: {len(ens_cache)} cases from {ens_dir}")

    corr_lookups = {}
    for cid in a.correctors:
        corr_dir = os.path.join(a.w2_cache_dir, "corrected", cid)
        corr_lookups[cid] = _load_corrected_cache(corr_dir, names)
        log(f"loaded corrected cache [{cid}]: {len(corr_lookups[cid])} cases")

    arms = ["backbone"] + [f"corrected_{cid}" for cid in a.correctors]
    lookups = {"backbone": backbone_lookup}
    lookups.update({f"corrected_{cid}": corr_lookups[cid] for cid in a.correctors})

    # ---- GATE: split seed 0 / channel 2 / backbone must reproduce published ----
    cal0, test0 = _split(test_pairs, 0)
    gate_res = conformal_one(cal0, test0, backbone_lookup, ens_cache,
                             channel=2, alpha=a.alpha)
    dq = abs(gate_res["q"] - _PUB_Q)
    dcov = abs(gate_res["coverage"] - _PUB_COVERAGE)
    gate_ok = gate_res.get("status") == "ok" and dq <= _GATE_TOL_Q and dcov <= _GATE_TOL_COV
    log(f"GATE (split 0, ch 2, backbone): q={gate_res['q']!r} "
        f"(pub {_PUB_Q!r}, |dq|={dq:.3e}), coverage={gate_res['coverage']!r} "
        f"(pub {_PUB_COVERAGE!r}, |dcov|={dcov:.3e}) -> "
        f"{'PASS' if gate_ok else 'FAIL'}")
    if not gate_ok:
        log("GATE FAILED — same code path was expected to reproduce the published "
            "numbers exactly. NOT writing results. Investigate before rerunning.")
        return 1

    # ---- the study: n_splits x channels x arms, q REFIT per split per arm ----
    per_channel: dict[str, dict[str, list[dict]]] = {
        _CHANNEL_NAMES.get(ch, str(ch)): {arm: [] for arm in arms} for ch in a.channels
    }
    for s in range(a.n_splits):
        t0 = time.time()
        cal, test = _split(test_pairs, s)
        for ch in a.channels:
            cname = _CHANNEL_NAMES.get(ch, str(ch))
            for arm in arms:
                r = conformal_one(cal, test, lookups[arm], ens_cache, ch, a.alpha)
                if r.get("status") != "ok":
                    raise RuntimeError(f"conformal_one failed: split={s} ch={ch} "
                                       f"arm={arm}: {r}")
                per_channel[cname][arm].append({
                    "split_seed": s,
                    "q": r["q"],
                    "coverage": r["coverage"],
                    "ece": r["ece"],
                })
        log(f"split {s}: {len(cal)} cal / {len(test)} test done "
            f"({time.time() - t0:.1f}s)")

    aggregates = {
        cname: {arm: _aggregate(recs) for arm, recs in arms_d.items()}
        for cname, arms_d in per_channel.items()
    }

    # ---- persist JSON ----
    out = {
        "meta": {
            "experiment": "multi-split resampling of split-conformal calibration",
            "n_splits": a.n_splits,
            "protocol": (
                f"For each split seed s in 0..{a.n_splits - 1}: the {len(test_pairs)}-case "
                "AirfRANS test pool is shuffled with np.random.default_rng(s)."
                "permutation and split first-half calibration / second-half test "
                "(identical to the published seed_split=0 protocol). Per channel "
                "and per arm, the conformal multiplier q is REFIT on the "
                "calibration scores (finite-sample-corrected quantile, alpha="
                f"{a.alpha}) and coverage/ECE are measured on the held-out half. "
                "Nonconformity score = |field - truth| / sigma per fluid cell "
                "(gt.mask), sigma = FROZEN physical 5-member ensemble std from "
                "the W2 cache (no std_out rescale) for ALL arms. Arms: backbone "
                "= ensemble-mean field; corrected_seedK = DEQ-corrected field "
                "from corrector seed K applied to the ensemble mean. Pure "
                "CPU+IO: all fields read from data/cache/w2 (no model forwards). "
                "Code path: _split/conformal_one imported from "
                "scripts/run_w2_conformal_corrected.py."
            ),
            "alpha": a.alpha,
            "target_coverage": 1.0 - a.alpha,
            "channels": a.channels,
            "arms": arms,
            "n_pool": len(test_pairs),
            "resolution": a.resolution,
            "task": a.task,
            "ensemble_cache": ens_dir,
            "corrected_cache": {cid: os.path.join(a.w2_cache_dir, "corrected", cid)
                                for cid in a.correctors},
            "runtime_sec": None,  # filled below
        },
        "gate": {
            "published_q": _PUB_Q,
            "published_coverage": _PUB_COVERAGE,
            "measured": gate_res,
            "tol_q": _GATE_TOL_Q,
            "tol_coverage": _GATE_TOL_COV,
            "abs_dq": dq,
            "abs_dcov": dcov,
            "passed": True,  # hard-gated above; we only get here on pass
        },
        "per_channel": per_channel,
        "aggregates": aggregates,
    }
    out["meta"]["runtime_sec"] = round(time.time() - t_start, 1)
    json_path = os.path.join(a.out_dir, "multisplit_conformal.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    log(f"wrote {json_path}")

    # ---- markdown summary ----
    def fmt(agg_metric: dict, digits: int = 3) -> str:
        return (f"{agg_metric['mean']:.{digits}f} ± "
                f"{agg_metric['std_ddof1']:.{digits}f}")

    lines = [
        "# Multi-split conformal calibration (mean ± std over "
        f"{a.n_splits} random splits)",
        "",
        f"Pool: {len(test_pairs)} AirfRANS test cases, split "
        f"{len(cal0)} cal / {len(test0)} test per seed; alpha={a.alpha} "
        f"(target coverage {1.0 - a.alpha:.2f}). q refit per split per arm; "
        "sigma = frozen 5-member ensemble std for all arms. std is sample std "
        "(ddof=1); ddof=0 values in the JSON.",
        "",
        f"Gate: split seed 0 / ch p / backbone reproduced the published "
        f"q={_PUB_Q:.6f}, coverage={_PUB_COVERAGE:.6f} "
        f"(|dq|={dq:.1e}, |dcov|={dcov:.1e}).",
        "",
        "| channel | arm | q | coverage | ECE |",
        "|---|---|---|---|---|",
    ]
    for ch in a.channels:
        cname = _CHANNEL_NAMES.get(ch, str(ch))
        for arm in arms:
            agg = aggregates[cname][arm]
            lines.append(
                f"| {cname} | {arm} | {fmt(agg['q'])} | "
                f"{fmt(agg['coverage'])} | {fmt(agg['ece'])} |"
            )
    lines.append("")
    md_path = os.path.join(a.out_dir, "multisplit_conformal.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    log(f"wrote {md_path}")

    # ---- console summary ----
    log("================= SUMMARY =================")
    for ch in a.channels:
        cname = _CHANNEL_NAMES.get(ch, str(ch))
        for arm in arms:
            agg = aggregates[cname][arm]
            log(f"  ch {cname:>3} | {arm:<16} | q={fmt(agg['q'])} | "
                f"cov={fmt(agg['coverage'])} | ece={fmt(agg['ece'])}")
    log(f"total runtime {time.time() - t_start:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
