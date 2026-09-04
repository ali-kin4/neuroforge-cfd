"""Why does a lossy round trip make the *exact* field converge better?

Section 5.2.3 reports an asymmetry it could not explain. Sending a seed through
a 256x64 body-fitted grid:

* helps the **exact** converged field a great deal -- ``oracle_bl`` +69.9% ->
  ``or_proj_coarse`` +86.1% on ``Cd_v``@1%, five cases of five;
* does **nothing** to the **network's** field -- ``nf_bl`` +14.6% ->
  ``nf_proj_coarse`` +14.5%.

A general "projection low-pass filters the field" story does not predict that
asymmetry: it should help both. One story does. Every ``*_bl`` arm hands the
solver a field inside the boundary layer and freestream outside, blended over a
ramp. The **exact** field has real near-wall structure -- in particular a large
eddy viscosity -- to be discontinuous *about* at that edge; the network's field
is smooth and wrong and has much less. If that is right, ``oracle_bl`` carries a
step across ``d ~ delta`` that the round trip softens, and the ``nf_*`` pair does
not, and the asymmetry is a **mask-edge** effect rather than a representation
effect.

That would matter twice over. It would replace an interpretation with a
measurement, and it would mean section 5.2.3's "region" contrast
(``oracle_mesh`` -> ``oracle_bl``, -22.5 points) is partly measuring a sharp mask
edge rather than the restriction itself.

Reads the seeds as the solver received them (``0/U``, ``0/p``, ``0/nut``) and
computes, per channel, the profile against wall distance and the size of the
step at the mask edge. **No solver runs.**

Usage
-----
    python scripts/mask_edge_probe.py --root runs/openfoam/placement2
"""

from __future__ import annotations

import argparse
import json
import os
import sys

# Must precede numpy: see the threading note in neuroforge/__init__.py.
import neuroforge  # noqa: F401
import numpy as np

from neuroforge.solver import cgrid as cg, openfoam as of, warmstart as ws

ARMS = ("oracle_mesh", "oracle_bl", "or_proj_coarse", "nf_bl", "nf_proj_coarse",
        "or_proj_fine", "nf_proj_fine")
CHANNELS = ("u", "v", "p", "nut")
N_BINS = 40          # log-spaced bins in wall distance, wall to 4*delta
EDGE_HALFWIDTH = 0.25  # the step is measured over d in [(1-h)*delta, (1+h)*delta]


def read_seed(case_dir: str, n_cells: int):
    """``(u, v, p, nut)`` as written into ``0/``, broadcast if uniform."""
    out = []
    for name, comp in (("U", 0), ("U", 1), ("p", None), ("nut", None)):
        path = os.path.join(case_dir, "0", name)
        if not os.path.isfile(path):
            return None
        raw = of.read_volfield(path)
        raw = np.asarray(raw, dtype=float)
        if raw.ndim == 1:
            raw = raw[:, None]
        col = raw[:, comp] if comp is not None else raw[:, 0]
        if len(col) == 1:
            col = np.broadcast_to(col, (n_cells,))
        elif len(col) != n_cells:
            return None
        out.append(np.asarray(col, dtype=float))
    return tuple(out)


def edge_step(distance: np.ndarray, values: np.ndarray, delta: float) -> float:
    """Relative size of the jump in ``values`` across ``d = delta``.

    The mean just inside the edge against the mean just outside, normalised by
    the field's own scale over the layer. Scale-free, so a channel that is
    simply larger is not called steppier.
    """
    inner = (distance >= (1.0 - EDGE_HALFWIDTH) * delta) & (distance < delta)
    outer = (distance >= delta) & (distance <= (1.0 + EDGE_HALFWIDTH) * delta)
    if inner.sum() < 5 or outer.sum() < 5:
        return float("nan")
    scale = float(np.mean(np.abs(values[distance <= delta])))
    if scale < 1e-30:
        return float("nan")
    return float(abs(np.mean(values[inner]) - np.mean(values[outer])) / scale)


def radial_roughness(distance: np.ndarray, values: np.ndarray,
                     delta: float) -> float:
    """Second difference of the wall-normal profile, over the layer and just past it.

    Bins on wall distance and measures how ragged the binned profile is. A field
    handed over inside the layer and freestream outside has a kink there; a
    field that was smoothed on the way in has less of one.
    """
    lo = max(float(distance.min()), 1e-8)
    edges = np.geomspace(lo, 4.0 * delta, N_BINS + 1)
    idx = np.digitize(distance, edges) - 1
    prof = np.array([values[idx == b].mean() if np.any(idx == b) else np.nan
                     for b in range(N_BINS)])
    prof = prof[np.isfinite(prof)]
    if prof.size < 5:
        return float("nan")
    scale = float(np.mean(np.abs(prof)))
    if scale < 1e-30:
        return float("nan")
    return float(np.sqrt(np.mean(np.diff(prof, n=2) ** 2)) / scale)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default=os.path.join("runs", "openfoam", "placement2"))
    ap.add_argument("--re", type=float, default=3e6)
    ap.add_argument("--out", default=os.path.join("results", "mask_edge.json"))
    args = ap.parse_args(argv)

    spec = cg.CGridSpec()
    delta = ws.bl_thickness(args.re)
    tags = sorted(d[: -len("_cold")] for d in os.listdir(args.root)
                  if d.endswith("_cold"))
    if not tags:
        print(f"no '<case>_cold' directories under {args.root}")
        return 1
    print(f"boundary layer {delta:.4f}; step measured over "
          f"d in [{1 - EDGE_HALFWIDTH:.2f}, {1 + EDGE_HALFWIDTH:.2f}] x delta\n")

    rows = []
    for tag in tags:
        code = tag.split("_")[0]
        cold_dir = os.path.join(args.root, f"{tag}_cold")
        centres = of.read_volfield(os.path.join(cold_dir, "0", "C"))
        inner, nw, ns = cg.inner_curve(code, spec)
        surface = inner[nw - 1: nw + ns - 1]
        distance = ws.wall_distance(centres, surface)

        row = {"case": tag, "arms": {}}
        for arm in ARMS:
            seed = read_seed(os.path.join(args.root, f"{tag}_{arm}"), len(centres))
            if seed is None:
                continue
            row["arms"][arm] = {
                "edge_step": {c: edge_step(distance, seed[i], delta)
                              for i, c in enumerate(CHANNELS)},
                "radial_roughness": {c: radial_roughness(distance, seed[i], delta)
                                     for i, c in enumerate(CHANNELS)},
            }
        rows.append(row)

    def mean_of(arm: str, kind: str, chan: str) -> float:
        vals = [r["arms"][arm][kind][chan] for r in rows
                if arm in r["arms"] and np.isfinite(r["arms"][arm][kind][chan])]
        return float(np.mean(vals)) if vals else float("nan")

    for kind, label in (("edge_step", "step across the mask edge (relative)"),
                        ("radial_roughness", "wall-normal profile roughness")):
        print(f"{label}, mean over {len(rows)} cases")
        print(f"   {'arm':>16} " + "  ".join(f"{c:>9}" for c in CHANNELS))
        for arm in ARMS:
            if not any(arm in r["arms"] for r in rows):
                continue
            print(f"   {arm:>16} "
                  + "  ".join(f"{mean_of(arm, kind, c):9.4f}" for c in CHANNELS))
        print()

    print("the pairs that matter (round trip softens the edge by):")
    for base, proj in (("oracle_bl", "or_proj_coarse"), ("nf_bl", "nf_proj_coarse")):
        if not (any(base in r["arms"] for r in rows)
                and any(proj in r["arms"] for r in rows)):
            continue
        bits = []
        for c in CHANNELS:
            a, b = mean_of(base, "edge_step", c), mean_of(proj, "edge_step", c)
            bits.append(f"{c}: {a:.3f} -> {b:.3f}"
                        + (f" ({100 * (1 - b / a):+.0f}%)" if a > 1e-12 else ""))
        print(f"   {base:>10} -> {proj:<16} " + "   ".join(bits))

    payload = {"root": args.root, "re": args.re, "delta": delta,
               "edge_halfwidth": EDGE_HALFWIDTH, "n_bins": N_BINS, "rows": rows,
               "mean": {arm: {kind: {c: mean_of(arm, kind, c) for c in CHANNELS}
                              for kind in ("edge_step", "radial_roughness")}
                        for arm in ARMS if any(arm in r["arms"] for r in rows)}}
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(payload, fh, indent=2)
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
