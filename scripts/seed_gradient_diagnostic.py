"""Why does a 10%-wrong prediction beat a seed built from the exact answer?

The measured fact this exists to explain: on total drag at a 1% band, ``nf_bl``
-- the trained NeuroForge backbone, restricted to the boundary layer, carrying
~10% error in ``u`` and ~110% error in ``nut`` -- converges 34% faster than a
cold start, while ``fitted_bl`` -- built from the *converged field itself*,
degraded only by a 256x64 wall-fitted projection -- is 218% slower. A seed made
from the exact answer loses to a seed made from a wrong one.

The hypothesis, and the reason to test it from disk before buying solves:
**viscous drag is the surface integral of the first-cell velocity gradient**, so
the quantity that decides ``Cd_v`` is not the seed's L2 error but ``du_t/dy`` at
the wall. A projection round-trip (mesh -> 256x64 wall-fitted grid -> mesh,
nearest-neighbour then linear) resamples across a first cell 4e-6 chords high
sitting under stations spaced ~0.01 apart. It cannot preserve that gradient, and
whatever it does to it, it does *unsmoothly* -- adjacent stations pick up
different interpolation error. A network evaluated pointwise at the same cell
centres has no round-trip and is smooth by construction.

This script measures both properties directly on the seeds already on disk:

* ``grad_err`` -- relative L2 error of the first-cell gradient ``u_t / y``
  against the converged field, station by station along the wall. The direct
  proxy for the seed's error in ``Cd_v``.
* ``roughness`` -- the second difference of that gradient along the surface,
  normalised by its own mean. How ragged the seed's wall shear is, independent
  of how wrong it is on average. The converged field sets the scale.
* ``profile_rough`` -- the same, wall-normal, through the first rings of cells.

The two are independent: a seed can be accurate and ragged (the projection's
signature, if the hypothesis holds) or wrong and smooth (the network's). If
``fitted_bl`` beats ``nf_bl`` on ``grad_err`` and loses on ``roughness``, the
smoothness reading is supported and the next move is the smoothed-projection arm
that establishes causation. If it loses on *both*, there is no smoothness story:
the projection simply destroys the gradient, which is a simpler paper and needs
no extra solve.

Reads ``0/U`` (the seed as the solver received it) and the latest time directory
of the cold arm (the converged answer). No solver runs.

Usage
-----
    python scripts/seed_gradient_diagnostic.py --root runs/openfoam/repr3
    python scripts/seed_gradient_diagnostic.py --only naca0012_aoa4 --json out.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys

# Must precede numpy/torch: see the threading note in neuroforge/__init__.py.
import neuroforge  # noqa: F401
import numpy as np
from scipy.spatial import cKDTree

from neuroforge.solver import cgrid as cg, openfoam as of, warmstart as ws

ARMS = ("cold", "oracle_mesh", "cartesian_128", "fitted_256x64",
        "fitted_bl", "nf_mesh", "nf_bl")

# Wall-normal stations, in chords, spanning the first cells to the edge of the
# boundary layer. The first C-grid cell centre sits ~4e-6 off the wall.
N_RINGS = 12


def wall_stations(code: str, spec: cg.CGridSpec) -> tuple[np.ndarray, np.ndarray]:
    """Surface midpoints and their outward unit normals, ``(M, 2)`` each."""
    inner, n_wake, n_surface = cg.inner_curve(code, spec)
    surf = inner[n_wake - 1: n_wake + n_surface - 1]
    mid = 0.5 * (surf[:-1] + surf[1:])
    tangent = surf[1:] - surf[:-1]
    tangent /= np.linalg.norm(tangent, axis=1, keepdims=True) + 1e-30
    normal = np.stack([-tangent[:, 1], tangent[:, 0]], axis=1)
    outward = mid - surf.mean(axis=0)
    flip = np.sign(np.sum(normal * outward, axis=1))
    flip[flip == 0] = 1.0
    return mid, normal * flip[:, None], surf


def tangential_profile(u, v, centres, mid, normal, heights):
    """``|u_t|`` at each ``(station, height)``, sampled at the nearest cell.

    Wall-normal rather than nearest-cell-to-the-surface: the C-grid's cells are
    ~2500x wider than they are tall near the wall, so a plain nearest-neighbour
    query off the surface walks along the wall instead of away from it.
    """
    tree = cKDTree(centres[:, :2])
    tangent = np.stack([-normal[:, 1], normal[:, 0]], axis=1)
    out = np.empty((len(mid), len(heights)))
    for j, h in enumerate(heights):
        pts = mid + h * normal
        _, idx = tree.query(pts)
        vel = np.stack([u[idx], v[idx]], axis=1)
        out[:, j] = np.abs(np.sum(vel * tangent, axis=1))
    return out


def roughness(values: np.ndarray) -> float:
    """Normalised second difference along the first axis.

    Scale-free, so a seed that is uniformly too small is not called rough. What
    it measures is station-to-station raggedness -- the signature a resampling
    round-trip leaves and a smooth field cannot.
    """
    v = np.asarray(values, dtype=float)
    scale = float(np.mean(np.abs(v)))
    if scale < 1e-30 or v.shape[0] < 3:
        return float("nan")
    return float(np.sqrt(np.mean(np.diff(v, n=2, axis=0) ** 2)) / scale)


def relative_l2(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(b))
    return float(np.linalg.norm(a - b) / denom) if denom > 0 else float("nan")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default=os.path.join("runs", "openfoam", "repr3"))
    ap.add_argument("--only", action="append", metavar="CASE_TAG")
    ap.add_argument("--arms", nargs="*", default=list(ARMS))
    ap.add_argument("--re", type=float, default=3e6)
    ap.add_argument("--json", default=os.path.join("results", "seed_gradient.json"))
    args = ap.parse_args(argv)

    spec = cg.CGridSpec()
    delta = ws.bl_thickness(args.re)
    heights = np.geomspace(4e-6, delta, N_RINGS)

    tags = sorted({d.rsplit("_", 1)[0] for d in os.listdir(args.root)
                   if os.path.isdir(os.path.join(args.root, d))
                   and d.rsplit("_", 1)[-1] in {"cold"}} |
                  {d[: -len("_cold")] for d in os.listdir(args.root)
                   if d.endswith("_cold")})
    if args.only:
        tags = [t for t in tags if t in set(args.only)]
    if not tags:
        print(f"no '<case>_cold' directories under {args.root}")
        return 1

    rows = []
    for tag in tags:
        code = tag.split("_")[0]
        cold_dir = os.path.join(args.root, f"{tag}_cold")
        latest = of._latest_time(cold_dir)
        if latest is None:
            print(f"{tag}: cold has no time directory > 0; skipped")
            continue
        centres = of.read_volfield(os.path.join(cold_dir, "0", "C"))
        truth = of.read_volfield(os.path.join(cold_dir, latest, "U"))
        mid, normal, _ = wall_stations(code, spec)

        ref = tangential_profile(truth[:, 0], truth[:, 1], centres, mid, normal, heights)
        ref_grad = ref[:, 0] / heights[0]

        print(f"== {tag} ==  converged wall-gradient roughness "
              f"{roughness(ref_grad):.4f}", flush=True)
        row = {"case": tag, "converged_roughness": roughness(ref_grad),
               "converged_mean_gradient": float(np.mean(ref_grad)), "arms": {}}

        for arm in args.arms:
            seed_u = os.path.join(args.root, f"{tag}_{arm}", "0", "U")
            if not os.path.isfile(seed_u):
                continue
            U0 = of.read_volfield(seed_u)
            # A cold start writes `uniform (ux uy uz)`, which comes back as one
            # row. It is a legitimate seed -- the one every other arm is scored
            # against -- so broadcast it rather than skipping it.
            if len(U0) == 1:
                U0 = np.broadcast_to(U0, (len(centres), U0.shape[1]))
            elif len(U0) != len(centres):
                print(f"   {arm:>14}: seed has {len(U0)} cells, mesh has "
                      f"{len(centres)}; skipped")
                continue
            prof = tangential_profile(U0[:, 0], U0[:, 1], centres, mid, normal, heights)
            grad = prof[:, 0] / heights[0]
            entry = {
                "grad_err": relative_l2(grad, ref_grad),
                "roughness": roughness(grad),
                "profile_rough": float(np.mean([roughness(prof[k]) for k in
                                                range(prof.shape[0])])),
                "field_err_bl": relative_l2(prof, ref),
                "mean_gradient": float(np.mean(grad)),
            }
            row["arms"][arm] = entry
            print(f"   {arm:>14}:  wall-gradient error {100 * entry['grad_err']:7.1f}%"
                  f"   roughness {entry['roughness']:7.4f}"
                  f"   (x{entry['roughness'] / max(row['converged_roughness'], 1e-30):6.1f}"
                  f" converged)"
                  f"   BL velocity error {100 * entry['field_err_bl']:6.1f}%", flush=True)
        rows.append(row)

    if rows:
        summary = {}
        for arm in args.arms:
            got = [r["arms"][arm] for r in rows if arm in r["arms"]]
            if got:
                summary[arm] = {
                    k: float(np.mean([g[k] for g in got]))
                    for k in ("grad_err", "roughness", "profile_rough", "field_err_bl")}
        print(f"\nmean over {len(rows)} case(s)")
        print(f"{'arm':>14}  {'wall-grad err':>14}  {'roughness':>10}  "
              f"{'x converged':>11}  {'BL vel err':>10}")
        base = float(np.mean([r["converged_roughness"] for r in rows]))
        for arm, v in summary.items():
            print(f"{arm:>14}  {100 * v['grad_err']:13.1f}%  {v['roughness']:10.4f}  "
                  f"{v['roughness'] / max(base, 1e-30):11.1f}  "
                  f"{100 * v['field_err_bl']:9.1f}%")
        print(f"{'(converged)':>14}  {'--':>14}  {base:10.4f}  {1.0:11.1f}  {'--':>10}")

        os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
        with open(args.json, "w", encoding="utf-8", newline="\n") as fh:
            json.dump({"root": args.root, "re": args.re,
                       "heights": heights.tolist(), "rows": rows,
                       "summary": summary,
                       "converged_roughness": base}, fh, indent=2)
        print(f"\nwrote {os.path.relpath(args.json)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
