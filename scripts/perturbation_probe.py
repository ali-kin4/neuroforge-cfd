"""Is it the *representation*, or just any departure from the discrete fixed point?

The objection this exists to answer is the strongest one against section 5.2, and
it is a good one. The oracle seed is the cold run's own converged solution
re-injected: to solver tolerance it is a **fixed point of the discrete
operator**, so it converges in ~50 iterations because its residual is already at
the floor, not because it is physically excellent. Every other arm is a
perturbation of that fixed point, and any perturbation restarts a transient. No
arm in the study isolates *which property* of the perturbation costs the solve --
so "storing the field on a raster is catastrophic" and "moving away from the
discrete fixed point at all is catastrophic" fit the data equally well, and the
second is a much weaker paper.

**The control.** Perturb the exact converged field by a *smooth* field carrying
the **same per-channel L2 norm** as the 128^2 raster round trip's error, and hand
it over mesh-native. Two properties make it a control rather than just another
bad seed:

* it is built from a handful of low-wavenumber Fourier modes, the shortest
  wavelength a third of a chord against a boundary layer 0.0187 chords thick, so
  it carries no structure on the scale the near-wall state lives at; and
* it is **ramped to zero inside the boundary layer**, so the near-wall field the
  solver receives is the converged one exactly. Without that ramp the control
  would be invalid: the whole-field error norm spread over every cell puts a
  velocity perturbation of order 0.1 into a first cell whose true velocity is
  0.03, which would wreck the near-wall state more thoroughly than the raster
  does and isolate nothing.

The probe measures and reports the first-ring gradient error rather than assuming
it, so a reader can check that the near-wall state really was left alone.

Then the two readings make opposite predictions:

* if ``smooth_perturb`` converges like ``cartesian_128`` (near zero saving),
  what section 5.2 measures is **distance from the discrete fixed point**, the
  representation claim does not survive, and the paper must say so;
* if it converges like ``oracle_mesh`` (near the control), then an equal-sized
  error that is *smooth* costs nothing while the same-sized error introduced by a
  raster costs everything, and the representation claim is enormously
  strengthened.

Arms (``cold`` and ``oracle_mesh`` are reused from the corpus tree if present):

* ``cold``            -- the baseline.
* ``oracle_mesh``     -- the converged field. The control.
* ``cartesian_128``   -- that field through a 128^2 raster. The arm being explained.
* ``smooth_perturb``  -- the converged field plus a smooth error of matched norm.
* ``smooth_perturb_x2`` -- the same at twice the norm, so a null is not just a
  perturbation too small to matter.

Usage
-----
    python scripts/perturbation_probe.py --only naca0012@0
    python scripts/reanalyse_depth.py --root runs/openfoam/perturb
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import warnings

# Must precede numpy/torch: see the threading note in neuroforge/__init__.py.
import neuroforge  # noqa: F401
import numpy as np

from neuroforge.core.types import FlowCase
from neuroforge.solver import cgrid as cg, openfoam as of, warmstart as ws

CASES = [("naca0012", 0.0), ("naca0012", 4.0), ("naca0015", 6.0),
         ("naca2412", 2.0), ("naca2415", 5.0)]
THRESHOLDS = (1e-3, 1e-4, 1e-5, 5e-6, 1e-6)
ARMS = ("oracle_mesh", "cartesian_128", "smooth_perturb", "smooth_perturb_x2")

# Fourier modes of the perturbation. The shortest wavelength here is 1/3 chord
# against a boundary layer 0.0187 chords thick, so the perturbation varies by
# ~6% of its amplitude across the layer: smooth on the scale that matters, by
# construction rather than by inspection.
N_MODES, MAX_WAVENUMBER, PERTURB_SEED = 6, 3.0, 20260904


def outer_weight(distance: np.ndarray, delta: float, ramp: float) -> np.ndarray:
    """0 inside the boundary layer, 1 beyond ``ramp * delta``, smooth between.

    A smoothstep rather than a linear ramp, so the weight has a continuous first
    derivative and the perturbation introduces no kink of its own at the layer
    edge -- which would be exactly the kind of structure this control must not
    contain.
    """
    t = np.clip((distance - delta) / max((ramp - 1.0) * delta, 1e-30), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def smooth_field(centres: np.ndarray, rng: np.random.Generator,
                 weight: np.ndarray | None = None) -> np.ndarray:
    """A unit-norm smooth field over the mesh: low-wavenumber modes, random phase.

    ``weight`` masks it away from the wall before normalising, so the returned
    field still has unit norm and still carries nothing near the wall.
    """
    x, y = centres[:, 0], centres[:, 1]
    out = np.zeros(len(centres), dtype=np.float64)
    for _ in range(N_MODES):
        kx, ky = rng.uniform(-MAX_WAVENUMBER, MAX_WAVENUMBER, size=2)
        phase = rng.uniform(0.0, 2 * np.pi)
        out += np.cos(2 * np.pi * (kx * x + ky * y) + phase)
    if weight is not None:
        out = out * weight
    norm = np.linalg.norm(out)
    return out / norm if norm > 0 else out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--re", type=float, default=3e6)
    ap.add_argument("--n-iter", type=int, default=6000)
    ap.add_argument("--only", action="append", metavar="AIRFOIL@AOA")
    ap.add_argument("--resolution", type=int, default=128)
    ap.add_argument("--ramp", type=float, default=3.0,
                    help="the perturbation reaches full strength at ramp*delta; "
                         "it is exactly zero inside the boundary layer")
    ap.add_argument("--work-dir", default=os.path.join("runs", "openfoam", "perturb"))
    ap.add_argument("--out", default=os.path.join("results", "perturbation.json"))
    ap.add_argument("--timeout", type=float, default=43200.0)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--dry-run", action="store_true",
                    help="build the seeds and report what they did to the field, "
                         "without solving -- the check that the control is valid")
    args = ap.parse_args(argv)

    if of.detect_openfoam() is None:
        print("OpenFOAM not found -- run scripts/openfoam_warm_start.py --check")
        return 1
    warnings.simplefilter("ignore")

    cases = CASES
    out_path = os.path.abspath(args.out)
    if args.only:
        wanted = set()
        for text in args.only:
            code, _, aoa = text.partition("@")
            wanted.add((code.strip(), float(aoa or 0.0)))
        cases = [c for c in CASES if c in wanted]
        if not cases:
            print("not in CASES: " + ", ".join(f"{c}@{a:g}" for c, a in sorted(wanted)))
            return 1
        stem, ext = os.path.splitext(out_path)
        out_path = f"{stem}_" + "_".join(f"{c}{a:g}" for c, a in cases) + ext
        print(f"running only: {', '.join(f'{c}@{a:g}' for c, a in cases)}\n"
              f"checkpointing to {out_path}\n")

    tags = {f"{code}_aoa{aoa:g}" for code, aoa in cases}
    busy = [p for p in of.running_solvers(os.path.basename(os.path.normpath(args.work_dir)))
            if any(os.path.basename(p).startswith(t + "_") for t in tags)]
    if busy and not args.force:
        print("already being solved by another process: "
              + ", ".join(os.path.basename(p) for p in busy[:6]))
        return 1

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    spec = cg.CGridSpec()
    delta = ws.bl_thickness(args.re)
    rows = []

    for code, aoa in cases:
        case = FlowCase.from_airfoil(airfoil=code, aoa=aoa, reynolds=args.re,
                                     u_inf=1.0, resolution=args.resolution)
        tag = f"{code}_aoa{aoa:g}"
        print(f"== {tag} ==", flush=True)

        def run(name, **kw):
            return cg.solve_cgrid(case, case_dir=os.path.join(args.work_dir,
                                                              f"{tag}_{name}"),
                                  spec=spec, n_iter=args.n_iter,
                                  timeout=args.timeout, **kw)

        cold = run("cold")
        u_inf, v_inf = of._freestream(case)
        nut_fs = of.NUTILDA_FREESTREAM_RATIO * float(case.fluid.kinematic_viscosity)
        inner, nw, ns = cg.inner_curve(code, spec)
        surface = inner[nw - 1: nw + ns - 1]
        distance = ws.wall_distance(cold.centres, surface)
        truth = (cold.u, cold.v, cold.p, cold.nut)

        cartesian, _ = ws.plain_seed(cold.to_grid(case.domain), case.domain,
                                     cold.centres, u_inf=u_inf, v_inf=v_inf,
                                     nut_freestream=nut_fs)

        # The norm to match, per channel, over the whole field -- exactly the
        # error the raster round trip introduces.
        target_norm = [float(np.linalg.norm(cartesian[i] - truth[i]))
                       for i in range(len(truth))]
        rng = np.random.default_rng(PERTURB_SEED)
        weight = outer_weight(distance, delta, args.ramp)
        basis = [smooth_field(cold.centres, rng, weight) for _ in truth]

        def perturbed(scale: float):
            return tuple(truth[i] + scale * target_norm[i] * basis[i]
                         for i in range(len(truth)))

        seeds = {"oracle_mesh": truth, "cartesian_128": cartesian,
                 "smooth_perturb": perturbed(1.0),
                 "smooth_perturb_x2": perturbed(2.0)}

        # What each seed did to the field, so the solve is interpretable. The
        # near-wall column is the point of the control: the smooth perturbation
        # must leave the first-cell gradient essentially alone while carrying the
        # raster's whole-field error.
        first = float(np.min(distance))
        ring = distance <= 1.5 * first
        in_bl = distance <= delta

        def err(a, b, mask=None):
            m = slice(None) if mask is None else mask
            denom = max(float(np.linalg.norm(b[m])), 1e-30)
            return float(100 * np.linalg.norm(a[m] - b[m]) / denom)

        damage = {}
        for name, seed in seeds.items():
            if name == "oracle_mesh":
                continue
            damage[name] = {
                "whole_field_pct": {f: err(seed[i], truth[i])
                                    for i, f in enumerate(ws.FIELDS)},
                "in_bl_pct": {f: err(seed[i], truth[i], in_bl)
                              for i, f in enumerate(ws.FIELDS)},
                "first_ring_u_pct": err(seed[0], truth[0], ring),
            }
            d = damage[name]
            print(f"   {name:>18}: whole-field u {d['whole_field_pct']['u']:7.2f}%"
                  f"   in BL {d['in_bl_pct']['u']:7.2f}%"
                  f"   first ring {d['first_ring_u_pct']:8.2f}%", flush=True)

        if args.dry_run:
            print("   (dry run: no solves)", flush=True)
            continue
        results = {n: run(n, mesh_initial=s) for n, s in seeds.items()}

        row = {"case": tag, "re": args.re, "first_cell_centre": first,
               "target_norm": target_norm, "damage": damage}
        for t in THRESHOLDS:
            k = f"{t:.0e}"
            row[f"cold@{k}"] = cold.iterations_to(t)
            for name, res in results.items():
                row[f"{name}@{k}"] = res.iterations_to(t)
        rows.append(row)
        with open(out_path, "w", encoding="utf-8", newline="\n") as fh:
            json.dump({"re": args.re, "n_iter": args.n_iter,
                       "n_modes": N_MODES, "max_wavenumber": MAX_WAVENUMBER,
                       "perturb_seed": PERTURB_SEED, "rows": rows}, fh, indent=2)

        for t in THRESHOLDS:
            k = f"{t:.0e}"
            base = row[f"cold@{k}"]
            bits = []
            for name in ARMS:
                v = row[f"{name}@{k}"]
                s = (1 - v / base) if (base and v) else None
                bits.append(f"{name}={v}" + (f" ({100 * s:+.0f}%)" if s is not None else ""))
            print(f"   @{k}: cold={base}  " + "  ".join(bits), flush=True)

    print(f"\nwrote {out_path}")
    print("\nThe forces are what decides this. Score with:\n"
          f"  python scripts/reanalyse_depth.py --root {args.work_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
