"""A warm start is not one thing: it wins on some quantities and loses on others.

Every warm start measured in this repo hands the solver *everything* the surrogate
predicts, and reports one number. Decomposing that number is what this experiment
is built on. From `repr3` -- five cases at Re 3e6, 6000 iterations, oracle control
passing at +88% to +99.9%, `naca4412@3` excluded because its steady solve has no
unique fixed point (its arms land 7% apart in Cd):

    quantity                  cold it    wall-fitted seed
    viscous drag  Cd_v            696             **+37%**
    lift          Cl              945             **+87%**
    pressure drag Cd_p           2391               -150%
    total drag    Cd              810               -141%

The wall-fitted seed **wins on everything it resolves and loses on pressure**,
and pressure drag is by a wide margin the slowest thing in a cold solve. That one
loss is what drags total Cd negative and it is why a single headline number for
"does warm starting work" has no stable answer.

The mechanism is the surrogate's *reach*. A wall-fitted grid clustered on the
boundary layer covers about a chord from the surface; beyond that the seed falls
back to freestream. Boundary layer and surface pressure are therefore right --
hence Cd_v and Cl -- while the global pressure field, which is elliptic and set by
the whole domain including the wake, is replaced by a uniform one.

**What the ablations then measured**, five cases, all 5/5, control passing:

    arm                        Cd_v@1%   Cl@1%   Cd_p@1%
    fitted_256x64  (all of it)   +37.2%  +86.5%   -166.2%
    fitted_outer   (no BL)        -0.5%   +1.7%   -127.2%
    fitted_p       (p only)       +0.1%   +0.1%     +0.2%
    potential      (baseline)     -0.5%   +3.3%    -41.6%

Three things fall out. **All of the surrogate's value is in the boundary layer** --
take it away and every gain goes with it. **None of it is in the pressure
channel** -- handing over the projected pressure alone does exactly nothing. And
**potential flow does not help either**, on any coefficient, so it cannot supply
the global pressure field the surrogate is missing and the composite arm inherits
the full seed's behaviour rather than improving on it.

So the seed is a boundary layer that helps plus an outer band -- from delta out to
the projection's one-chord reach -- that hurts. Keep the first, drop the second.
That is ``fitted_bl``, and it is what the ablations point at rather than what was
guessed in advance.

Arms per case, same mesh, same budget, all from the same wall-fitted 256x64
projection so they differ in what they hand over and never in what they know:

* ``cold``          -- uniform freestream. Baseline. (reused)
* ``oracle_mesh``   -- the converged field at mesh resolution. Control. (reused)
* ``fitted_256x64`` -- the whole wall-fitted seed. The measured trade. (reused)
* ``fitted_p``      -- pressure only; velocity and nuTilda start cold.
* ``fitted_outer``  -- everything *except* the boundary layer.
* ``potential``     -- ``potentialFoam`` alone. The baseline that decides whether
  any of this is practical: no model, no training data, no GPU, seconds to run,
  and what industry already does. NVIDIA's hybrid initialisation blends its
  surrogate *with* potential flow rather than replacing it (arXiv:2503.15766).
* ``composite``     -- potential flow outside, surrogate boundary layer inside.
* ``fitted_bl``     -- **the boundary layer and nothing else.** The arm the
  ablations point at.

`cold`, `oracle_mesh` and `fitted_256x64` are reused from the representation
probe's tree when one is pointed at, so this costs five solves per case, not
eight.

A positive result here is worth more than the representation result on its own:
it is a recipe rather than an observation, it needs no new model, and it applies
to any surrogate anyone already has.

Usage
-----
    python scripts/selective_seed_probe.py --work-dir runs/openfoam/repr3
    python scripts/reanalyse_depth.py --root runs/openfoam/repr3 --per-case
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

CASES = [("naca0012", 4.0), ("naca2412", 2.0), ("naca0015", 6.0),
         ("naca0012", 0.0), ("naca4412", 3.0), ("naca2415", 5.0)]
THRESHOLDS = (1e-3, 1e-4, 1e-5, 5e-6, 1e-6)
FORCE_TOLS = (0.01, 0.005)
NEW_ARMS = ("fitted_p", "fitted_outer", "potential", "composite", "fitted_bl")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--re", type=float, default=3e6)
    ap.add_argument("--n-iter", type=int, default=6000)
    ap.add_argument("--only", action="append", metavar="AIRFOIL@AOA",
                    help="run just this case (repeatable), e.g. naca0012@4")
    ap.add_argument("--n-s", type=int, default=256)
    ap.add_argument("--n-n", type=int, default=64)
    ap.add_argument("--first", type=float, default=2.5e-4)
    ap.add_argument("--ramp", type=float, default=3.0,
                    help="where the prediction takes over, in multiples of delta")
    ap.add_argument("--work-dir", default=os.path.join("runs", "openfoam", "repr3"),
                    help="shares the representation probe's tree, so cold and "
                         "oracle_mesh are reused rather than re-solved")
    ap.add_argument("--out", default=os.path.join("results", "selective_seed.json"))
    ap.add_argument("--timeout", type=float, default=43200.0)
    args = ap.parse_args(argv)

    if of.detect_openfoam() is None:
        print("OpenFOAM not found -- run scripts/openfoam_warm_start.py --check")
        return 1
    warnings.simplefilter("ignore")

    out_path = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    cases = CASES
    if args.only:
        wanted = set()
        for text in args.only:
            code, _, aoa = text.partition("@")
            wanted.add((code.strip(), float(aoa or 0.0)))
        cases = [c for c in CASES if c in wanted]
        if not cases:
            print("not in CASES: " + ", ".join(f"{c}@{a:g}" for c, a in sorted(wanted)))
            return 1
        # One checkpoint per process; several sharing one path would interleave.
        if out_path == os.path.abspath(ap.get_default("out")):
            stem, ext = os.path.splitext(out_path)
            out_path = f"{stem}_" + "_".join(f"{c}{a:g}" for c, a in cases) + ext
        print(f"running only: {', '.join(f'{c}@{a:g}' for c, a in cases)}\n"
              f"checkpointing to {out_path}\n")

    def checkpoint(rows):
        tmp = out_path + ".tmp"
        with open(tmp, "w", encoding="utf-8", newline="\n") as fh:
            json.dump({"re": args.re, "n_iter": args.n_iter, "rows": rows}, fh, indent=2)
        os.replace(tmp, out_path)

    spec = cg.CGridSpec()
    delta = ws.bl_thickness(args.re)
    print(f"Re {args.re:.0e} | boundary layer {delta:.4f} chord | velocity handed "
          f"back to the solver inside it, ramped to the prediction by "
          f"{args.ramp:g} delta\n")

    rows = []
    for code, aoa in cases:
        case = FlowCase.from_airfoil(airfoil=code, aoa=aoa, reynolds=args.re,
                                     u_inf=1.0, resolution=128)
        tag = f"{code}_aoa{aoa:g}"
        print(f"== {tag} ==", flush=True)

        def run(name, **kw):
            return cg.solve_cgrid(case, case_dir=os.path.join(args.work_dir, f"{tag}_{name}"),
                                  spec=spec, n_iter=args.n_iter, timeout=args.timeout, **kw)

        cold = run("cold")
        u_inf, v_inf = of._freestream(case)
        nut_fs = of.NUTILDA_FREESTREAM_RATIO * float(case.fluid.kinematic_viscosity)
        inner, nw, ns = cg.inner_curve(code, spec)
        surface = inner[nw - 1: nw + ns - 1]

        # One projection, used three ways -- the arms differ in what they hand
        # over, never in what they know.
        fitted, rep = ws.clustered_seed(
            (cold.u, cold.v, cold.p, cold.nut), cold.centres, surface,
            n_s=args.n_s, n_n=args.n_n, first=args.first,
            u_inf=u_inf, v_inf=v_inf, nut_freestream=nut_fs)

        p_only, rep_p = ws.masked_seed(fitted, cold.centres, surface, fields=("p",),
                                       u_inf=u_inf, v_inf=v_inf, nut_freestream=nut_fs)
        outer, rep_o = ws.masked_seed(fitted, cold.centres, surface,
                                      free_within=delta, ramp=args.ramp,
                                      u_inf=u_inf, v_inf=v_inf, nut_freestream=nut_fs)
        results = {"fitted_p": run("fitted_p", mesh_initial=p_only),
                   "fitted_outer": run("fitted_outer", mesh_initial=outer)}

        # potentialFoam overwrites the 0/ fields it is run on, so it gets a
        # scratch case of its own and only its result is carried across.
        src = os.path.join(args.work_dir, f"{tag}_potential_src")
        cg.write_cgrid_case(case, src, spec=spec, n_iter=1)
        of.run_openfoam("blockMesh", src, timeout=args.timeout, log_name="log.blockMesh")
        pu, pv, pp = of.potential_flow_seed(src, timeout=args.timeout)
        potential = (pu, pv, pp, np.full_like(pu, nut_fs))
        results["potential"] = run("potential", mesh_initial=potential)

        # The composition the measurement points at: potential flow supplies the
        # global pressure and outer velocity -- free, untrained, and exactly what
        # the surrogate ruins -- and the wall-fitted surrogate supplies the
        # boundary layer, which potential flow does not have at all.
        both, rep_c = ws.masked_seed(potential, cold.centres, surface,
                                     fields=("u", "v", "p"), background=fitted,
                                     free_within=delta, ramp=args.ramp,
                                     u_inf=u_inf, v_inf=v_inf, nut_freestream=nut_fs)
        results["composite"] = run("composite", mesh_initial=both)

        # The arm the two ablations point at. `fitted_outer` -- the surrogate
        # everywhere *except* the boundary layer -- keeps none of the gain
        # (Cd_v +37% -> -0.5%, Cl +87% -> +1.7%) and all of the harm (Cd_p
        # -127%). `fitted_p` -- the surrogate's pressure alone -- does nothing at
        # all (+0.1% on everything). So the whole seed is a boundary layer that
        # helps plus an outer band, from delta out to the projection's one-chord
        # reach, that hurts. Keep the first and drop the second: freestream is
        # the primary field, and the surrogate is the background it falls back to
        # inside delta.
        only_bl, rep_b = ws.masked_seed(
            (np.full_like(pu, u_inf), np.full_like(pu, v_inf),
             np.zeros_like(pu), np.full_like(pu, nut_fs)),
            cold.centres, surface, background=fitted,
            free_within=delta, ramp=args.ramp,
            u_inf=u_inf, v_inf=v_inf, nut_freestream=nut_fs)
        results["fitted_bl"] = run("fitted_bl", mesh_initial=only_bl)

        row = {"case": tag, "re": args.re, "delta": delta,
               "covered_fraction": rep["covered_fraction"],
               "blended_fraction": rep_o["blended_fraction"],
               "seeded_fields": rep_p["fields"],
               "composite_blended": rep_c["blended_fraction"],
               "bl_blended": rep_b["blended_fraction"]}
        for t in THRESHOLDS:
            k = f"{t:.0e}"
            row[f"cold@{k}"] = cold.iterations_to(t)
            for name, res in results.items():
                row[f"{name}@{k}"] = res.iterations_to(t)
        rows.append(row)
        checkpoint(rows)

        for t in THRESHOLDS:
            k = f"{t:.0e}"
            base = row[f"cold@{k}"]
            bits = []
            for name in NEW_ARMS:
                v = row[f"{name}@{k}"]
                s = (1 - v / base) if (base and v) else None
                bits.append(f"{name}={v}" + (f" ({100 * s:+.0f}%)" if s is not None else ""))
            print(f"   @{k}: cold={base}  " + "  ".join(bits), flush=True)

    print("\nper-threshold mean")
    for t in THRESHOLDS:
        k = f"{t:.0e}"
        bits = []
        for name in NEW_ARMS:
            vals = [1 - r[f"{name}@{k}"] / r[f"cold@{k}"] for r in rows
                    if r.get(f"cold@{k}") and r.get(f"{name}@{k}")]
            bits.append(f"{name} {100 * np.mean(vals):+6.1f}% (n={len(vals)})"
                        if vals else f"{name} --")
        print(f"  @{k}: " + "   ".join(bits))
    checkpoint(rows)
    print(f"\nwrote {os.path.relpath(out_path)}")
    print("\nScore both metrics with:\n"
          f"  python scripts/reanalyse_depth.py --root {args.work_dir} --per-case")
    return 0


if __name__ == "__main__":
    sys.exit(main())
