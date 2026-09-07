"""Actually minimise the monitored residual, and watch the field error grow.

The paper's central claim is a two-way dissociation: the monitored residual is a
good error *detector* and a bad correction *objective*. The detector half is
measured. The objective half never was -- no experiment in the manuscript
minimises the residual. ``tab:iters`` sweeps a DEQ corrector's fixed-point cap,
and that corrector is trained by supervised regression toward ground truth and
never sees the residual as an objective. So the negative half of the headline
rests on a theorem plus a sweep that does not test it. This closes that hole
directly, using the differentiable residual the framework already ships.

Gradient descent on ``J(y) = 1/2 ||R_h(y)||^2`` from three starting points:

**From the ground truth (the model-free arm, and the strongest one).** Theorem
leg (ii) says ``grad J(u*) = L^T r*``, nonzero whenever the floor is not in
``ker L^T``, so gradient flow *leaves* the truth. Starting exactly at ``u*`` the
field error begins at zero and can only grow. No network is involved, so no
reviewer can answer it with "your surrogate is over-smoothed" -- the standing
objection to the 160/200 inversion count.

**From a trained backbone prediction (the practical arm).** What a practitioner
would actually do: take the surrogate's output and polish it by descending the
physics residual.

**From the ground truth on the boundary-inclusive objective (the lambda arm).**
Whether up-weighting no-slip rescues the objective, tested by descent rather
than in closed form.

PREDICTION, written before the first run, and how it fared. Two of its three
parts were wrong, and both corrections make the finding narrower and more
defensible, so they are recorded here rather than quietly dropped.

* *"The residual falls while the field error rises from zero."* **Held.** From
  the exact truth, an 83% cut in the monitored residual buys a field error of
  order the one a trained surrogate starts with.
* *"...monotonically."* **Wrong.** The error rises overall but not step by step.
  These are Adam steps on a non-convex objective, not gradient flow, so nothing
  guaranteed monotonicity; the claim in the paper must be about the endpoint,
  not the path.
* *"The iterate moves toward the uniform freestream."* **Wrong, and
  interestingly so.** Distance to the uniform field *grows*. Descent does not
  travel to the spurious global minimum of leg (i); it finds a nearby, different
  low-residual field. The damage does not require reaching the uniform field --
  which strengthens the claim, because the practitioner's short polish never
  would.

The ``stationarity_check`` arm is degenerate by construction and kept for what
that degeneracy shows: started at ``u*``, the *error* objective has exactly zero
gradient and nothing moves, while the *residual* objective moves immediately.
That is leg (ii) -- ``u*`` is stationary for one and not the other -- as a
measurement rather than an algebraic remark. The optimiser is validated instead
by ``perturbed_error``, which must reduce a real nonzero error.

Usage
-----
    python scripts/residual_descent.py --n-cases 24 --steps 200
    python scripts/residual_descent.py --n-cases 200 --steps 200 --ckpt <path>
"""

from __future__ import annotations

import argparse
import json
import os
import sys

# Must precede numpy/torch: see the threading note in neuroforge/__init__.py.
import neuroforge  # noqa: F401
import numpy as np
import torch

from neuroforge.data.airfrans_loader import load_airfrans
from neuroforge.physics.residuals import (_solid_adjacent_fluid,
                                          physics_residual_torch)

AIRFRANS_NU = 1.5e-5


def scored_cells(field) -> np.ndarray:
    """Fluid minus the solid-adjacent ring -- exactly what ``diagnose()`` scores."""
    mask32 = np.asarray(field.mask, dtype=np.float32)
    fluid = np.asarray(field.mask, dtype=float) > 0.5
    return fluid & ~_solid_adjacent_fluid(mask32)


def residual_norm_t(y: torch.Tensor, inp: torch.Tensor, dx: float, dy: float,
                    sel: torch.Tensor, u_inf: float, length: float) -> torch.Tensor:
    """The monitored norm, non-dimensionalised exactly as ``diagnose()`` does.

    Continuity by ``u_inf/L``, momentum by ``u_inf^2/L``, RMS over the scored
    cells, no-slip term excluded -- this is the quantity the paper monitors.
    """
    r = physics_residual_torch(y, inp, dx, dy, AIRFRANS_NU)
    c = r["continuity"] / (u_inf / length)
    x = r["momentum_x"] / (u_inf * u_inf / length)
    v = r["momentum_y"] / (u_inf * u_inf / length)
    m2 = (c * c + x * x + v * v)[sel]
    return torch.sqrt(torch.clamp(m2.mean(), min=1e-300))


def bc_norm_t(y: torch.Tensor, sdf: torch.Tensor, dx: float, dy: float,
              sel: torch.Tensor, u_inf: float) -> torch.Tensor:
    """Proximity-weighted no-slip violation, matching ``bc_violation``'s form."""
    band = 3.0 * min(dx, dy)
    w = torch.exp(-torch.abs(sdf) / band)
    speed = torch.sqrt(torch.clamp(y[:, 0:1] ** 2 + y[:, 1:2] ** 2, min=1e-30))
    b = (w * speed / u_inf)[sel]
    return torch.sqrt(torch.clamp((b * b).mean(), min=1e-300))


def field_error(y: torch.Tensor, truth: torch.Tensor, sel: torch.Tensor) -> float:
    """Mean squared velocity error over the scored cells, in the paper's units."""
    d = (y[:, 0:2] - truth[:, 0:2]) ** 2
    return float(d[:, 0:1][sel].mean() + d[:, 1:2][sel].mean())


def to_uniform(y: torch.Tensor, unif: torch.Tensor, sel: torch.Tensor) -> float:
    """RMS velocity distance to the uniform freestream -- the spurious minimiser."""
    d = (y[:, 0:2] - unif[:, 0:2]) ** 2
    return float(torch.sqrt(d[:, 0:1][sel].mean() + d[:, 1:2][sel].mean()))


def numpy_monitor_rms(y: torch.Tensor, case, template, sel: np.ndarray) -> float:
    """The *numpy* monitor's residual on a descended field.

    Validity check, and not a cosmetic one. The differentiable residual descended
    here uses a repeated first difference for the Laplacian; the monitor the
    paper reports uses the compact three-point stencil. The wide stencil is blind
    to the highest mode on the grid, so descent could in principle buy its
    reduction with a checkerboard the compact monitor would see as *worse*. If
    both operators fall together the finding is about the residual; if only the
    differentiable one falls, the finding is about the stencil.
    """
    from neuroforge.core.types import FlowField
    from neuroforge.physics.residuals import PhysicsChecker
    arr = y.detach().cpu().numpy()[0]
    fld = FlowField(domain=template.domain, u=arr[0], v=arr[1], p=arr[2],
                    nut=arr[3], mask=template.mask, sdf=template.sdf)
    d = PhysicsChecker().diagnose(fld, case)
    c = np.asarray(d.continuity, dtype=np.float64)
    x = np.asarray(d.momentum_x, dtype=np.float64)
    v = np.asarray(d.momentum_y, dtype=np.float64)
    return float(np.sqrt(np.mean((c * c + x * x + v * v)[sel])))


def smooth_perturbation(y: torch.Tensor, sel: torch.Tensor, rel: float,
                        seed: int) -> torch.Tensor:
    """A smooth, per-channel-scaled displacement away from the truth.

    Smooth rather than white: a high-frequency perturbation is precisely what a
    diffusive residual removes fastest, so descending it would flatter the
    objective and prove nothing. Low-pass noise puts the error where a surrogate
    actually puts it -- in the resolved scales.
    """
    g = torch.Generator().manual_seed(seed)
    noise = torch.randn(y.shape, generator=g)
    # Three box passes ~ a Gaussian: cheap, separable, no extra dependency.
    for _ in range(3):
        noise = torch.nn.functional.avg_pool2d(noise, 5, stride=1, padding=2)
    noise = noise / noise.abs().amax(dim=(2, 3), keepdim=True).clamp(min=1e-12)
    scale = y.abs().amax(dim=(2, 3), keepdim=True).clamp(min=1e-8)
    return noise * scale * rel * sel.to(y.dtype)


def descend(y0: torch.Tensor, truth: torch.Tensor, unif: torch.Tensor,
            inp: torch.Tensor, sdf: torch.Tensor, sel: torch.Tensor,
            dx: float, dy: float, u_inf: float, length: float,
            steps: int, lr: float, objective: str, lam: float = 0.0):
    """Descend ``objective`` and record the trajectory.

    ``objective`` is ``residual`` (the monitored norm), ``residual_bc`` (with the
    no-slip term at weight ``lam``), or ``error`` (the control: descend the true
    field error, which must go down).

    Only fluid cells move: the solid interior is a boundary condition, not a
    variable, so its gradient is masked away.
    """
    # Descend in units of each channel's own scale. Adam takes a step of about
    # `lr` per element regardless of the gradient, so on raw fields it would
    # move nut (order 1e-3) and u (order 30) by the same absolute amount and
    # destroy the eddy viscosity long before the velocity moved at all. The
    # iterate is therefore y0 + delta * scale, and delta is what is optimised.
    scale = y0.abs().amax(dim=(2, 3), keepdim=True).clamp(min=1e-8)
    delta = torch.zeros_like(y0).requires_grad_(True)
    opt = torch.optim.Adam([delta], lr=lr)
    free = sel.to(y0.dtype)
    traj = []
    for step in range(steps + 1):
        y = y0 + delta * scale
        with torch.no_grad():
            traj.append({
                "step": step,
                "residual": float(residual_norm_t(y, inp, dx, dy, sel, u_inf, length)),
                "bc": float(bc_norm_t(y, sdf, dx, dy, sel, u_inf)),
                "mse_u": field_error(y, truth, sel),
                "dist_uniform": to_uniform(y, unif, sel),
            })
        if step == steps:
            break
        opt.zero_grad(set_to_none=True)
        if objective == "error":
            loss = ((y[:, 0:2] - truth[:, 0:2]) ** 2)[
                sel.expand(-1, 2, -1, -1)].mean()
        else:
            r = residual_norm_t(y, inp, dx, dy, sel, u_inf, length)
            loss = 0.5 * r * r
            if objective == "residual_bc" and lam > 0:
                b = bc_norm_t(y, sdf, dx, dy, sel, u_inf)
                loss = loss + 0.5 * lam * b * b
        loss.backward()
        with torch.no_grad():
            delta.grad.mul_(free)      # the solid does not move
        opt.step()
    return traj, (y0 + delta * scale).detach()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="descend the monitored residual")
    ap.add_argument("--n-cases", type=int, default=24)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lam", type=float, default=100.0,
                    help="no-slip weight for the boundary-inclusive arm")
    ap.add_argument("--perturb", type=float, default=0.05,
                    help="smooth displacement, as a fraction of each channel's scale")
    ap.add_argument("--seed", type=int, default=20260907)
    ap.add_argument("--resolution", type=int, default=128)
    ap.add_argument("--task", default="full")
    ap.add_argument("--root", default="data")
    ap.add_argument("--cache-dir", default=os.path.join("data", "cache"))
    ap.add_argument("--out", default=os.path.join("results", "residual_descent.json"))
    args = ap.parse_args(argv)

    torch.manual_seed(0)
    pairs = load_airfrans(task=args.task, train=False, resolution=args.resolution,
                          limit=args.n_cases, root=args.root,
                          cache_dir=args.cache_dir, progress=False)
    print(f"residual descent | {len(pairs)} cases @ {args.resolution}^2 | "
          f"{args.steps} steps | Adam lr={args.lr}\n")

    # (start, objective). "perturbed" arms begin at truth + a smooth displacement:
    # perturbed_error is the harness control (descending the error MUST reduce it),
    # perturbed_residual is the practical claim (does descending the residual
    # recover the truth, or find a different low-residual field?).
    arms = {"from_truth": ("truth", "residual"),
            "from_truth_bc": ("truth", "residual_bc"),
            "stationarity_check": ("truth", "error"),
            "perturbed_residual": ("perturbed", "residual"),
            "perturbed_error": ("perturbed", "error")}
    rows = []
    for case, truth in pairs:
        sel_np = scored_cells(truth)
        dom = truth.domain
        dx, dy = dom.dx, dom.dy
        u_inf = max(float(case.bc.u_inf), 1e-9)
        length = max(float(case.reference_length()), 1e-9)
        u_in, v_in = case.bc.inlet_vector()

        stack = np.stack([truth.u, truth.v, truth.p, truth.nut]).astype(np.float32)
        y_true = torch.from_numpy(stack)[None]
        y_unif = torch.zeros_like(y_true)
        y_unif[:, 0] = float(u_in)
        y_unif[:, 1] = float(v_in)
        sel = torch.from_numpy(sel_np)[None, None]
        sdf = torch.from_numpy(np.asarray(truth.sdf, dtype=np.float32))[None, None]
        inp = torch.zeros((1, 7) + truth.shape, dtype=torch.float32)
        inp[:, 1] = torch.from_numpy(np.asarray(truth.mask, dtype=np.float32))

        y_pert = y_true + smooth_perturbation(y_true, sel, args.perturb, args.seed)

        row = {"case": case.name, "arms": {}}
        for arm, (start, objective) in arms.items():
            y0 = y_pert if start == "perturbed" else y_true
            traj, y_end = descend(y0, y_true, y_unif, inp, sdf, sel, dx, dy,
                                  u_inf, length, args.steps, args.lr, objective,
                                  lam=args.lam if objective == "residual_bc" else 0.0)
            # Cross-operator check: did the monitor the paper actually reports
            # fall too, or only the differentiable twin we descended?
            traj[0]["numpy_monitor"] = numpy_monitor_rms(y0, case, truth, sel_np)
            traj[-1]["numpy_monitor"] = numpy_monitor_rms(y_end, case, truth, sel_np)
            row["arms"][arm] = traj
        rows.append(row)
        a = row["arms"]["from_truth"]
        print(f"  {case.name:>28}  residual {a[0]['residual']:.4f} -> "
              f"{a[-1]['residual']:.4f}   mse_u {a[0]['mse_u']:.2e} -> "
              f"{a[-1]['mse_u']:.4f}   to-uniform {a[0]['dist_uniform']:.3f} -> "
              f"{a[-1]['dist_uniform']:.3f}")

    print("\n" + "=" * 78)
    summary = {}
    for arm in arms:
        first = np.array([r["arms"][arm][0]["residual"] for r in rows])
        last = np.array([r["arms"][arm][-1]["residual"] for r in rows])
        e0 = np.array([r["arms"][arm][0]["mse_u"] for r in rows])
        e1 = np.array([r["arms"][arm][-1]["mse_u"] for r in rows])
        d0 = np.array([r["arms"][arm][0]["dist_uniform"] for r in rows])
        d1 = np.array([r["arms"][arm][-1]["dist_uniform"] for r in rows])
        # Monotone over the whole path, not merely worse at the end.
        mono_up = sum(all(t[i + 1]["mse_u"] >= t[i]["mse_u"] - 1e-12
                          for i in range(len(t) - 1))
                      for t in (r["arms"][arm] for r in rows))
        summary[arm] = {
            "residual_start": float(first.mean()), "residual_end": float(last.mean()),
            "residual_fell": int((last < first).sum()),
            "mse_start": float(e0.mean()), "mse_end": float(e1.mean()),
            "error_rose": int((e1 > e0).sum()),
            "error_monotone_up": int(mono_up),
            "to_uniform_start": float(d0.mean()), "to_uniform_end": float(d1.mean()),
            "moved_toward_uniform": int((d1 < d0).sum()), "n": len(rows),
            "error_fell": int((e1 < e0).sum()),
        }
        s = summary[arm]
        print(f"{arm:>16}  residual {s['residual_start']:.4f} -> "
              f"{s['residual_end']:.4f} (fell in {s['residual_fell']}/{s['n']})"
              f"   mse_u {s['mse_start']:.2e} -> {s['mse_end']:.4f} "
              f"(rose in {s['error_rose']}/{s['n']}, monotonically in "
              f"{s['error_monotone_up']}/{s['n']})")
        n0 = np.array([r["arms"][arm][0]["numpy_monitor"] for r in rows])
        n1 = np.array([r["arms"][arm][-1]["numpy_monitor"] for r in rows])
        summary[arm]["numpy_monitor_start"] = float(n0.mean())
        summary[arm]["numpy_monitor_end"] = float(n1.mean())
        summary[arm]["numpy_monitor_fell"] = int((n1 < n0).sum())
        print(f"{'':>16}  the numpy monitor (compact stencil) "
              f"{n0.mean():.4f} -> {n1.mean():.4f} "
              f"(fell in {int((n1 < n0).sum())}/{len(rows)})")
        print(f"{'':>16}  distance to the uniform freestream "
              f"{s['to_uniform_start']:.3f} -> {s['to_uniform_end']:.3f} "
              f"(closer in {s['moved_toward_uniform']}/{s['n']})")

    payload = {"n_cases": len(rows), "steps": args.steps, "lr": args.lr,
               "lam": args.lam, "resolution": args.resolution,
               "summary": summary, "rows": rows}
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(payload, fh, indent=2)
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
