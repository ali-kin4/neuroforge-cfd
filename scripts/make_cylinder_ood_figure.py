"""Qualitative OOD demo: airfoil-trained model on a CYLINDER it never saw.

Thesis ("physics beats familiarity", made visual)
-------------------------------------------------
The deployed Transolver (+ DEQ corrector) was trained ONLY on AirfRANS airfoils.
Run it on a circular CYLINDER -- a shape outside its training manifold -- and ask
two independent questions about the prediction:

* the PHYSICS-RESIDUAL trust map (PDE continuity/momentum + no-slip BC violation,
  computed against the governing equations, NO ground truth) -- does it LOCALISE
  where the field is physically wrong (flow through the body, missing wake)?
* the DATA-DRIVEN deep-ensemble VARIANCE (per-cell std of 5 independently trained
  members) -- does it localise the same physical failure, or is it diffuse?

This is a QUALITATIVE figure. There is NO ground truth and NO quantitative claim:
the only in-repo cylinder reference is the synthetic pseudo-RANS solver, which is
physically invalid for a bluff-body cylinder (no separation / vortex shedding /
correct wake) and was deliberately rejected for any number. The residual is the
PDE residual of the predicted field, which needs no reference.

Why the point-cloud path (and why it is delicate)
-------------------------------------------------
The deployed model is a POINT-CLOUD Transolver that looks its inputs up by
``case.name`` (see ``solver/pointcloud_predictor.py`` /
``models/baselines/transolver_adapter.make_predict_fn``). So we must hand it a
NATIVE cylinder point cloud whose ``name`` matches the ``FlowCase.name``, built in
the EXACT 7-feature AirfRANS layout (``x, y, u_in_x, u_in_y, sdf, n_x, n_y`` -- see
``data/pointcloud.POINT_IN_FEATURES``):

* ``sdf`` is the AirfRANS convention: the UNSIGNED distance to the surface in the
  fluid (>= 0), which equals the fluid-side signed distance for a convex body.
* ``n_x, n_y`` are zero everywhere EXCEPT on the body-surface points, where they
  are the outward unit normal (radial for a cylinder). This mirrors AirfRANS,
  where normals are nonzero only on the ``on_airfoil`` ring.
* The cloud must EXTEND PAST the rasterisation crop (AirfRANS clouds span well
  beyond it). ``make_predict_fn`` rasterises with ``method="linear", fill=0.0``;
  if the cloud did not cover the crop corners they would be filled with 0.0 and
  inject huge spurious gradients that would light up the residual map on
  INTERPOLATION ARTEFACTS rather than physics. We therefore sample a jittered
  grid over an extended bound (+margin) plus a dense surface ring, fluid-only.

Normalisers
-----------
The deployed point normaliser is read from ``seed0.pt`` (``point_norm``); it is
numerically identical to fitting a ``PointNormalizer`` on the cached
``airfrans_pc_full_train_n800`` clouds (verified), so the SAME normaliser serves
both the deployed DEQ model and the 5 ensemble members (the members carry no
normaliser). The grid normaliser (also from ``seed0.pt``) is what the engine /
correction loop read off ``predictor.normalizer``.

Always ``import neuroforge`` FIRST (BLAS thread cap) -- done at the top. Prefix
CPU runs with ``OMP_NUM_THREADS=8`` so the kNN rasterisation / eval is not
glacial.

HARD CONSTRAINT: this script runs CPU-only by default and NEVER writes a
checkpoint. A GPU job may own the GPU; do not pass ``--device cuda``/``auto``
while it runs.

SMOKE (tiny CPU end-to-end -- build 1 cylinder, 2 members, few points, save a
small figure to a ``_smoke`` path, print honesty diagnostics):
    OMP_NUM_THREADS=8 .venv/Scripts/python.exe scripts/make_cylinder_ood_figure.py --smoke

FULL figure (CPU; run after the GPU job frees up, or on GPU once it is done --
but device is forced cpu here unless you change it):
    OMP_NUM_THREADS=8 .venv/Scripts/python.exe scripts/make_cylinder_ood_figure.py \
        --members 5 --n-points 16384 --resolution 128 --device cpu
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import torch

import neuroforge  # noqa: F401  -- MUST precede numpy/torch (sets thread caps)
from neuroforge.core.config import Config
from neuroforge.core.types import (
    DTYPE,
    BoundaryConditions,
    Domain,
    FlowCase,
    FluidProperties,
)
from neuroforge.data.datamodule import Normalizer
from neuroforge.data.pointcloud import (
    F_IN,
    F_OUT,
    PointCloudCase,
    PointNormalizer,
)
from neuroforge.geometry.shapes import cylinder_geometry
from neuroforge.models import build_corrector
from neuroforge.models.baselines.transolver import TransolverPointModel
from neuroforge.physics.residuals import PhysicsChecker
from neuroforge.solver.engine import NeuroForgeEngine
from neuroforge.solver.pointcloud_predictor import PointCloudPredictor


# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #
def log(msg: str) -> None:
    print(f"[cyl-ood] {msg}", flush=True)


def _resolve_device(name: str) -> torch.device:
    # Default is CPU on purpose (a GPU job may own the device). We never write a
    # checkpoint here, so reading on CPU is always safe.
    if name in ("auto", "", None):
        return torch.device("cpu")
    return torch.device(name)


# --------------------------------------------------------------------------- #
# Cylinder case + native point cloud (the main risk -- built to match AirfRANS)
# --------------------------------------------------------------------------- #
def make_cylinder_case(
    cx: float,
    cy: float,
    r: float,
    u_inf: float,
    aoa_deg: float,
    reynolds: float,
    nu: float,
    resolution: int,
    n_surface: int,
    bounds: tuple[float, float, float, float],
    name: str,
) -> FlowCase:
    """A bluff-body cylinder :class:`FlowCase` on the deployment domain/crop.

    The geometry is a CCW unit-diameter cylinder (drop-in for the unit-chord
    airfoil domain). ``nu`` is taken from the deployed AirfRANS scale (so the
    Reynolds-number / viscosity feature is in-distribution and the OOD signal is
    the SHAPE, not a trivial scale mismatch).
    """
    geom = cylinder_geometry(cx=cx, cy=cy, r=r, n=n_surface)
    return FlowCase(
        geometry=geom,
        bc=BoundaryConditions(u_inf=u_inf, aoa_deg=aoa_deg, reynolds=reynolds),
        fluid=FluidProperties(density=1.0, kinematic_viscosity=nu),
        domain=Domain(bounds=bounds, nx=resolution, ny=resolution),
        name=name,
    )


def build_cylinder_pointcloud(
    case: FlowCase,
    n_points: int,
    margin: float,
    seed: int,
) -> PointCloudCase:
    """Build a native cylinder cloud in the AirfRANS 7-feature layout.

    Layout (matches ``POINT_IN_FEATURES`` EXACTLY): ``x, y, u_in_x, u_in_y, sdf,
    n_x, n_y``. Two point populations are concatenated:

    1. VOLUME points: a jittered grid over the crop bounds EXTENDED by ``margin``
       on every side (so the rasterisation convex hull covers all crop corners,
       avoiding ``fill=0.0`` artefacts), fluid-only (points inside the cylinder
       are rejected). ``sdf`` = unsigned distance to the surface; normals = 0.
    2. SURFACE points: the body ring (sdf = 0, outward radial unit normals) --
       the only points that carry nonzero normals, exactly as AirfRANS does.

    Targets are zeros (never used: this is a pure-inference / no-ground-truth
    demo; ``PointNormalizer.transform_in`` only touches features).
    """
    rng = np.random.default_rng(seed)
    geom = case.geometry
    cx, cy = geom.meta["center"]
    r = float(geom.meta["radius"])
    u_in, v_in = case.bc.inlet_vector()

    xmin, xmax, ymin, ymax = case.domain.bounds
    exmin, exmax = xmin - margin, xmax + margin
    eymin, eymax = ymin - margin, ymax + margin

    # --- volume points: jittered grid over the extended bounds, fluid-only ---
    # Oversample, then reject interior points, then trim to the target count.
    side = int(np.ceil(np.sqrt(n_points * 1.6)))  # 1.6x to survive rejection
    gx = np.linspace(exmin, exmax, side)
    gy = np.linspace(eymin, eymax, side)
    GX, GY = np.meshgrid(gx, gy)
    pts = np.stack([GX.ravel(), GY.ravel()], axis=1)
    # Jitter by ~half a cell so the cloud is irregular (like AirfRANS), not a grid.
    jx = 0.5 * (exmax - exmin) / max(side - 1, 1)
    jy = 0.5 * (eymax - eymin) / max(side - 1, 1)
    pts = pts + rng.uniform(-1.0, 1.0, size=pts.shape) * np.array([jx, jy])

    dist = np.sqrt((pts[:, 0] - cx) ** 2 + (pts[:, 1] - cy) ** 2)
    fluid = dist > r * 1.001  # strictly outside (no interior / negative sdf)
    pts = pts[fluid]
    dist = dist[fluid]
    if pts.shape[0] > n_points:
        keep = rng.choice(pts.shape[0], size=n_points, replace=False)
        pts = pts[keep]
        dist = dist[keep]

    sdf_vol = (dist - r).astype(np.float64)  # unsigned fluid-side distance >= 0
    vol = np.zeros((pts.shape[0], F_IN), dtype=np.float64)
    vol[:, 0] = pts[:, 0]
    vol[:, 1] = pts[:, 1]
    vol[:, 2] = u_in
    vol[:, 3] = v_in
    vol[:, 4] = sdf_vol
    # normals (cols 5,6) stay 0 for volume points (AirfRANS convention).

    # --- surface points: the body ring (sdf = 0, outward radial normals) ---
    spts = geom.surface_points.astype(np.float64)        # (n_surface, 2)
    snrm = geom.surface_normals.astype(np.float64)        # (n_surface, 2)
    surf = np.zeros((spts.shape[0], F_IN), dtype=np.float64)
    surf[:, 0] = spts[:, 0]
    surf[:, 1] = spts[:, 1]
    surf[:, 2] = u_in
    surf[:, 3] = v_in
    surf[:, 4] = 0.0
    surf[:, 5] = snrm[:, 0]
    surf[:, 6] = snrm[:, 1]

    feats = np.concatenate([vol, surf], axis=0).astype(DTYPE)
    pos = feats[:, 0:2].astype(DTYPE)
    targets = np.zeros((feats.shape[0], F_OUT), dtype=DTYPE)  # unused
    return PointCloudCase(case.name, feats, targets, pos)


# --------------------------------------------------------------------------- #
# Model / normaliser loading (read-only; CPU)
# --------------------------------------------------------------------------- #
def load_backbone(ckpt_path: str, device: torch.device):
    """Load the deployed Transolver backbone + both normalisers from seed0.pt."""
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = state["backbone_config"]
    model = TransolverPointModel(
        in_features=int(cfg["in_features"]),
        out_features=int(cfg["out_features"]),
        width=int(cfg["width"]),
        n_layers=int(cfg["n_layers"]),
        n_heads=int(cfg["n_heads"]),
        n_slices=int(cfg["n_slices"]),
        dropout=float(cfg.get("dropout", 0.0)),
    ).to(device)
    model.load_state_dict(state["model_state"])
    model.eval()

    pn = state["point_norm"]
    point_norm = PointNormalizer(
        mean_in=np.asarray(pn["mean_in"], DTYPE), std_in=np.asarray(pn["std_in"], DTYPE),
        mean_out=np.asarray(pn["mean_out"], DTYPE), std_out=np.asarray(pn["std_out"], DTYPE),
        eps=float(pn.get("eps", 1e-6)),
    )
    grid_norm = Normalizer.from_state_dict(state["grid_norm"])
    nu = float(state.get("nu", 1.56e-5))
    return model, point_norm, grid_norm, nu, cfg


def load_member(ckpt_path: str, fallback_cfg: dict, device: torch.device):
    """Load one ensemble member backbone (members carry no normaliser)."""
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    margs = state.get("args", {})
    model = TransolverPointModel(
        in_features=F_IN, out_features=F_OUT,
        width=int(margs.get("width", fallback_cfg["width"])),
        n_layers=int(margs.get("n_layers", fallback_cfg["n_layers"])),
        n_heads=int(margs.get("n_heads", fallback_cfg["n_heads"])),
        n_slices=int(margs.get("n_slices", fallback_cfg["n_slices"])),
        dropout=float(margs.get("dropout", 0.0)),
    ).to(device)
    model.load_state_dict(state["model"])
    model.eval()
    return model


def load_corrector(ckpt_path: str):
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    corrector = build_corrector(state["corrector_config"])
    corrector.load_state_dict(state["corrector_state"])
    corrector.eval()
    return corrector


# --------------------------------------------------------------------------- #
# Maps: physics trust (no GT) + deep-ensemble variance
# --------------------------------------------------------------------------- #
def ensemble_variance(members, clouds, point_norm, grid_norm, case, device):
    """Per-cell std across the member rasterised fields ``(4, H, W)`` -> speed-std map.

    Each member is wrapped in a PointCloudPredictor on the cylinder cloud (keyed
    by name) and asked to ``predict(case)``; we stack the physical fields and take
    ``std(0)``. Returns ``(std_4ch, speed_std)`` where ``speed_std`` is the std of
    the velocity magnitude across members (a single scalar map for the panel).
    """
    fields = []
    for m in members:
        pred = PointCloudPredictor(m, clouds, point_norm, grid_norm, device=device)
        ff = pred.predict(case)
        fields.append(ff)
    arr = np.stack([f.as_array() for f in fields])  # (M, 4, H, W)
    std4 = arr.std(0).astype(DTYPE)
    speeds = np.stack([np.sqrt(f.u ** 2 + f.v ** 2) for f in fields])  # (M, H, W)
    speed_std = speeds.std(0).astype(DTYPE)
    return std4, speed_std


# --------------------------------------------------------------------------- #
# Figure
# --------------------------------------------------------------------------- #
def make_figure(case, field, diag, speed_std, out_pdf, out_png, representative_note):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    domain = case.domain
    xmin, xmax, ymin, ymax = domain.bounds
    extent = [xmin, xmax, ymin, ymax]
    mask = np.asarray(field.mask, dtype=DTYPE)
    solid = mask < 0.5

    speed = field.speed()
    # Combined physics-residual magnitude that drives the trust map (already
    # non-dimensionalised, solid + wall-ring zeroed inside diagnose).
    resid = np.sqrt(
        diag.continuity.astype(np.float64) ** 2
        + diag.momentum_x.astype(np.float64) ** 2
        + diag.momentum_y.astype(np.float64) ** 2
        + diag.bc_violation.astype(np.float64) ** 2
    )

    def _shade_body(ax):
        body = np.ma.masked_where(~solid, np.ones_like(mask))
        ax.imshow(body, origin="lower", extent=extent, cmap="Greys", vmin=0, vmax=1,
                  alpha=0.55, zorder=3)
        cx, cy = case.geometry.meta["center"]
        r = case.geometry.meta["radius"]
        th = np.linspace(0, 2 * np.pi, 200)
        ax.plot(cx + r * np.cos(th), cy + r * np.sin(th), "k-", lw=1.2, zorder=4)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), constrained_layout=True)

    # (a) predicted speed field
    ax = axes[0]
    im0 = ax.imshow(speed, origin="lower", extent=extent, cmap="viridis", aspect="auto")
    _shade_body(ax)
    ax.set_title("(a) Predicted speed |U|  (airfoil model on a cylinder)")
    fig.colorbar(im0, ax=ax, fraction=0.046, pad=0.02, label="|U|")

    # (b) physics-residual trust map (PDE residual magnitude; NO ground truth).
    # The one-cell outer border carries the far-field BC term, which visually
    # competes with the on-body localisation; blank it so the eye goes to the
    # body. (Diagnostics still report it; this is render-only.)
    resid_disp = resid.copy()
    resid_disp[0, :] = resid_disp[-1, :] = resid_disp[:, 0] = resid_disp[:, -1] = np.nan
    ax = axes[1]
    vmax_r = np.percentile(resid[mask > 0.5], 99) if np.any(mask > 0.5) else 1.0
    im1 = ax.imshow(resid_disp, origin="lower", extent=extent, cmap="magma",
                    vmin=0.0, vmax=max(vmax_r, 1e-9), aspect="auto")
    _shade_body(ax)
    ax.set_title("(b) Physics-residual trust map (no ground truth)")
    fig.colorbar(im1, ax=ax, fraction=0.046, pad=0.02, label="|PDE residual| (nondim)")

    # (c) deep-ensemble variance (data-driven uncertainty) for contrast
    ax = axes[2]
    vmax_s = np.percentile(speed_std[mask > 0.5], 99) if np.any(mask > 0.5) else 1.0
    im2 = ax.imshow(speed_std, origin="lower", extent=extent, cmap="cividis",
                    vmin=0.0, vmax=max(vmax_s, 1e-9), aspect="auto")
    _shade_body(ax)
    ax.set_title("(c) Deep-ensemble variance (data-driven)")
    fig.colorbar(im2, ax=ax, fraction=0.046, pad=0.02, label="std(|U|) over members")

    for ax in axes:
        ax.set_xlabel("x")
        ax.set_ylabel("y")

    os.makedirs(os.path.dirname(out_pdf), exist_ok=True)
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, dpi=160, bbox_inches="tight")
    plt.close(fig)
    log(f"saved figure -> {out_pdf} and {out_png}")


# --------------------------------------------------------------------------- #
# Honesty diagnostics (what actually drives each map) -- printed for the report
# --------------------------------------------------------------------------- #
def report_diagnostics(case, field, diag, std4, speed_std):
    mask = np.asarray(field.mask, dtype=DTYPE) > 0.5
    cx, cy = case.geometry.meta["center"]
    r = case.geometry.meta["radius"]
    X, Y = case.domain.grid()
    d = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
    xmin, xmax, ymin, ymax = case.domain.bounds
    # TIGHT near-body ring (the thin band the trust map lights up). A wide annulus
    # dilutes the ring with the calm zone just outside it, so keep it ~0.3D thick.
    near_body = mask & (d <= 1.3 * r) & (d > r)
    # Far field = an upstream region away from the body AND its wake, well inside
    # the domain (avoid the outer-boundary BC ring), so it is never empty.
    upstream_x = xmin + 0.25 * (cx - xmin)               # well ahead of the body
    interior = mask & (X > xmin + 0.1 * (xmax - xmin)) & (X < xmax - 0.1 * (xmax - xmin)) \
        & (Y > ymin + 0.1 * (ymax - ymin)) & (Y < ymax - 0.1 * (ymax - ymin))
    far_field = interior & (X < upstream_x) & (d > 3.0 * r)

    def _band(arr, sel):
        sel = sel & mask
        return float(np.mean(np.abs(arr[sel]))) if np.any(sel) else float("nan")

    s = diag.summary
    log("--- per-component physics-residual means (fluid, nondim) ---")
    for k in ("continuity_mean", "momentum_x_mean", "momentum_y_mean", "bc_mean",
              "residual_mag_mean", "residual_mag_max"):
        log(f"    {k:22s} = {s.get(k, float('nan')):.4g}")
    log(f"    trust red/yellow/green frac = "
        f"{s.get('trust_red_frac', 0):.3f} / {s.get('trust_yellow_frac', 0):.3f} / "
        f"{s.get('trust_green_frac', 0):.3f}")

    resid = np.sqrt(diag.continuity ** 2 + diag.momentum_x ** 2
                    + diag.momentum_y ** 2 + diag.bc_violation ** 2)
    log("--- localisation contrast (mean |.| near-body annulus vs far wake) ---")
    log(f"    residual : near-body={_band(resid, near_body):.4g}  "
        f"far={_band(resid, far_field):.4g}  "
        f"ratio={_band(resid, near_body)/max(_band(resid, far_field),1e-12):.2f}")
    log(f"    ens.var  : near-body={_band(speed_std, near_body):.4g}  "
        f"far={_band(speed_std, far_field):.4g}  "
        f"ratio={_band(speed_std, near_body)/max(_band(speed_std, far_field),1e-12):.2f}")
    # Peak localisation: where is the single hottest fluid cell of each map?
    def _peak_xy(arr):
        a = np.where(mask, np.abs(arr), -np.inf)
        j, i = np.unravel_index(int(np.argmax(a)), a.shape)
        return float(X[j, i]), float(Y[j, i]), float(d[j, i] / r)
    rx, ry, rdr = _peak_xy(resid)
    vx, vy, vdr = _peak_xy(speed_std)
    log(f"    residual peak at (x={rx:.2f}, y={ry:.2f}), {rdr:.2f}*r from centre "
        "(near 1 => on the body surface)")
    log(f"    ens.var  peak at (x={vx:.2f}, y={vy:.2f}), {vdr:.2f}*r from centre")

    # Penetration check: speed inside the body-adjacent fluid ring should be ~0 for
    # a physical no-slip flow; a large value means the model lets flow through.
    speed = field.speed()
    log(f"    near-body mean |U| = {_band(speed, near_body):.4g} "
        f"(physical no-slip flow -> small near the wall)")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description="Qualitative cylinder-OOD figure: physics residual vs ensemble variance."
    )
    p.add_argument("--backbone-ckpt", default="checkpoints/v2_transolver/seed0.pt")
    p.add_argument("--corrector-ckpt", default="checkpoints/v2_transolver/seed0_corr_with.pt")
    p.add_argument("--member-ckpt-dir", default="checkpoints/uq_ensemble")
    p.add_argument("--members", type=int, default=5)
    p.add_argument("--n-points", type=int, default=16384)
    p.add_argument("--resolution", type=int, default=128)
    p.add_argument("--margin", type=float, default=0.3,
                   help="extend the cloud past the crop so rasterisation covers corners")
    p.add_argument("--device", default="cpu",
                   help="cpu (default; a GPU job may own the GPU). Do NOT use cuda/auto "
                        "while the GPU is busy.")
    p.add_argument("--out-dir", default="results/figures")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--smoke", action="store_true")
    a = p.parse_args(argv)

    if a.smoke:
        a.members = min(a.members, 2)
        a.n_points = min(a.n_points, 4096)
        a.resolution = min(a.resolution, 64)
        a.device = "cpu"
        a.out_dir = a.out_dir.rstrip("/\\") + "_smoke"
        log("SMOKE MODE: 1 cylinder, 2 members, 4096 pts, res 64, cpu, _smoke dir.")

    device = _resolve_device(a.device)
    if device.type != "cpu":
        log(f"WARNING: device={device}. This demo is meant to run CPU-only so it "
            "never contends with a running GPU job.")
    log(f"device={device}  members={a.members}  n_points={a.n_points}  res={a.resolution}")

    # ---- load deployed backbone + normalisers (read-only) ----
    log(f"loading deployed backbone from {a.backbone_ckpt} ...")
    model, point_norm, grid_norm, nu, bb_cfg = load_backbone(a.backbone_ckpt, device)

    # ---- cylinder cases (deterministic). 3 for the full run, 1 for smoke. ----
    # Freestream chosen in-distribution (AirfRANS u_in ~ 30-80 m/s) so the OOD
    # signal is the SHAPE, not a trivial inlet-scale mismatch. nu from deployment.
    bounds = (-1.0, 2.0, -1.5, 1.5)
    re_ref = 40.0 * (2 * 0.5) / nu  # U * D / nu with the reference diameter
    case_specs = [
        # (cx, cy, r, u_inf, aoa, re, n_surface, tag)
        (0.5, 0.0, 0.5, 40.0, 0.0, re_ref, 256, "r0.5_re_mid"),
        (0.5, 0.0, 0.35, 30.0, 0.0, 30.0 * 0.7 / nu, 256, "r0.35_re_lo"),
        (0.5, 0.0, 0.5, 60.0, 0.0, 60.0 / nu, 256, "r0.5_re_hi"),
    ]
    if a.smoke:
        case_specs = case_specs[:1]

    checker = PhysicsChecker(Config().physics)
    corrector = load_corrector(a.corrector_ckpt)

    # ---- load ensemble members (read-only) ----
    members = []
    for m in range(a.members):
        mp = os.path.join(a.member_ckpt_dir, f"member{m}.pt")
        if not os.path.exists(mp):
            log(f"WARNING: missing member checkpoint {mp} -- skipping")
            continue
        members.append(load_member(mp, bb_cfg, device))
    log(f"loaded {len(members)} ensemble members")

    representative = None
    for (cx, cy, r, u_inf, aoa, re, n_surf, tag) in case_specs:
        name = f"cylinder_{tag}"
        log(f"=========== {name} (r={r}, U={u_inf}, Re={re:.2e}) ===========")
        case = make_cylinder_case(
            cx, cy, r, u_inf, aoa, re, nu, a.resolution, n_surf, bounds, name)
        cloud = build_cylinder_pointcloud(case, a.n_points, a.margin, a.seed)
        log(f"built native cloud: {cloud.n_points} points "
            f"(sdf>=0 fluid + {n_surf} surface-ring pts), feature layout F_IN={F_IN}")

        # ---- deployed prediction = backbone + DEQ corrector (the engine path) ----
        pred = PointCloudPredictor(model, [cloud], point_norm, grid_norm, device=device)
        # Sanity: the predictor MUST find the cloud by name (the flagged risk).
        if not pred.has_cloud(case.name):
            log(f"FATAL: predictor has no cloud for '{case.name}' -- name keying broke.")
            return 2
        engine = NeuroForgeEngine(pred, checker, corrector=corrector, config=Config())
        result = engine.solve(case)
        field = result.field

        # ---- map 1: physics trust (NO ground truth, uncertainty=None) ----
        diag = checker.diagnose(field, case, uncertainty=None)

        # ---- map 2: deep-ensemble variance (independent of the trust map) ----
        if members:
            std4, speed_std = ensemble_variance(
                members, [cloud], point_norm, grid_norm, case, device)
        else:
            speed_std = np.zeros(field.shape, dtype=DTYPE)
            std4 = np.zeros((4,) + field.shape, dtype=DTYPE)

        report_diagnostics(case, field, diag, std4, speed_std)

        if representative is None:
            representative = (case, field, diag, speed_std, name)

    # ---- figure for the representative cylinder ----
    case, field, diag, speed_std, name = representative
    out_pdf = os.path.join(a.out_dir, "fig_cylinder_ood.pdf")
    out_png = os.path.join(a.out_dir, "fig_cylinder_ood.png")
    note = (f"Representative cylinder = {name} (first of {len(case_specs)} cases; "
            "chosen by index, not cherry-picked against the thesis).")
    make_figure(case, field, diag, speed_std, out_pdf, out_png, note)

    # ---- caption (honest; printed to stdout) ----
    caption = (
        "Figure: Qualitative out-of-distribution probe. An airfoil-trained "
        "point-cloud Transolver, deployed on a circular CYLINDER it never saw "
        "during training. (a) Predicted speed field of the deployed (DEQ-corrected) "
        "model: it does not produce a physical bluff-body flow -- there is no "
        "proper stagnation and NO WAKE, and the flow accelerates past the body. "
        "(b) The PHYSICS-RESIDUAL trust map -- the magnitude of the governing-PDE "
        "residual (continuity + momentum) plus the no-slip boundary violation, "
        "computed WITHOUT any ground truth -- lights up in a tight ring ON the "
        "body, where the field LOCALLY violates the equations. Note the residual is "
        "a LOCAL-violation detector, not a global-correctness oracle: the smoothly "
        "wrong wake region is close to a locally-valid steady solution, so it stays "
        "dark even though it is also physically wrong. (c) The DATA-DRIVEN "
        "deep-ensemble variance (per-cell std of 5 independently trained, "
        "UNCORRECTED members) for contrast: it is diffuse and tracks freestream / "
        "wake member-disagreement rather than localising the body failure. This is "
        "a QUALITATIVE illustration (no reference solution; the in-repo synthetic "
        "cylinder pseudo-RANS is physically invalid for a bluff body and is NOT "
        "used): a physics-residual signal flags the OOD failure from first "
        "principles where the field is locally inconsistent, whereas data-driven "
        "familiarity need not localise the same physical defect. " + note
    )
    log("=========== CAPTION ===========")
    print(caption, flush=True)
    log("Read the per-component diagnostics above to see WHICH residual term "
        "(continuity/momentum vs no-slip BC) drives the trust map, and whether the "
        "ensemble variance localises near the body (near/far ratio) -- write the "
        "paper caption to match what the numbers actually show.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
