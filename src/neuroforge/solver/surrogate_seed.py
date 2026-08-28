"""Seed an OpenFOAM solve from a **trained** NeuroForge prediction.

Every warm start measured in this track so far uses an *oracle* -- the converged
answer degraded only by projection. That isolates the mechanism and bounds what
any surrogate could achieve, and it is deliberately not a claim about a model.
This closes that gap: it queries the deployed point backbone and hands the result
to the solver.

Two facts make it possible, and one makes it delicate.

``TransolverPointModel`` maps ``(B, N, 7) -> (B, N, 4)`` over an **arbitrary
point cloud**, not a fixed grid. So the model can be evaluated directly at the
C-grid cell centres -- no Cartesian round-trip, no resolution ceiling. Its
training clouds come from body-fitted OpenFOAM meshes, so its points are already
dense where the boundary layer is.

Its features are ``x, y, u_in_x, u_in_y, sdf, n_x, n_y`` -- surface normals, and
**no Reynolds channel**. AirfRANS fixes ``nu = 1.56e-5`` and varies the inlet
speed, so Reynolds enters through ``|u_in|`` alone and the model must be queried
in *dimensional* units: ``|u_in| = Re * nu / c``. At Re 3e6 that is 46.8 m/s,
against a training distribution centred on 61.6 +/- 17.8 -- comfortably inside
it. The prediction comes back dimensional and is rescaled to the solver's
non-dimensional convention here.

The delicate part is **reach**. The training ``sdf`` distribution is centred on
0.23 chords with a standard deviation of 0.48; the C-grid runs to 20. Asking for
a prediction out there is a forty-sigma extrapolation, so :func:`predict_on_mesh`
returns freestream beyond ``max_sdf`` and reports how much of the mesh it
covered. That is the same "reach" limit the projection arms hit, arriving this
time from the training distribution rather than from the grid.
"""

from __future__ import annotations

import numpy as np

from neuroforge.core.types import DTYPE

__all__ = [
    "AIRFRANS_NU",
    "load_point_backbone",
    "surface_normals",
    "predict_on_mesh",
    "dimensional_speed",
]

# AirfRANS is dimensional and fixes the kinematic viscosity of air at 20 C.
AIRFRANS_NU = 1.56e-5

# Beyond this many chords from the wall the query is far outside the training
# sdf distribution (mean 0.23, std 0.48) and the prediction is not usable.
DEFAULT_MAX_SDF = 2.0


def dimensional_speed(reynolds: float, chord: float = 1.0,
                      nu: float = AIRFRANS_NU) -> float:
    """Inlet speed in m/s that puts a case at ``reynolds`` in AirfRANS units."""
    return float(reynolds) * float(nu) / float(chord)


def load_point_backbone(ckpt_path: str, device: str = "cpu"):
    """Restore the deployed point Transolver and its normaliser from a seed file.

    The normaliser travels inside the checkpoint, so no training data is touched
    -- the same restore path the Paper-1 scripts use.
    """
    import torch

    from neuroforge.data.pointcloud import PointNormalizer
    from neuroforge.models.baselines.transolver import TransolverPointModel

    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = state["backbone_config"]
    model = TransolverPointModel(
        in_features=int(cfg.get("in_features", 7)),
        out_features=int(cfg.get("out_features", 4)),
        width=int(cfg["width"]), n_layers=int(cfg["n_layers"]),
        n_heads=int(cfg["n_heads"]), n_slices=int(cfg["n_slices"]),
        dropout=float(cfg.get("dropout", 0.0)),
    ).to(device)
    model.load_state_dict(state["model_state"])
    model.eval()
    pn = state["point_norm"]
    norm = PointNormalizer(
        mean_in=np.asarray(pn["mean_in"], DTYPE), std_in=np.asarray(pn["std_in"], DTYPE),
        mean_out=np.asarray(pn["mean_out"], DTYPE), std_out=np.asarray(pn["std_out"], DTYPE),
        eps=float(pn.get("eps", 1e-6)),
    )
    return model, norm


def surface_normals(surface: np.ndarray) -> np.ndarray:
    """Outward unit normals of a closed body polyline, ``(M, 2)``.

    The training features carry the normal on body points and zero everywhere
    else, so the surface points have to be included in the query cloud with
    theirs: Transolver pools globally across the cloud, and a cloud missing the
    body carries no information about where the body is beyond ``sdf``.

    Rotating the tangent gives a normal whose sign follows the polyline's
    winding, which is not something a caller should have to know -- and AirfRANS
    normals point *outward*, so getting it backwards feeds the model the opposite
    of what it was trained on. Each normal is therefore flipped to point away
    from the body's centroid, which is well defined for any section a wall-normal
    grid can be built around.
    """
    surf = np.asarray(surface, dtype=np.float64)[:, :2]
    # An airfoil polyline runs trailing edge around to trailing edge, so it is
    # closed. `np.gradient` falls back to a one-sided difference at the ends,
    # which puts the two worst normals exactly at the trailing edge -- where the
    # boundary layer this seed exists to supply is thickest. Difference
    # periodically when the ends meet.
    closed = np.allclose(surf[0], surf[-1], atol=1e-9)
    ring = surf[:-1] if closed else surf
    if closed:
        tangent = 0.5 * (np.roll(ring, -1, axis=0) - np.roll(ring, 1, axis=0))
        tangent = np.concatenate([tangent, tangent[:1]], axis=0)
    else:
        tangent = np.gradient(surf, axis=0)
    tangent /= np.linalg.norm(tangent, axis=1, keepdims=True) + 1e-30
    normal = np.stack([-tangent[:, 1], tangent[:, 0]], axis=1)
    outward = surf - surf.mean(axis=0)
    flip = np.sign(np.sum(normal * outward, axis=1))
    flip[flip == 0] = 1.0
    return normal * flip[:, None]


def predict_on_mesh(
    checkpoints,
    centres: np.ndarray,
    surface: np.ndarray,
    *,
    reynolds: float,
    aoa_deg: float,
    wall_distance: np.ndarray | None = None,
    max_sdf: float = DEFAULT_MAX_SDF,
    chord: float = 1.0,
    u_inf: float = 1.0,
    nut_freestream: float = 0.0,
    device: str = "cpu",
) -> tuple[tuple[np.ndarray, ...], dict]:
    """Predict ``(u, v, p, nut)`` at mesh cell centres, non-dimensionalised.

    ``checkpoints`` may be one path or several; several are averaged, which is
    the deployed NeuroForge ensemble rather than a single draw.

    Returns the four mesh-order arrays and a report carrying the covered
    fraction, the dimensional inlet speed used, and how far that speed sits from
    the training mean in standard deviations -- all three belong in any table
    that quotes the result.
    """
    import torch

    from neuroforge.solver.warmstart import wall_distance as _wall_distance

    paths = [checkpoints] if isinstance(checkpoints, str) else list(checkpoints)
    if not paths:
        raise ValueError("no checkpoints given")

    centres = np.asarray(centres, dtype=np.float64)
    surf = np.asarray(surface, dtype=np.float64)[:, :2]
    sdf = (_wall_distance(centres, surf) if wall_distance is None
           else np.asarray(wall_distance, dtype=np.float64))

    speed = dimensional_speed(reynolds, chord)
    aoa = np.deg2rad(float(aoa_deg))
    u_in = np.array([speed * np.cos(aoa), speed * np.sin(aoa)])

    inside = sdf <= float(max_sdf)
    mesh = np.zeros((int(inside.sum()), 7), dtype=DTYPE)
    mesh[:, 0:2] = centres[inside, :2]
    mesh[:, 2], mesh[:, 3] = u_in
    mesh[:, 4] = sdf[inside]
    # n_x, n_y stay zero off the wall, exactly as the training features do.

    body = np.zeros((len(surf), 7), dtype=DTYPE)
    body[:, 0:2] = surf
    body[:, 2], body[:, 3] = u_in
    body[:, 5:7] = surface_normals(surf)

    query = np.concatenate([mesh, body], axis=0)
    n_mesh = mesh.shape[0]

    total = None
    for path in paths:
        model, norm = load_point_backbone(path, device)
        with torch.no_grad():
            tensor = torch.from_numpy(norm.transform_in(query)).unsqueeze(0).to(device)
            out = model(tensor).squeeze(0).cpu().numpy()
        raw = norm.inverse_out(out)[:n_mesh]
        total = raw if total is None else total + raw
    pred = total / len(paths)

    # Dimensional -> the solver's convention: velocity by the inlet speed,
    # kinematic pressure by its square, eddy viscosity by speed x chord.
    u = np.full(centres.shape[0], u_inf * np.cos(aoa))
    v = np.full(centres.shape[0], u_inf * np.sin(aoa))
    p = np.zeros(centres.shape[0])
    nut = np.full(centres.shape[0], nut_freestream)
    u[inside] = pred[:, 0] / speed * u_inf
    v[inside] = pred[:, 1] / speed * u_inf
    p[inside] = pred[:, 2] / speed**2 * u_inf**2
    nut[inside] = np.maximum(pred[:, 3] / (speed * chord) * u_inf, 0.0)

    _, norm = load_point_backbone(paths[0], device)
    report = {
        "mode": "neuroforge",
        "checkpoints": [str(p) for p in paths],
        "covered_fraction": float(inside.mean()),
        "max_sdf": float(max_sdf),
        "speed_m_s": speed,
        # How far the query sits from the inlet speeds the model was trained on.
        "speed_sigma": float((speed - norm.mean_in[2]) / max(norm.std_in[2], 1e-30)),
    }
    return (u, v, p, nut), report
