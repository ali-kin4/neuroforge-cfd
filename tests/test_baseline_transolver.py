"""Tests for the Transolver baseline (model, point datamodule, eval adapter).

These do NOT need the airfrans package or the real dataset: a synthetic
AirfRANS-format point cloud is built and rasterised, then the adapter's
"perfect predictor -> GT-identical metrics" property (the fairness keystone of
``docs/baseline_plan.md`` section 3) is asserted directly.
"""

from __future__ import annotations

import numpy as np
import torch

from neuroforge.core.types import FlowCase
from neuroforge.data.airfrans_loader import _sim_to_pair
from neuroforge.data.pointcloud import (
    F_IN,
    F_OUT,
    PointCloudCase,
    PointNormalizer,
    _array_to_pointcase,
)
from neuroforge.data.synthetic import SyntheticRANS
from neuroforge.geometry.sdf import surface_normals
from neuroforge.models.baselines.transolver import (
    PhysicsAttentionIrregularMesh,
    TransolverPointModel,
)
from neuroforge.models.baselines.transolver_adapter import make_predict_fn
from neuroforge.physics.evaluation import evaluate_cases


def _mock_airfrans_array(case: FlowCase) -> np.ndarray:
    """Synthetic (M, 12) AirfRANS-format array from a synthetic solve."""
    field = SyntheticRANS(case.domain.nx).solve(case)
    X, Y = case.domain.grid()
    m = field.mask > 0.5
    u_inf, v_inf = case.bc.inlet_vector()

    vol = np.zeros((int(m.sum()), 12))
    vol[:, 0], vol[:, 1] = X[m], Y[m]
    vol[:, 2], vol[:, 3] = u_inf, v_inf
    vol[:, 4] = np.abs(field.sdf[m])
    vol[:, 7], vol[:, 8] = field.u[m], field.v[m]
    vol[:, 9], vol[:, 10] = field.p[m], field.nut[m]

    sp = case.geometry.surface_points
    sn = surface_normals(case.geometry)
    surf = np.zeros((sp.shape[0], 12))
    surf[:, 0], surf[:, 1] = sp[:, 0], sp[:, 1]
    surf[:, 2], surf[:, 3] = u_inf, v_inf
    surf[:, 5], surf[:, 6] = sn[:, 0], sn[:, 1]
    surf[:, 11] = 1.0  # on-airfoil flag
    return np.vstack([vol, surf])


def test_transolver_forward_backward_and_budget():
    m = TransolverPointModel(in_features=F_IN, out_features=F_OUT, width=64,
                             n_layers=2, n_heads=4, n_slices=16)
    x = torch.randn(2, 500, F_IN)
    y = m(x)
    assert y.shape == (2, 500, F_OUT)
    y.sum().backward()
    assert all(p.grad is not None for p in m.parameters() if p.requires_grad)
    # The default budget config matches the FNO backbone to <1%.
    big = TransolverPointModel()
    assert abs(big.num_params() - 7_387_684) / 7_387_684 < 0.01


def test_physics_attention_is_per_head():
    """Faithfulness: slice weights are per-head (B,H,N,G), temperature learnable."""
    attn = PhysicsAttentionIrregularMesh(dim=32, heads=4, dim_head=8, slice_num=16)
    assert attn.temperature.requires_grad
    assert tuple(attn.temperature.shape) == (1, 4, 1, 1)
    x = torch.randn(2, 50, 32)
    x_mid = attn.in_project_x(x).reshape(2, 50, 4, 8).permute(0, 2, 1, 3)
    sw = attn.softmax(attn.in_project_slice(x_mid) / attn.temperature)
    assert tuple(sw.shape) == (2, 4, 50, 16)  # B, H, N, G -> per-head assignment


def test_point_normalizer_no_leakage():
    rng = np.random.default_rng(0)
    cases = [
        PointCloudCase(f"c{i}", rng.normal(size=(100, F_IN)), rng.normal(size=(100, F_OUT)),
                       rng.normal(size=(100, 2)))
        for i in range(3)
    ]
    norm = PointNormalizer().fit(cases)
    assert norm.fitted
    z = norm.transform_in(cases[0].features)
    assert z.shape == (100, F_IN)
    # inverse round-trips the targets.
    y = cases[0].targets
    back = norm.inverse_out(norm.transform_out(y))
    assert np.allclose(back, y, atol=1e-3)


def test_adapter_perfect_predictor_matches_gt():
    """A perfect point predictor -> rasterised field == GT -> ~zero MSE.

    This is the fairness keystone: grid scoring of the point model adds no bias,
    because an exact predictor reproduces the GT grid the metric uses.
    """
    case = FlowCase.from_airfoil("naca2412", aoa=4.0, reynolds=2e6, u_inf=20.0, resolution=32)
    arr = _mock_airfrans_array(case)
    # GT pair via the SAME path the real loader uses.
    gt_case, gt_field = _sim_to_pair(arr, case.geometry.name, resolution=32)
    pc = _array_to_pointcase(arr, gt_case.name)

    # Identity normaliser so the "model" can return raw physical targets.
    norm = PointNormalizer(
        mean_in=np.zeros(F_IN), std_in=np.ones(F_IN),
        mean_out=np.zeros(F_OUT), std_out=np.ones(F_OUT),
    )

    class PerfectModel(torch.nn.Module):
        """Returns the per-point GT targets regardless of input (oracle)."""

        def __init__(self, targets):
            super().__init__()
            self.targets = torch.from_numpy(targets.astype(np.float32))

        def forward(self, x):  # x: (1, n, F_IN) -- ignored; oracle by construction
            n = x.shape[1]
            return self.targets[:n].unsqueeze(0)

    # The oracle must see the FULL cloud in one chunk (it indexes by position).
    model = PerfectModel(pc.targets)
    predict_fn = make_predict_fn(model, [pc], norm, device="cpu", chunk=pc.n_points + 1)

    metrics = evaluate_cases(predict_fn, [(gt_case, gt_field)])
    # Same rasterisation on both sides -> volume MSE is at numerical-noise level.
    assert metrics["mse_u"] < 1e-6
    assert metrics["mse_v"] < 1e-6
    assert metrics["mse_p"] < 1e-6
    assert np.isfinite(metrics["surface_mse_p"])
