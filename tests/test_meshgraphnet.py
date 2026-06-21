"""Tests for the MeshGraphNet point-cloud baseline (faithfulness + plumbing)."""

from __future__ import annotations

import neuroforge  # noqa: F401  -- thread caps before torch

import torch

from neuroforge.models.baselines.meshgraphnet import (
    MeshGraphNetPointModel,
    edge_features,
    knn_edges,
)


def _toy_graph(n=40, k=6, seed=0):
    g = torch.Generator().manual_seed(seed)
    pos = torch.rand(n, 2, generator=g)
    ei = knn_edges(pos, k=k)
    l_xy = pos.std(0).clamp_min(1e-6)
    ea = edge_features(pos, ei, l_xy, float(torch.linalg.norm(l_xy)))
    x = torch.rand(1, n, 7, generator=g)
    return x, ei, ea


def test_forward_shape_b1():
    model = MeshGraphNetPointModel(in_features=7, out_features=4, width=16, n_layers=3)
    x, ei, ea = _toy_graph()
    out = model(x, ei, ea)
    assert out.shape == (1, x.shape[1], 4)
    assert torch.isfinite(out).all()


def test_knn_no_self_loops_and_symmetric():
    g = torch.Generator().manual_seed(1)
    pos = torch.rand(50, 2, generator=g)
    ei = knn_edges(pos, k=5)
    src, dst = ei[0], ei[1]
    # no self loops
    assert int((src == dst).sum()) == 0
    # symmetric: the reverse of every edge is also present (set equality)
    fwd = set(zip(src.tolist(), dst.tolist()))
    rev = set(zip(dst.tolist(), src.tolist()))
    assert fwd == rev
    # no duplicate directed edges
    assert len(fwd) == src.shape[0]


def test_grad_checkpoint_numerically_exact():
    """grad_checkpoint on vs off: outputs AND gradients match to ~1e-5.

    Checkpointing recomputes activations in backward, so it must change only
    memory, not numerics. Uses the SAME model instance for both passes so weights
    are identical; compares output and a representative parameter gradient.
    """
    torch.manual_seed(0)
    model = MeshGraphNetPointModel(in_features=7, out_features=4, width=24, n_layers=4)
    model.train()
    x, ei, ea = _toy_graph(n=60, k=6, seed=2)

    # --- checkpointing OFF ---
    model.zero_grad(set_to_none=True)
    out_off = model(x, ei, ea, grad_checkpoint=False)
    out_off.pow(2).mean().backward()
    ref_param = next(p for p in model.parameters() if p.grad is not None)
    grad_off = ref_param.grad.detach().clone()
    out_off = out_off.detach().clone()

    # --- checkpointing ON (same weights) ---
    model.zero_grad(set_to_none=True)
    out_on = model(x, ei, ea, grad_checkpoint=True)
    out_on.pow(2).mean().backward()
    grad_on = ref_param.grad.detach().clone()

    out_diff = (out_on.detach() - out_off).abs().max().item()
    grad_diff = (grad_on - grad_off).abs().max().item()
    assert out_diff < 1e-5, f"output mismatch {out_diff}"
    assert grad_diff < 1e-5, f"grad mismatch {grad_diff}"


def test_grad_checkpoint_noop_in_eval():
    """In eval/no_grad the checkpoint path is skipped; output matches the plain path."""
    torch.manual_seed(0)
    model = MeshGraphNetPointModel(in_features=7, out_features=4, width=16, n_layers=3)
    model.eval()
    x, ei, ea = _toy_graph(seed=3)
    with torch.no_grad():
        a = model(x, ei, ea, grad_checkpoint=True)
        b = model(x, ei, ea, grad_checkpoint=False)
    assert torch.allclose(a, b, atol=1e-6)


def test_eval_edge_chunk_exact():
    """Chunked edge update == dense (eval memory bound is numerically exact)."""
    torch.manual_seed(0)
    model = MeshGraphNetPointModel(in_features=7, out_features=4, width=16, n_layers=3)
    model.eval()
    x, ei, ea = _toy_graph(n=80, k=6, seed=4)
    with torch.no_grad():
        dense = model(x, ei, ea, edge_chunk=None)
        chunked = model(x, ei, ea, edge_chunk=37)
    assert torch.allclose(dense, chunked, atol=1e-6)
