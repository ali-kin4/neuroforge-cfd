"""Adapter: score a point-cloud model with the SAME ``evaluate_cases`` as NeuroForge.

``evaluate_cases(predict_fn, pairs)`` expects ``predict_fn: FlowCase ->
FlowField`` on the structured grid. A point model predicts per-point
``(u, v, p, nut)`` on the native cloud, so to score it apples-to-apples we
rasterise those predictions onto the structured crop using the **byte-identical**
projection that built the ground-truth fields in
``airfrans_loader._sim_to_pair``:

* same ``Domain``/crop (taken from the GT ``FlowCase``),
* ``rasterize_point_cloud(..., fill=0.0, method="linear")``,
* the same solid-zeroing (``u, v, nut`` zeroed inside the body, ``p`` kept) using
  the GT case's geometry-derived ``sdf`` / ``mask``.

Because a perfect point predictor rasterises to a field identical to the GT, the
projection adds **no bias between methods** — it only fails to reward sub-grid
boundary-layer detail, which the grid backbone also cannot represent (a shared,
documented limitation; see ``docs/baseline_plan.md`` §3).
"""

from __future__ import annotations

import numpy as np
import torch

from neuroforge.core.types import FlowCase, FlowField
from neuroforge.data.pointcloud import PointCloudCase, PointNormalizer
from neuroforge.data.rasterize import rasterize_point_cloud
from neuroforge.geometry.sdf import signed_distance, solid_mask


def _predict_points(
    model: torch.nn.Module,
    pc: PointCloudCase,
    normalizer: PointNormalizer,
    device: torch.device,
    chunk: int | None = None,
) -> np.ndarray:
    """Run the point model over the FULL cloud in ONE forward, return physical ``(M, 4)``.

    Whole-cloud inference is the faithful eval: Transolver's physics-attention
    couples ALL input points through the learned slice decomposition, so a point's
    prediction depends on the whole cloud it is forwarded with. Splitting the
    cloud into chunks would give each chunk its own slice tokens — a *different*,
    unfaithful prediction. ``chunk`` is therefore ``None`` by default (single
    forward) and should only be set as an **OOM fallback**, at the documented cost
    of partitioning the slice attention (no longer equivalent to the model the
    authors run).
    """
    model.eval()
    feats = normalizer.transform_in(pc.features)  # (M, F_IN) normalised
    with torch.no_grad():
        if chunk is None or feats.shape[0] <= chunk:
            xb = torch.from_numpy(feats).to(device).unsqueeze(0)  # (1, M, F)
            yb = normalizer.inverse_out(model(xb).squeeze(0))
            return yb.detach().cpu().numpy().astype(np.float32)
        # OOM fallback: chunked forward. NOT equivalent to whole-cloud inference
        # (each chunk gets its own slice decomposition); use only if the full
        # cloud does not fit in memory.
        out = np.empty((feats.shape[0], 4), dtype=np.float32)
        for s in range(0, feats.shape[0], chunk):
            xb = torch.from_numpy(feats[s:s + chunk]).to(device).unsqueeze(0)
            yb = normalizer.inverse_out(model(xb).squeeze(0))
            out[s:s + chunk] = yb.detach().cpu().numpy().astype(np.float32)
        return out


def make_predict_fn(
    model: torch.nn.Module,
    pointclouds: list[PointCloudCase],
    normalizer: PointNormalizer,
    device: torch.device | str = "cpu",
    chunk: int = 65536,
):
    """Build a ``FlowCase -> FlowField`` predictor for ``evaluate_cases``.

    ``pointclouds`` are the native clouds aligned (by ``name``) to the GT
    ``FlowCase`` objects passed to ``evaluate_cases``. The returned closure looks
    up the cloud by ``case.name``, predicts per-point fields on the full cloud,
    and rasterises them onto ``case.domain`` exactly as ``_sim_to_pair`` did.
    """
    device = torch.device(device) if isinstance(device, str) else device
    model = model.to(device)
    by_name = {pc.name: pc for pc in pointclouds}

    def predict_fn(case: FlowCase) -> FlowField:
        pc = by_name.get(case.name)
        if pc is None:
            raise KeyError(
                f"no point cloud for case '{case.name}'. The point clouds and GT "
                "pairs must come from the same af.dataset.load order/task/split."
            )
        preds = _predict_points(model, pc, normalizer, device, chunk=chunk)  # (M, 4) physical

        domain = case.domain
        # Identical rasterisation path to airfrans_loader._sim_to_pair.
        raster = rasterize_point_cloud(pc.pos, preds, domain, fill=0.0, method="linear")
        sdf = signed_distance(case.geometry, domain)
        mask = solid_mask(case.geometry, domain)
        solid = mask < 0.5
        u = np.where(solid, 0.0, raster[0])
        v = np.where(solid, 0.0, raster[1])
        p = raster[2]
        nut = np.where(solid, 0.0, np.maximum(raster[3], 0.0))
        return FlowField(
            domain=domain, u=u, v=v, p=p, nut=nut, mask=mask, sdf=sdf,
            meta={"source": "transolver_baseline", "case": case.name},
        )

    return predict_fn
