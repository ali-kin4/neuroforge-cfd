"""NeuroForge v2 predictor: a Transolver point-cloud backbone behind the engine.

NeuroForge v1's :class:`~neuroforge.solver.engine.Predictor` runs a *grid* backbone
(FNO/UNet/...) on the encoded ``(N_IN, H, W)`` stack. The grid FNO trails the
strong point-cloud Transolver by ~16-40x on the AirfRANS field MSEs (paper Table
2). :class:`PointCloudPredictor` closes that gap **without touching the frozen
grid-based** :class:`~neuroforge.core.types.FlowField` **contract**: the backbone
becomes Transolver (predicting per-point ``(u, v, p, nut)`` on the *native* cloud),
and its predictions are rasterised onto the structured grid via the byte-identical
projection in :mod:`neuroforge.models.baselines.transolver_adapter`. The result is
a denormalised grid :class:`FlowField` with ``mask``/``sdf`` — exactly what
:class:`~neuroforge.solver.engine.NeuroForgeEngine` and the certified
Neural-Residual-Iteration loop consume, so the whole self-correction machinery
(PhysicsChecker residuals, DEQ / local corrector, trust map, conformal coverage)
runs **unchanged** on a SOTA-quality prediction.

Two normalisers are held deliberately:

* a :class:`~neuroforge.data.pointcloud.PointNormalizer` standardises the *point*
  features/targets for the Transolver (the representation it was trained on);
* a *grid* :class:`~neuroforge.data.datamodule.Normalizer` (``norm_in`` /
  ``norm_out`` / ``denorm_out`` / ``std_out``) is what the correction loop reads
  off ``predictor.normalizer`` to map fields/residuals into the corrector's
  normalised grid space. It is fitted on the **ground-truth grid pairs** (the same
  field scale the corrector's ``delta_phys = delta_norm * std_out`` denorm needs),
  NOT on Transolver outputs.

The predictor looks the cloud up by ``case.name`` — so it must be built with the
clouds relevant to the call site (test clouds for evaluation, train clouds for
corrector training), aligned to the GT pairs exactly as ``run_baselines`` does.
"""

from __future__ import annotations

import numpy as np
import torch

from neuroforge.core.types import FlowCase, FlowField
from neuroforge.data.pointcloud import PointCloudCase, PointNormalizer
from neuroforge.models.baselines.transolver_adapter import make_predict_fn

__all__ = ["PointCloudPredictor"]


class PointCloudPredictor:
    """Bridge a trained Transolver point model into the v1 engine surface.

    Exposes the same interface :class:`NeuroForgeEngine` / the correction loop
    expect of a v1 :class:`~neuroforge.solver.engine.Predictor`:

    * ``predict(case) -> FlowField`` — denormalised grid field with mask/sdf,
    * ``normalizer`` — a *grid* :class:`~neuroforge.data.datamodule.Normalizer`
      (used by the loop to (de)normalise fields/residuals), and
    * ``device`` — the torch device the backbone runs on.

    Parameters
    ----------
    model : torch.nn.Module
        Trained :class:`~neuroforge.models.baselines.transolver.TransolverPointModel`.
    pointclouds : list of PointCloudCase
        Native clouds aligned (by ``name``) to the cases this predictor will be
        asked to predict. Test clouds for eval; train clouds for corrector
        training.
    point_normalizer : PointNormalizer
        Fitted on the TRAIN clouds only (no leakage); standardises point I/O.
    grid_normalizer : Normalizer
        Grid normaliser fitted on the GT grid pairs; what the correction loop
        reads via ``predictor.normalizer``.
    device : str or torch.device, optional
        Torch device for the backbone (``'auto'`` -> cuda if available).
    chunk : int, optional
        Per-point-forward OOM fallback chunk size (see ``transolver_adapter``).
        ``None`` (default) = faithful single whole-cloud forward.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        pointclouds: list[PointCloudCase],
        point_normalizer: PointNormalizer,
        grid_normalizer,
        device: str | torch.device = "cpu",
        chunk: int | None = None,
    ) -> None:
        self.device = _resolve_device(device)
        self.model = model.to(self.device)
        self.model.eval()
        self.point_normalizer = point_normalizer
        # The attribute the engine / correction loop read is ``.normalizer`` and
        # it must be the GRID normaliser (norm_in/norm_out/denorm_out/std_out).
        self.normalizer = grid_normalizer
        self.chunk = chunk
        self._predict_fn = make_predict_fn(
            model, pointclouds, point_normalizer, device=self.device, chunk=chunk,
        )
        self._by_name = {pc.name: pc for pc in pointclouds}

    def predict(self, case: FlowCase) -> FlowField:
        """Predict ``case`` via Transolver-on-cloud -> rasterise -> grid FlowField.

        Delegates to the *same* rasterise path as
        :func:`neuroforge.models.baselines.transolver_adapter.make_predict_fn`,
        so the field here is byte-identical to the one the baseline table scores —
        making the "backbone alone" vs "backbone + certified loop" comparison
        clean (the only difference is the loop).
        """
        return self._predict_fn(case)

    def predict_tensor(self, x: torch.Tensor) -> torch.Tensor:
        """Run the point backbone on a normalised point tensor ``(B, M, F_IN)``.

        Provided for parity with the v1 :class:`Predictor` surface; the grid
        engine path uses :meth:`predict` instead.
        """
        x = x.to(self.device)
        with torch.no_grad():
            return self.model(x)

    def has_cloud(self, name: str) -> bool:
        """Whether a native cloud is registered for ``name`` (alignment check)."""
        return name in self._by_name


def _resolve_device(name: str | torch.device) -> torch.device:
    if isinstance(name, torch.device):
        return name
    if name in ("auto", "", None):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)
