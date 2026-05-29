"""Optional local classical-CFD fallback patch.

When the engine's uncertainty / trust map flags a region as unreliable, a
classical solver can in principle be run on just that sub-region and stitched
back in. This module provides the *interface* and a no-op ``stub`` backend that
documents what would run; the real ``openfoam`` / ``su2`` backends are deferred
(they raise :class:`NotImplementedError` with guidance). The module imports with
nothing installed.
"""

from __future__ import annotations

import numpy as np

from neuroforge.core.types import FlowCase, FlowField

__all__ = ["ClassicalFallback"]


class ClassicalFallback:
    """Patch a flagged region of a flow field with a classical CFD solve.

    Parameters
    ----------
    backend : str, optional
        ``'stub'`` (default) reports what would run without changing the field;
        ``'openfoam'`` / ``'su2'`` raise :class:`NotImplementedError`.
    """

    _SUPPORTED = ("stub", "openfoam", "su2")

    def __init__(self, backend: str = "stub") -> None:
        backend = str(backend).lower()
        if backend not in self._SUPPORTED:
            raise ValueError(
                f"unknown fallback backend '{backend}' "
                f"(expected one of {self._SUPPORTED})"
            )
        self.backend = backend

    def patch(
        self, field: FlowField, case: FlowCase, region_mask: np.ndarray
    ) -> FlowField:
        """Return a (possibly patched) field over the cells in ``region_mask``.

        Parameters
        ----------
        field : FlowField
            Current flow field.
        case : FlowCase
            Problem definition (passed to the classical solver in real backends).
        region_mask : numpy.ndarray
            Boolean / ``{0,1}`` map ``(ny, nx)`` selecting the region to re-solve.

        Returns
        -------
        FlowField
            For the ``stub`` backend, the same field with ``meta['fallback']``
            describing the (hypothetical) classical solve and the region size.
        """
        region = np.asarray(region_mask) > 0.5
        n_cells = int(region.sum())

        if self.backend == "stub":
            field.meta = dict(field.meta)
            field.meta["fallback"] = {
                "backend": "stub",
                "ran": False,
                "region_cells": n_cells,
                "region_fraction": float(region.mean()) if region.size else 0.0,
                "note": (
                    "stub backend: would extract the flagged sub-region, run a "
                    "local steady incompressible solve (e.g. simpleFoam) with the "
                    "engine prediction as the initial/boundary condition, then "
                    "blend the result back over the trust gradient. No external "
                    "solver was invoked."
                ),
            }
            return field

        # Real backends are deferred.
        guidance = {
            "openfoam": (
                "OpenFOAM fallback not yet implemented. Install OpenFOAM and a "
                "Python bridge (e.g. PyFoam), export the region to a blockMesh, "
                "run simpleFoam, and re-import the patched field."
            ),
            "su2": (
                "SU2 fallback not yet implemented. Install SU2 with its Python "
                "wrapper, write the region as an SU2 mesh + config, run the RANS "
                "solver, and re-import the patched field."
            ),
        }[self.backend]
        raise NotImplementedError(guidance)
