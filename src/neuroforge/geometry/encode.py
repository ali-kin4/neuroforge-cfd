"""Encode a :class:`FlowCase` into the network input stack.

The encoder is the single place that materialises the 7-channel
``INPUT_CHANNELS`` tensor consumed by every backbone. Channel order is frozen:

``[sdf, mask, x_norm, y_norm, u_in_field, v_in_field, log10_re]``.
"""

from __future__ import annotations

import numpy as np

from neuroforge.core.types import DTYPE, N_IN, FlowCase
from neuroforge.geometry.sdf import signed_distance, solid_mask

__all__ = ["encode_case", "case_geometry_fields"]


def case_geometry_fields(case: FlowCase) -> tuple[np.ndarray, np.ndarray]:
    """Compute the geometry-derived fields for a case.

    Parameters
    ----------
    case : FlowCase
        The problem statement (geometry + domain).

    Returns
    -------
    tuple of numpy.ndarray
        ``(sdf, mask)`` each ``(ny, nx)`` float32. ``sdf`` is negative inside
        the solid; ``mask`` is ``1.0`` in fluid and ``0.0`` in the body.
    """
    sdf = signed_distance(case.geometry, case.domain)
    mask = solid_mask(case.geometry, case.domain)
    return sdf, mask


def encode_case(case: FlowCase) -> np.ndarray:
    """Build the ``(N_IN, ny, nx)`` network input stack for a case.

    Channels, in :data:`~neuroforge.core.types.INPUT_CHANNELS` order:

    0. ``sdf`` — signed distance (negative inside the solid).
    1. ``mask`` — ``1`` in fluid, ``0`` in solid.
    2. ``x`` — x-coordinate scaled to ``~[-1, 1]`` over the domain x-bounds.
    3. ``y`` — y-coordinate scaled to ``~[-1, 1]`` over the domain y-bounds.
    4. ``u_in`` — inlet x-velocity (``case.bc.inlet_vector()``), broadcast.
    5. ``v_in`` — inlet y-velocity, broadcast.
    6. ``log_re`` — ``log10(case.bc.reynolds)``, broadcast.

    Parameters
    ----------
    case : FlowCase
        The problem statement.

    Returns
    -------
    numpy.ndarray
        ``(N_IN, ny, nx)`` float32 input stack.
    """
    domain = case.domain
    ny, nx = domain.shape

    sdf, mask = case_geometry_fields(case)

    X, Y = domain.grid()
    xmin, xmax, ymin, ymax = domain.bounds
    # Scale to [-1, 1] over each axis (guard against degenerate extents).
    xspan = max(xmax - xmin, 1e-12)
    yspan = max(ymax - ymin, 1e-12)
    x_norm = (2.0 * (X - xmin) / xspan - 1.0).astype(DTYPE)
    y_norm = (2.0 * (Y - ymin) / yspan - 1.0).astype(DTYPE)

    u_in, v_in = case.bc.inlet_vector()
    u_field = np.full((ny, nx), u_in, dtype=DTYPE)
    v_field = np.full((ny, nx), v_in, dtype=DTYPE)

    re = max(float(case.bc.reynolds), 1.0)
    log_re = np.full((ny, nx), np.log10(re), dtype=DTYPE)

    stack = np.stack(
        [sdf, mask, x_norm, y_norm, u_field, v_field, log_re], axis=0
    ).astype(DTYPE)

    assert stack.shape[0] == N_IN, f"expected {N_IN} channels, got {stack.shape[0]}"
    return stack
