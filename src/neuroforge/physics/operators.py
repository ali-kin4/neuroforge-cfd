"""Backend-agnostic finite-difference (and optional spectral) operators.

Every operator here works for **both** ``numpy.ndarray`` and ``torch.Tensor``
inputs shaped ``(..., ny, nx)`` and returns an array of the *same* type and
shape as its input. The backend is selected by duck typing on ``type(f)``: if
the input is a ``torch.Tensor`` we use torch ops (so gradients flow), otherwise
we use numpy.

Grid convention (see :mod:`neuroforge.core.types`)
--------------------------------------------------
Structured fields are ``(ny, nx)`` with row index = y (j) and column index =
x (i), built via ``np.meshgrid(x, y, indexing="xy")``. Therefore:

* ``d/dx`` differentiates along the **last** axis (``axis = -1``, the ``nx``
  columns), with spacing ``dx``.
* ``d/dy`` differentiates along the **second-to-last** axis (``axis = -2``, the
  ``ny`` rows), with spacing ``dy``.

Finite differences are second-order central in the interior and first-order
one-sided at the two edges of the differentiated axis, so the output keeps the
input shape (no padding loss). These are the defaults used everywhere in the
physics layer; the FFT-based :func:`spectral_ddx` / :func:`spectral_ddy` are
provided for periodic experiments only.
"""

from __future__ import annotations

from typing import Any

import numpy as np

try:  # torch is an optional backend selected at call time, never at import.
    import torch

    _TORCH_OK = True
except Exception:  # pragma: no cover - torch is a hard dep here but be safe.
    torch = None  # type: ignore[assignment]
    _TORCH_OK = False


# --------------------------------------------------------------------------- #
# Backend helpers
# --------------------------------------------------------------------------- #


def _is_torch(f: Any) -> bool:
    """Return ``True`` if ``f`` is a :class:`torch.Tensor`."""
    return _TORCH_OK and isinstance(f, torch.Tensor)


def _empty_like(f: Any) -> Any:
    """Allocate an uninitialised array of the same type/shape/dtype/device."""
    if _is_torch(f):
        return torch.empty_like(f)
    return np.empty_like(f)


# --------------------------------------------------------------------------- #
# First derivatives (central interior, one-sided edges)
# --------------------------------------------------------------------------- #


def _diff_axis(f: Any, h: float, axis: int) -> Any:
    """First derivative of ``f`` along ``axis`` with uniform spacing ``h``.

    Second-order central differences in the interior and first-order one-sided
    differences at the first/last index of ``axis``. Output has the same type
    and shape as ``f``.

    Parameters
    ----------
    f : numpy.ndarray or torch.Tensor
        Field shaped ``(..., ny, nx)``.
    h : float
        Grid spacing along ``axis``.
    axis : int
        Axis to differentiate (``-1`` for x, ``-2`` for y).

    Returns
    -------
    numpy.ndarray or torch.Tensor
        ``df`` with the same shape/type as ``f``.
    """
    n = f.shape[axis]
    if n < 2:
        # Degenerate axis: derivative is undefined; return zeros to stay safe.
        if _is_torch(f):
            return torch.zeros_like(f)
        return np.zeros_like(f)

    inv2h = 1.0 / (2.0 * h)
    invh = 1.0 / h
    out = _empty_like(f)

    # Index helpers that slice a single, arbitrary axis.
    def sl(start: int | None, stop: int | None) -> tuple[slice, ...]:
        idx: list[slice] = [slice(None)] * f.ndim
        idx[axis] = slice(start, stop)
        return tuple(idx)

    def at(i: int) -> tuple[slice | int, ...]:
        idx: list[Any] = [slice(None)] * f.ndim
        idx[axis] = i
        return tuple(idx)

    if n >= 3:
        # Central in the interior: (f[i+1] - f[i-1]) / (2h).
        out[sl(1, -1)] = (f[sl(2, None)] - f[sl(None, -2)]) * inv2h
    # First-order one-sided at the two edges.
    out[at(0)] = (f[at(1)] - f[at(0)]) * invh
    out[at(n - 1)] = (f[at(n - 1)] - f[at(n - 2)]) * invh
    if n == 2:
        # Only two points: both edges already set above (forward/backward equal).
        pass
    return out


def ddx(f: Any, dx: float) -> Any:
    """Partial derivative ``df/dx`` (along the last axis, the ``nx`` columns).

    Parameters
    ----------
    f : numpy.ndarray or torch.Tensor
        Field shaped ``(..., ny, nx)``.
    dx : float
        Grid spacing in x.

    Returns
    -------
    numpy.ndarray or torch.Tensor
        ``df/dx`` with the same type and shape as ``f``.
    """
    return _diff_axis(f, float(dx), axis=-1)


def ddy(f: Any, dy: float) -> Any:
    """Partial derivative ``df/dy`` (along the second-to-last axis, ``ny`` rows).

    Parameters
    ----------
    f : numpy.ndarray or torch.Tensor
        Field shaped ``(..., ny, nx)``.
    dy : float
        Grid spacing in y.

    Returns
    -------
    numpy.ndarray or torch.Tensor
        ``df/dy`` with the same type and shape as ``f``.
    """
    return _diff_axis(f, float(dy), axis=-2)


# --------------------------------------------------------------------------- #
# Second derivatives / Laplacian
# --------------------------------------------------------------------------- #


def _second_diff_axis(f: Any, h: float, axis: int) -> Any:
    """Second derivative along ``axis`` with uniform spacing ``h``.

    Second-order central in the interior ``(f[i+1] - 2 f[i] + f[i-1]) / h^2``;
    at the two edges the one-sided three-point second-derivative stencil
    ``(f[0] - 2 f[1] + f[2]) / h^2`` is used so the output keeps the input shape.
    """
    n = f.shape[axis]
    out = _empty_like(f)

    def sl(start: int | None, stop: int | None) -> tuple[slice, ...]:
        idx: list[slice] = [slice(None)] * f.ndim
        idx[axis] = slice(start, stop)
        return tuple(idx)

    def at(i: int) -> tuple[Any, ...]:
        idx: list[Any] = [slice(None)] * f.ndim
        idx[axis] = i
        return tuple(idx)

    if n < 3:
        # Cannot form a second derivative; flat assumption -> 0.
        if _is_torch(f):
            return torch.zeros_like(f)
        return np.zeros_like(f)

    invh2 = 1.0 / (h * h)
    out[sl(1, -1)] = (f[sl(2, None)] - 2.0 * f[sl(1, -1)] + f[sl(None, -2)]) * invh2
    # One-sided three-point stencils at the edges.
    out[at(0)] = (f[at(0)] - 2.0 * f[at(1)] + f[at(2)]) * invh2
    out[at(n - 1)] = (f[at(n - 1)] - 2.0 * f[at(n - 2)] + f[at(n - 3)]) * invh2
    return out


def laplacian(f: Any, dx: float, dy: float) -> Any:
    """Laplacian ``d2f/dx2 + d2f/dy2``.

    Parameters
    ----------
    f : numpy.ndarray or torch.Tensor
        Field shaped ``(..., ny, nx)``.
    dx, dy : float
        Grid spacings in x and y.

    Returns
    -------
    numpy.ndarray or torch.Tensor
        The Laplacian with the same type and shape as ``f``.
    """
    return _second_diff_axis(f, float(dx), axis=-1) + _second_diff_axis(f, float(dy), axis=-2)


# --------------------------------------------------------------------------- #
# Vector operators
# --------------------------------------------------------------------------- #


def divergence(u: Any, v: Any, dx: float, dy: float) -> Any:
    """Divergence of a 2-D vector field, ``du/dx + dv/dy``.

    Parameters
    ----------
    u, v : numpy.ndarray or torch.Tensor
        Velocity components shaped ``(..., ny, nx)`` (same type/shape).
    dx, dy : float
        Grid spacings.

    Returns
    -------
    numpy.ndarray or torch.Tensor
        ``du/dx + dv/dy``.
    """
    return ddx(u, dx) + ddy(v, dy)


def gradient(f: Any, dx: float, dy: float) -> tuple[Any, Any]:
    """Spatial gradient of a scalar field.

    Parameters
    ----------
    f : numpy.ndarray or torch.Tensor
        Field shaped ``(..., ny, nx)``.
    dx, dy : float
        Grid spacings.

    Returns
    -------
    tuple
        ``(df/dx, df/dy)``, each the same type and shape as ``f``.
    """
    return ddx(f, dx), ddy(f, dy)


# --------------------------------------------------------------------------- #
# Optional spectral (FFT) derivatives — periodic domains only.
# --------------------------------------------------------------------------- #


def _fft_wavenumbers(n: int, h: float, like: Any) -> Any:
    """Angular wavenumbers ``k`` for an ``n``-point periodic axis of spacing ``h``."""
    if _is_torch(like):
        return torch.fft.fftfreq(n, d=h, device=like.device) * (2.0 * np.pi)
    return np.fft.fftfreq(n, d=h).astype(np.float64) * (2.0 * np.pi)


def spectral_ddx(f: Any, dx: float) -> Any:
    """FFT-based ``df/dx`` along the last axis (assumes x is periodic).

    Provided for periodic experiments; the finite-difference :func:`ddx` is the
    default everywhere else. Returns the real part, matching the input type.
    """
    n = f.shape[-1]
    k = _fft_wavenumbers(n, float(dx), f)
    if _is_torch(f):
        shape = [1] * f.ndim
        shape[-1] = n
        ik = (1j * k).reshape(shape)
        fh = torch.fft.fft(f.to(torch.complex64), dim=-1)
        return torch.fft.ifft(fh * ik, dim=-1).real.to(f.dtype)
    ik = 1j * k
    fh = np.fft.fft(f, axis=-1)
    return np.real(np.fft.ifft(fh * ik, axis=-1)).astype(f.dtype)


def spectral_ddy(f: Any, dy: float) -> Any:
    """FFT-based ``df/dy`` along the second-to-last axis (assumes y is periodic)."""
    n = f.shape[-2]
    k = _fft_wavenumbers(n, float(dy), f)
    if _is_torch(f):
        shape = [1] * f.ndim
        shape[-2] = n
        ik = (1j * k).reshape(shape)
        fh = torch.fft.fft(f.to(torch.complex64), dim=-2)
        return torch.fft.ifft(fh * ik, dim=-2).real.to(f.dtype)
    ik = (1j * k).reshape((-1, 1))
    fh = np.fft.fft(f, axis=-2)
    return np.real(np.fft.ifft(fh * ik, axis=-2)).astype(f.dtype)


__all__ = [
    "ddx",
    "ddy",
    "laplacian",
    "divergence",
    "gradient",
    "spectral_ddx",
    "spectral_ddy",
]
