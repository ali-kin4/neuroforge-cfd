"""Zero-download synthetic pseudo-RANS flow-field generator.

:class:`SyntheticRANS` produces physically-plausible 2-D airfoil flow fields
*without a real solver*. The model is a superposition of analytic building
blocks chosen so the result looks like a converged RANS solution rather than
noise, while remaining CPU-cheap and download-free:

1. **Potential core** — uniform freestream plus a source-panel distribution on
   the body surface (with a bound vortex enforcing the Kutta condition). The
   source panels make the body a streamline, so the inviscid field is
   divergence-free almost everywhere (continuity ``div(u) ≈ 0``).
2. **Viscous correction** — a near-wall damping layer driving the velocity to
   zero at the surface (approximate no-slip) and a Gaussian wake deficit
   trailing the body, mimicking the boundary layer and momentum wake of a RANS
   solution.
3. **Pressure** — incompressible Bernoulli, ``p ≈ p_inf + 0.5 (U_inf^2 - |u|^2)``
   in kinematic (``p/rho``) units.
4. **Turbulent viscosity** — a simple mixing-length-style ``nut`` field
   concentrated in the boundary layer and wake.

Inside the solid the velocity and ``nut`` are set to zero via the mask.
"""

from __future__ import annotations

import numpy as np

from neuroforge.core.types import (
    DTYPE,
    FlowCase,
    FlowField,
)
from neuroforge.geometry.sdf import signed_distance, solid_mask

__all__ = ["SyntheticRANS"]


# Airfoils sampled across the synthetic dataset (4- and 5-digit families).
_AIRFOIL_BANK: tuple[str, ...] = (
    "naca0012",
    "naca2412",
    "naca4412",
    "naca0015",
    "naca2415",
    "naca6409",
    "naca23012",
    "naca23015",
)


class SyntheticRANS:
    """Analytic pseudo-RANS flow-field generator for airfoils.

    Parameters
    ----------
    resolution : int, optional
        Grid resolution; the domain is discretised ``resolution x resolution``.
    seed : int, optional
        Seed for the case sampler (airfoil / AoA / Re selection).
    domain_bounds : tuple of float, optional
        ``(xmin, xmax, ymin, ymax)`` computational domain.
    """

    def __init__(
        self,
        resolution: int = 128,
        seed: int = 0,
        domain_bounds: tuple[float, float, float, float] = (-1.0, 2.0, -1.5, 1.5),
    ) -> None:
        self.resolution = int(resolution)
        self.seed = int(seed)
        self.domain_bounds = tuple(float(b) for b in domain_bounds)
        self._rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------ #
    # Case sampling
    # ------------------------------------------------------------------ #

    def sample_case(self, idx: int) -> FlowCase:
        """Sample a reproducible :class:`FlowCase` for dataset index ``idx``.

        Airfoil, angle of attack, Reynolds number and freestream speed are
        drawn from a per-index RNG so the dataset is deterministic given the
        constructor seed.

        Parameters
        ----------
        idx : int
            Sample index.

        Returns
        -------
        FlowCase
            A NACA-airfoil case with consistent ``nu = U * c / Re``.
        """
        rng = np.random.default_rng((self.seed * 1_000_003 + idx) & 0xFFFFFFFF)
        airfoil = _AIRFOIL_BANK[idx % len(_AIRFOIL_BANK)]
        aoa = float(rng.uniform(-6.0, 12.0))
        reynolds = float(10.0 ** rng.uniform(5.7, 6.7))   # ~5e5 .. 5e6
        u_inf = float(rng.uniform(20.0, 50.0))
        return FlowCase.from_airfoil(
            airfoil=airfoil,
            aoa=aoa,
            reynolds=reynolds,
            u_inf=u_inf,
            resolution=self.resolution,
            domain_bounds=self.domain_bounds,
        )

    # ------------------------------------------------------------------ #
    # Potential-flow core (source panels + bound vortex)
    # ------------------------------------------------------------------ #

    @staticmethod
    def _panel_geometry(geom) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Build panel midpoints, normals, tangents and lengths from a loop.

        Returns
        -------
        tuple of numpy.ndarray
            ``(mid (P,2), normal (P,2), tangent (P,2), length (P,))``.
        """
        p = np.asarray(geom.surface_points, dtype=np.float64)
        if not np.allclose(p[0], p[-1]):
            p = np.vstack([p, p[0]])
        a = p[:-1]
        b = p[1:]
        mid = 0.5 * (a + b)
        edge = b - a
        length = np.hypot(edge[:, 0], edge[:, 1])
        length[length < 1e-12] = 1e-12
        tangent = edge / length[:, None]
        # Outward normal for a CCW loop: rotate tangent by -90 deg.
        normal = np.stack([tangent[:, 1], -tangent[:, 0]], axis=1)
        return mid, normal, tangent, length

    def _source_strengths(
        self, mid: np.ndarray, normal: np.ndarray, length: np.ndarray,
        u_inf: float, v_inf: float,
    ) -> np.ndarray:
        """Solve a source-panel system so the body is (near) impermeable.

        Enforces zero normal velocity at each panel midpoint from the
        superposition of the freestream and all panel sources (a classic
        Hess–Smith source-panel method, simplified to point sources at the
        midpoints). A small Tikhonov term keeps the system well conditioned.

        Returns
        -------
        numpy.ndarray
            ``(P,)`` source strengths per unit length.
        """
        n = mid.shape[0]
        # Influence: normal velocity at panel i from a unit point source at j.
        dx = mid[:, 0][:, None] - mid[:, 0][None, :]
        dy = mid[:, 1][:, None] - mid[:, 1][None, :]
        r2 = dx * dx + dy * dy
        np.fill_diagonal(r2, np.inf)  # self-influence handled by the 0.5 term
        inv = 1.0 / (2.0 * np.pi * r2)
        ux = dx * inv
        uy = dy * inv
        # Project induced velocity onto panel-i normal, weight by source area.
        A = (ux * normal[:, 0][:, None] + uy * normal[:, 1][:, None]) * length[None, :]
        np.fill_diagonal(A, 0.5)  # source sheet self term
        rhs = -(u_inf * normal[:, 0] + v_inf * normal[:, 1])
        # Regularised solve (robust for coincident/!convex panels).
        lam = 1e-6 * np.trace(A) / max(n, 1)
        sigma = np.linalg.solve(A + lam * np.eye(n), rhs)
        return sigma

    @staticmethod
    def _circulation(
        geom, u_inf: float, v_inf: float, chord: float
    ) -> float:
        """Thin-airfoil bound circulation enforcing the Kutta condition.

        Uses ``Gamma = -0.5 * U * c * Cl`` with ``Cl = 2*pi*(alpha - alpha_0)``
        where the zero-lift angle ``alpha_0`` is estimated from the camber.

        Returns
        -------
        float
            Bound circulation ``Gamma`` (sign per the chosen vortex convention).
        """
        u_mag = float(np.hypot(u_inf, v_inf)) + 1e-12
        alpha = np.arctan2(v_inf, u_inf)
        # Estimate zero-lift angle from max camber (rough, from the meta data).
        # Camber proxy: signed mean of y on the loop (positive => cambered up).
        ys = np.asarray(geom.surface_points, dtype=np.float64)[:, 1]
        camber_proxy = float(np.mean(ys))
        alpha0 = -2.0 * camber_proxy / max(chord, 1e-9)
        cl = 2.0 * np.pi * (alpha - alpha0)
        gamma = -0.5 * u_mag * chord * cl
        return float(gamma)

    def _potential_velocity(
        self, X: np.ndarray, Y: np.ndarray, geom, u_inf: float, v_inf: float,
        chord: float, eps: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Inviscid velocity field: freestream + source panels + bound vortex.

        The singular ``1/r^2`` source/vortex kernels are *desingularised* with a
        smoothing (blob) radius ``eps`` (Rosenhead–Moore regularisation): the
        denominator becomes ``r^2 + eps^2``. This caps the velocity within a
        cell of the surface — removing the spurious near-wall spikes that would
        otherwise violate continuity — while leaving the far field unchanged.

        Parameters
        ----------
        eps : float
            Desingularisation radius (order of the grid spacing / panel length).

        Returns
        -------
        tuple of numpy.ndarray
            ``(u, v)`` each ``(ny, nx)``.
        """
        mid, normal, _tangent, length = self._panel_geometry(geom)
        sigma = self._source_strengths(mid, normal, length, u_inf, v_inf)
        gamma = self._circulation(geom, u_inf, v_inf, chord)
        eps2 = float(eps) ** 2

        shape = X.shape
        gx = X.ravel()
        gy = Y.ravel()
        u = np.full(gx.shape, u_inf, dtype=np.float64)
        v = np.full(gy.shape, v_inf, dtype=np.float64)

        # Source-panel induced velocity (chunk over grid to bound memory).
        sig_len = sigma * length  # effective point-source strength
        chunk = 8192
        cx = mid[:, 0]
        cy = mid[:, 1]
        for s in range(0, gx.size, chunk):
            e = s + chunk
            dx = gx[s:e][:, None] - cx[None, :]
            dy = gy[s:e][:, None] - cy[None, :]
            r2 = dx * dx + dy * dy + eps2
            inv = sig_len[None, :] / (2.0 * np.pi * r2)
            u[s:e] += np.sum(dx * inv, axis=1)
            v[s:e] += np.sum(dy * inv, axis=1)

        # Single bound vortex at the quarter-chord enforcing circulation.
        x_q = geom.bounding_box()[0] + 0.25 * chord
        ys = np.asarray(geom.surface_points, dtype=np.float64)[:, 1]
        y_q = float(np.mean(ys))
        dxq = gx - x_q
        dyq = gy - y_q
        r2q = dxq * dxq + dyq * dyq + eps2
        vfac = gamma / (2.0 * np.pi * r2q)
        u += vfac * dyq
        v += -vfac * dxq

        return u.reshape(shape), v.reshape(shape)

    # ------------------------------------------------------------------ #
    # Viscous boundary-layer & wake model
    # ------------------------------------------------------------------ #

    @staticmethod
    def _viscous_correction(
        u: np.ndarray,
        v: np.ndarray,
        sdf: np.ndarray,
        X: np.ndarray,
        Y: np.ndarray,
        geom,
        u_inf: float,
        v_inf: float,
        chord: float,
        reynolds: float,
        cell: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Apply near-wall damping and a wake deficit; build a ``nut`` field.

        Parameters
        ----------
        u, v : numpy.ndarray
            Inviscid velocity components ``(ny, nx)``.
        sdf : numpy.ndarray
            Signed distance (negative inside the body).
        X, Y : numpy.ndarray
            Coordinate grids.
        geom : Geometry
            The body (for wake origin / orientation).
        u_inf, v_inf : float
            Freestream components.
        chord : float
            Reference length.
        reynolds : float
            Chord Reynolds number (sets boundary-layer / wake thickness).
        cell : float
            Representative grid spacing; the boundary-layer thickness is floored
            to a few cells so the no-slip ramp is resolved on the grid (an
            unresolved ramp would manifest as a large, spurious ``div(u)``).

        Returns
        -------
        tuple of numpy.ndarray
            ``(u_visc, v_visc, nut)`` each ``(ny, nx)``.
        """
        u_mag = float(np.hypot(u_inf, v_inf)) + 1e-12
        re = max(float(reynolds), 1.0)

        # Boundary-layer thickness ~ c / sqrt(Re) (turbulent: a touch thicker),
        # floored to ~2.5 cells so the wall ramp is always grid-resolved.
        delta = chord * (5.0 / np.sqrt(re)) ** 0.5
        delta = float(np.clip(delta, 0.01 * chord, 0.12 * chord))
        delta = max(delta, 2.5 * float(cell))

        d = np.maximum(sdf, 0.0)  # distance into the fluid from the surface
        # Smooth no-slip ramp: 0 at the wall, ->1 beyond ~delta.
        wall_ramp = 1.0 - np.exp(-(d / delta) ** 1.4)
        u_visc = u * wall_ramp
        v_visc = v * wall_ramp

        # --- Wake deficit: Gaussian transverse profile aligned with freestream.
        te_x = geom.bounding_box()[1]
        ys = np.asarray(geom.surface_points, dtype=np.float64)[:, 1]
        te_y = float(np.mean(ys))
        # Streamwise / transverse coordinates relative to the trailing edge.
        ex, ey = u_inf / u_mag, v_inf / u_mag      # streamwise unit vector
        nx_, ny_ = -ey, ex                          # transverse unit vector
        s = (X - te_x) * ex + (Y - te_y) * ey       # downstream distance
        n = (X - te_x) * nx_ + (Y - te_y) * ny_     # lateral offset

        downstream = s > 0.0
        # Wake half-width grows with sqrt(distance); deficit decays with distance.
        half_width = delta * (1.0 + 2.5 * np.sqrt(np.maximum(s, 0.0) / chord))
        half_width = np.maximum(half_width, 1e-3)
        deficit_amp = 0.45 * np.exp(-np.maximum(s, 0.0) / (3.0 * chord))
        wake = deficit_amp * np.exp(-0.5 * (n / half_width) ** 2)
        wake = np.where(downstream, wake, 0.0)
        u_visc = u_visc * (1.0 - wake)
        v_visc = v_visc * (1.0 - wake)

        # --- Turbulent viscosity: concentrated in BL + wake, scaled by nu_t ~ U*delta.
        nut_scale = 0.02 * u_mag * delta
        bl_band = np.exp(-(d / (2.0 * delta)) ** 2)              # near-wall band
        nut = nut_scale * (bl_band + 1.6 * wake)
        nut = np.maximum(nut, 0.0).astype(DTYPE)

        return u_visc.astype(DTYPE), v_visc.astype(DTYPE), nut

    @staticmethod
    def _smooth(f: np.ndarray, passes: int = 1) -> np.ndarray:
        """Light separable 1-2-1 (binomial) blur, edge-replicating.

        Real RANS fields are diffused and smooth; a couple of binomial passes
        remove the residual grid-scale kinks left by the algebraic boundary
        layer (which dominate the discrete divergence) without changing the
        bulk flow. Pure NumPy — no SciPy dependency.

        Parameters
        ----------
        f : numpy.ndarray
            ``(ny, nx)`` field.
        passes : int, optional
            Number of smoothing passes.

        Returns
        -------
        numpy.ndarray
            The smoothed field, same shape and dtype.
        """
        out = f.astype(np.float64)
        for _ in range(int(passes)):
            # x-direction
            left = np.empty_like(out)
            right = np.empty_like(out)
            left[:, 1:] = out[:, :-1]
            left[:, 0] = out[:, 0]
            right[:, :-1] = out[:, 1:]
            right[:, -1] = out[:, -1]
            out = 0.25 * left + 0.5 * out + 0.25 * right
            # y-direction
            up = np.empty_like(out)
            down = np.empty_like(out)
            up[1:, :] = out[:-1, :]
            up[0, :] = out[0, :]
            down[:-1, :] = out[1:, :]
            down[-1, :] = out[-1, :]
            out = 0.25 * up + 0.5 * out + 0.25 * down
        return out.astype(f.dtype)

    # ------------------------------------------------------------------ #
    # Solve
    # ------------------------------------------------------------------ #

    def solve(self, case: FlowCase) -> FlowField:
        """Generate the analytic pseudo-RANS field for ``case``.

        Parameters
        ----------
        case : FlowCase
            The problem statement.

        Returns
        -------
        FlowField
            Fields ``u, v, p, nut`` with ``mask`` and ``sdf`` populated. Inside
            the solid, velocity and ``nut`` are exactly zero.
        """
        domain = case.domain
        geom = case.geometry
        X, Y = domain.grid()
        Xd = X.astype(np.float64)
        Yd = Y.astype(np.float64)

        u_inf, v_inf = case.bc.inlet_vector()
        chord = max(geom.chord(), 1e-9)

        sdf = signed_distance(geom, domain)
        mask = solid_mask(geom, domain)

        # 1) inviscid potential core. Desingularise at ~1.5 cell diagonals so
        #    near-wall kernel spikes are smoothed to grid scale (keeps div small).
        eps = 1.5 * float(np.hypot(domain.dx, domain.dy))
        u_pot, v_pot = self._potential_velocity(Xd, Yd, geom, u_inf, v_inf, chord, eps)

        # 2) viscous boundary layer + wake + turbulent viscosity.
        cell = 0.5 * (domain.dx + domain.dy)
        u, v, nut = self._viscous_correction(
            u_pot, v_pot, sdf.astype(np.float64), Xd, Yd, geom,
            u_inf, v_inf, chord, case.bc.reynolds, cell,
        )

        # Clip non-physical overspeed (panel singularities near the wall).
        u_mag = float(np.hypot(u_inf, v_inf)) + 1e-12
        speed_cap = 2.5 * u_mag
        spd = np.hypot(u, v)
        over = spd > speed_cap
        scale = np.ones_like(spd)
        scale[over] = speed_cap / spd[over]
        u = u * scale
        v = v * scale

        # Diffuse to a RANS-like smoothness (removes grid-scale kinks; lowers
        # the discrete divergence) while leaving the bulk flow intact.
        u = self._smooth(u, passes=2)
        v = self._smooth(v, passes=2)

        # 3) kinematic Bernoulli pressure: p = 0.5 (U_inf^2 - |u|^2), p_inf = 0.
        speed2 = u * u + v * v
        p = 0.5 * (u_mag * u_mag - speed2)

        # 4) enforce solid interior (no-slip, zero eddy viscosity).
        solid = mask < 0.5
        u = np.where(solid, 0.0, u)
        v = np.where(solid, 0.0, v)
        nut = np.where(solid, 0.0, nut)
        # Stagnation pressure inside the body (kinematic), a mild, smooth value.
        p = np.where(solid, 0.5 * u_mag * u_mag, p)

        return FlowField(
            domain=domain,
            u=u.astype(DTYPE),
            v=v.astype(DTYPE),
            p=p.astype(DTYPE),
            nut=nut.astype(DTYPE),
            mask=mask.astype(DTYPE),
            sdf=sdf.astype(DTYPE),
            meta={"synthetic": True, "case": case.name},
        )

    # ------------------------------------------------------------------ #
    # Dataset generation
    # ------------------------------------------------------------------ #

    def generate(self, n: int) -> list[tuple[FlowCase, FlowField]]:
        """Generate ``n`` (case, field) pairs.

        Parameters
        ----------
        n : int
            Number of samples.

        Returns
        -------
        list of tuple
            ``[(FlowCase, FlowField), ...]`` of length ``n``.
        """
        pairs: list[tuple[FlowCase, FlowField]] = []
        for i in range(int(n)):
            case = self.sample_case(i)
            field = self.solve(case)
            pairs.append((case, field))
        return pairs
