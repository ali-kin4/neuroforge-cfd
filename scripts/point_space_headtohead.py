"""The point-space head-to-head: Transolver vs parameter interpolation at the
NATIVE AirfRANS cloud nodes, with NO rasterisation anywhere in the comparison path.

WHY THIS RUN EXISTS
-------------------
``docs/paper/review/r2_round2.md`` scores one objection fatal (section 1, section 6): the
headline ``8.4x`` is a *measure choice* and the manuscript cannot distinguish it
from a finding. ``docs/paper/review/measure_asymmetry.md`` then measured the
weighting half of that objection and CONCEDED it (D3 -> ``ARTIFACT``: re-weighting
the identical grid errors by AirfRANS's own node measure takes ``mse_p`` from
``8.39x`` in the interpolator's favour to ``0.440x``). It also measured the half a
re-weighting cannot reach: at the native nodes the r128 raster's OWN round-trip
error exceeds Transolver's per-node error by ``418x`` pooled and ``495x`` inside
``0.005c``. So no grid-scored number -- for either method -- adjudicates
boundary-layer skill.

This script runs the experiment both documents name as the single most valuable
follow-up, and which the manuscript's limitations bullet wrongly says is
impossible. That bullet claims the point-space head-to-head needs a
``PointNormalizer`` "the checkpoints do not store". **That is false.**
``checkpoints/v2_transolver/seed{m}.pt`` carries ``point_norm`` AND ``grid_norm``,
and ``scripts/recompute_force_vs_official.py:134-158`` has loaded both since June.

THE CONSTRUCTION, AND WHY IT IS THE FAIREST AVAILABLE
-----------------------------------------------------
Transolver at native nodes is direct: load the deployed checkpoint with its own
``point_norm`` and infer on the cloud. That arm needs no construction at all.

The interpolator does. Its published definition is a weight matrix over the 800
train cases applied to the *nondimensional* field,
``pred_t = sum_j W[t,j] * fhat_j``, redimensionalised with the TEST case's own
``(U, alpha)``. The weights ``W`` are a function of the seven name-parsed scalars
only -- they are identical here and in ``interp_full.json``; nothing about them is
grid-specific. What IS grid-specific in the published run is the *representation of
``fhat_j``*: a 128^2 raster. At native nodes the same estimator can be evaluated
without any raster:

    pred_t(x) = sum_j W[t,j] * fhat_j(x),  x = a native node of test case t,

where ``fhat_j(x)`` is the linear (Delaunay barycentric) interpolation of TRAIN
CASE j's OWN nondimensional field on TRAIN CASE j's OWN native cloud, evaluated at
``x``. This is the exact same interpolant ``rasterize_point_cloud`` and
``airfrans_loader._sim_to_pair`` use to build every grid field in this project, so
no new numerical operator is introduced.

Why this is the fairest construction, stated as four properties:

  (1) **It has no representation ceiling.** Every train field is evaluated on its
      own native cloud, at the resolution AirfRANS actually simulated. This is the
      decisive contrast with the r128 protocol, whose own round-trip error is
      418x Transolver's per-node error (Block D of ``measure_asymmetry.md``).
      Nothing in this comparison path discards sub-grid structure from EITHER arm.
  (2) **It is the estimator the paper published, not a new one.** The weight matrix
      is rebuilt by the published ``build_weights`` from the published
      CV-selected config, read out of ``interp_full.json`` and never re-selected
      (gate G1). Only the field representation changes -- from a raster to the
      native cloud -- and it changes in the interpolator's favour.
  (3) **It cannot be accused of down-sampling the baseline.** The alternative --
      bilinearly sampling the interpolator's r128 output at the native nodes --
      would hand it the 418x raster ceiling as a floor and is NOT run as the
      primary arm for that reason. It is reported as arm ``r128_resample`` purely
      to quantify what the grid protocol costs.
  (4) **The one genuinely ambiguous choice is disclosed and resolved in the
      interpolator's favour.** A test node lying at 4e-5c off a THIN test airfoil
      can fall INSIDE a thicker train airfoil's body, where train case j has no
      nodes. Two conventions exist and both are run:
        ``bridge``  -- the Delaunay triangulation spans the body (the "Delaunay
                       bridge" ``interpolation_baseline.md`` section 6(b) already
                       studies), giving a smooth linear extension;
        ``nearfill`` -- the value at train case j's nearest cloud node, the point
                       analogue of the published ``--fill nearest``.
      P1 reads the BETTER of the two per channel (see the pre-registration).

WHAT THIS CONSTRUCTION COSTS THE INTERPOLATOR, STATED IN ADVANCE
-----------------------------------------------------------------
Physical coordinates. A test node at wall distance ``d_t`` sits at wall distance
``d_j != d_t`` under train case j's geometry, because the airfoils differ. Inside a
boundary layer that is resolved to ``4.4e-5 c`` this mismatch is not a rounding
error: it samples a different part of the profile. The 0.0234c raster hid this by
averaging over it. We therefore MEASURE the mismatch directly (``geom_mismatch``:
``|d_j - d_t|`` against ``d_t``, per band, |W|-weighted over j) so the mechanism is
a number and not an argument, and we bound its consequence with an ORACLE (P3)
that is handed the test answer.

We deliberately do NOT build a body-fitted or wall-aligned variant of the
interpolator. Such an arm would introduce a blend length scale and an arc-length
correspondence with no published provenance -- a straw steelman a reviewer would
reopen. P3's oracle does that job with no free parameters: if the single
best-matching training field IN THE WHOLE TRAINING SET, selected by an oracle that
is shown the test answer, still cannot approach Transolver near the wall, then no
choice of weights over those fields can, and the near-wall result is a property of
the representation rather than of the estimator or its hyperparameters.

PRE-REGISTERED DECISION RULES -- written and committed BEFORE any number existed
--------------------------------------------------------------------------------
All errors below are PER NODE squared error against the native AirfRANS targets,
pooled over every native node of all 200 ``full`` test cases, in physical units.
Transolver = the mean over the three deployed seeds of the per-seed pooled MSE.
Interpolator = the better (lower MSE) of ``bridge`` and ``nearfill``, chosen
independently per channel: benefit of the doubt to the interpolator.

    R_c = MSE_interp(c) / MSE_transolver(c)      ( > 1 means Transolver is better )

P1 -- IS THE INTERPOLATOR COMPETITIVE AT NATIVE RESOLUTION?  [PRIMARY]
      Read on the three volume channels u, v, p, with the SAME 2x threshold that
      ``0dcde75`` pre-registered for the grid comparison (``COMPETITIVE`` there
      required within-2x on all three volume channels):
        R_c <= 2 for ALL of u, v, p   -> COMPETITIVE-AT-NATIVE. The interpolation
              finding survives the strongest test available and the paper is far
              stronger than it is today; the headline should be restated in the
              node measure and kept.
        R_c  > 2 for ALL of u, v, p   -> NOT-COMPETITIVE-AT-NATIVE. The
              interpolation headline is dead in every measure. The surviving
              claims are the localisation thesis and the measure-dependence
              finding, and they must be stated without hedging.
        otherwise                     -> MIXED-AT-NATIVE. Report per channel; no
              aggregate headline may be quoted in either direction.
      ``nut`` is reported and may not be substituted for any of u, v, p.
      The standardised mean over u, v, p (paper's ``Var_train`` divisors
      u 341.88 / v 52.45 / p 135590) is reported alongside; it is NOT the verdict.

P2 -- DOES THE LOCALISATION THESIS SURVIVE WITH THE RASTER REMOVED?
      r_b = MSE_interp(b) / MSE_transolver(b) on u over BANDS7 (cloud sdf, the
      identical band edges as ``tab:interp_bands`` and Block B). S = r_first/r_last.
        S >= 10 and r_b decreasing outward (at most one inversion)
                  -> CONFIRMED AT NATIVE RESOLUTION;
        S < 3     -> NOT CONFIRMED: the grid D2 verdict was a rasterisation
                     artifact and section 7.5 of ``measure_asymmetry.md`` must be
                     withdrawn;
        otherwise -> PARTIAL.

P3 -- IS THE NEAR-WALL RESULT THE WEIGHTS, OR THE COORDINATES?
      For each test case t and band b let ``O(t,b) = min_j MSE(fhat_j -> t, b)``
      on u -- the single best training field out of 800, chosen with knowledge of
      the test answer. Let ``O_b`` be its case-mean, and
      ``Q_b = O_b / MSE_transolver(b)``.
        Q_b > 10 in band 0-0.005c -> NO-PARAMETER-COMBINATION: an oracle handed
              the answer cannot approach Transolver near the wall from these 800
              fields in physical coordinates. The near-wall gap is representational
              and cannot be closed by better weights, a better kernel, or more
              training cases.
        Q_b <= 1 in band 0-0.005c -> WEIGHTS-BOUND: a better parameter-space
              estimator could close the gap and the P1 verdict must be scoped to
              "this estimator" rather than "parameter interpolation".
        otherwise -> PARTIAL.

P4 -- INCONCLUSIVE TRIGGERS, declared before the run. A band is flagged
      INCONCLUSIVE and is excluded from P1/P2 if either:
        (a) the |W|-weighted fraction of that band's node queries falling OUTSIDE
            a train cloud's convex hull exceeds 0.20 -- the construction would be
            extrapolating rather than interpolating; or
        (b) | MSE_bridge(b) - MSE_nearfill(b) | exceeds
            | MSE_interp_best(b) - MSE_transolver(b) | -- i.e. the verdict in that
            band would be decided by an arbitrary in-body convention rather than
            by the two methods.
      If P1 cannot be read on the pooled aggregate because a band carrying more
      than 20% of the nodes is INCONCLUSIVE, the overall verdict is INCONCLUSIVE
      and the blocking quantity is named.

GATES -- the run aborts if any fails
------------------------------------
G1  The weight matrix is the PUBLISHED one. ``build_weights`` is called with the
    config read from ``results/interpolation/interp_full.json`` (never
    re-selected) on the same 800 train / 200 test names in manifest order, and the
    resulting GRID prediction ``W @ Y`` reproduces that file's
    mse_u / mse_v / mse_p to < 1e-7 relative.

    AMENDMENT 1 (recorded; gate tolerance only, no decision rule touched). The
    threshold as first committed in ``a54cb75`` was < 1e-9, copied from
    ``measure_asymmetry.py`` G1. That script reaches 0.00e+00 because it reuses
    the published per-case metric path verbatim; this script rebuilds the same
    ``W @ Y`` and computes the same per-case MSE inline, so the float32 GEMM over
    ``800 x 65536`` terms and the reduction associate differently. Measured:
    ``mse_v`` 2.371e-09 (``0.03361523297`` vs ``0.03361523305``) and ``mse_p``
    1.624e-08 (``75.03145160`` vs ``75.03145038``) -- agreement to eight and nine
    significant figures. The tolerance is raised to 1e-7, all three measured
    values are logged by the gate and stored in the artifact, and P1-P4 and their
    thresholds are untouched.
G2  The Transolver arm is the DEPLOYED backbone. Its per-node, per-band pooled MSE
    reproduces ``results/interpolation/measure_asymmetry.json``
    ``D_node_space.mse_full`` (same 200 cases, same band edges, same checkpoints)
    to < 1e-9 relative.
G3  CONSTRUCTION VALIDITY, reported not thresholded (declared in advance). The
    point-space interpolator prediction, rasterised back to r128 by the identical
    ``rasterize_point_cloud`` call the Transolver arm uses and scored by the
    identical fluid mask, is compared to the published per-case grid values in
    ``measure_asymmetry.json:per_case_area_uniform["interp"]``. These are NOT
    expected to agree to machine precision: the published arm rasterises then
    combines with ``--fill nearest`` ON THE RASTER; this arm combines then
    rasterises, across clouds with different triangulations. The reading is fixed
    in advance: if the point-space construction scores BETTER on the grid measure
    than the published grid construction, it is a strictly stronger interpolator
    and any adverse P1 verdict is conservative; if it scores far worse (> 3x on
    any of u, v, p) the construction is treated as buggy and the run is stopped
    pending diagnosis.
G4  Band tables reconstruct the pooled total: ``sum_b n_b * MSE_b == pooled SE``
    to < 1e-10 relative, every arm, every channel.
G5  The worker partition over test cases is disjoint and its union is exactly the
    200 cached test names, in cache order.

COST / SAFETY.  CPU-only for the interpolator arm (scipy, single-BLAS-thread per
the package cap, process-parallel over test cases). One short GPU inference pass
for the Transolver arm, identical in kind to ``measure_asymmetry.py --stage full``.
No training. Nothing is written to ``results/mgn``, ``checkpoints/mgn`` or
``mgn_run.log``.

RUN
    .venv/Scripts/python.exe scripts/point_space_headtohead.py --stage cache
    .venv/Scripts/python.exe scripts/point_space_headtohead.py --stage pilot
    .venv/Scripts/python.exe scripts/point_space_headtohead.py --stage interp
    .venv/Scripts/python.exe scripts/point_space_headtohead.py --stage transolver
    .venv/Scripts/python.exe scripts/point_space_headtohead.py --stage reduce
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time

import neuroforge  # noqa: F401  -- MUST precede numpy/torch (BLAS thread caps)

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from parameter_interpolation_baseline import (  # noqa: E402
    CHORD,
    build_weights,
    case_U_alpha,
    case_features,
)

INF = 1e9
BANDS7 = [(0.0, 0.005), (0.005, 0.01), (0.01, 0.02), (0.02, 0.05),
          (0.05, 0.15), (0.15, 0.50), (0.50, INF)]
NAMES7 = ["0-0.005c", "0.005-0.01c", "0.01-0.02c", "0.02-0.05c",
          "0.05-0.15c", "0.15-0.5c", ">0.5c"]
NAMES8 = ["wall"] + NAMES7                       # band 0 = nodes with sdf == 0
CHANS = ("u", "v", "p", "nut")
NB = len(NAMES8)

#: the paper's train-set per-channel variance (a common divisor; ratios unaffected)
VAR_TRAIN = {"u": 341.88, "v": 52.45, "p": 135590.0, "nut": 1.3199e-6}

DEF_SCRATCH = os.environ.get(
    "NF_PSH_SCRATCH",
    os.path.join(os.environ.get("TEMP", "/tmp"), "nf_point_space_headtohead"),
)


def log(msg: str) -> None:
    print(f"[psh] {msg}", flush=True)


# --------------------------------------------------------------------------- #
# Bands on the cloud's own sdf (AirfRANS column 4). Band 0 is the wall itself.
# Bands 1..7 are BANDS7, byte-identical edges to measure_asymmetry / tab:interp_bands.
# --------------------------------------------------------------------------- #
def band_index8(sdf: np.ndarray) -> np.ndarray:
    out = np.full(sdf.shape, -1, np.int64)
    out[sdf <= 0.0] = 0
    for k, (lo, hi) in enumerate(BANDS7):
        out[(sdf > lo) & (sdf <= hi)] = k + 1
    return out


# --------------------------------------------------------------------------- #
# Compact per-case caches (scratch), so 8 workers do not each re-read .vtu files.
# --------------------------------------------------------------------------- #
def _train_cache_path(scratch: str, name: str) -> str:
    return os.path.join(scratch, "train", f"{name}.npz")


def _test_cache_path(scratch: str, name: str) -> str:
    return os.path.join(scratch, "test", f"{name}.npz")


def build_train_cache(names: list[str], data_root: str, scratch: str) -> None:
    """One compact ``.npz`` per TRAIN case: pos, nondimensional targets, sdf, airfoil."""
    import airfrans.simulation as afsim

    os.makedirs(os.path.join(scratch, "train"), exist_ok=True)
    t0 = time.time()
    todo = [n for n in names if not os.path.exists(_train_cache_path(scratch, n))]
    log(f"train cache: {len(names) - len(todo)} present, {len(todo)} to build")
    for i, nm in enumerate(todo):
        sim = afsim.Simulation(root=data_root, name=nm)
        U, al = case_U_alpha(nm)
        ar = np.radians(al)
        tg = np.concatenate([np.asarray(sim.velocity, np.float64),
                             np.asarray(sim.pressure, np.float64),
                             np.asarray(sim.nu_t, np.float64)], axis=1)
        # the PUBLISHED `nd` representation, using the TRAIN case's own (U, alpha)
        tg[:, 0] = (tg[:, 0] - U * np.cos(ar)) / U
        tg[:, 1] = (tg[:, 1] - U * np.sin(ar)) / U
        tg[:, 2] = tg[:, 2] / U ** 2
        tg[:, 3] = tg[:, 3] / (U * CHORD)
        np.savez(
            _train_cache_path(scratch, nm),
            pos=np.asarray(sim.position, np.float32),
            yhat=tg.astype(np.float32),
            sdf=np.asarray(sim.sdf, np.float32).ravel(),
            apos=np.asarray(sim.airfoil_position, np.float32),
            anrm=np.asarray(sim.airfoil_normals, np.float32),
        )
        if (i + 1) % 50 == 0:
            log(f"train cache {i+1}/{len(todo)} ({time.time()-t0:.0f}s)")
    log(f"train cache ready ({time.time()-t0:.0f}s)")


def build_test_cache(pcs, pairs, scratch: str) -> None:
    """One compact ``.npz`` per TEST case: pos, raw targets, sdf, crop mask."""
    os.makedirs(os.path.join(scratch, "test"), exist_ok=True)
    by = {pc.name: pc for pc in pcs}
    for case, _ref in pairs:
        p = _test_cache_path(scratch, case.name)
        if os.path.exists(p):
            continue
        pc = by[case.name]
        pos = np.asarray(pc.pos, np.float64)
        xmin, xmax, ymin, ymax = case.domain.bounds
        incrop = ((pos[:, 0] >= xmin) & (pos[:, 0] <= xmax)
                  & (pos[:, 1] >= ymin) & (pos[:, 1] <= ymax))
        np.savez(p,
                 pos=pos.astype(np.float32),
                 tgt=np.asarray(pc.targets, np.float32),
                 sdf=np.asarray(pc.features[:, 4], np.float32),
                 incrop=incrop)
    log(f"test cache ready ({len(pairs)} cases)")


def load_test_case(scratch: str, name: str):
    z = np.load(_test_cache_path(scratch, name))
    return (np.asarray(z["pos"], np.float64), np.asarray(z["tgt"], np.float64),
            np.asarray(z["sdf"], np.float64), np.asarray(z["incrop"], bool))


def load_train_case(scratch: str, name: str):
    z = np.load(_train_cache_path(scratch, name))
    return (np.asarray(z["pos"], np.float64), np.asarray(z["yhat"], np.float64),
            np.asarray(z["apos"], np.float64), np.asarray(z["anrm"], np.float64))


# --------------------------------------------------------------------------- #
# Band accumulator over NODES (no cells, no raster anywhere).
# --------------------------------------------------------------------------- #
class NodeAcc:
    """Per-band squared error over native nodes, for one arm."""

    def __init__(self) -> None:
        self.n = np.zeros(NB)
        self.n_crop = np.zeros(NB)
        self.se = {c: np.zeros(NB) for c in CHANS}
        self.se_crop = {c: np.zeros(NB) for c in CHANS}

    def add(self, b: np.ndarray, incrop: np.ndarray, pred: np.ndarray, tgt: np.ndarray) -> None:
        keep = b >= 0
        bk = b[keep]
        self.n += np.bincount(bk, minlength=NB)
        bc = b[keep & incrop]
        self.n_crop += np.bincount(bc, minlength=NB)
        for j, c in enumerate(CHANS):
            e2 = (pred[:, j] - tgt[:, j]) ** 2
            self.se[c] += np.bincount(bk, weights=e2[keep], minlength=NB)
            self.se_crop[c] += np.bincount(bc, weights=e2[keep & incrop], minlength=NB)

    def merge(self, other: "NodeAcc") -> None:
        self.n += other.n
        self.n_crop += other.n_crop
        for c in CHANS:
            self.se[c] += other.se[c]
            self.se_crop[c] += other.se_crop[c]

    def to_dict(self) -> dict:
        return {"n": self.n.tolist(), "n_crop": self.n_crop.tolist(),
                "se": {c: self.se[c].tolist() for c in CHANS},
                "se_crop": {c: self.se_crop[c].tolist() for c in CHANS}}

    @staticmethod
    def from_dict(d: dict) -> "NodeAcc":
        a = NodeAcc()
        a.n = np.asarray(d["n"], np.float64)
        a.n_crop = np.asarray(d["n_crop"], np.float64)
        for c in CHANS:
            a.se[c] = np.asarray(d["se"][c], np.float64)
            a.se_crop[c] = np.asarray(d["se_crop"][c], np.float64)
        return a


def redim_nodes(vals: np.ndarray, U: float, alpha_deg: float) -> np.ndarray:
    """Invert the ``nd`` transform with the QUERY case's own (U, alpha)."""
    ar = np.radians(alpha_deg)
    out = np.empty_like(vals)
    out[:, 0] = vals[:, 0] * U + U * np.cos(ar)
    out[:, 1] = vals[:, 1] * U + U * np.sin(ar)
    out[:, 2] = vals[:, 2] * U ** 2
    out[:, 3] = vals[:, 3] * (U * CHORD)
    return out


# --------------------------------------------------------------------------- #
# The worker: stream TRAIN clouds, accumulate sum_j W[t,j] * fhat_j(X_t).
# --------------------------------------------------------------------------- #
def run_chunk(args: tuple) -> dict:
    """One chunk of (test cases) x (train cases).

    Returns per-test-case band accumulators, oracle statistics and diagnostics.
    With ``return_pred`` the full nondimensional prediction sums are returned too
    (used by ``--stage pilot``, where the chunking is over TRAIN cases).
    """
    import neuroforge  # noqa: F401
    import numpy as np
    from scipy.interpolate import LinearNDInterpolator
    from scipy.spatial import Delaunay, cKDTree

    (scratch, test_names, train_names, Wsub, return_pred, kd_workers, tag) = args

    nt = len(test_names)
    T = [load_test_case(scratch, nm) for nm in test_names]
    bands = [band_index8(t[2]) for t in T]
    accum = [np.zeros((t[0].shape[0], 4)) for t in T]        # bridge, nd units
    accum_nf = [np.zeros((t[0].shape[0], 4)) for t in T]     # nearest-fill, nd units

    # oracle + diagnostics, per test case
    ora_se = [np.full((NB, 4), np.inf) for _ in T]           # min over j of band SE
    ora_arg = [np.full((NB, 4), -1, np.int64) for _ in T]
    w_out = [np.zeros(NB) for _ in T]      # |W|-weighted outside-hull node mass
    w_in = [np.zeros(NB) for _ in T]       # |W|-weighted inside-train-body node mass
    w_tot = [np.zeros(NB) for _ in T]      # |W|-weighted total node mass
    gm_sum = [np.zeros(NB) for _ in T]     # |W|-weighted sum of |d_j - d_t|
    gm_dt = [np.zeros(NB) for _ in T]      # |W|-weighted sum of d_t

    t_start = time.time()
    for jj, jname in enumerate(train_names):
        pos_j, yhat_j, apos_j, anrm_j = load_train_case(scratch, jname)
        tri = Delaunay(pos_j)
        lin = LinearNDInterpolator(tri, yhat_j, fill_value=np.nan)
        kt_pos = cKDTree(pos_j)
        kt_surf = cKDTree(apos_j)

        for ti, nm in enumerate(test_names):
            w = float(Wsub[ti, jj])
            pos_t, tgt_t, sdf_t, incrop_t = T[ti]
            b = bands[ti]
            keep = b >= 0

            vals = np.asarray(lin(pos_t), np.float64)         # (M, 4), nan outside hull
            out = np.isnan(vals[:, 0])
            if out.any():                                     # outside hull -> nearest node
                _d, idx = kt_pos.query(pos_t[out], k=1, workers=kd_workers)
                vals[out] = yhat_j[idx]

            # wall distance under TRAIN case j's geometry + the in-body test
            dj, isurf = kt_surf.query(pos_t, k=1, workers=kd_workers)
            # AirfRANS airfoil normals point INTO the body: (q - s).n > 0 <=> inside
            inbody = np.einsum("ij,ij->i", pos_t - apos_j[isurf], anrm_j[isurf]) > 0.0

            vals_nf = vals
            if inbody.any():
                vals_nf = vals.copy()
                _d2, idx2 = kt_pos.query(pos_t[inbody], k=1, workers=kd_workers)
                vals_nf[inbody] = yhat_j[idx2]

            accum[ti] += w * vals
            accum_nf[ti] += w * vals_nf

            aw = abs(w)
            bk = b[keep]
            w_tot[ti] += aw * np.bincount(bk, minlength=NB)
            w_out[ti] += aw * np.bincount(bk, weights=out[keep].astype(np.float64), minlength=NB)
            w_in[ti] += aw * np.bincount(bk, weights=inbody[keep].astype(np.float64), minlength=NB)
            gm_sum[ti] += aw * np.bincount(
                bk, weights=np.abs(dj[keep] - sdf_t[keep]), minlength=NB)
            gm_dt[ti] += aw * np.bincount(bk, weights=sdf_t[keep], minlength=NB)

            # P3 oracle: this SINGLE train field, redimensionalised to case t
            U_t, al_t = case_U_alpha(nm)
            single = redim_nodes(vals, U_t, al_t)
            nb_ = np.bincount(bk, minlength=NB)
            for ci in range(4):
                e2 = (single[:, ci] - tgt_t[:, ci]) ** 2
                se = np.bincount(bk, weights=e2[keep], minlength=NB)
                with np.errstate(invalid="ignore", divide="ignore"):
                    mse = np.where(nb_ > 0, se / np.maximum(nb_, 1), np.inf)
                better = mse < ora_se[ti][:, ci]
                ora_se[ti][better, ci] = mse[better]
                ora_arg[ti][better, ci] = jj

        if (jj + 1) % 25 == 0:
            el = time.time() - t_start
            log(f"[{tag}] train {jj+1}/{len(train_names)}  {el:.0f}s "
                f"({el/(jj+1):.2f}s/train, {el/(jj+1)/max(nt,1):.3f}s/train/test)")

    # ---- finalise ---------------------------------------------------------
    out: dict = {"test_names": list(test_names), "per_case": [], "acc": {}}
    acc_bridge, acc_nf = NodeAcc(), NodeAcc()
    preds = {}
    for ti, nm in enumerate(test_names):
        pos_t, tgt_t, sdf_t, incrop_t = T[ti]
        b = bands[ti]
        U_t, al_t = case_U_alpha(nm)
        pb = redim_nodes(accum[ti], U_t, al_t)
        pn = redim_nodes(accum_nf[ti], U_t, al_t)
        if return_pred:
            preds[nm] = {"bridge": accum[ti].astype(np.float32),
                         "nearfill": accum_nf[ti].astype(np.float32)}
        acc_bridge.add(b, incrop_t, pb, tgt_t)
        acc_nf.add(b, incrop_t, pn, tgt_t)
        keep = b >= 0
        row = {
            "name": nm, "n_nodes": int(pos_t.shape[0]), "n_banded": int(keep.sum()),
            "mse_bridge": {c: float(np.mean((pb[:, i] - tgt_t[:, i]) ** 2))
                           for i, c in enumerate(CHANS)},
            "mse_nearfill": {c: float(np.mean((pn[:, i] - tgt_t[:, i]) ** 2))
                             for i, c in enumerate(CHANS)},
            "oracle_band_mse": {c: ora_se[ti][:, i].tolist() for i, c in enumerate(CHANS)},
            "oracle_argmin": {c: ora_arg[ti][:, i].tolist() for i, c in enumerate(CHANS)},
            "w_frac_outside_hull": (w_out[ti] / np.maximum(w_tot[ti], 1e-30)).tolist(),
            "w_frac_inside_body": (w_in[ti] / np.maximum(w_tot[ti], 1e-30)).tolist(),
            "geom_mismatch_mean": (gm_sum[ti] / np.maximum(w_tot[ti], 1e-30)).tolist(),
            "d_t_mean": (gm_dt[ti] / np.maximum(w_tot[ti], 1e-30)).tolist(),
        }
        out["per_case"].append(row)
    out["acc"] = {"bridge": acc_bridge.to_dict(), "nearfill": acc_nf.to_dict()}
    out["wallclock_sec"] = time.time() - t_start
    if return_pred:
        out["_preds"] = preds
        out["_accum_nd"] = {nm: (accum[ti].astype(np.float32), accum_nf[ti].astype(np.float32))
                            for ti, nm in enumerate(test_names)}
    return out


# --------------------------------------------------------------------------- #
# Setup shared by every stage.
# --------------------------------------------------------------------------- #
def setup(a):
    """Names, published config, weight matrix, and gate G1."""
    man = json.load(open(os.path.join(a.data_root, "manifest.json"), encoding="utf-8"))
    names_tr = list(man[f"{a.task}_train"])[:a.n_train]
    names_te = list(man[f"{a.task}_test"])[:a.n_test]
    assert not (set(names_tr) & set(names_te)), "train/test name overlap"
    base = json.load(open(os.path.join(a.out_dir, f"interp_{a.task}.json"), encoding="utf-8"))
    cfg = base["variants"][a.rep]["cv_selected_cfg_raw"]
    Xtr = np.stack([case_features(n) for n in names_tr])
    Xte = np.stack([case_features(n) for n in names_te])
    W = build_weights(cfg, Xtr, Xte)
    return names_tr, names_te, cfg, W, base


def gate_g1(a, names_tr, names_te, cfg, W, base) -> dict:
    """The weight matrix IS the published one: W @ Y reproduces interp_full.json."""
    from parameter_interpolation_baseline import build_stack, load_pairs, redimensionalise

    tr = load_pairs(a.cache_dir, a.task, True, 128, a.n_train)
    te = load_pairs(a.cache_dir, a.task, False, 128, a.n_test)
    assert [c.name for c, _ in tr] == names_tr, "train cache order != manifest order"
    assert [c.name for c, _ in te] == names_te, "test cache order != manifest order"
    H, Wd = te[0][1].u.shape
    hw = H * Wd
    Y, _M, _ = build_stack(tr, a.rep, a.fill)
    pred = redimensionalise(W @ Y, names_te, a.rep, hw)
    got = {}
    for ci, c in enumerate(CHANS):
        vals = []
        for i, (case, ref) in enumerate(te):
            fluid = np.asarray(ref.mask).ravel() > 0.5
            g = np.asarray(getattr(ref, c), np.float64).ravel()
            vals.append(float(np.mean((pred[i, ci * hw:(ci + 1) * hw][fluid] - g[fluid]) ** 2)))
        got[f"mse_{c}"] = float(np.mean(vals))
    pub = {k: float(base["variants"][a.rep]["test_metrics"][k])
           for k in ("mse_u", "mse_v", "mse_p")}
    rel = {k: abs(got[k] - pub[k]) / pub[k] for k in pub}
    log(f"G1 relative reproduction error, all channels: "
        + ", ".join(f"{k} {v:.3e}" for k, v in rel.items()))
    for k, v in rel.items():
        assert v < 1e-7, f"G1 FAILED {k}: {got[k]} vs {pub[k]} (rel {v:.3e})"  # amendment 1
    log(f"G1 PASS  published weights reproduce interp_{a.task}.json: {pub} (rel {rel})")
    return {"got": got, "published": pub, "rel": rel}


# --------------------------------------------------------------------------- #
# Stages
# --------------------------------------------------------------------------- #
def stage_cache(a) -> None:
    from neuroforge.data.pointcloud import load_airfrans_pointclouds
    from parameter_interpolation_baseline import load_pairs

    names_tr, names_te, _cfg, _W, _b = setup(a)
    os.makedirs(a.scratch, exist_ok=True)
    missing_te = [n for n in names_te if not os.path.exists(_test_cache_path(a.scratch, n))]
    if missing_te:
        pairs = load_pairs(a.cache_dir, a.task, False, 128, a.n_test)
        pcs = load_airfrans_pointclouds(root="data", task=a.task, train=False,
                                        limit=a.n_test, cache_dir=a.cache_dir)
        build_test_cache(pcs, pairs, a.scratch)
        del pcs, pairs
    else:
        log(f"test cache ready ({len(names_te)} cases, all present)")
    build_train_cache(names_tr, a.data_root, a.scratch)


def _dispatch(chunks, n_proc, tag):
    if n_proc <= 1:
        return [run_chunk(c) for c in chunks]
    from concurrent.futures import ProcessPoolExecutor
    outs = []
    with ProcessPoolExecutor(max_workers=n_proc) as ex:
        for r in ex.map(run_chunk, chunks):
            outs.append(r)
            log(f"[{tag}] chunk done ({len(outs)}/{len(chunks)}, {r['wallclock_sec']:.0f}s)")
    return outs


def stage_pilot(a) -> dict:
    """G1 + G3: validate the construction on a few cases before the full sweep."""
    from neuroforge.data.rasterize import rasterize_point_cloud
    from parameter_interpolation_baseline import load_pairs

    names_tr, names_te, cfg, W, base = setup(a)
    g1 = gate_g1(a, names_tr, names_te, cfg, W, base)

    pte = names_te[:a.n_pilot]
    idx = [names_te.index(n) for n in pte]
    # chunk over TRAIN cases (few test cases -> partial sums are small)
    ntr = len(names_tr)
    bounds = np.linspace(0, ntr, a.n_proc + 1).astype(int)
    chunks = [(a.scratch, pte, names_tr[bounds[k]:bounds[k + 1]],
               W[np.ix_(idx, np.arange(bounds[k], bounds[k + 1]))],
               True, a.kd_workers, f"pilot{k}")
              for k in range(a.n_proc) if bounds[k + 1] > bounds[k]]
    t0 = time.time()
    outs = _dispatch(chunks, a.n_proc, "pilot")
    log(f"pilot chunks done ({time.time()-t0:.0f}s)")

    # sum the nondimensional partials, then score on r128 through the identical raster
    pairs = load_pairs(a.cache_dir, a.task, False, 128, a.n_test)
    by_case = {c.name: (c, r) for c, r in pairs}
    ma = json.load(open(os.path.join(a.out_dir, "measure_asymmetry.json"), encoding="utf-8"))
    # the published per-case rows are positional, in cache order
    pub_rows = ma["per_case_area_uniform"]["interp"]
    pub_names = [c.name for c, _ in pairs]

    rows = []
    for i, nm in enumerate(pte):
        tot = {k: None for k in ("bridge", "nearfill")}
        for o in outs:
            for k in ("bridge", "nearfill"):
                v = np.asarray(o["_preds"][nm][k], np.float64)
                tot[k] = v if tot[k] is None else tot[k] + v
        case, ref = by_case[nm]
        pos, tgt, sdf, incrop = load_test_case(a.scratch, nm)
        U_t, al_t = case_U_alpha(nm)
        H, Wd = ref.u.shape
        fluid = np.asarray(ref.mask) > 0.5
        row = {"name": nm, "published_grid": pub_rows[pub_names.index(nm)]}
        for k in ("bridge", "nearfill"):
            phys = redim_nodes(tot[k], U_t, al_t)
            row[f"node_mse_{k}"] = {c: float(np.mean((phys[:, j] - tgt[:, j]) ** 2))
                                    for j, c in enumerate(CHANS)}
            raster = rasterize_point_cloud(pos.astype(np.float32), phys.astype(np.float32),
                                           case.domain, fill=0.0, method="linear")
            gm = {}
            for j, c in enumerate(CHANS):
                pr = np.asarray(raster[j], np.float64)
                if c in ("u", "v", "nut"):
                    pr = np.where(fluid, pr, 0.0)
                if c == "nut":
                    pr = np.maximum(pr, 0.0)
                g = np.asarray(getattr(ref, c), np.float64)
                gm[f"mse_{c}"] = float(np.mean((pr[fluid] - g[fluid]) ** 2))
            row[f"grid_mse_{k}"] = gm
        rows.append(row)
        log(f"pilot {nm}: grid(bridge) {row['grid_mse_bridge']} | published {row['published_grid']}")

    # G3 reading, fixed in advance
    g3 = {}
    for c in ("u", "v", "p"):
        pub = float(np.mean([r["published_grid"][f"mse_{c}"] for r in rows]))
        got = float(np.mean([r["grid_mse_bridge"][f"mse_{c}"] for r in rows]))
        g3[f"mse_{c}"] = {"published_grid_arm": pub, "point_space_arm_rasterised": got,
                          "ratio": got / pub}
    log(f"G3 construction check (point-space arm, rasterised back to r128): {g3}")
    bad = [c for c in ("u", "v", "p") if g3[f"mse_{c}"]["ratio"] > 3.0]
    assert not bad, f"G3 FAILED: point-space construction >3x worse on {bad} -- stop and diagnose"

    out = {"meta": {"stage": "pilot", "n_pilot": a.n_pilot, "cfg": cfg,
                    "n_train": len(names_tr), "wallclock_sec": time.time() - t0},
           "G1": g1, "G3": g3, "rows": rows}
    os.makedirs(a.out_dir, exist_ok=True)
    write_json(os.path.join(a.out_dir, "point_space_pilot.json"), out)
    return out


def stage_interp(a) -> dict:
    names_tr, names_te, cfg, W, base = setup(a)
    # G5: disjoint cover of the 200 cached test names, in cache order
    bounds = np.linspace(0, len(names_te), a.n_proc + 1).astype(int)
    parts = [names_te[bounds[k]:bounds[k + 1]] for k in range(a.n_proc)]
    parts = [p for p in parts if p]
    flat = [n for p in parts for n in p]
    assert flat == names_te, "G5 FAILED: worker partition is not a disjoint cover in order"
    log(f"G5 PASS  {len(parts)} workers cover {len(names_te)} test cases disjointly, in order")

    chunks = []
    off = 0
    for k, p in enumerate(parts):
        idx = np.arange(off, off + len(p))
        chunks.append((a.scratch, p, names_tr, W[idx, :], False, a.kd_workers, f"w{k}"))
        off += len(p)
    t0 = time.time()
    outs = _dispatch(chunks, a.n_proc, "interp")
    dt = time.time() - t0

    acc = {k: NodeAcc() for k in ("bridge", "nearfill")}
    per_case = []
    for o in outs:
        for k in ("bridge", "nearfill"):
            acc[k].merge(NodeAcc.from_dict(o["acc"][k]))
        per_case.extend(o["per_case"])
    out = {"meta": {"stage": "interp", "task": a.task, "rep": a.rep, "cfg": cfg,
                    "n_train": len(names_tr), "n_test": len(names_te),
                    "n_proc": a.n_proc, "wallclock_sec": dt},
           "bands": NAMES8,
           "acc": {k: acc[k].to_dict() for k in acc},
           "per_case": per_case}
    write_json(os.path.join(a.out_dir, "point_space_interp.json"), out)
    log(f"interp stage done in {dt:.0f}s ({dt/60:.1f} min)")
    return out


def stage_transolver(a) -> dict:
    import torch
    from neuroforge.data.pointcloud import load_airfrans_pointclouds
    from recompute_force_vs_official import load_backbone

    _names_tr, names_te, _cfg, _W, _b = setup(a)
    device = torch.device(a.device if a.device not in ("auto", "") else
                          ("cuda" if torch.cuda.is_available() else "cpu"))
    log(f"transolver stage: device={device}")
    pcs = load_airfrans_pointclouds(root="data", task=a.task, train=False,
                                    limit=a.n_test, cache_dir=a.cache_dir)
    by = {pc.name: pc for pc in pcs}
    models = {}
    for s in a.seeds:
        m, pn, _gn, _nu = load_backbone(os.path.join(a.ckpt_dir, f"seed{s}.pt"), device)
        models[s] = (m, pn)
        log(f"loaded backbone seed{s}")

    accs = {f"seed{s}": NodeAcc() for s in a.seeds}
    per_case = {f"seed{s}": [] for s in a.seeds}
    t0 = time.time()
    for ci, nm in enumerate(names_te):
        pc = by[nm]
        pos, tgt, sdf, incrop = load_test_case(a.scratch, nm)
        b = band_index8(sdf)
        for s in a.seeds:
            m, pn = models[s]
            feats = pn.transform_in(pc.features)
            with torch.no_grad():
                y = m(torch.from_numpy(feats).to(device).unsqueeze(0)).squeeze(0)
                y = pn.inverse_out(y)
            pr = np.asarray(y.detach().cpu().numpy(), np.float64)
            accs[f"seed{s}"].add(b, incrop, pr, tgt)
            per_case[f"seed{s}"].append(
                {"name": nm, **{f"mse_{c}": float(np.mean((pr[:, j] - tgt[:, j]) ** 2))
                                for j, c in enumerate(CHANS)}})
        if (ci + 1) % 25 == 0:
            log(f"transolver {ci+1}/{len(names_te)} ({time.time()-t0:.0f}s)")
    dt = time.time() - t0

    # G2: reproduce measure_asymmetry Block D (bands 1..7, sdf > 0, full cloud)
    ma = json.load(open(os.path.join(a.out_dir, "measure_asymmetry.json"), encoding="utf-8"))
    D = ma["D_node_space"]
    g2 = {}
    for s in a.seeds:
        A = accs[f"seed{s}"]
        for c in CHANS:
            mine = (A.se[c][1:] / np.maximum(A.n[1:], 1e-30))
            pub = np.asarray(D["mse_full"][f"transolver_seed{s}"][c], np.float64)
            rel = float(np.max(np.abs(mine - pub) / np.maximum(np.abs(pub), 1e-30)))
            g2[f"seed{s}/{c}"] = rel
            assert rel < 1e-9, (f"G2 FAILED seed{s}/{c}: max rel {rel:.3e} vs "
                               f"measure_asymmetry D_node_space.mse_full")
        nrel = float(np.max(np.abs(A.n[1:] - np.asarray(D["n_nodes_full"])) /
                            np.maximum(np.asarray(D["n_nodes_full"]), 1.0)))
        assert nrel < 1e-12, f"G2 FAILED seed{s}: band node counts differ (rel {nrel:.3e})"
    log(f"G2 PASS  per-node band MSE reproduces measure_asymmetry D_node_space "
        f"(max rel {max(g2.values()):.2e})")

    out = {"meta": {"stage": "transolver", "seeds": list(a.seeds), "device": str(device),
                    "n_test": len(names_te), "wallclock_sec": dt},
           "bands": NAMES8, "G2": g2,
           "acc": {k: v.to_dict() for k, v in accs.items()},
           "per_case": per_case}
    write_json(os.path.join(a.out_dir, "point_space_transolver.json"), out)
    log(f"transolver stage done in {dt:.0f}s")
    return out


# --------------------------------------------------------------------------- #
# Reduce: apply P1-P4 exactly as pre-registered.
# --------------------------------------------------------------------------- #
def _mse(acc: dict, c: str, crop: bool = False):
    n = np.asarray(acc["n_crop" if crop else "n"], np.float64)
    se = np.asarray(acc["se_crop" if crop else "se"][c], np.float64)
    band = se / np.maximum(n, 1e-30)
    pooled = float(se.sum() / max(n.sum(), 1e-30))
    return band, pooled, n


def stage_reduce(a) -> dict:
    I = json.load(open(os.path.join(a.out_dir, "point_space_interp.json"), encoding="utf-8"))
    Tr = json.load(open(os.path.join(a.out_dir, "point_space_transolver.json"), encoding="utf-8"))
    seeds = Tr["meta"]["seeds"]

    res: dict = {"meta": {"interp": I["meta"], "transolver": Tr["meta"]},
                 "bands": NAMES8, "G2": Tr["G2"]}

    # --- pooled + band MSE, both arms, both crops --------------------------
    tab = {}
    for arm in ("bridge", "nearfill"):
        tab[f"interp_{arm}"] = {}
        for crop in (False, True):
            key = "in_crop" if crop else "full"
            tab[f"interp_{arm}"][key] = {}
            for c in CHANS:
                band, pooled, n = _mse(I["acc"][arm], c, crop)
                tab[f"interp_{arm}"][key][c] = {"band": band.tolist(), "pooled": pooled}
                tab[f"interp_{arm}"][key]["n"] = n.tolist()
    for s in seeds:
        k = f"seed{s}"
        tab[f"transolver_{k}"] = {}
        for crop in (False, True):
            key = "in_crop" if crop else "full"
            tab[f"transolver_{k}"][key] = {}
            for c in CHANS:
                band, pooled, n = _mse(Tr["acc"][k], c, crop)
                tab[f"transolver_{k}"][key][c] = {"band": band.tolist(), "pooled": pooled}
                tab[f"transolver_{k}"][key]["n"] = n.tolist()
    # Transolver = mean over seeds
    tso = {}
    for key in ("full", "in_crop"):
        tso[key] = {}
        for c in CHANS:
            tso[key][c] = {
                "band": np.mean([tab[f"transolver_seed{s}"][key][c]["band"] for s in seeds],
                                axis=0).tolist(),
                "pooled": float(np.mean([tab[f"transolver_seed{s}"][key][c]["pooled"]
                                         for s in seeds])),
            }
    tab["transolver_mean"] = tso
    res["tables"] = tab

    # --- G4: band tables reconstruct the pooled total -----------------------
    g4 = {}
    for armk, src in (("interp_bridge", I["acc"]["bridge"]), ("interp_nearfill", I["acc"]["nearfill"]),
                      *[(f"transolver_seed{s}", Tr["acc"][f"seed{s}"]) for s in seeds]):
        for c in CHANS:
            n = np.asarray(src["n"], np.float64)
            se = np.asarray(src["se"][c], np.float64)
            lhs = float((n * (se / np.maximum(n, 1e-30))).sum())
            rhs = float(se.sum())
            rel = abs(lhs - rhs) / max(abs(rhs), 1e-300)
            g4[f"{armk}/{c}"] = rel
            assert rel < 1e-10, f"G4 FAILED {armk}/{c}: rel {rel:.3e}"
    res["G4"] = {"max_rel": max(g4.values()), "n_checked": len(g4)}
    log(f"G4 PASS  band tables reconstruct the pooled total (max rel {max(g4.values()):.1e})")

    # --- P4 inconclusive triggers ------------------------------------------
    pc = I["per_case"]
    n_band = np.asarray(I["acc"]["bridge"]["n"], np.float64)
    wout = np.average(np.stack([r["w_frac_outside_hull"] for r in pc]), axis=0)
    win = np.average(np.stack([r["w_frac_inside_body"] for r in pc]), axis=0)
    gmm = np.average(np.stack([r["geom_mismatch_mean"] for r in pc]), axis=0)
    dtm = np.average(np.stack([r["d_t_mean"] for r in pc]), axis=0)
    incon = []
    for bi in range(NB):
        trig = []
        if wout[bi] > 0.20:
            trig.append(f"outside-hull mass {wout[bi]:.3f} > 0.20")
        mb = tab["interp_bridge"]["full"]["u"]["band"][bi]
        mn = tab["interp_nearfill"]["full"]["u"]["band"][bi]
        mt = tso["full"]["u"]["band"][bi]
        best = min(mb, mn)
        if abs(mb - mn) > abs(best - mt):
            trig.append(f"bridge/nearfill spread {abs(mb-mn):.4g} > interp-Transolver gap "
                        f"{abs(best-mt):.4g}")
        if trig:
            incon.append({"band": NAMES8[bi], "triggers": trig})
    res["P4_inconclusive_bands"] = incon
    res["diagnostics"] = {
        "w_frac_outside_hull": wout.tolist(),
        "w_frac_inside_body": win.tolist(),
        "geom_mismatch_mean_abs_dj_minus_dt": gmm.tolist(),
        "d_t_mean": dtm.tolist(),
        "geom_mismatch_over_dt": (gmm / np.maximum(dtm, 1e-30)).tolist(),
        "node_frac_per_band": (n_band / max(n_band.sum(), 1e-30)).tolist(),
    }

    # --- P1 -----------------------------------------------------------------
    R = {}
    best_arm = {}
    for c in CHANS:
        mb = tab["interp_bridge"]["full"][c]["pooled"]
        mn = tab["interp_nearfill"]["full"][c]["pooled"]
        best_arm[c] = "bridge" if mb <= mn else "nearfill"
        mi = min(mb, mn)
        mt = tso["full"][c]["pooled"]
        R[c] = mi / mt
    inc_frac = sum(n_band[NAMES8.index(d["band"])] for d in incon) / max(n_band.sum(), 1e-30)
    if inc_frac > 0.20:
        p1 = "INCONCLUSIVE"
    elif all(R[c] <= 2.0 for c in ("u", "v", "p")):
        p1 = "COMPETITIVE-AT-NATIVE"
    elif all(R[c] > 2.0 for c in ("u", "v", "p")):
        p1 = "NOT-COMPETITIVE-AT-NATIVE"
    else:
        p1 = "MIXED-AT-NATIVE"
    std_i = float(np.mean([min(tab["interp_bridge"]["full"][c]["pooled"],
                               tab["interp_nearfill"]["full"][c]["pooled"]) / VAR_TRAIN[c]
                           for c in ("u", "v", "p")]))
    std_t = float(np.mean([tso["full"][c]["pooled"] / VAR_TRAIN[c] for c in ("u", "v", "p")]))
    res["P1"] = {"R_per_channel": R, "interp_arm_used": best_arm,
                 "inconclusive_node_fraction": inc_frac,
                 "standardised_mean_uvp": {"interp": std_i, "transolver": std_t,
                                           "ratio_interp_over_transolver": std_i / std_t},
                 "verdict": p1}

    # --- P2 -----------------------------------------------------------------
    rb = []
    for bi in range(1, NB):                       # BANDS7 only, comparable to D2
        mi = min(tab["interp_bridge"]["full"]["u"]["band"][bi],
                 tab["interp_nearfill"]["full"]["u"]["band"][bi])
        rb.append(mi / tso["full"]["u"]["band"][bi])
    rb = np.asarray(rb)
    inver = int(np.sum(np.diff(rb) > 0))
    S = float(rb[0] / rb[-1]) if rb[-1] > 0 else float("inf")
    p2 = ("CONFIRMED-AT-NATIVE" if (S >= 10 and inver <= 1)
          else "NOT-CONFIRMED" if S < 3 else "PARTIAL")
    res["P2"] = {"r_b_u": rb.tolist(), "bands": NAMES7, "S": S,
                 "n_inversions": inver, "verdict": p2}

    # --- P3 oracle ----------------------------------------------------------
    O = np.mean(np.stack([np.asarray(r["oracle_band_mse"]["u"], np.float64) for r in pc]), axis=0)
    Q = O / np.maximum(np.asarray(tso["full"]["u"]["band"], np.float64), 1e-30)
    q1 = float(Q[1])                              # band 0-0.005c
    p3 = "NO-PARAMETER-COMBINATION" if q1 > 10 else "WEIGHTS-BOUND" if q1 <= 1 else "PARTIAL"
    res["P3"] = {"oracle_band_mse_u": O.tolist(), "Q_b": Q.tolist(),
                 "Q_0_0.005c": q1, "verdict": p3}
    # same, all channels, for the report
    res["P3_all_channels"] = {
        c: {"oracle_band_mse": np.mean(np.stack(
            [np.asarray(r["oracle_band_mse"][c], np.float64) for r in pc]), axis=0).tolist(),
            "Q_b": (np.mean(np.stack([np.asarray(r["oracle_band_mse"][c], np.float64)
                                      for r in pc]), axis=0)
                    / np.maximum(np.asarray(tso["full"][c]["band"], np.float64), 1e-30)).tolist()}
        for c in CHANS}

    # --- paired per-case sign test on the headline channels -----------------
    tmap = {}
    for s in seeds:
        for r in Tr["per_case"][f"seed{s}"]:
            tmap.setdefault(r["name"], []).append(r)
    signs = {}
    for c in CHANS:
        wins = 0
        tot = 0
        for r in pc:
            mi = min(r["mse_bridge"][c], r["mse_nearfill"][c])
            mt = float(np.mean([x[f"mse_{c}"] for x in tmap[r["name"]]]))
            tot += 1
            wins += int(mt < mi)
        signs[c] = {"transolver_better_cases": wins, "n": tot}
    res["paired_sign_test"] = signs

    write_json(os.path.join(a.out_dir, "point_space_headtohead.json"), res)
    log(json.dumps({"P1": res["P1"]["verdict"], "R": R, "P2": p2, "S": S,
                    "P3": p3, "Q_wall": q1}, indent=2))
    return res


def write_json(path: str, obj) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(obj, f, indent=2)
    log(f"wrote {path}")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--stage", required=True,
                   choices=("cache", "pilot", "interp", "transolver", "reduce"))
    p.add_argument("--task", default="full")
    p.add_argument("--rep", default="nd")
    p.add_argument("--fill", default="nearest")
    p.add_argument("--n-train", type=int, default=800)
    p.add_argument("--n-test", type=int, default=200)
    p.add_argument("--n-pilot", type=int, default=3)
    p.add_argument("--n-proc", type=int, default=8)
    p.add_argument("--kd-workers", type=int, default=2)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--device", default="auto")
    p.add_argument("--data-root", default=os.path.join("data", "Dataset"))
    p.add_argument("--cache-dir", default=os.path.join("data", "cache"))
    p.add_argument("--ckpt-dir", default=os.path.join("checkpoints", "v2_transolver"))
    p.add_argument("--out-dir", default=os.path.join("results", "interpolation"))
    p.add_argument("--scratch", default=DEF_SCRATCH)
    a = p.parse_args(argv)

    t0 = time.time()
    if a.stage == "cache":
        stage_cache(a)
    elif a.stage == "pilot":
        stage_pilot(a)
    elif a.stage == "interp":
        stage_interp(a)
    elif a.stage == "transolver":
        stage_transolver(a)
    elif a.stage == "reduce":
        stage_reduce(a)
    log(f"stage {a.stage} total {time.time()-t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
