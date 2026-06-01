# NeuroForge — Pre-Registered Experimental Protocol (Paper A1)

This protocol fixes **what we measure, how, and what would falsify the claims**,
*before* running the experiments — so the results are reviewer-proof and not
cherry-picked. The harness that produces every number is committed
(`benchmarks/ablation.py`, `neuroforge.physics.evaluation`,
`neuroforge.physics.calibration`); the analysis is therefore reproducible by
re-running the commands below.

## Claims under test (and how each can fail)

| # | Hypothesis | Metric | **Falsified if** |
|---|---|---|---|
| H1 | The corrector improves **accuracy**, not just the residual | val field MSE, ρ_Cd | `backbone + corrector` does **not** beat `backbone` on MSE/ρ_Cd (mean over seeds) |
| H2 | The physics residual is a **valid trust signal** | `residual_error_spearman` | the correlation is **≤ 0** (low residual does not track low error) |
| H3 | The **DEQ** corrector is at least as good as the feed-forward one | val MSE, ρ_Cd | DEQ is worse than `local` beyond seed noise |
| H4 | The conformal trust is **calibrated** | empirical coverage at α=0.1 | coverage is outside ~[0.85, 0.95] |
| H5 | The DEQ corrector **converges** | empirical contraction factor | contraction ≥ 1 (it does not; guaranteed by spectral norm — reported, not assumed) |

We will **report all five honestly**, including negative outcomes. If H1 or H2
fail, the paper's contribution is reframed as a *safety wrapper with a no-harm
guarantee + a calibrated surrogate*, not "self-correction improves accuracy".

## Metrics (AirfRANS community protocol — `evaluate_cases`)
- **Volume** per-channel MSE: `mse_u, mse_v, mse_p, mse_nut` (well-conditioned;
  *not* relative-L2, which blows up on the near-zero-mean cross-stream `v`).
- **Surface** pressure MSE on the body (`surface_mse_p`) — the design-relevant field.
- **Force coefficients**: Cd/Cl mean ± std relative error, and the headline
  **Spearman rank correlation ρ_Cd / ρ_Cl** across the test set (what early-design
  ranking actually needs).
- **Boundary-layer** velocity profiles at chord stations (diagnostic; reported as
  a known resolution limitation until a body-fitted/point-cloud backbone lands).
- **Trust diagnostics**: `residual_error_spearman` (H2) and conformal coverage / a
  reliability check (H4).

## Datasets & splits
- **In-distribution**: AirfRANS `full` (800 train / 200 test).
- **Out-of-distribution** (the generalization claim): the `reynolds` and `aoa`
  splits — train on the train range, test only on the held-out range. Report the
  **in-dist → OOD metric gap**, and whether the correction loop reduces it.

## Ablations (`benchmarks/ablation.py`)
Arms, each trained over **≥ 3 seeds**, reported as **mean ± std**:
1. `backbone` (one-shot neural operator),
2. `backbone (no physics loss)` — does the PINN term help end metrics?,
3. `backbone + local corrector` (feed-forward),
4. `backbone + DEQ corrector` (contractive fixed point).
Plus toggles: trust-gating on/off, acceptance-test on/off — each measured on
**field MSE and ρ_Cd**, never on residual alone.

## Baselines (honest — no strawmen)
The toy CPU benchmark is a **CI sanity check only** and must not appear in a
results table. The paper compares against properly-tuned, matched-budget
implementations of: the **real Transolver** (ICML 2024), **GINO**, and
**MeshGraphNet** on the *native point cloud* (not the rasterised grid, which
hides the boundary layer). Report each on the metric set above.

## Statistical rigor & reproducibility
- ≥ 3 seeds; mean ± std on every metric; no single-seed claims.
- Pin: `airfrans` version, split indices, resolution, crop bounds, seeds — emitted
  alongside each results table.
- One-command reproduction:
  ```bash
  python benchmarks/ablation.py --source airfrans --task full \
      --n-train 400 --n-val 120 --seeds 0 1 2 --cache-dir data/cache
  ```
  → writes `results/ablation.md` + `results/ablation.csv` (the paper's main table).

## Reporting artifacts
- **Table 1** — the ablation (arms × metrics, mean ± std).
- **Table 2** — vs. baselines (Transolver/GINO/MeshGraphNet) on AirfRANS.
- **Table 3** — OOD gap (in-dist vs `reynolds`/`aoa`) and its reduction by the loop.
- **Fig.** — ρ_Cd scatter (pred vs CFD), Cp profiles, residual↔error scatter,
  reliability diagram, DEQ convergence curve.

> The single result that decides the paper is **H1 ∧ H2**: the corrector must
> improve accuracy *and* the residual must track error. The harness above
> produces exactly that evidence; we commit to reporting it as-is.

## Status

A **first preliminary run** (1 seed, undertrained: 40 epochs, 150/800 train sims,
res 128) has been completed on real AirfRANS. Directional outcome: **H1 ✓** on the
design-relevant metrics (ρ_Cd 0.773→0.908, ρ_Cl 0.924→0.958, surface-pressure MSE
−25 %), **H2 ✓** (residual↔error Spearman 0.436→0.650), **H3 ✓** (DEQ ≥ local on
those metrics), with an honest `mse_v` volume-regression caveat under
investigation. See §5.3 of `docs/paper/neuroforge_cfd.md`. **The pre-registered
≥ 3-seed, full-budget run with error bars is still pending and remains the
deciding result** — these preliminary numbers are not final.
