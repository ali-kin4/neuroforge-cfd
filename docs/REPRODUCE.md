# Reproducing NeuroForge CFD

This document maps **every headline number in the paper**
(`docs/paper/neuroforge_cfd.tex`) to the committed script that produces it, the
result file it lands in, and the cost to reproduce it. It is the artifact-evaluation
entry point.

## TL;DR — two tiers of reproduction

Reproduction splits cleanly into two tiers. **Verify** is what an artifact
reviewer can do on a laptop in minutes; **Regenerate** is the multi-day GPU work
that produced the committed JSONs (already done — do not re-run unless auditing the
numbers themselves).

| Tier | What it does | Cost / hardware | Inputs needed |
|---|---|---|---|
| **VERIFY** | re-derive every paper table/figure from the committed result JSONs | CPU, ~minutes | repo only (JSONs are committed) |
| **REGENERATE** | re-run training + evaluation to recreate the JSONs | GPU (RTX 4070 Ti class), **days**, ~13 GB AirfRANS download | AirfRANS dataset + a GPU |

The committed result files are the ground truth of the paper. The headline-number
inventory with **real SHA-256 hashes** lives in
[`results/MANIFEST.json`](../results/MANIFEST.json) (regenerate with
`python scripts/make_manifest.py`, audit with `--check`).

### Verify the paper on a laptop (no GPU, no download)

```bash
pip install -e ".[dev]"                 # core + pytest/ruff (CPU torch is fine)
python scripts/make_manifest.py --check # confirm committed result hashes are intact
python scripts/make_figures.py          # rebuild every figure from committed JSONs -> results/figures/
PYTHONPATH=src python -m pytest -q      # 144 fast tests (~15 s); slow tests need -m slow
neuroforge demo                         # end-to-end engine on a synthetic case (uses bundled assets/demo.pt)
```

`scripts/make_figures.py` hand-types **no numbers** — every figure is read from the
committed `results/*.json` / `*.csv`. So rebuilding the figures and re-reading the
tables off the JSONs is a complete CPU-only verification of the paper's plots.

> **CPU-from-scratch caveat.** The *cpu-tier* probe/control/sensitivity scripts
> (`run_sensitivity.py`, `probe_residual_floor.py`, `control_*`, `verify_conformal_*`)
> re-evaluate cheaply on CPU **but consume trained checkpoints**
> (`checkpoints/certificates_deq.pt`, `checkpoints/v2_transolver/seed*.pt`), which are
> **gitignored** (`*.pt`). So re-running them from a clean clone first requires the
> GPU regeneration step that produces those checkpoints. Verifying the *committed
> JSONs* (above) needs no checkpoints.

---

## Claim → command → file map

Runtime tier: **GPU** = produced by a multi-day GPU run; **CPU** = re-derivable on a
laptop *from a (gitignored) checkpoint*; **VERIFY** = pure plotting/inventory off
committed files. All GPU runs: AirfRANS, RTX 4070 Ti / 12 GB, torch 2.6.0+cu124,
Python 3.13 (see `results/full_research/MANIFEST.json`). All "GPU" scripts are
resumable (atomic per-epoch checkpoints).

### Abstract + headline (SOTA Transolver)

| Paper claim / number | Command | Output file | Tier |
|---|---|---|---|
| Trust rho = **0.611**±0.054 (Transolver bare, **5 seeds** — supersedes the 3-seed 0.625±0.019); MC-dropout coverage **0.902**±0.008 | `python scripts/run_w1_capture.py --task full --n-train 800 --n-val 200 --epochs 80 --resolution 128 --seeds 0 1 2 3 4` (3-seed original: `scripts/run_v2.py ... --conformal`) | `results/control/w1_capture.json` (coverage: `results/v2/v2_results.json`) | GPU |
| Self-correction **mse_u −8% / mse_v −21% / mse_p −25%** (5 seeds; the 3-seed run reported −9% on mse_u), rho_Cl 0.999 | same run (`loop_per_seed` vs `backbone_per_seed`) | `results/control/w1_capture.json`, `results/v2/v2_results.json` | GPU |
| **W1 control**: null-residual corrector matches/exceeds residual-fed (gain not from residual input) | `python scripts/run_w1_capture.py --task full --n-train 800 --n-val 200 --epochs 80 --resolution 128 --seeds 0 1 2` | `results/control/w1_capture.json` | GPU |
| Third backbone: **MeshGraphNet bare trust rho = 0.851**±0.058 | `python scripts/run_mgn.py --task full --n-train 800 --n-val 200 --epochs 80 --seeds 0 1 2` | `results/mgn/mgn_results.json` | GPU |
| Deep-ensemble adaptive band: **coverage 0.915, ECE 0.074, q 2.35** | `python scripts/run_ensemble_uq.py --task full --n-train 800 --n-val 200 --epochs 80 --members 5` | `results/uq_ensemble/uq_results.json` | GPU |

### Tables

| Table | Claim | Command | Output file | Tier |
|---|---|---|---|---|
| **tab:v2** | Transolver alone vs +DEQ loop | `scripts/run_v2.py` (above) | `results/v2/v2_results.json` | GPU |
| **tab:uq** | MC-dropout vs deep ensemble conformal | `run_v2.py` + `run_ensemble_uq.py` | `results/v2/v2_results.json`, `results/uq_ensemble/uq_results.json` | GPU |
| **tab:indist** | weak-grid ablation, trust 0.40→0.83 | `python scripts/run_full_research.py --preset full` | `results/full_research/in_dist/ablation.{md,csv}` | GPU |
| **tab:ood** | OOD ablation, aoa trust 0.31→0.75 | `python scripts/run_full_research.py --preset full` (OOD stage) | `results/full_research/ood/ablation_ood.{md,csv}` | GPU |
| **tab:iters** | residual rises (0.11→0.62) while error falls (3.92→2.29) | `python scripts/run_sensitivity.py` | `results/sensitivity/iters.csv` (+ `.json`) | CPU* |
| **tab:transolver** | matched-budget baseline, grid trails 4–60× | `python scripts/run_baselines.py --model transolver --task full --n-train 800 --n-val 200 --epochs 80 --n-points 16384 --seeds 0 1 2` | `results/baselines/table2.{md,csv}` | GPU |

### Certificates, theorem, controls

| Claim | Command | Output file | Tier |
|---|---|---|---|
| **H4** in-dist conformal coverage 0.911/0.928/0.942 | `python scripts/run_certificates.py` | `results/certificates/h4_coverage.json` | CPU* |
| **H5** DEQ contraction median 0.78 < κ 0.9 | `python scripts/run_certificates.py` | `results/certificates/h5_contraction.json` | CPU* |
| **Residual floor** (theorem H2): ‖r*‖ mean 0.192, uniform=0 | `python scripts/probe_residual_floor.py` | `results/certificates/residual_floor_realdata.json` | CPU* |
| BC-inclusive residual: negative survives no-slip term | `python scripts/control_bc_inclusive_residual.py` | `results/control/bc_inclusive_sweep.json` | CPU* |
| Per-cell vs per-case trust (0.22 Spearman / 0.60 Pearson) | `python scripts/control_percell_residual_error.py` | `results/control/percell_residual_error.json` | CPU* |
| Conformal survives DEQ (probe + verify) | `scripts/probe_conformal_after_deq.py`, `scripts/verify_conformal_corrected_field.py` | `results/certificates/probe_conformal_after_deq.json`, `.../verify_conformal_corrected_field.json` | CPU* |
| **W2**: conformal on the deployed corrected Transolver field | `python scripts/run_w2_conformal_corrected.py` | `results/uq_ensemble/w2_conformal_corrected.json` | GPU |
| OOD conformal coverage stays in-band | `python scripts/run_sensitivity.py` | `results/sensitivity/ood_coverage.json` | CPU* |
| Toggles are structural no-ops on DEQ | `python scripts/run_sensitivity.py` | `results/sensitivity/toggles.json` | CPU* |
| Cylinder cross-geometry OOD (fig:cylinder, ~7× far-field residual; near-wall ring control) | `python scripts/control_cylinder_nearwall_artifact.py` | `results/figures/cylinder_control_airfoil_ratio.json` | CPU* |
| **Acceptance gate** admits a step on **99.8%** of deployed cases (full step 50.7%); admitted step lowers true error on **89.3%** by median **5.8%** | `python scripts/measure_acceptance_gate.py --mode both --backbone-seeds 0 1 2` | `results/control/acceptance_gate.json` (+ `.md`) | GPU (ensemble arm: CPU) |
| **Selective prediction**: AUROC 0.87–0.91 residual, **0.905** fused; ~**91%** of oracle error reduction at 10% rejection (residual alone ~67%) | `python scripts/run_selective_prediction.py` | `results/selective/selective_prediction.json`, `selective_percase.json` | CPU |
| **Multi-split conformal** (20 re-draws): coverage 0.895–0.902 every arm; corrected-p q 2.26→2.02–2.07 | `python scripts/run_multisplit_conformal.py` | `results/uq_ensemble/multisplit_conformal.json` | CPU |
| **Bootstrap 95% CIs** on the trust-signal Spearmans; every arm excludes zero (ensemble 0.610 [0.51, 0.70]) | `python scripts/bootstrap_spearman_ci.py` | `results/control/bootstrap_spearman_ci.json` | CPU |
| **Repaired force integrator**: control-volume lift rho_L **0.998**±0.001 from predicted fields, median 3.6–3.9% | `python scripts/design_force_integrator.py`, `python scripts/recompute_force_multischeme.py` | `results/control/integrator_design.json`, `force_vs_official_multischeme.json` | GPU |
| **Spatial trust scale**: cell-level rho 0.166 → **0.323** at 16×16 patch pooling (93% of cases improve) | `python scripts/percell_trust_localization.py` | `results/control/percell_localization.json` (+ `.md`) | CPU |
| **Audit cost** median **1.7 ms/case** (1 CPU thread), ~1e-6 of a 16-core RANS solve | `python scripts/measure_audit_cost.py` | `results/control/audit_cost.json` | CPU |
| **Ensemble-size study** M=2..5 (all 26 subsets): fused AUROC 0.905–0.912 at every M; band adaptivity needs M≈4–5 (ECE 0.074→0.199) | `python scripts/run_ensemble_mstudy.py` | `results/uq_ensemble/mstudy.json` (+ `.md`) | GPU |

### Figures

| Figure(s) | Command | Output | Tier |
|---|---|---|---|
| fig1–fig6 (all paper figures) | `python scripts/make_figures.py` | `results/figures/fig*.{pdf,png}` | VERIFY |
| fig:cylinder qualitative panels | `python scripts/make_cylinder_ood_figure.py` | `results/figures/fig_cylinder_ood.{pdf,png}` | CPU* |
| experimental-scope figure | `python scripts/make_scope_figure.py` | `results/figures/fig_experimental_scope.{pdf,png}` | VERIFY |

\* CPU re-evaluation, **but requires a gitignored checkpoint** — see the
CPU-from-scratch caveat above. The committed JSON is the artifact; re-running these
from a clean clone first needs the GPU step that produces the checkpoint.

---

## Reproducibility holes (flagged honestly)

These are paper statements **not** backed by a committed script+file, or backed only
indirectly. None invalidates a headline number; they are gaps an artifact reviewer
should know about.

1. **Accepted-step count (acceptance-test claim).** The paper itself flags this
   (sec:iters): *"we do not have a committed artifact that quantifies the
   accepted-step count, and flag it as the thinnest leg of the residual-as-objective
   negative."* The claim rests on the acceptance-test definition + the tab:indist
   feed-forward result, not a dedicated artifact. **[hole — acknowledged in-paper]**

2. **Checkpoints are gitignored.** `*.pt` is excluded by `.gitignore`, so
   `checkpoints/certificates_deq.pt` and `checkpoints/v2_transolver/seed*.pt` are not
   in the repo. Every "CPU" script depends on one of them, so the cpu-tier scripts are
   **not runnable from a clean clone** without the GPU regeneration step. The committed
   JSONs remain verifiable. **[needs decision — host checkpoints, e.g. a release/Zenodo
   artifact, to make cpu-tier scripts clone-runnable]**

3. **Determinism.** Seeds are pinned in every headline script
   (`torch.manual_seed` / `np.random.seed` / `np.random.default_rng`), but
   `torch.use_deterministic_algorithms` is **not** set, so GPU runs are not
   bit-reproducible. This is expected and the W1 gate explicitly tolerates "3-seed GPU
   non-determinism." **[acceptable, documented here]**

Files present on disk but **not cited by the paper** (work-in-progress, *not* holes):
`results/control/residual_input_ablation.json`,
`scripts/control_residual_input_ablation.py`, `scripts/retrain_without_residual.py`.
The cited W1 source is `results/control/w1_capture.json` (tracked).

---

## Environment

The headline numbers were produced on: Windows 11, Python 3.13.1, torch 2.6.0+cu124,
CUDA 12.4, NVIDIA RTX 4070 Ti (12 GB), 24 cores — recorded in
`results/full_research/MANIFEST.json` and mirrored into `results/MANIFEST.json`.

CPU verification works on any Python ≥ 3.10 with CPU torch
(`pip install torch --index-url https://download.pytorch.org/whl/cpu`).

Dependency floors are in `pyproject.toml` (`>=` bounds). For bit-faithful
regeneration, pin to the tested versions above (numpy 2.4.6, scipy 1.17.1).
