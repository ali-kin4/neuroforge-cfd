# Reproducing the paper

This document maps **every headline number in the manuscript**
(`docs/paper/body.tex` + `abstract.tex`, built as `neuroforge_cfd_elsevier.tex` for the
Journal of Computational Science and `neuroforge_cfd.tex` for arXiv) to the committed script
that produces it and the result file it lands in. It is the artifact-evaluation entry point.

> **Rewritten 2026-09-23.** The previous version of this file mapped an earlier manuscript
> (calibrated trust layer, conformal coverage, DEQ correction). Those claims are no longer in
> the paper; their artifacts stay in `results/MANIFEST.json` for provenance and are listed at
> the foot of this file, but nothing below depends on them.

## TL;DR — two tiers

| Tier | What it does | Cost / hardware | Inputs needed |
|---|---|---|---|
| **VERIFY** | check every number the paper quotes against the committed JSONs, rebuild the figure, confirm artifact hashes | CPU, minutes | the repository only |
| **REGENERATE** | re-run the measurements that produced the JSONs | CPU sweeps of hours (18-core workstation) plus two short GPU inference passes; AirfRANS download (~13 GB) | AirfRANS + the deployed Transolver checkpoints |

### Verify the paper on a laptop (no GPU, no download)

```bash
pip install -e ".[dev]"
python scripts/make_manifest.py --check        # every cited artifact present, hashes intact
python scripts/audit_paper_numbers.py          # every number the manuscript quotes, read from its JSON
python scripts/audit_paper_numbers.py --verbose | grep -c SKIP    # must print 0
python scripts/make_fig_bandratio.py           # rebuilds Figure 1 from the committed JSONs
python scripts/check_submission.py             # abstract/keyword/highlight limits for the journal
PYTHONPATH=src python -m pytest -q             # fast test suite
```

`audit_paper_numbers.py` is the primary verification: it holds a claim table of every number
the manuscript states, each with a reader that pulls the value out of the committed result
file and a tolerance, and it fails if any disagrees. The figure script transcribes no values.

## Claim → command → file map

Tier: **CPU** = re-derivable on a workstation from the AirfRANS download (and, where marked
†, the gitignored Transolver checkpoints); **GPU** = one inference pass over the three
deployed checkpoints; **VERIFY** = read off committed files.

### The native-resolution head-to-head (`tab:native`, abstract, §1)

| Claim | Command | Output file | Tier |
|---|---|---|---|
| per-node MSE ratio interpolation/Transolver 144.2 / 133.9 / 4.85 / 18.3 on u/v/p/ν_t; 200/200 cases on u, v, ν_t and 142/200 on p; rule P1 `NOT-COMPETITIVE-AT-NATIVE` | `python scripts/point_space_headtohead.py --stage {cache,pilot,interp,transolver,r128resample,reduce}` | `results/interpolation/point_space_headtohead.json` (+ `point_space_interp.json`, `point_space_transolver.json`, `point_space_pilot.json`) | CPU + GPU† |
| memo rows: published r128 output at the nodes (400.2 on u, 3.06e6 on p) and the raster's own round trip | `... --stage r128resample` | `results/interpolation/point_space_r128resample.json` | CPU |
| representation ceiling 413× pooled, 495× inside 0.005c | `... --stage reduce` | `point_space_headtohead.json` | CPU |
| single-field oracle 343.6× inside 0.005c | `... --stage interp` then `reduce` | `point_space_interp.json` | CPU |
| mechanism: query 7.8× further from the training wall; 35% of weight inside the training airfoil | `... --stage interp` | `point_space_interp.json` (geometry diagnostics) | CPU |

### The family bound, physical coordinates (§4.1)

| Claim | Command | Output file | Tier |
|---|---|---|---|
| least-squares bound 21.3× on 3 cases (amendment 2) | `python scripts/point_space_headtohead.py --stage oracle_ls` | `results/interpolation/point_space_oracle_ls.json` | CPU |
| least-squares bound **25.0×** on 200 cases, `FAMILY-BOUND-CLOSED` (amendment 3); gate G7 reproduces the n=3 artifact at 0.00e+00; 25.2× at λ = 10⁻⁶ tr/n | `python scripts/point_space_headtohead.py --stage oracle_ls_n200 --n-ls 200 --n-proc 18` | `results/interpolation/point_space_oracle_ls_n200.json` | CPU (~5.5 h) |

### The body-fitted arm — the headline (§4.1)

| Claim | Command | Output file | Tier |
|---|---|---|---|
| test-surface cache the frame needs | `python scripts/bodyfit_bound.py --stage surfcache` | scratch only | CPU |
| **B1 = 0.0044×**, `BODY-FITTED-BOUND-OPEN`; body-fitted/physical 1.8×10⁻⁴, better on 200/200; physical arm on the same restricted nodes 25.21×; frame undefined on 4.98% of the strip (1.36% of the verdict band); GB2a 0 collisions, GB2b 2.06×10⁻¹³; pressure bounds and deployed-estimator errors in both frames (ratios on matched nodes: next table) | `python scripts/bodyfit_bound.py --stage run --n-ls 200 --n-proc 18` | `results/interpolation/bodyfit_bound.json` | CPU (~9.6 h) |
| gate GB1: identity frame reproduces the n=200 artifact at 0.000e+00 (20 cases) | `git checkout 94db42d -- scripts/bodyfit_bound.py && python scripts/bodyfit_bound.py --stage run --frame physical --n-ls 20` | `results/interpolation/bodyfit_bound_physical.json` | CPU |

### Paired per-case statistics (§4.1)

| Claim | Command | Output file | Tier |
|---|---|---|---|
| Transolver per-case, per-band error on all nodes and on the restricted set; gate T1 pools back to `point_space_transolver.json` at 2.5×10⁻¹³ | `PYTHONPATH=src python scripts/transolver_percase_bands.py` | `results/interpolation/transolver_percase_bands.json` | GPU† (~5 min) |
| paired physical bound: median 36.6, 185/200 above 10, none ≤ 1; paired body-fitted bound: 200/200 below 1; B1 on matched nodes 0.0049; deployed estimator body-fitted: median 0.83, 115/200, case-mean 1.71 (improvement 112×); pressure on matched nodes: physical bound 0.00103× (974×), body-fitted 0.00162×, frame worse on 185/200, deployed 5.34× → 4.39×; gate S1 | `python scripts/paired_band_stats.py` | `results/interpolation/paired_band_stats.json` | VERIFY |

### The measure reversal and the grid comparison (§4.1)

| Claim | Command | Output file | Tier |
|---|---|---|---|
| `tab:interp`: the interpolation row (mse_p 75.0, the floor ladder, the permuted control) | `python scripts/parameter_interpolation_baseline.py --task full` | `results/interpolation/interp_full.json` | CPU |
| `tab:interp`: the Transolver row (a separate training of the same architecture) | `python scripts/run_baselines.py --model transolver --task full --n-train 800 --n-val 200 --epochs 80 --n-points 16384 --seeds 0 1 2` | `results/baselines/table2.csv` | GPU (training) |
| `tab:measure`: 8.39× area-uniform → 0.440× node-weighted, rule D3 `ARTIFACT`; `tab:bandratio`; rule D2 | `python scripts/measure_asymmetry.py --stage {nodes,full}` | `results/interpolation/measure_asymmetry.json`, `measure_asymmetry_nodes.json` | CPU + GPU† |
| five node-measure discretisations, mse_p ∈ [0.43, 1.61] | `python scripts/measure_asymmetry.py --stage sensitivity` | `results/interpolation/measure_asymmetry_sensitivity.json` | CPU |
| `tab:interp_bands`: 92% of u error inside 0.02c; R² ≥ 0.9996 beyond 0.05c on the raster | `python scripts/interpolation_band_control.py` | `results/interpolation/interp_band_control_full.json` | CPU |
| resolution ladder 128²/256²/512², rule `PARTIAL` | `python scripts/interpolation_resolution_ladder.py` | `results/interpolation/interp_resolution_ladder.json` (+ `_pfill`) | CPU |
| Figure 1 | `python scripts/make_fig_bandratio.py` | `results/figures/fig_bandratio.{pdf,png}` | VERIFY |

### Node-measure variance, the surface comparator and surrogate competence (§4.1)

| Claim | Command | Output file | Tier |
|---|---|---|---|
| surface-band node-measure Var(p) 2.06×10⁷; arms at 0.73% and 0.48% of it; Transolver 0.073×10⁻² (u) and 0.099×10⁻² (p) in the AirfRANS paper's convention; native standardised mean 24.7× (area-uniform divisors) / 80× (node-measure divisors) | `python scripts/node_measure_variance.py` | `results/interpolation/node_measure_variance.json` | CPU |

### Force columns (§4.2)

| Claim | Command | Output file | Tier |
|---|---|---|---|
| ρ_D = 0.84 against official labels, same on exact fields; interpolation 0.8389 vs exact 0.8394 | `python scripts/recompute_force_vs_official.py` | `results/control/force_vs_official.json` | GPU† |
| round trip through AirfRANS's own integrator: ρ_D = 0.700, ρ_L = 0.9967 | `python scripts/drag_observability_roundtrip.py` | `results/control/drag_observability_roundtrip.json` | CPU |
| control-volume lift ρ_L = 0.998 from predicted fields | `python scripts/recompute_force_multischeme.py` | `results/control/force_vs_official_multischeme.json` | GPU† |

### The physics residual (§4.3)

| Claim | Command | Output file | Tier |
|---|---|---|---|
| floor of the exact field ‖r*‖ mean 0.192; uniform freestream exactly 0 | `python scripts/probe_residual_floor.py` | `results/certificates/residual_floor_realdata.json` | CPU |
| no no-slip weight rescues it (10 cases, closed form) | `python scripts/bc_weight_sweep.py` | `results/control/bc_weight_sweep.json` | CPU |
| floor rises under refinement 0.0624 → 0.0779 → 0.1067, 21/24, p = −0.387 | `python scripts/floor_resolution_decomposition.py` | `results/certificates/floor_resolution_decomposition.json` | CPU |
| thinning the source cloud raises the floor 16/16 | `python scripts/floor_cloud_decimation.py` | `results/certificates/floor_cloud_decimation.json` | CPU |
| descent from the exact truth raises field error 200/200 (Armijo); from Transolver 96.5/94.0/98.0%; iterate selection 24/24 | `python scripts/residual_descent_test.py` | `results/residual_descent/*.json` | CPU† |
| ν_t mean 8.2×10⁻⁵ ≈ 5.2 ν, 84% of ν_eff | `python scripts/check_nut_learned.py` | `results/control/nut_learned.json` | CPU† |

† consumes the gitignored checkpoints `checkpoints/v2_transolver/seed{0,1,2}.pt`.

## Pre-registration record

Every decision rule was written into a script docstring and committed before the run it
governs. The commits are public; this table lets a reader check the order.

| Rule(s) | Governs | Committed | Notes |
|---|---|---|---|
| D1–D3, G1–G6 | measure asymmetry (`tab:measure`, `tab:bandratio`) | `cdf08f5`, 2026-09-12 09:38 | amendments `5746f58` and the sensitivity block are diagnostics only |
| P1–P4 | native head-to-head (`tab:native`) | `a54cb75`, 2026-09-12 10:21 | |
| amendment 3 (n=200 bound, gate G7) | `point_space_oracle_ls_n200.json` | `b0d42d7`, 2026-09-16 17:21 | the stage body was committed here; the two-line **dispatch** wiring that lets `--stage oracle_ls_n200` run was made in the working tree before the run and only committed in `141c2c5` (2026-09-23). The stage logic that ran is the one in `b0d42d7`. |
| B1, B2, GB1–GB3 | body-fitted arm | `94db42d`, 2026-09-16 17:27 | before any body-fitted number existed |
| amendment B-1 (interior-projection exclusion, GB2a/GB2b, both arms in one pass) | `bodyfit_bound.json` | `9cbb398`, 2026-09-17 07:27 | written after the original GB2 **failed** on training fields, before any B1 number existed; **committed after** a two-case smoke test of the amended code and while the 200-case run was executing. The script's content did not change between the smoke test and the commit. The B1 rule and thresholds are those of `94db42d`. |
| resolution-ladder rule | `interp_resolution_ladder.json` | `10f71a6`, 2026-09-11 | tolerance fix `d009cc2` touched a gate, not a rule |

## Reproducibility holes (flagged honestly)

1. **Checkpoints are gitignored.** `checkpoints/v2_transolver/seed*.pt` are not in the
   repository, so every † row needs them. The committed JSONs remain verifiable without them.
2. **GB1 was produced by an earlier script version.** Amendment B-1 removed the `--frame`
   switch; `bodyfit_bound_physical.json` is reproducible from `94db42d` (command above).
3. **Determinism.** Seeds are pinned, but `torch.use_deterministic_algorithms` is not set, so
   GPU passes are not bit-reproducible across hardware. The CPU bounds are deterministic.
4. **The GB1 and T1/S1 gates are exact on this machine** (0.000e+00, 2.5×10⁻¹³); on other
   BLAS builds expect agreement to ~10⁻¹², inside every gate's tolerance.

## Environment

Windows 11, Python 3.13.1, torch 2.6.0+cu124, CUDA 12.4, NVIDIA RTX 4070 Ti (12 GB), 24 cores,
numpy 2.4.6, scipy 1.17.1. Recorded in `results/MANIFEST.json`. CPU verification works on any
Python ≥ 3.10 with CPU torch.

## Artifacts of earlier versions (not cited by the current manuscript)

`results/MANIFEST.json` also hashes the artifacts of the earlier trust-layer manuscript
(`results/v2/`, `results/uq_ensemble/`, `results/selective/`, `results/certificates/h4_*`,
`h5_*`, `results/full_research/`, …) and of the separate metadata-null paper
(`results/nullbench/`). They are kept for provenance; none of them supports a claim in the
current paper.
