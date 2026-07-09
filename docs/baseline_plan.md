# Baseline plan — a FAIR Transolver on AirfRANS (paper Table 2)

Author: benchmarking engineer (neutral). Goal: a Transolver baseline that the
*authors of Transolver would endorse* — native point-cloud representation,
matched budget, and scored by the **exact same** `evaluate_cases` metric code as
NeuroForge. No strawman.

## 1. Implementation path — DECISION

**Chosen: vendor the AUTHORS' official Transolver `Physics_Attention_Irregular_Mesh`
block (MIT-licensed) with attribution, and wrap it in a thin point-cloud head.**

Why this path over a from-scratch reimplementation:

- The official repo `thuml/Transolver` is **MIT-licensed**
  (`Copyright (c) 2024 THUML @ Tsinghua University`), which permits vendoring with
  attribution. We copy the irregular-mesh physics-attention block verbatim (it is
  the architecturally load-bearing part) into
  `src/neuroforge/models/baselines/transolver.py` with a header crediting the
  source file and commit. This is the lowest-risk, highest-fidelity option: the
  attention math is exactly the authors' own.
- The official repo has a dedicated **AirfRANS / irregular-mesh experiment**, so
  the block we vendor is the same one the authors apply to this dataset family.

### Why the existing `models/transformer.py` is NOT a valid baseline

`PhysicsTransformer` is explicitly a Transolver-*style* block and differs from the
real thing in a way that matters for faithfulness:

| Aspect | Existing `PhysicsAttention` (NOT the baseline) | Official `Physics_Attention_Irregular_Mesh` (the baseline) |
|---|---|---|
| Slice assignment | **Shared across heads** (one `(B,N,S)` softmax) | **Per-head** (`(B,H,N,G)` softmax) — each head learns its own point→slice map |
| Slice projection input | the token features `x` | a separate **per-head projected** `x_mid` (`in_project_x`) with a distinct value path `fx_mid` (`in_project_fx`) |
| Temperature | fixed `head_dim**-0.5` | **learnable** per-head `nn.Parameter` initialised to 0.5 |
| Slice-proj init | default | **orthogonal** init on `in_project_slice` |
| Representation | rasterised **grid** tokens (N=H·W) | **native irregular point cloud** (N = mesh nodes) |

The per-head slicing, the dual `x`/`fx` projections, the learnable temperature and
the native-cloud input are the defining features of Transolver. We reproduce all
of them. (The smoke test asserts the per-head `(B,H,N,G)` slice-weight shape and
the expected slice/token counts, so faithfulness is checked, not assumed.)

## 2. Native representation (no rasterisation of the INPUT)

`docs/EXPERIMENTS.md` explicitly demands the real Transolver on the **native point
cloud**, not the rasterised grid (which hides the boundary layer). So:

- The model **consumes** the AirfRANS native `(M, F)` point cloud directly
  (`src/neuroforge/data/pointcloud.py`). Per-point input features
  (`F = 7`): `x, y, u_in_x, u_in_y, sdf, n_x, n_y` — the official AirfRANS input
  features. Targets (`4`): `u, v, p, nut`.
- AirfRANS raw arrays are `(M, 12)` in airfrans 0.1.2:
  `[0,1]=pos, [2,3]=u_in, [4]=sdf, [5,6]=normals, [7,8]=u,v, [9]=p, [10]=nut,
  [11]=on-airfoil bool`. (The loader docstring's "11" predates the boolean
  column; targets at `[7:11]` are unchanged.)

## 3. Same eval code — the fairness keystone

`evaluate_cases(predict_fn, pairs)` takes `predict_fn: FlowCase -> FlowField` and
scores volume MSE per channel, `surface_mse_p`, and Spearman `rho_cl/rho_cd`.

To score the point model with **identical** code:

1. Model predicts per-point `(u,v,p,nut)` on the case's **full** native cloud.
2. The adapter rasterises those predictions onto the structured crop using the
   **byte-identical projection that built the ground truth** in
   `airfrans_loader._sim_to_pair`: same `_CROP = (-1,2,-1.5,1.5)`,
   `rasterize_point_cloud(..., fill=0.0, method="linear")`, and the same
   solid-zeroing (`u,v,nut` zeroed inside the body, `p` kept) with the same
   `sdf`/`mask` from the geometry.
3. The resulting `FlowField` is fed to the *same* `evaluate_cases`.

**Why grid scoring is unbiased here (not a NeuroForge advantage):** a perfect
point predictor rasterises to a field identical to the GT → identical metrics. So
the projection adds no bias *between* methods. What grid scoring cannot reward is
sub-grid boundary-layer detail — but NeuroForge predicts on the grid too, so
neither method gets credit for it. This is a shared, documented limitation (the
EXPERIMENTS.md "rasterised grid hides the boundary layer" caveat), not a bias in
the comparison. Transolver is scored against the **same cached GT pairs**
NeuroForge uses (`data/cache/airfrans_full_test_r128_n200.pkl`), aligned by
`case.name`, so the references are literally identical.

## 4. Matched budget (documented, every hyperparameter)

Anchor = NeuroForge's full-run FNO backbone at the ablation config
(`width=48, modes=20, n_layers=4`) = **7,387,684 params** (measured).

| Knob | NeuroForge (FNO backbone) | Transolver baseline | Match policy |
|---|---|---|---|
| Params | 7,387,684 | `width=256, n_layers=10, n_slices=32, n_heads=8` = **7,350,420** (within 0.5%; printed at runtime, emitted to the table) | matched to <1% |
| Data | task `full`, n_train 800 | same task, same n_train | identical |
| Epochs | 80 (ablation default) | 80 | same epoch count (see note) |
| Seeds | 0,1,2 | 0,1,2 | identical |
| Optimizer | Adam + WarmupCosine, lr 1e-3, wd 1e-5 | **AdamW + OneCycle, lr 1e-3, wd 1e-5** (Transolver's own recipe) | *Deviation, intentional:* we give the baseline its authors' recipe so it is not handicapped. Documented here and in the table footnote. A `--optimizer adam-cosine` flag can mirror NeuroForge exactly if a referee prefers the stricter match. |
| Normalisation | per-channel grid `Normalizer`, fit on **train only** | per-point feature + target z-score, fit on **train only**, persisted, applied to test | same leakage-free policy |
| Point sampling | n/a (grid) | random subsample of `n_points` (default 16384) per training step; **full** cloud (chunked) at eval | matches the official Transolver sampling style; the number is documented, not implied "standard" |

**Note on gradient-step count (honest caveat):** "80 epochs" is matched, but the
*number of optimizer steps* is not. NeuroForge batches 8 grids/step (n_train=800,
bs=8 → ~100 steps/epoch → ~8k steps). The point model processes one cloud/step
(point clouds have heterogeneous M and can't be grid-batched without padding/
masking) → ~800 steps/epoch → ~64k steps, ≈8× more updates. This asymmetry runs
**in the baseline's favour** (more optimization for Transolver), so it is not a
strawman; it is disclosed here and could be equalized later via gradient
accumulation or cloud batching if a referee wants exact step-count parity.

## 5. Metric columns reported

`mse_u, mse_v, mse_p, mse_nut, surface_mse_p, rho_cl, rho_cd,
cl_rel_err_mean(±std), cd_rel_err_mean(±std)` — mean ± std over seeds.

`residual_error_spearman` is **NOT** reported for Transolver: it requires the
NeuroForge `PhysicsChecker` and is a property of NeuroForge's trust map, not of a
plain surrogate. Fabricating it for the baseline would be dishonest; the column is
left as `n/a`.

## 6. Known risks / gaps (read before the overnight run)

- **Whole-cloud eval is mandatory (not chunked).** Transolver's physics-attention
  couples ALL input points via the learned slice decomposition, so a point's
  prediction depends on the whole cloud it is forwarded with. The adapter
  therefore runs **one whole-cloud forward** per case (default `--eval-chunk 0`).
  Measured: ~0.4 s GPU forward at ~180k pts, ~3.1 GB peak (fits the 4070 Ti's
  12 GB easily). `--eval-chunk N>0` exists only as an OOM fallback and is
  explicitly NOT equivalent (it partitions the slice attention) — do not use it
  for the headline table.
- **Eval rasterisation cost.** The slow part of eval is the CPU-bound (SciPy/Qhull)
  linear interpolation of the ~180k predicted points onto 128² — measured ~2.7 s
  per case → ~9 min for 200 cases × 3 seeds.
- **Measured ETA (RTX 4070 Ti).** Train step ~90 ms at `n_points=16384`. Full run
  = 90 ms × 800 clouds × 80 epochs × 3 seeds ≈ **4.8 h** training + ~0.5 h eval +
  one-time ~4 min data load → **~5.3 h total** for 3 seeds. Peak GPU ~3 GB.
- **Optimizer deviation** (AdamW/OneCycle vs Adam/WarmupCosine) is a deliberate
  pro-baseline choice, flagged above and footnoted in the table.
- **No GINO / MeshGraphNet yet.** This deliverable is Transolver only; the script
  is structured (`--model`) to add them later.
- **Point subsample size** (16384) is a defensible default, not a value lifted
  from a canonical AirfRANS Transolver config (the public repo's exact AirfRANS
  sampling could not be pinned during this build). It is a documented knob.
