# The floor's mechanism, the conformal certificate, and what the manuscript must change

Date 2026-09-07. Branch `paper1/reframe-after-jcp`. **No manuscript file was edited**;
§6 lists every sentence that must change, with replacement text.

This report answers `docs/paper/review/theorem_targets.md`, which argued that the
paper's central *mechanism* claim — that the residual floor's growth under refinement
is due to **operator provenance** and is *not* a resolution or rasterisation artifact —
is wrong. Four experiments were run. Two of them damage the paper. Both are reported
at full strength.

## 0. Verdicts

| # | question | verdict | the number that decides it |
|---|---|---|---|
| **1** | is the floor's growth operator provenance, or representation error? | **PARTIALLY-RESOLVED — the paper's attribution must be deleted, not rewritten** | at fixed `h`, thinning the cloud raises the floor in **16/16 cases**, all 3 bands, both rungs, `q = +0.55 ± 0.27`. Provenance predicts `q = 0`: **refuted**. Representation predicts `q = 1.64`: **also refuted** (1/16 reach `q ≥ 1`) |
| **2.1** | band ordering | **assessment correct**, and now quantitative | floor is monotone in `h/s` across bands: `h/s` = 9.27/8.07/6.35 at 128², floors 0.0574/0.0459/0.0377 |
| **2.2** | does the MMS raster null reverse? | **assessment correct in direction, wrong in magnitude** | it rises at the fine end in **16/16 cases**; slope `−0.49`, not the assessment's `−0.80` |
| **2.3** | viscous block | **assessment correct** | 5.0–6.8% of the floor, `p = −0.823`; the first-derivative block is **93–95%** |
| **3** | does the certificate carry its guarantee? | **RESOLVED** | coverage `0.891–0.895 → 0.901–0.903`; **width `5.6–6.8× → 6.5–7.9×`** |
| **4** | is the bound wide *because* of the floor? | **CONCEDED-WITH-MITIGATION** | removing the floor *exactly* makes the bound **wider**: `7.4× → 11.3×`, 3/3 seeds. The causal clause is **backwards** |

**Artifacts.** `scripts/floor_cloud_decimation.py` → `results/certificates/floor_cloud_decimation.json`;
`scripts/floor_collapse_analysis.py` → `results/certificates/floor_collapse_analysis.json`;
`scripts/floor_subtraction_gate.py` → `results/review/floor_subtraction_gate.json`;
`scripts/functional_audit_gate.py` (fixed) → `results/review/functional_audit_gate_followup.json`.
Decision rules were committed in `0fed12b` and `02ea6ff` **before** the corresponding
runs. `results/MANIFEST.json` regenerated and verified (68 files, 0 missing, 0 hash
mismatches after re-checkout).

---

## 1. Experiment 1 — decimation at fixed `h`

### 1.1 What was held fixed, and how that is gated

The geometry is built from the **full** cloud, once, before any decimation, so the
surface loop, SDF, solid mask and therefore the measurement bands are identical at
every decimation `D`. `CaseLadder.geometry_on` asserts the surface-loop hash, and that
assert is a live gate here. Subsets are **nested** (`D=8 ⊂ D=4 ⊂ D=2 ⊂ D=1`) with a
fixed per-case seed, so the ladder is paired per case. `s` is **measured** per
(case, `D`, band) as the median nearest-neighbour spacing inside that band, not
assumed to be `√D` times the original — the AirfRANS cloud is graded and the realized
factor differs by band.

| gate | rule | result |
|---|---|---|
| reproduction at `D=1` | must match `floor_resolution_decomposition.json` per-case band floors at 128² and 256² to `<1e-9` relative | **PASS**, `worst_rel = 0.0` over 128 comparisons — bit-for-bit |
| hull fallback | `<1e-3` of query points may fall outside the decimated triangulation | **PASS**, exactly `0` at every `D` |

The reproduction gate at `worst_rel = 0.0` also confirms `data/cache/airfrans_pc_full_test_n24.pkl`
is order-identical to the n=100 cache the committed ladder used.

### 1.2 The pre-registered prediction, restated

The registered model (`theorem_targets.md` Prop A4) was
`floor = ‖ε‖·(s/2h)^{β/2}` with `‖ε‖ ∝ s`, giving `floor ∝ s^{1+β/2} h^{−β/2}`.
Since the committed ladder measured `p_h = −0.639` (band 0.1), the model makes a
parameter-free prediction for the `s`-exponent `q = 1 − p_h = 1.64` (band 0.1) and
`1.77` (band 0.25). Operator provenance predicts `q ≈ 0`. Derived secondary: a rise of
`×4.8` to `×8.0` at `D = 8`.

### 1.3 Result — 16 cases, 3 bands, 2 rungs, 4 decimations

Means over 16 cases. `s` is the measured in-band cloud spacing.

| band | rung | `s` at `D`=1/2/4/8 | floor at `D`=1/2/4/8 | `q` | rise at `D=8` |
|---|---:|---|---|---:|---:|
| 0.05 | 128² | 0.00255 / 0.00310 / 0.00442 / 0.00666 | 0.05743 / 0.05935 / 0.07108 / 0.08747 | **+0.490 ± 0.302** | ×1.65 (16/16) |
| 0.05 | 256² | same | 0.07374 / 0.07773 / 0.09555 / 0.11717 | **+0.478 ± 0.309** | ×1.63 (16/16) |
| **0.10** | **128²** | 0.00293 / 0.00357 / 0.00521 / 0.00833 | 0.04593 / 0.04816 / 0.06168 / 0.08008 | **+0.545 ± 0.271** | ×1.82 (16/16) |
| **0.10** | **256²** | same | 0.07192 / 0.07603 / 0.09436 / 0.11552 | **+0.453 ± 0.306** | ×1.67 (16/16) |
| 0.25 | 128² | 0.00372 / 0.00449 / 0.00653 / 0.01092 | 0.03768 / 0.03992 / 0.05555 / 0.07269 | **+0.616 ± 0.290** | ×1.97 (16/16) |
| 0.25 | 256² | same | 0.07139 / 0.07477 / 0.09202 / 0.10823 | **+0.402 ± 0.319** | ×1.60 (16/16) |

**Per case, band 0.1, `n=128`** (the primary cell; reported in full, not as a mean):

| case (`airFoil2D_` prefix stripped) | `q` | rise at `D=8` |
|---|---:|---:|
| SST_31.812_1.334_0.371_3.287_0.0_19.548 | +0.596 | ×1.91 |
| SST_39.741_14.642_3.175_3.316_1.0_10.583 | +0.391 | ×1.48 |
| SST_85.752_6.341_0.382_6.715_8.448 | +0.783 | ×2.15 |
| SST_80.247_1.078_1.298_5.983_0.0_5.593 | +0.657 | ×1.97 |
| SST_61.728_9.994_3.311_7.957_1.0_7.685 | +1.245 | ×3.52 |
| SST_65.202_-3.411_1.499_5.063_12.467 | +0.322 | ×1.37 |
| SST_58.451_-4.262_3.372_4.615_1.0_19.965 | +0.373 | ×1.47 |
| SST_86.75_14.13_0.471_5.71_17.904 | +0.438 | ×1.56 |
| SST_37.251_6.299_1.193_5.155_17.506 | +0.574 | ×1.73 |
| SST_43.677_-1.764_0.243_7.109_0.0_19.17 | +0.567 | ×1.77 |
| SST_90.574_8.095_5.69_1.526_10.267 | +0.949 | ×2.77 |
| SST_90.102_0.155_4.394_4.125_19.322 | +0.598 | ×1.83 |
| SST_40.676_-2.366_6.027_6.14_15.943 | +0.201 | ×1.20 |
| SST_57.682_-2.023_5.676_5.094_12.787 | +0.314 | ×1.35 |
| SST_60.295_-0.953_1.701_6.885_1.0_6.221 | +0.147 | ×1.16 |
| SST_82.537_6.598_2.148_5.703_1.0_12.892 | +0.572 | ×1.82 |
| **mean** | **+0.545** | **×1.82** |

### 1.4 What this proves, and what it does not

**Operator provenance as the sole mechanism is refuted.** `q > 0` in **16/16 cases on
every band at every rung** (sign test `p = 3×10⁻⁵` two-sided). The monitor-versus-
reference operator mismatch does not change when the cloud is thinned; the floor does,
by a factor of 1.8 at `D=8`, with a per-case exponent that never touches zero (minimum
`+0.147`). A claim that the floor is "the residue of operator provenance" cannot
survive a measurement in which the floor moves by 80% while the operator and the
reference solution are untouched.

**Representation error as the *driver of the h-growth* is not established either.** The
registered exponent `q = 1.64` is refuted: `q = 0.55`, and only **1/16** cases reach
`q ≥ 1`. The registered `×4.8–8.0` rise at `D=8` is refuted: the measured rise is
**×1.82**. The amplitude `‖ε‖` is therefore not linear in `s`, and the model that
generated the prediction is wrong in its amplitude scaling.

**Registered branch: `REPRESENTATION-PARTIAL`.** Its committed reading is that the
paper may claim **neither** mechanism: the attribution sentences are deleted and
replaced by the measured facts. That is the reading §6 implements, and this report
does not exceed it.

### 1.5 The post-hoc `s/h` question — reported, and *not* headlined

`scripts/floor_collapse_analysis.py` pools the decimation grid with the committed
5-rung ladder (11 points per case per band) and asks whether the floor is a function of
the single group `s/h`, i.e. whether `q = r` where `r = −p_h`.

| band | `q` | `r` | `q − r` | `|q−r| ≤ 0.35` | `|q−1−r| ≤ 0.35` | adj `R²` full / collapse / registered |
|---|---:|---:|---:|---:|---:|---|
| 0.05 | +0.484 | +0.524 | −0.041 | 6/16 | 3/16 | 0.886 / 0.664 / −1.074 |
| 0.10 | +0.496 | +0.647 | −0.151 | **7/16** | **1/16** | 0.927 / 0.702 / −0.821 |
| 0.25 | +0.509 | +0.771 | −0.262 | 4/16 | 0/16 | 0.927 / 0.725 / −0.646 |

The one-group `s/h` form beats the registered `s`-linear-amplitude form on adjusted
`R²` in **13–14/16** cases. **That is all it establishes.** It is not a collapse:

* `q − r` on the primary band is `−0.151 ± 0.448` and within 0.35 in only 7/16 cases,
  and the gap grows monotonically with band (−0.04 / −0.15 / −0.26).
* Pooled over all 33 (band, rung, `D`) cells, `floor ∝ (h/s)^{−0.484}` gives `R² = 0.71`
  against 0.775 for the two-slope fit; median scatter 12%, max 33%.
* **The per-case test fails.** Two routes to `h/s = 2.84` on band 0.1 — thinning the
  cloud ×8 at 128², versus refining to 362² with the full cloud — give means that agree
  to 13% (0.0801 vs 0.0905), but the **per-case** discrepancy has median **43%** and the
  two are **anti-correlated across cases, Spearman `ρ = −0.61`**.

A negative per-case correlation at matched `h/s` is not a collapse with noise; it says
`h/s` is not the governing group. The aggregate agreement is two 16-case averages
landing near each other. **The manuscript must not claim that thinning the cloud
reproduces refinement.** The defensible statement is the narrow one of §1.4.

### 1.6 One structural fact, verified by reading the code rather than measured

`CaseLadder.__init__` builds `self.tri = Delaunay(pos)` **once**, and
`CaseLadder.rasterise` changes only the query grid. The reconstruction is therefore
provably `h`-independent, and Prop A1's hypothesis (Lipschitz, `h`-independent
reconstruction ⇒ the central difference is exactly a segment mean, hence bounded
uniformly in `h`) holds as a property of this pipeline. This is independent of
Experiment 1 and it licenses the wording change from "grows" to "does not decay"
regardless of which mechanism branch had fired.

---

## 2. Experiment 2 — the three checks against committed artifacts

### 2.1 Band ordering — the assessment is right, and the argument can be made quantitative

The committed ladder separates the bands at 128² and converges them at 512²:

| rung | `h` | band 0.05 | band 0.10 | band 0.25 |
|---:|---:|---:|---:|---:|
| 128 | 0.02362 | 0.05743 | 0.04593 | 0.03768 |
| 512 | 0.00587 | 0.10711 | 0.10737 | 0.10933 |
| ratio | | ×1.87 | ×2.34 | ×2.90 |

The further-out band starts **lowest** and rises **fastest**. Operator provenance
predicts the opposite ordering: the body-fitted/Cartesian mismatch is a near-wall
phenomenon, largest where the reference mesh is most anisotropic and the flow most
structured, so band 0.25 — which excludes that region — should carry the *least*
mismatch and the *shallowest* growth. It carries the steepest, in **16/16** cases.

Experiment 1 supplies the per-band cloud spacing the committed ladder never logged,
which turns this from a narrative into a number:

| band | `s` (chord) | `h/s` at 128 / 181 / 256 / 362 / 512 |
|---|---:|---|
| 0.05 | 0.00255 | 9.27 / 6.54 / 4.62 / 3.26 / 2.31 |
| 0.10 | 0.00293 | 8.07 / 5.69 / 4.02 / 2.84 / 2.01 |
| 0.25 | 0.00372 | 6.35 / 4.48 / 3.16 / 2.23 / 1.58 |

At 128² the floor is **monotone increasing in `h/s`** across the three bands
(9.27 → 0.0574, 8.07 → 0.0459, 6.35 → 0.0377), and the bands converge exactly as their
`h/s` values compress toward ~2. The ordering that provenance gets backwards is the
ordering `h/s` gets right. `floor_resolution_study.md` §6 reads band 0.25 as "the
strongest series, not a hand-picked one"; it is also the series most exposed to the
source cloud.

### 2.2 The MMS raster control — the assessment is right in direction, wrong in magnitude

`mms_raster_band_0.1` rung means: `6.906e-4, 3.723e-4, 2.762e-4, 3.006e-4, 3.823e-4`.
It falls to a minimum at 256² and then **rises**.

| statistic | value |
|---|---|
| whole-range fit on the rung means | `p = +0.402` (the committed per-case mean is `+0.32 ± 0.27`) |
| 128 → 256 | `p = +1.315` |
| **256 → 512** | **`p = −0.469`** |
| **362 → 512** | **`p = −0.692`** |
| per-case fine-half (256–362–512) | `p = −0.494 ± 0.137`, **negative in 16/16** |
| per-case: raster value higher at 512² than at 256² | **16/16** |
| per-case: raster value higher at 512² than at 362² | **16/16** |
| location of the per-case minimum (rung index 0–4) | 0 / 1 / **11** / 4 / 0 |

**Correction to the assessment.** It quotes `p = −0.80` (256→512) and `−0.79`
(362→512). Those are wrong; the correct figures on the rung means are `−0.469` and
`−0.692`. The assessment's own §1.5(b) disavows the quadrature subtraction that
presumably produced them. The **count** is stronger than the assessment claimed
(16/16 per case, against its aggregate 3/16 rising over the whole range) and the
**slope** is shallower.

**Is it "indistinguishable from the real data's slope"?** Same sign and same order, not
the same slope: the truth's `band_0.1` over the same rungs is `p = −0.576` (256→512)
and `−0.492` (362→512), against the null's `−0.469` and `−0.692`.

**The analytic contribution does not explain it.** At 512² `mms_analytic` is
`4.15e-5` against the raster's `3.823e-4` — **10.9% in amplitude, 1.18% in mean
square** — so over the fine half the raster numbers *are* the pipeline-induced
component to better than 1%, and they rise monotonically.

**Consequence, stated plainly.** `floor_resolution_study.md` §1 cites this control for
the claim that "the rasteriser does not manufacture rising residuals out of nothing",
and `body.tex` §sec:floor_ladder cites it as "the pipeline does not manufacture rising
residuals … falls with refinement (fine/coarse 0.41–0.74)". Over the fine half of its
own ladder it manufactures exactly that, in 16/16 cases. **As cited, this control
supports the opposite of what the paper says it supports.** The `0.41–0.74` fine/coarse
ratio is true of the *whole* range and is not false; it is incomplete in a way that
reverses its meaning, which is worse.

### 2.3 The viscous block — the assessment is right

| rung | 128 | 181 | 256 | 362 | 512 |
|---|---:|---:|---:|---:|---:|
| `truth_term_visc_band_0.1` | 0.00229 | 0.00370 | 0.00512 | 0.00637 | 0.00729 |
| floor `band_0.1` | 0.04593 | 0.05590 | 0.07192 | 0.09049 | 0.10737 |
| viscous share | 5.0% | 6.6% | 7.1% | 7.0% | 6.8% |
| **first-derivative block** | **95.0%** | **93.4%** | **92.9%** | **93.0%** | **93.2%** |

The viscous block fits `p = −0.823` over five rungs (`−0.833` on the 128/512 pair)
against the floor's `−0.626`, so it diverges faster than the floor and is a distinct
object. Confirmed: **5.0–6.8% of the floor**, `p ≈ −0.83`.

**Consequence for the C¹ control.** `floor_resolution_study.md` §6.1 and `body.tex`
§sec:floor_ladder present CloughTocher re-rasterisation as the sharpest control against
the interpolation objection. The mechanism it tests — "second-differencing across a
kink scales like `Δs/h`" — lives entirely in the **viscous block**, i.e. in 5–7% of the
floor. It has no purchase on the 93–95% that is first-derivative. The committed numbers
agree that it is not measuring what it is cited for: cubic/linear is
`1.114 / 1.119 / 1.064` at 128²/256²/512² — the ratio **decreases** with refinement,
which is two interpolants approaching the same limit with slightly different
amplitudes, not a kink mechanism being switched off.

---

## 3. Experiment 3 — the conformal quantile defect

`scripts/functional_audit_gate.py:876` used
`np.quantile(err[cal]/score[cal], 0.90)`. Split conformal requires the
`⌈(1−α)(n+1)⌉`-th order statistic. At `n_cal = 100` the plain quantile lands at
position `1 + 0.9×99 = 90.1`, predicting coverage `90.1/101 = 0.892`. The measured
coverages were `0.891 / 0.8945 / 0.891 / 0.893` — the predicted deficit to three
decimals on all four arms, not sampling noise.

Fixed by a new `conformal_quantile()` helper (written as an order statistic, not as
`np.quantile(..., method="higher")`, which lands on the 92nd rather than the required
91st and is one step conservative). Applied at both call sites.

| arm | coverage before → after | `c` before → after | width before → after | **× median error** |
|---|---|---|---|---|
| raw_seed0 / \|ΔC_D\| | 0.8910 → **0.9008** | 0.05176 → 0.06561 | 0.00788 → 0.00918 | **5.57 → 6.49 (+16.6%)** |
| raw_seed1 / \|ΔC_D\| | 0.8945 → **0.9033** | 0.06983 → 0.07877 | 0.01000 → 0.01172 | **6.73 → 7.89 (+17.2%)** |
| raw_seed2 / \|ΔC_D\| | 0.8910 → **0.9011** | 0.05570 → 0.06352 | 0.00861 → 0.00986 | **6.76 → 7.74 (+14.5%)** |
| ensemble_mean / field rel-L2 | 0.8930 → **0.9019** | 0.03796 → 0.04001 | 0.00532 → 0.00546 | **1.61 → 1.65 (+2.6%)** |

**The width was not unchanged.** The assessment warned that the ratio distribution is
heavy-tailed and that moving one order statistic could move the width by more than
intuition suggests. It moved it by **15–17%** on the drag arms. `5.6–6.8×` is now
`6.5–7.9×` and `1.6×` is now `1.7×`. These are paper numbers.

**Two things checked so the fix is not partial.** (i) The second site, `_conformal` at
line 829, had the same defect; it is fixed, though it is dead code under the gate's
actual decision branch (`test_c` is built only on branch D and the gate returns branch
A), so no committed number moves. (ii) `src/neuroforge/physics/calibration.py:80`, the
library's `ConformalCalibrator`, **already applies** the finite-sample correction
(`level = ceil((n+1)(1−α))/n`) and is valid; every other `np.quantile` in `scripts/` is
a decile threshold for labelling, not a conformal quantile. The defect was confined to
the two sites now fixed.

Re-running `--analyse` after the fix leaves the gate's DECISION unchanged (Design A,
both tests fail) and changes only float noise; that file was reverted.

---

## 4. Experiment 4 — the floor-subtraction gate

### 4.1 Instrument

`σ'_i := ‖R_h(û_i) − R_h(u*_i)‖`, the **field difference** — because
`‖R(û)‖ − ‖r*‖` is not a valid norm decomposition and is negative on **2.5–4.0%** of
cases. Reduction identical to `Diagnostics.residual_norm()`: RMS over all cells of
`√(c² + mx² + my²)`, applied component-wise to the difference of residual maps.
200 cases, 3 seeds, 400 random half-splits, exact conformal quantile.

### 4.2 Result — the causal clause is not merely unsupported, it is backwards

| scale | seed 0 | seed 1 | seed 2 | mean |
|---|---:|---:|---:|---:|
| uninformative (`σ ≡ const`) | 15.40× | 15.17× | 21.03× | **17.20×** |
| deployed `‖R_h(û)‖` | 6.49× | 7.84× | 7.73× | **7.35×** |
| **oracle floor-subtracted `σ'`** | **10.46×** | **10.52×** | **13.02×** | **11.33×** |
| perfect oracle (`σ ∝ E`) | 1.00× | 1.02× | 1.00× | 1.00× |

Pre-registered thresholds were `≤2×` = FLOOR-CAUSAL, `≥4×` = FLOOR-NOT-CAUSAL. The
result is **11.33×**, on the wrong side of the *deployed* number. Removing the floor
**exactly** — an oracle that no deployed monitor can have — makes the certificate
**54% wider**, on 3/3 seeds. The assessment predicted a drop to 3–4×; it does not drop
at all.

### 4.3 Why, so that the result cannot be dismissed as a bug

The conformal width ratio is essentially the dispersion of the nonconformity ratio
`E/σ`, and that is what moves:

| scale | `Q_0.9 / median` of `E/σ` | measured width × |
|---|---:|---:|
| uninformative | 15.07 | 15.40 |
| deployed `‖R_h(û)‖` | 6.32 | 6.49 |
| oracle `σ'` | 10.50 | 10.46 |

The width tracks the ratio's dispersion almost exactly. Floor subtraction **improves
the ranking** — Spearman(σ, E) `0.616 / 0.592 / 0.626` → Spearman(σ', E)
`0.681 / 0.703 / 0.601` — while **degrading the scale**. The floor acts as a
stabilising pedestal: it damps the case-to-case variability of the score, which is what
a conformal *scale* needs, at the cost of ranking resolution. Subtracting it buys a
better ranker and a worse certificate.

**Scale-invariance gate: PASS, exactly.** `σ'` has 0.42–0.46× the median of `σ`, so the
first objection is that the comparison is unfair. It is not: `c` is the 0.90 quantile
of `E/σ'`, so `σ' → λσ'` sends `c → c/λ` and leaves `c·σ'` unchanged. Measured over
`λ ∈ {0.01, 0.5, 2, 100}`: **max relative change `0.0`**.

### 4.4 What survives, and what this closes for free

* **Mitigation, and it is favourable and currently unstated.** The deployed monitor
  buys **2.34×** over using no monitor at all (17.20× → 7.35×). That belongs in the
  "what remains deployable" argument.
* **Target C is closed by the same number.** The oracle `r*` dominates every predictor
  `r̂*` of it. The oracle makes the certificate worse, so no learned floor predictor can
  make it better. The constructive half of Target C needs no further experiment.
* **Caveats, stated regardless of direction.** (i) `σ'` uses `r*`, unavailable at
  deployment: an oracle counterfactual, the right instrument for an impossibility
  argument and the wrong thing to present as a method. (ii) `σ'` cancels *shared*
  rasterisation noise as well as the floor, since `û` is rasterised through the same
  pipeline as `u*`. That caveat cuts in the safe direction here: floor removal was
  given every advantage and still lost.

---

## 5. What none of this touches

Unchanged and unweakened: `thm:consistency-floor` (a proof about the exact continuum
solution, involving no rasteriser); `thm:residual-floor` (i)–(iv); `prop:kernel`
(K1)–(K4); `‖R_h(u_∞)‖ = 0` exactly at every `h`; the 24/24 residual-descent divergence
from the exact truth; 0/16 decay under refinement; the 5-seed W1 ablation; the
functional gate's double failure; AUROC 0.952 on worst-decile drag error; the
`+16.6%`-style measured negatives that carry the paper. The refinement leg's
*conclusion* — refinement does not recover the floor — survives, and is strengthened by
§1.6 into a bounded-and-convergent statement rather than a fitted negative exponent.

What does not survive is the sentence explaining *why the ladder rises*, and the
sentence explaining *why the certificate is wide*.

---

## 6. Every sentence that must change

Anchor phrases below were verified present in the current files. The source is
hard-wrapped; match on the phrase, not on the line breaks. **Nothing in this section
was applied — another process owns these files.**

**One instruction to whoever applies this list, so the split is not "harmonised" in the
wrong direction.** The word *grows* is deliberately **kept** where it reports the
measurement (the contribution bullet at `body.tex` l.110, the ladder result paragraph)
and **removed** where it carries mechanism weight (the thesis sentence B1, the
subsection title B4, the discussion B8). That inconsistency is intentional: the
measurement is real and stays; the claim that refinement *causes* the growth by
resolving more of an operator mismatch is what was refuted. Do not unify them.

### `docs/paper/sections/residual_floor_theorem.tex`

**(T1) §"Assumptions, stated plainly" — delete the provenance attribution.** *Licensed
by:* `floor_cloud_decimation.json` (16/16 rise at fixed `h`), §1.6 (the reconstruction
is `h`-independent by construction).
Anchor: `It follows from operator provenance: the floor grows`.
Replace

> "(H2) therefore holds here for a reason that is \emph{not} grid resolution. It
> follows from operator provenance: the floor grows under refinement, the individually
> grid-converged convective and pressure-gradient terms ($0.183\!\to\!0.199$ and
> $0.180\!\to\!0.181$) fail to cancel under a discretisation different from the one
> that produced the labels, and restoring the omitted stress divergence moves the floor
> by under $0.1\%$."

with

> "(H2) therefore holds here for a reason that is \emph{not} the deployed grid's
> resolution: the floor does not decay under refinement, and the individually
> grid-converged convective and pressure-gradient terms ($0.183\!\to\!0.199$ and
> $0.180\!\to\!0.181$) do not cancel under a discretisation different from the one that
> produced the labels, while restoring the omitted stress divergence moves the floor by
> under $0.1\%$. We do \emph{not} attribute the floor's \emph{growth} to operator
> provenance. Holding $h$ fixed and thinning the source point cloud by a factor of
> eight raises the floor in $16/16$ cases on every band and at both rungs tested
> (fitted $\mathrm{d}\log\|r^\star\|/\mathrm{d}\log s = +0.55\pm0.27$;
> \texttt{results/certificates/floor\_cloud\_decimation.json}), so the measured floor
> depends on the reconstruction of the label as well as on the operator, and no
> measurement we have separates the two contributions."

**(T2) same paragraph — keep and strengthen the existing representation caveat.** It is
already correct and now load-bearing; add one clause. Anchor:
`so the measured floor mixes the theorem's term with that error`. Append:

> "The decimation experiment above bounds that mixing from below: at fixed $h$ the
> representation term is large enough to move the measured floor by $80\%$."

**(T3) §46 abstract-of-the-theorem line — the width number.** *Licensed by:*
`functional_audit_gate_followup.json` (corrected quantile).
Anchor: `certificate is $5.6$--$6.8\times$ the quantity being certified`. Replace
`$5.6$--$6.8\times$` with **`$6.5$--$7.9\times$`**.

**(T4) §"Ranking survives…" — coverage.** *Licensed by:* same artifact.
Anchor: `ours attains $0.891$--$0.895$ coverage against a $0.90$ target`. Replace with

> "ours attains $0.901$--$0.903$ coverage against a $0.90$ target, using the exact
> $\lceil(1-\alpha)(n+1)\rceil$-th order statistic rather than the plain empirical
> quantile, which at $n_{\mathrm{cal}}=100$ under-covers by the predictable $0.892$."

**(T5) same paragraph — DELETE the causal clause. This is the most damaging change.**
*Licensed by:* `floor_subtraction_gate.json`.
Anchor: `a fixed, irreducible part of every score is present at zero error`.
Replace

> "What the floor costs is the bound's \textbf{width}. Because $u^\star$ itself is
> rejected at every threshold below $\|r^\star\|$ … a fixed, irreducible part of every
> score is present at zero error: the median truth-to-prediction residual ratio on the
> deployed arm is $0.864$, so $86\%$ of a typical score is floor. The bound that
> results is $5.6$--$6.8\times$ the median drag error it certifies and $1.6\times$ the
> median field error."

with

> "What the floor leaves is a wide bound: $6.5$--$7.9\times$ the median drag error it
> certifies and $1.7\times$ the median field error. We do \emph{not} attribute that
> width to the floor. The median truth-to-prediction residual ratio on the deployed arm
> is $0.864$, so $86\%$ of a typical score is present at zero error---but removing the
> floor \emph{exactly}, via the oracle field difference
> $\sigma'=\|R_h(\hat u)-R_h(u^\star)\|$ (a difference of norms is not a valid
> decomposition, and $\|R_h(\hat u)\|<\|r^\star\|$ on $2.5$--$4.0\%$ of cases), makes
> the bound \emph{wider}, $7.4\times\!\to\!11.3\times$ on $3/3$ seeds
> (\texttt{results/review/floor\_subtraction\_gate.json}). Floor subtraction improves
> the score's \emph{ranking} (Spearman $0.61\!\to\!0.66$) and degrades its use as a
> conformal \emph{scale}. (Floor subtraction improves the ranking on $2$ of $3$ seeds,
> Spearman $0.616/0.592/0.626\to0.681/0.703/0.601$.) The width tracks the dispersion of the nonconformity
> ratio $E/\sigma$ ($Q_{0.9}/\mathrm{median}$: $15.1$ for an uninformative score,
> $6.3$ for the monitor, $10.5$ after floor subtraction). The width is the heavy tail
> of the drag-error distribution, not the floor: the marginal interval with no monitor
> at all is $17.2\times$, so the monitor already buys $2.3\times$. Since the oracle
> $r^\star$ dominates any predictor of it, no learned floor correction can improve this
> certificate either."

**(T5b) the usability sentence between T5 and T6 — it falls in neither anchor and
carries two stale numbers.** *Licensed by:* `functional_audit_gate_followup.json`.
Anchor: `An interval $1.6\times$ the median error is usable; one $6\times$ the median
drag error is not a design tool.` Replace with

> "An interval $1.7\times$ the median error is usable; one $8\times$ the median drag
> error is not a design tool."

**(T6) same paragraph — the refinement clause.** *Licensed by:* §1.4 and §1.6.
Anchor: `And refinement cannot recover it, because the floor grows`. Replace with

> "And refinement does not recover it: the floor does not decay to zero under
> refinement, and the reconstruction the ladder differentiates is fixed independently
> of $h$, so the fitted order is a crossover slope rather than the exponent of a
> scaling law."

**(T7) add the exchangeability hypothesis where the certificate is claimed.** *Licensed
by:* it is the load-bearing assumption and is currently unnamed. Append to the
"certify" paragraph:

> "The guarantee is marginal and rests on exchangeability of the calibration and test
> cases, here a random split of one pooled AirfRANS partition. It says nothing about
> the out-of-distribution regime in which a certificate is most wanted."

**(T8) soften the two over-strong "cannot form it" sentences.** *Licensed by:*
`theorem_targets.md` §3.2, and now by §4.4 — the correction term has free supervised
labels at training time, so the honest claim is that the monitor cannot form it
*exactly*. Replace "exactly the object a deployment-time monitor cannot form, since it
would require the fine operator we are trying to avoid running" with "an object a
deployment-time monitor cannot form \emph{exactly}; it is a supervised regression
target with free labels at training time, but its prediction error then lower-bounds
the monitor's own, and our oracle counterfactual shows even a \emph{perfect} predictor
would not narrow the certificate." Apply the same softening to the
`zhang2026phymgn`-remedy sentence.

### `docs/paper/body.tex`

**(B1) l.42–46, thesis paragraph.** Anchor: `by a margin that grows under grid
refinement`. Replace with `by a margin that does not decay under grid refinement`.
*Licensed by:* §1.4 — "grows" is measured but its mechanism is now unattributed, and
§1.6 shows the fitted order is a crossover slope. The weaker word costs nothing: the
argument only needs non-decay.

**(B2) l.49–60, "Why this is not a statement about our grid" — the cubic control's
role.** Anchor: `The sharpest control is on the real data rather than a manufactured
one`. Replace that sentence and its cubic clause with

> "A higher-order interpolant does not remove the floor either: rasterising the
> \emph{same} reference cloud with cubic instead of linear interpolation \emph{raises}
> it by $6$--$12\%$ and lowers it in only $2$--$4$ of $16$ cases. That control bears on
> the second-derivative block---$5$--$7\%$ of the floor---and not on the
> first-derivative block that carries $93$--$95\%$ of it, so we do not present it as
> the sharpest control."

*Licensed by:* §2.3 (viscous share 5.0–6.8%; cubic/linear ratio *decreasing* with
refinement).

**(B3) l.110–125, contribution bullet.** Anchor: `What sets its size is the mismatch
between our Cartesian stencil and the body-fitted discrete balance the reference
solved.` Replace with

> "What sets its size is not settled: the floor depends on the mismatch between our
> Cartesian stencil and the body-fitted discrete balance the reference solved, and also
> on the reconstruction of the reference onto our grid---at fixed $h$, thinning the
> source cloud eightfold raises the floor in $16/16$ cases. We report both and separate
> neither."

*Licensed by:* `floor_cloud_decimation.json`.

**(B4) l.1328, subsection title.** `\subsection{The floor does not refine away---it
grows}` → **`\subsection{The floor does not refine away}`**. *Licensed by:* §1.6; the
body may still report the measured rise.

**(B5) l.1403–1404, the MMS-raster control. This is the sentence a referee would call
a fatal oversight.** Anchor: `gives a residual that \emph{falls} with refinement
(fine/coarse $0.41$--$0.74$) where the real truth's rises`. Replace

> "\emph{The pipeline does not manufacture rising residuals.} Pushing a manufactured
> analytic solution through the identical rasteriser gives a residual that \emph{falls}
> with refinement (fine/coarse $0.41$--$0.74$) where the real truth's rises."

with

> "\emph{The pipeline null, reported in full because its fine end reverses.} Pushing a
> manufactured analytic solution through the identical rasteriser gives a residual that
> falls over the ladder as a whole (fine/coarse $0.41$--$0.74$), but it reaches a
> minimum at $256^2$ and then \emph{rises}, in $16/16$ cases, at $p=-0.49\pm0.14$
> against the real truth's $-0.58$ over the same rungs. At $512^2$ the pure-truncation
> curve is $10.9\%$ of the raster curve in amplitude and $1.2\%$ in mean square, so
> over that range the raster numbers are the pipeline-induced component to better than
> $1\%$. The null therefore does \emph{not} show that the rasteriser cannot manufacture
> a rising residual; it shows that on a field smooth at the cloud scale the effect is
> ${\sim}280\times$ smaller than on the real data."

*Licensed by:* §2.2.

**(B6) l.1408–1412, "What this leaves".** Anchor: `is set by the mismatch between our
Cartesian stencil and the body-fitted discrete balance the AirfRANS reference actually
solved---a mismatch common to every equation, which is why continuity and momentum rise
at the same rate in the same cases.` Replace with

> "has two contributions we can name and cannot separate. One is the mismatch between
> our Cartesian stencil and the body-fitted discrete balance the AirfRANS reference
> actually solved. The other is the reconstruction of that reference onto our grid: at
> fixed $h$, thinning the source point cloud eightfold raises the floor in $16/16$
> cases, on every band and at both rungs
> (\texttt{results/certificates/floor\_cloud\_decimation.json}). The lockstep of
> continuity and momentum does not discriminate between them---both carry the same
> gradient reconstruction error in every equation---and no measurement we have
> separates their shares."

*Licensed by:* §1.4, §2.1.

**(B7) l.1412–1417, the ladder's limitation.** Anchor: `at $512^2$ the raster spacing
is only $1.27\times$ the source cloud's far-field spacing`. Append:

> "That limit is not merely a caveat about the finest rung: the ratio $h/s$ is
> $8.1\!\to\!2.0$ on the primary band across the whole ladder, and the three exclusion
> bands sit at $h/s=9.3/8.1/6.4$ at $128^2$, which is the order in which their floors
> are ranked and the order in which they converge."

*Licensed by:* §2.1.

**(B8) l.2185–2196, discussion.** Anchor: `and refinement resolves more of that
mismatch, not less.` Replace the trailing clause with

> "and refinement does not remove it. We stop short of naming what makes the measured
> floor larger on a finer grid: the same experiment that refines the grid also samples
> a fixed reconstruction more finely, and holding the grid fixed while thinning the
> source cloud raises the floor in $16/16$ cases."

*Licensed by:* §1.4.

### `docs/paper/abstract.tex` — outside the two files named in the brief, and the
most-read file in the package

**(A1) l.22 carries both stale numbers.** *Licensed by:*
`functional_audit_gate_followup.json`. Anchor: `supports a valid split-conformal band
at $0.89$ coverage, at a width $5.6$--$6.8\times$ the drag error it bounds.` Replace
with

> "supports a valid split-conformal band at $0.90$ coverage, at a width
> $6.5$--$7.9\times$ the drag error it bounds."

The `$0.89$` was the *defect*, not a rounding: the plain empirical quantile delivers
`0.892` by construction. Shipping the abstract with it would advertise the bug.

**(B9) the residual grep.** Sites found and listed: `abstract.tex` l.22 (A1),
`residual_floor_theorem.tex` l.46 (T3), l.387 (T4), l.393 (T5), l.395 (T5b).
`neuroforge_cfd.tex` and `neuroforge_cfd_elsevier.tex` are wrappers and carry none.
`docs/paper/submission/arxiv_v4/stage/` is a **frozen snapshot of a shipped package**
and must not be edited in place; it will be regenerated from the corrected sources.
Re-run before submission:
`grep -rn "5\.6\|6\.8\|0\.891\|0\.895\|0\.89 coverage" docs/paper/*.tex docs/paper/sections/`.

### `docs/paper/review/floor_resolution_study.md`

Not a manuscript file, but it is the source the manuscript quotes and it currently
carries the four wrong sentences. §1a's "The cause consistent with that is the
**reference-operator mismatch**", §4's "refinement resolves more of the very structure
the monitored operator omits", §6.2's "it is the residue of **operator provenance**",
and §1's reading of the MMS control all need the same treatment as B5/B6/B8. Its §9
rebuttal line should lose "refinement makes the mismatch worse".

---

## 7. Rebuttal lines

> **Reviewer:** *your rising residual floor is a rasterisation artifact, and your own
> pipeline null says so — it rises at the fine end too.*
> **We did:** held the grid spacing fixed and varied the source cloud instead,
> decimating the AirfRANS point cloud by nested factors 1/2/4/8 with the geometry built
> from the full cloud so the scoring region is identical at every factor
> (`scripts/floor_cloud_decimation.py`, decision rule committed in `0fed12b` before the
> run; `D=1` reproduces the committed ladder bit-for-bit, hull fallback exactly zero).
> **Evidence:** the floor rises in **16/16 cases** on every band at every rung,
> `d log‖r*‖/d log s = +0.55 ± 0.27`. We therefore **withdraw** the operator-provenance
> attribution of the growth, report both contributions, and separate neither — and we
> report the null's fine-end reversal ourselves, at `p = −0.49 ± 0.14` in 16/16 cases.
> The refinement leg's conclusion is unchanged and now rests on a stronger footing: the
> reconstruction is `h`-independent by construction, so the floor is bounded and does
> not decay, and the fitted order is a crossover slope rather than a scaling exponent.

> **Reviewer:** *your certificate does not carry the guarantee you claim, and your
> explanation for its width assumes its conclusion.*
> **We did:** replaced the plain empirical quantile with the exact
> `⌈(1−α)(n+1)⌉`-th order statistic, and ran the oracle counterfactual that removes the
> floor exactly (`scripts/floor_subtraction_gate.py`, thresholds committed in `02ea6ff`
> before the run, with a scale-invariance gate).
> **Evidence:** coverage `0.891–0.895 → 0.901–0.903`, width `5.6–6.8× → 6.5–7.9×`.
> Removing the floor exactly makes the bound **wider** (`7.4× → 11.3×`, 3/3 seeds), so
> we **delete** the causal clause: the width is the heavy tail of the drag-error
> distribution, and the monitor already buys `2.3×` over using no monitor at all. The
> same number closes the constructive question of whether a *learned* floor correction
> would help — the oracle dominates it, and the oracle loses.

---

## 8. What a fuller run would add

* **A 1024² rung on `band_0.25`.** §1.6 says the first-derivative block must saturate;
  the ladder stops at `h/s = 1.58` on that band. The registered prediction to make
  before running it: `band_0.25` flattens while `truth_term_visc` continues at
  `p ≈ −1`.
* **A reconstruction whose support is fixed in physical units** (moving least squares at
  `r = 0.02c`, independent of `h`). That removes the sampling mechanism by construction
  and would *isolate* the provenance block — the number this report says nobody has.
* **The dropout-FNO contrast for Experiment 4.** `û` from a grid-native backbone does
  not share the rasterisation pipeline with `u*`, so `σ'` would not cancel shared
  noise. The direction of the present result makes this a robustness check rather than
  a load-bearing one.
* **More than 16 cases and more than 2 rungs for the decimation `q`.** The per-case
  spread (`+0.147` to `+1.245`) is wide; the sign is not in doubt but the exponent is
  soft.
