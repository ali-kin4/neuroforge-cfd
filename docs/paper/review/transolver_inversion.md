# Does the prediction-below-truth-floor inversion survive on the SOTA backbone?

**Status:** complete — 3 seeds x 200 cases, both identity gates PASS.
**Verdict: the objection STANDS. The inversion DISAPPEARS on Transolver.**

Script: `scripts/probe_transolver_inversion.py`
Result JSON: `results/certificates/transolver_inversion.json`
(per-stage resume caches: `results/certificates/inversion_cache/`)

---

## 1. The objection, in its strongest form

`docs/paper/sections/residual_floor_theorem.tex` ("Empirical confirmation") claims:

> A trained backbone's over-smoothed prediction (here the dropout-FNO of
> `checkpoints/certificates_deq.pt`, distinct from the Transolver headline) sits
> *below* the truth floor in 160/200 cases.

The steelmanned attack:

> Your prediction has a lower residual than the truth because your model is
> over-smoothed. That is a fact about your weak model, not a fact about residual
> monitors. The paper's own sentence concedes the model is over-smoothed, and the
> supporting statistics show it: `norm_pred_std` = 0.034 against `norm_truth_std`
> = 0.162 — the prediction's residual barely varies across cases, exactly what a
> blurred field would do. Re-run it on the Transolver you actually deploy and the
> effect will vanish.

This is the paper's weakest load-bearing number because it is measured on the one
backbone the paper elsewhere calls weak, and it is used to support a claim about
residual monitors *in general*.

## 2. Why a negative was the expected outcome (stated before the run)

Linearising the monitored operator (paper Eq. `rf-decomp`), with `e = û − u*`:

```
‖R_h(û)‖²  ≈  ‖r*‖²  +  2 r*·Le  +  ‖Le‖²
```

Inversion (`‖R_h(û)‖ < ‖r*‖`) requires `r*·Le < −‖Le‖²/2` — the prediction error
must be systematically **anti-aligned** with the floor. Error that is merely
*large* but unbiased drives the monitored residual **up**. Since `r*` is dominated
by unresolved near-wall/closure structure, "anti-aligned with `r*`" is close to a
definition of over-smoothing. So the reviewer's mechanism was a priori plausible,
and the experiment had to be allowed to return NO. It did.

## 3. Method

Same 200-case AirfRANS `full` test split, one ruler for every field.

- **Monitored norm** — `PhysicsChecker.diagnose` → RMS over fluid cells of
  `sqrt(cont² + mom_x² + mom_y²)`, **BC term excluded** (the monitored objective
  does not see no-slip); solid and solid-adjacent wall ring zeroed; per-equation
  non-dimensionalisation. Predicted fields are re-wrapped with the **truth's**
  mask/sdf so masking is identical across truth / uniform / prediction.
- **Fields scored** — (1) truth `u*` (the floor); (2) uniform freestream;
  (3) Transolver **backbone alone**; (4) Transolver **+ DEQ**, the deployed field
  the user actually receives. (3) and (4) are separate rows and are never
  conflated: the DEQ path applies its fixed-point delta with **no acceptance
  test**, so it can move the monitored residual.
- **Deployed path** — `neural_residual_iteration(field0, …, Config().correction)`
  with the seed-matched `seed{m}_corr_with.pt` DEQ corrector: byte-identical to
  `NeuroForgeEngine.solve` (predict once, then loop).
- **Checkpoints** — `checkpoints/v2_transolver/seed{0,1,2}.pt` (the paper's
  headline seeds; seeds 3 and 4 also exist and were not scored).

### Measurement identity — proven, not assumed

The norm is defined **locally** in the new script rather than imported from
`scripts/probe_residual_floor.py`, because a concurrent agent is editing that file
and an import would be an unguarded race. Identity is then enforced at runtime by
two gates:

| Gate | Check | Result |
|---|---|---|
| **A — ruler identity** | recompute `norm_truth` on all 200 cases; require exact match (names, order, values) against the committed `per_case` block of `residual_floor_realdata.json` | **PASS**, `max_abs_diff = 0.000e+00`, order identical |
| **B — reproduce the number under attack** | re-derive the dropout-FNO inversion count from scratch | **PASS**, 160/200, and `norm_mean` = 0.11358129431521029, `norm_std` = 0.034288340792058995 reproduce the published values **to all digits** |

Because the harness reproduces the published dropout-FNO row bit-for-bit, any
difference on Transolver is attributable to the **backbone**, not to the ruler.

### Smoothness diagnostic

`norm_pred_std` is reported for direct comparability with the published number,
but note plainly: **it is the spread across cases of a scalar norm, not a
smoothness measure**, and it cannot settle a smoothing question. The verdict is
carried by a banded **gradient-energy ratio**

```
G = ⟨|∇u|² + |∇v|²⟩ · (L/u_∞)²        (dimensionless)
ratio = G_pred / G_truth               (<1 ⇒ smoother than truth)
```

measured in the **pre-registered** sdf bands already committed in
`scripts/control_cylinder_nearwall_artifact.py` (near-body `0 < sdf ≤ 0.15`,
far-field `sdf > 1.0`), intersected with fluid and with the **wall ring removed**,
so smoothing is measured exactly where `diagnose` evaluates the residual.

*Caveat:* the FNO predicts on the grid while Transolver predicts on the native
cloud and is then rasterised, so the cross-backbone gradient-energy comparison
carries a rasterisation confound. Banding localises but does not remove it. The
inversion counts themselves are free of this confound.

## 4. Results

Truth floor (identical for both backbones, n=200):
`‖r*‖` mean **0.1920**, std **0.1619**, median 0.1325.
Uniform freestream: **exactly 0 on 200/200**.

### Inversion counts — the headline comparison

| backbone | field | below floor | `‖R_h‖` mean | `‖R_h‖` std | median signed gap `pred − truth` | sign test |
|---|---|---|---|---|---|---|
| dropout-FNO (published, reproduced) | backbone | **160/200 (80%)** | 0.1136 | 0.0343 | **−0.0241** | p = 3.4e−18 |
| Transolver seed 0 | backbone alone | **8/200 (4.0%)** | 0.2116 | 0.1608 | **+0.0135** | p = 7.2e−47 |
| Transolver seed 1 | backbone alone | **5/200 (2.5%)** | 0.2119 | 0.1601 | **+0.0152** | p = 3.2e−51 |
| Transolver seed 2 | backbone alone | **5/200 (2.5%)** | 0.2108 | 0.1618 | **+0.0137** | p = 3.2e−51 |
| Transolver seed 0 | **+ DEQ (deployed)** | **12/200 (6.0%)** | 0.2149 | 0.1579 | **+0.0133** | p = 8.1e−42 |
| Transolver seed 1 | **+ DEQ (deployed)** | **9/200 (4.5%)** | 0.2124 | 0.1577 | **+0.0148** | p = 1.5e−45 |
| Transolver seed 2 | **+ DEQ (deployed)** | **14/200 (7.0%)** | 0.2123 | 0.1580 | **+0.0135** | p = 1.6e−39 |

Seed agreement is complete: all three seeds land in 2.5-4.0% (backbone) and
4.5-7.0% (deployed), against 80% for the dropout-FNO. There is no seed on which
the inversion survives.

The sign test is two-sided exact binomial; note both are astronomically
significant but **in opposite directions** — the FNO sits below the floor, the
Transolver sits above it.

### Smoothness — the objection's stated mechanism

Gradient-energy ratio `G_pred / G_truth` (median over cases; `<1` = smoother):

| backbone | near-body | far-field | all fluid | `norm_*_std` |
|---|---|---|---|---|
| truth `u*` (reference) | 1.000 | 1.000 | 1.000 | 0.1619 |
| dropout-FNO | **1.960** | 4.544 | **1.746** | **0.0343** |
| Transolver seed 0, backbone | **1.002** | 1.041 | **1.001** | 0.1608 |
| Transolver seed 1, backbone | **1.003** | 1.091 | **1.008** | 0.1601 |
| Transolver seed 2, backbone | **1.000** | 1.066 | **1.000** | 0.1618 |
| Transolver seed 0, + DEQ | 1.009 | 1.538 | 1.005 | 0.1579 |
| Transolver seed 1, + DEQ | 1.006 | 1.955 | 1.003 | 0.1577 |
| Transolver seed 2, + DEQ | 1.015 | 2.054 | 1.006 | 0.1580 |

Two things follow, and they are different:

1. **Transolver is not smoothed.** Its near-body and all-fluid gradient energy
   matches the truth to within 0.8% on every seed, and its residual-norm spread
   (0.160-0.162) tracks the truth's (0.162) — i.e. its monitored residual varies
   with case difficulty. The FNO's spread (0.034) is nearly flat: its residual
   barely responds to the case at all. (The DEQ step adds some far-field
   structure, ratio 1.5-2.1, while leaving the near-body band at ~1.01; this is
   the corrector writing into the wake, and it does not change any count.)
2. **The word "over-smoothed" in the current paper text is itself unsupported.**
   The dropout-FNO has ~1.75x the truth's gradient energy overall and ~1.96x
   near the body — it is *rougher*, not smoother. So the reviewer's stated
   *mechanism* is wrong for this backbone, even though the reviewer's
   *conclusion* is right. The FNO's low residual comes from its residual being
   nearly case-independent, not from a blurred field. The paper should stop
   asserting the smoothing mechanism it never measured.

## 5. Verdict

**CONCEDED — the objection stands. The 160/200 inversion is a property of the
weak backbone and cannot support a general claim about residual monitors.**

- On the paper's SOTA backbone the inversion essentially vanishes: **2.5-4.0%**
  (backbone) and **4.5-7.0%** (deployed) across 3 seeds, versus **80%** on the
  dropout-FNO. No seed dissents.
- The direction *reverses*: the Transolver's monitored residual sits a median
  **+0.013 to +0.015 above** the floor, the FNO's a median **−0.024 below** it.
- The deployed Transolver+DEQ field — the one a user actually receives — also
  sits above the floor. The DEQ correction nudges the monitored residual slightly
  *up* on every seed (e.g. 0.2116 → 0.2149), consistent with the paper's existing
  `bc_inclusive_sweep` finding and with the "bad fixer" leg.

**This is not a loss for the paper.** It is a cleaner story than the one being
defended, for three reasons:

1. **The model-free leg is untouched and is the real theorem.** The uniform
   freestream scores **exactly 0** on **200/200** while the truth scores
   **0.192**. A constant field annihilates every finite-difference derivative, so
   this holds independently of grid, Reynolds number, closure, and backbone. That
   alone carries Theorem `thm:residual-floor`(i). It needs no model, so no
   reviewer can attack it via the backbone.
2. **The detector/fixer duality gets *stronger*, not weaker.** On the deployed
   backbone the residual sits above the floor and its spread tracks the truth's,
   which is exactly the regime in which `‖R_h‖ ≈ ‖Le‖` is a two-sided proxy for
   field error — and indeed `residual_error_spearman` = 0.61 / 0.65 / 0.61 across
   the three seeds (`results/v2/v2_results.json`, already published). The
   objective is still globally broken (leg 1). Good detector, bad fixer, now
   demonstrated on the *deployed* model rather than on a weak one.
3. It removes a number that a reviewer would otherwise have used to discredit the
   section.

## 6. Exact replacement text for the "Empirical confirmation" paragraph

Replace the current paragraph (`residual_floor_theorem.tex`, lines 109-117) with:

```latex
\paragraph{Empirical confirmation.} On $200/200$ real AirfRANS test cases, the monitored
residual of the ground-truth field is substantially nonzero (\textbf{(H2) holds}):
$\|r^\star\|$ has mean $0.192$ (median $0.133$), while the uniform-freestream field gives
$\|R_h(u_\infty)\|=0$ in \emph{every} case---the objective strictly prefers a physically
wrong field to the truth (\texttt{results/certificates/residual\_floor\_realdata.json}).
This leg is model-free: a spatially constant field annihilates every finite-difference
derivative, so it holds independently of backbone, grid, Reynolds number and closure.
Whether a \emph{trained} prediction also falls below the floor is, by contrast,
backbone-specific, and we report it as such. Writing the first-order expansion
\eqref{eq:rf-decomp}, $\|R_h(\hat u)\|^2\approx\|r^\star\|^2+2\,r^\star\!\cdot\!Le+\|Le\|^2$,
a prediction sits \emph{below} the floor only when its error is systematically
anti-aligned with $r^\star$; unbiased error of any magnitude raises the monitored residual.
Accordingly, on the deployed Transolver backbone the monitored residual sits \emph{above}
the floor---in $192/195/195$ of $200$ cases for the backbone alone and $188/191/186$ for the
deployed Transolver$+$DEQ field (seeds $0,1,2$; median gap $+0.013$ to $+0.015$)---whereas
the weaker dropout-FNO of
\texttt{checkpoints/certificates\_deq.pt} sits \emph{below} it in $160/200$ cases (median
gap $-0.024$). The two backbones are measured with an identical ruler: the harness
reproduces the published dropout-FNO row to all digits
(\texttt{scripts/probe\_transolver\_inversion.py},
\texttt{results/certificates/transolver\_inversion.json}). The dropout-FNO's inversion
tracks a residual that is nearly case-independent (spread $0.034$ versus the truth's
$0.162$), not a smoother field---its velocity-gradient energy is in fact $1.75\times$ the
truth's, against $1.00\times$ for Transolver. We therefore scope the inversion to that
backbone and do \emph{not} claim it as a general property of residual monitors; the
general claims are the model-free minimum above and the detector/fixer duality below.
```

Changes made and why:

- Deletes the unsupported word **"over-smoothed"** (measured: the FNO is
  *rougher* than the truth, 1.75x gradient energy).
- Deletes the causal clause **"consistent with the loop's drive away from `u*` in
  Table `tab:iters`"** — that table is the dropout-FNO sweep, so the link was
  scoped to that backbone anyway and reads as a general claim.
- Names the **backbone and the corrector state** explicitly in every count; the
  vagueness is part of what let the objection land.
- Leads with the **model-free** leg and demotes the inversion to a scoped,
  backbone-specific observation with its mechanism stated correctly.

## 7. Rebuttal line

> **Reviewer:** "The 160/200 inversion is an artifact of your weak, over-smoothed
> backbone, not a fact about residual monitors."
>
> **Response:** Agreed, and we have measured it. We re-ran the statistic on our
> deployed SOTA Transolver over the same 200 cases with a harness that reproduces
> the published dropout-FNO row to all digits (identical residual operator,
> masking and non-dimensionalisation). The inversion does not survive on any of
> the three headline seeds: 8/5/5 of 200 (backbone) and 12/9/14 of 200 (deployed
> Transolver+DEQ) versus 160/200 for the dropout-FNO, with the sign of the median
> gap reversing (+0.013…+0.015 vs −0.024). We
> have scoped the claim to that backbone in the revised text and removed the
> "over-smoothed" characterisation, which our gradient-energy diagnostic shows is
> also wrong (the FNO carries 1.75x the truth's gradient energy; Transolver
> 1.00x). The theorem's load-bearing leg is model-free and unaffected: the
> uniform-freestream field scores exactly 0 on 200/200 cases while the truth
> scores 0.192, so the monitored objective's global minimiser is a physically
> wrong field regardless of backbone.

## 8. Cost and reproduction

- **Wall clock:** truth floor + uniform ~30 s; dropout-FNO stage 298 s (**CPU**);
  Transolver 1645 / 1322 / 1029 s for seeds 0 / 1 / 2 (**GPU**, RTX 4070 Ti,
  inference only, ~5-8 s/case; the spread is contention from other jobs on the
  box, not a change in the work done). Total measurement ~68 min.
  Note `wall_clock_seconds_this_invocation` in the JSON is the cost of that
  invocation only — on a cache-served merge pass it is ~0 and is **not** the
  measurement cost; per-seed cost is `transolver.seed*.seconds`.
- **GPU used:** yes, for the Transolver forwards only (78 s/case on CPU would be
  ~13 h for the 3x200 matrix). The dropout-FNO stage stayed on CPU. No other
  heavy GPU job was running; `nvidia-smi` was checked before launch.
- **Bug found and fixed during the run:** each AirfRANS cloud has a different
  point count, so every forward requests a differently-sized block and torch's
  caching allocator fragments steadily; unchecked it reached the 12 GB cap and
  the run degraded from 7.8 s/case to a standstill. The script now calls
  `torch.cuda.empty_cache()` every 10 cases, which keeps the footprint flat.
- **Resumability:** each stage/seed is written atomically to
  `--cache-dir` the moment it completes and is skipped on re-run (an earlier run
  was killed mid-seed and lost 50 min of GPU; that cannot recur).

```bash
OMP_NUM_THREADS=8 PYTHONPATH=src .venv/Scripts/python.exe \
    scripts/probe_transolver_inversion.py --n 200 --seeds 0 1 2 --device cuda
```
