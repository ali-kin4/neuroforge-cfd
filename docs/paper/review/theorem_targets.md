# Theorem targets A / B / C — feasibility assessment

Assessor: theory / formal-guarantees. Date 2026-09-07.
Branch: `paper1/reframe-after-jcp`. **No manuscript file was edited.** This is a
feasibility assessment against the question *"can a month of theory save this
paper?"*

**Execution constraint, stated first because it bounds everything below.** No
shell was available in this session (`Bash` disabled). Every number here is
either (a) copied from a committed artifact, (b) arithmetic on committed
numbers, or (c) an analytic derivation. Nothing was measured. Claims of type (c)
are labelled as derivations and each carries the measurement that would falsify
it.

---

## 0. Verdict, ranked

| target | verdict | what is actually provable | days |
|---|---|---|---:|
| **A** — growth exponent | **PARTIALLY PROVABLE, and it is the most important of the three — but not in the direction the brief expects.** The *exponent* is not derivable as a sharp number. What **is** rigorously provable, in half a page and with no modelling assumption, is that the growth of the dominant residual block **must stop**: it converges to a finite positive limit, so a power law in `h` is the wrong framing. The mechanism responsible for the ***`h`-dependence*** is **representation error in the rasterised label, not operator provenance**. **The paper's thesis is intact** — `thm:consistency-floor` proves the `h`-independent part is positive by a wholly separate route that nothing here touches — but **four mechanism sentences in one section are wrong** and must be edited. | 3–4 (theory) + 2 (the one experiment) |
| **B** — conformal width lower bound | **NOT PROVABLE as an impossibility result.** The closed form that reproduces `5.6×` is a one-free-parameter model that assumes its own conclusion, and the distribution-free version is true but weak and not floor-specific. Worse: the measured `5.6×` is **mostly the heavy tail of the drag-error distribution, not the floor** — the monitor already buys 2.7× over using no monitor at all. The paper's causal attribution of the width to the floor is unsupported. One half-day experiment can decide whether any version of B survives. | 0.5 (gate) + 4–6 (only if the gate passes) |
| **C** — conditions for a consistent cheap monitor | **NOT PROVABLE as an interesting theorem.** The characterisation is a tautology (consistency ⟺ the floor is a computable function of the deployment inputs) and the constructive version — *learn the floor, it has free labels at training time* — is `zhang2026phymgn`'s remedy and `stetter1978defect`'s τ-correction in kind. One genuinely non-obvious observation survives, and it **weakens a sentence currently in the theorem file**. | 0 as theory; free with B's gate |

**Direct answer to the question asked.** No. None of A, B or C yields a theorem
that would move a desk rejection on originality. A yields something more useful
than a theorem: **a proof that strengthens the refinement leg, plus a surgical
correction to four mechanism sentences**. My recommendation is one week, not one
month, and the week is spent defending and sharpening committed claims rather
than adding theory. Details in §5.

**Scope of the bad news, stated up front so it is not over-read.** The
`h`-*dependence* of the measured floor is representation error. The floor's
`h`-*independent* part is a different object, is proved positive by
`thm:consistency-floor` via an argument that involves no rasteriser at all, and
is untouched by everything below. The thesis — *you cannot certify a surrogate
with a residual operator that is not the one its labels solve* — survives intact.
What does not survive is the sentence explaining *why the ladder rises*.

**Three concrete findings that do not depend on any of the three targets, and
that should be actioned regardless.** They are in §1.5, §2.4 and §2.7.

---

## 1. TARGET A — the growth exponent

### 1.1 Restating the puzzle sharply, because it has a clean resolution

The brief's own argument is correct and it is decisive: as `h → 0` with the
reference solution *fixed*, the monitored operator applied to that solution
should converge to a fixed mismatch — the reference solver's own consistency
defect plus our omitted closure term — and the floor should **plateau**.

That argument proves more than it says. It proves that **any mechanism which
produces unbounded growth must be one whose amplitude depends on `h` through the
representation of the label**, because the operator is fixed, the label's
underlying object is fixed, and the only remaining `h`-dependent object in the
pipeline is the map from cloud to grid. That is not a candidate among several. It
is the only surviving class, by elimination.

So the correct question is not "which mechanism produces `p = −0.6`". It is
"which part of the representation error grows, how far can it grow, and does the
measured range sit inside or outside the growth regime". Those have answers.

### 1.2 Setup and notation

Fix a case. Let `{x_k}` be the source point cloud in the crop and let
`Ū : Ω → R⁴` be the reconstruction the pipeline uses — barycentric-linear on the
Delaunay triangulation of `{x_k}` (`airfrans_loader.py`, `method="linear"`), or
CloughTocher for the C¹ control. The load-bearing structural fact:

> **`Ū` does not depend on `h`.** The triangulation is built once from the cloud;
> refinement only changes where `Ū` is *sampled*. So `u*|_{Ω_h} = Ū|_{Ω_h}` for
> every rung of the ladder.

Let `K ⊂⊂ Ω_f` be the fixed physical band (`band_0.1`, `band_0.25`, or the
`sdf ≥ 0.0352c` exclusion of the 24-case ladder). Split the monitored operator
of `eq:monitored`:

```
R_h = R_h^(1) + R_h^(2),
R_h^(1) = ( D_x u + D_y v ,  u D_x u + v D_y u + D_x p ,  u D_x v + v D_y v + D_y p )
R_h^(2) = ( 0 , -ν_eff Δ_h u , -ν_eff Δ_h v )
```

`R_h^(1)` is the **first-derivative block** — continuity, convection, pressure
gradient. `R_h^(2)` is the viscous block. The committed decomposition
(`floor_resolution_decomposition.json`, band 0.1, 16 cases) puts the viscous term
at `0.0023` of a `0.0459` floor at 128² and `0.0073` of `0.1074` at 512²: **5% to
7%**. The floor is `R_h^(1)`.

### 1.3 Proposition A1 — the dominant block converges. The growth is provably pre-asymptotic.

> **Proposition A1.** Let `Ū ∈ W^{1,∞}(K)` (satisfied by the linear interpolant,
> which is Lipschitz on `K`, and by CloughTocher). Let `K` be fixed in physical
> units and at distance `≥ 3h` from the crop border, so only interior stencils
> act on `K`. Then for every `x ∈ K` and every admissible `h`,
> ```
> D_x^h Ū(x) = (1/2h) ∫_{-h}^{h} ∂_x Ū(x + t e₁) dt                    (A1.1)
> ```
> **exactly**. Consequently
> ```
> |D_x^h Ū| ≤ ‖∂_x Ū‖_{L^∞(K)}   uniformly in h,                       (A1.2)
> D_x^h Ū → ∂_x Ū   a.e. and in L^p(K) for every p < ∞,                 (A1.3)
> ```
> and therefore
> ```
> lim_{h→0} ‖ R_h^(1)( Ū|_{Ω_h} ) ‖_{RMS, K}  =  F_K  :=  |K|^{-1/2} ‖ R^(1)(Ū) ‖_{L²(K)}  <  ∞.   (A1.4)
> ```
> The first-derivative block of the residual floor is **bounded above uniformly
> in `h` and converges**. Its measured growth is a monotone approach to `F_K`
> from below.

*Proof.* (A1.1): `Ū` is Lipschitz along the segment, hence absolutely continuous,
so `Ū(x+he₁) − Ū(x−he₁) = ∫_{-h}^{h} ∂_xŪ(x+te₁) dt`; divide by `2h`. The
central difference is *exactly* the segment mean of the derivative — no error
term, no smoothness assumption beyond Lipschitz. (A1.2): the mean of a function
bounded by `‖∂_xŪ‖_∞`. (A1.3): `∂_xŪ ∈ L^∞ ⊂ L¹_loc`, so a.e. `x` is a Lebesgue
point and the symmetric averages converge there; (A1.2) then gives `L^p`
convergence by dominated convergence. The same holds for `D_y`. The convective
terms are products of `Ū` (continuous, converging uniformly on `K` since it is
sampled exactly) with the converging difference quotients, so they converge in
`L²(K)`; likewise `D_xp, D_yp`. (A1.4): the measured RMS over cells of `K` is a
Riemann sum for `|K|^{-1}∫_K |R^(1)|²`; the integrands are uniformly bounded and
converge a.e., so the Riemann sums converge to the integral. ∎

**Three remarks the statement needs to survive a referee.**

1. **No noise model, no correlation structure, no exponent.** A1 assumes only
   that the reconstruction is Lipschitz and `h`-independent. That is a property
   of the pipeline, checkable by reading `airfrans_loader.py`.
2. **It covers the C¹ control.** CloughTocher is also `W^{1,∞}` and also
   `h`-independent, so its floor also converges — to a *different* `F_K`. This is
   why the C¹ control raising the floor by 6–12% is not evidence against a
   representation mechanism (§1.5c).
3. **It does not contradict `thm:consistency-floor`.** That theorem is a
   statement about the *exact continuum solution* `u_c` and gives a positive
   lower limit. A1 is a statement about the *rasterised label* `Ū` and gives a
   finite upper limit. They bracket, and the paper's existing caveat ("the
   measured floor mixes the theorem's term with representation error") is exactly
   the gap they leave.

### 1.4 Corollary A2 — what the plateau level is, and the honest budget

Write `u_c` for the exact continuum RANS solution the reference approximates.

```
R^(1)(Ū) = [ R^(1)(Ū) − R^(1)(u_c) ]  +  R^(1)(u_c)
             └── reconstruction ──┘      └─ provenance ─┘
```

* **Provenance block, `R^(1)(u_c)`.** `h`-independent by construction. Contains
  the omitted closure term `T = 2S∇ν_t` of `thm:consistency-floor` (measured
  proxy `truth_omit_gradnu_band_0.1` = 0.0016 → 0.0048) and the reference FV
  scheme's own consistency defect. This is the paper's thesis and A1 does not
  touch it.
* **Reconstruction block.** Controlled by `‖∇Ū − ∇u_c‖`, the *gradient*
  reconstruction error of the interpolant, of size `O(s |∇²u_c|)` with `s` the
  local cloud spacing. `h`-independent in the limit, but **the measured quantity
  at finite `h` is a segment-average of it (A1.1), which suppresses it by an
  amount that shrinks as `h` shrinks** — which is precisely the observed growth.

So `F_K² ≈ ‖T‖² + ‖τ_ref‖² + ‖E_grad‖²` (up to cross terms), and everything the
ladder measures as *`h`-dependence* lives in the third term's approach to its own
limit.

**What I explicitly do NOT claim.** I ran a two-parameter fit
`floor(h)² = A² + B²(s/h)` on the band-0.1 rungs and it returns `A² < 0`, which
would say the entire measured floor is reconstruction error. **That is a fit
artifact, not a finding**: the fit hard-codes `p = −1/2`, the data's slope is
steeper than `−1/2`, and imposing the steeper slope returns `A² > 0`. There is
**no committed measurement that separates `‖T‖ + ‖τ_ref‖` from `‖E_grad‖`.** The
defensible statement is that both are present, the `h`-dependence is entirely the
second, and their split is unmeasured. Anything stronger — in either direction —
is not supported.

### 1.5 The exponent: a band, not a number, plus three confirmations already committed

> **Proposition A3 (viscous block, the one component that genuinely diverges).**
> For a piecewise-linear `Ū`, `∂²Ū` is a measure supported on the triangulation
> edges. At a node whose 3-point stencil straddles an edge,
> `Δ_h Ū = [∂_n Ū]·O(1/h)` with `[∂_nŪ] = O(s|∇²u_c|)` the gradient jump. The
> fraction of nodes affected is `min(1, c·h/s)`. Hence
> ```
> ‖Δ_h Ū‖_{RMS,K}  ~  s|∇²u_c| · h^{-1} · min(1, c h/s)^{1/2}
>                   =  O(h^{-1/2}) for h ≫ s,   O(h^{-1}) for h ≳ s (fraction saturated).
> ```
> This block does **not** converge. It is unambiguously an artifact of twice-
> differencing a C⁰ reconstruction.
>
> *Measured:* `truth_term_visc_band_0.1` = 0.00229 → 0.00729 over `h`-ratio 4.02,
> i.e. `p = −0.83`, inside the predicted `[−1, −1/2]`. It is 5–7% of the floor
> and rising.

> **Proposition A4 (the first-derivative block's approach rate — a band).**
> Model the triangle-wise gradient error as a piecewise-constant field with
> correlation length `s`. By (A1.1) the measured derivative averages `N ≈ 2h/s`
> pieces, so the surviving RMS is `‖ε‖ N^{-β/2}` with `β = 1` for uncorrelated
> triangle errors and `β = 2` for the anti-correlated case (the two triangles of
> a locally regular quad carry equal and opposite leading-order errors). Hence
> ```
> ‖r*‖(h) ∝ h^{-β/2},   p ∈ [−1, −1/2],   saturating at ‖ε‖ once h ≲ s.
> ```

**`β` is a property of the Delaunay triangulation's local regularity. It is
case-dependent, region-dependent, and unmeasured. A sharp exponent is therefore
NOT derivable from available information.** What is derivable is the band, and
every committed exponent lies in or near it:

| series | measured `p` | in `[−1,−1/2]`? |
|---|---:|---|
| band 0.25 (best-resolved) | −0.77 ± 0.16 | yes |
| band 0.10 (primary) | −0.64 ± 0.29 | yes |
| band 0.05 | −0.45 (from rung means) | edge |
| viscous term (Prop A3) | −0.83 | yes |
| omitted ∇ν_t term | −0.93 ± 0.20 | yes |
| 24-case ladder, physical-units exclusion | −0.387 | **no — shallower** |
| MMS raster, finest rung pair (§1.5b) | −0.79 to −0.80 | yes |

**A4 also predicts the one discrepancy the reports currently shrug off.** The
24-case ladder fits `p = −0.387`; the 16-case ladder fits `−0.64 ± 0.29` on band
0.1 and `−0.77 ± 0.16` on band 0.25. `floor_resolution_study.md` §2 reads this as
"different exclusion policies, same verdict". Under A4 it is *predicted*: a
crossover has no single slope, and the two studies use different exclusion radii,
therefore sample different local cloud spacings, therefore sit at different
`h/s` positions on the same crossover curve. The 24-case exclusion
(`sdf ≥ 0.0352c`) admits cells closer to the wall, where `s` is smaller and `h/s`
larger, so it sits further into the averaged regime and reads a shallower slope —
exactly the ordering observed, and exactly the ordering the band series (a)
reproduces internally. A power-law reading has no explanation for it.

**Three confirmations sitting in the committed data, none of which required a new
run.**

**(a) The band ordering is diagnostic, and it favours representation.** The three
bands separate at 128² (0.0574 / 0.0459 / 0.0377 for 0.05 / 0.10 / 0.25) and
**converge at 512²** (0.1071 / 0.1074 / 0.1093). The further-out band starts
lowest and rises fastest. Representation predicts exactly this: the cloud is
coarsest far from the wall (`floor_resolution_study.md` §1a: median NN spacing
`4.4e-5` chord at the wall, `4.6e-3` in the far field), so `h/s` is smallest in
band 0.25, the averaging of (A1.1) is weakest there, and band 0.25 saturates
first — steepest apparent slope, and all bands meeting at the same saturated
value. Operator provenance predicts the **opposite** ordering: the body-fitted /
Cartesian mismatch is largest near the wall, where the reference mesh is most
anisotropic and the flow most structured, and should be smallest in the smooth
outer field. The report reads band 0.25 as "the strongest series, not a
hand-picked one" (`floor_resolution_study.md` §6); under this analysis it is the
series that most cleanly indicts the rasteriser.

**(b) The MMS-raster null reverses at the fine end, at the predicted slope.**
`mms_raster_band_0.1` = `6.906e-4, 3.723e-4, 2.762e-4, 3.006e-4, 3.823e-4`. It
falls to a **minimum at 256²** and then **rises**: `p = −0.80` over 256→512,
`p = −0.79` over 362→512. The report quotes the whole-range fit `p = +0.32` and
reads the control as showing "the rasteriser does not manufacture rising
residuals out of nothing". Over the fine half of the ladder it manufactures
exactly that, at a slope indistinguishable from the real data's.

No separation of the two components is needed to say this. Over 256→362→512 the
analytic (truncation) curve is `1.69e-4 → 8.36e-5 → 4.15e-5` while the raster
curve is `2.762e-4 → 3.006e-4 → 3.823e-4`; at 512² the analytic contribution is
**below 11% of the raster value in amplitude, under 1.2% in the mean square**, so
the raster numbers *are* the pipeline-induced component over that range to better
than 1%. Quote them directly: rising monotonically at `p ≈ −0.8`. (I originally
isolated the component by quadrature subtraction; do not do that — it assumes the
truncation and interpolation contributions are orthogonal, which they are not
since `R_h(Π_h u) − R_h(u) ≈ L(Π_h u − u)`, and at 128² it differences two
numbers agreeing to four digits.)

Same mechanism as the real data, amplitude ~280× smaller because the manufactured
potential flow's cloud-scale curvature is ~280× smaller — which is what
`mms_null_caveat` already says ("smooth at the cloud scale"). **This is the
single strongest committed evidence for the representation mechanism, and it is
currently reported as evidence against it.**

**(c) The C¹ control does not refute this mechanism; it refutes a different
one.** §6.1 tests "second-differencing across a kink scales like `Δs/h`" and
correctly refutes it. But that mechanism lives entirely in the viscous block,
which the paper's own budget puts at 5–7% of the floor (Prop A3). The
first-derivative block never sees a kink singularity — (A1.1) says it sees a
segment *average*, which is bounded. So C¹ has no reason to help, and the numbers
agree: cubic/linear = 1.114 / 1.119 / 1.064 at 128² / 256² / 512² — the ratio
**decreases with refinement**, consistent with both interpolants approaching the
same saturation from below with slightly different amplitudes.
`truth_cubic_native` = 541 / 1163 / 1438 confirms CloughTocher's reconstruction
is violently worse somewhere excluded, i.e. its gradient error is not smaller.

**Anisotropy (the brief's mechanism 3) is disfavoured** by the same band ordering
as (a): the raster/mesh anisotropy mismatch is a near-wall phenomenon and band
0.25 excludes it, yet band 0.25 grows fastest.

**The existing `interpolant_dominated` gate does not cover this.** It is
`bool(domain.dx < s_local)` (`floor_resolution_decomposition.py:783`) — it
excludes rungs *below* the cloud spacing. It says nothing about whether the floor
at `h > s` is representation error. A reviewer answering "we already gated that"
would be wrong, and so would we.

### 1.6 What is at risk in already-committed text

This is the part of Target A that is worth more than any theorem. Specific
sentences that Prop A1 + §1.5 put at risk:

| location | sentence | status under A1 |
|---|---|---|
| `floor_resolution_study.md` §1a | "they move together, which points at a cause common to every equation … The cause consistent with that is the **reference-operator mismatch**" | **Representation error is also common to every equation** — `∇·Ū`, `u·∇Ū` and `∇P̄` all carry the same gradient reconstruction error and all are averaged the same way by (A1.1). The lockstep does not discriminate between the two mechanisms, and is currently presented as if it does. |
| `floor_resolution_study.md` §4 | "refining the grid makes it **worse**, because refinement resolves more of the very structure the monitored operator omits" | The mechanism clause is wrong under A1 (the first-derivative block converges) and is also inconsistent with the paper's own budget (repairing the omitted structure moves the floor `<0.1%`). |
| `floor_resolution_study.md` §6.2 | "it is the residue of **operator provenance**" | Unsupported: no committed measurement separates provenance from reconstruction. |
| `sections/residual_floor_theorem.tex`, "Assumptions, stated plainly" | "(H2) is a statement about auditing a finite-volume, body-fitted reference with a different discrete operator" | The `h`-*dependence* is not. The `h`-independent part may well be, but that is exactly the part not isolated. |
| same file, "certify" paragraph | "And refinement cannot recover it, because the floor grows" | **Conclusion survives, mechanism does not.** A1 gives a positive finite limit, so refinement still cannot recover the width. Reword to "does not decay to zero" and the sentence is safe. |

The report's own §6 honest bound already says the defensible claim is **"does not
decay", not "grows without bound"**. Prop A1 turns that from caution into a
theorem, and simultaneously removes the licence for the mechanism sentences.

### 1.7 The experiment that decides it — and it is cheap

**Cloud decimation at fixed `h`.** Decimate the source cloud (random or
Poisson-disk, factors 2 / 4 / 8 in point count, i.e. `s → √2 s, 2s, 2.8s`),
re-triangulate, re-rasterise at 128² and 256², measure `band_0.1` and
`band_0.25`. No new solves; the point-cloud cache
(`data/cache/airfrans_pc_full_test_n100.pkl`) is committed and
`floor_resolution_decomposition.py` already does 16 cases × 5 rungs in 17.8 min.

**The prediction, derived rather than guessed.** A4's model is
`RMS = ‖ε‖·(s/2h)^{β/2}` with amplitude `‖ε‖ ∝ s|∇²u_c|`, so

```
RMS  ∝  s^{1 + β/2} · h^{-β/2} ,      β ∈ [1, 2].
```

At **fixed `h`** the `s`-exponent is therefore `1 + β/2 ∈ [1.5, 2]` — *steeper*
than linear, not shallower. In 2-D, decimating the point count by a factor `D`
gives `s → √D · s`:

| decimation `D` | `s` factor | predicted floor rise, `β=1` (`s^{1.5}`) | `β=2` (`s²`) |
|---:|---:|---:|---:|
| 2 | 1.41 | ×1.68 | ×2.0 |
| 4 | 2.00 | ×2.83 | ×4.0 |
| 8 | 2.83 | **×4.8** | **×8.0** |

| mechanism | prediction at fixed `h` |
|---|---|
| representation | floor rises as `s^{1.5}` to `s²`: **×4.8 to ×8.0 at `D = 8`** |
| operator provenance | floor **insensitive to `s`** |

These are not close, and a null result is as informative as a positive one.
**This one experiment is worth more than all three theorem targets**, because it
decides whether four committed sentences survive. Register the table above before
running it.

**Second experiment, cheaper.** Add a 1024² rung and score `band_0.25`. The
existing ladder stops at 512² because `h/s → 1`; that is exactly where A1 says
the first-derivative block must saturate while the viscous block (Prop A3) keeps
rising. A1 predicts `band_0.25` flattens and `truth_term_visc` continues at
`p ≈ −1`. Provenance predicts continued growth of the whole floor. Report as a
prediction registered before the run.

**Third, and the cleanest if you want a positive result.** Rasterise with a
*smoothing* reconstruction whose support is fixed in physical units — moving
least-squares gradients at radius `r = 0.02c`, independent of `h`. A1's averaging
mechanism is then removed by construction: the floor should become
`h`-independent immediately and equal the provenance block plus the MLS
reconstruction error. That would *isolate* `‖T‖ + ‖τ_ref‖` and give the paper the
number it currently does not have.

### 1.8 Verdict on A

**Provable:** Prop A1 (rigorous, half a page, no modelling assumption), Prop A3
(rigorous up to the standard fraction-of-affected-cells counting), Cor A2 as a
decomposition statement.

**Not provable:** the exponent as a number. Prop A4 gives `[−1, −1/2]` and the
measurements sit in it, but `β` is a triangulation property that is not measured
and would differ per case and per region.

**Keep A1 and A4 strictly separate — this is the report's most important
bookkeeping and it is easy to blur.**

* **A1 alone** (no modelling assumption, needs only that the reconstruction is
  Lipschitz and `h`-independent) establishes that the first-derivative block
  converges to a finite `F_K`. That kills the *power-law framing* as `h → 0`:
  whatever `p` the ladder fits, it cannot continue. **A1 says the growth must
  stop. It says nothing about where.**
* **A4** — which rests on the unmeasured correlation length `s` and correlation
  exponent `β` — is what places saturation at `h ≈ s` and therefore what licenses
  the claim that the ladder's `h/s ∈ [2, 8]` is *already inside* the crossover.
  That claim is a model prediction, not a theorem, and the 1024² rung (§1.7) is
  what would test it.

So the defensible sentence for the manuscript is A1's: *the floor is bounded and
convergent under refinement, hence the fitted order is not an exponent of a
scaling law.* The stronger sentence — *and the ladder is already at the knee* —
must be flagged as a prediction until the 1024² rung reports.

**One scope note so no reader conflates two numbers.** A1 and §1.5 concern the
`h`-dependence of the **band-restricted** floors (`0.0459 → 0.1074`). They say
nothing about the magnitude of the deployed whole-fluid `‖r*‖ = 0.192` at 128²,
which is a different scoring region measured once and not refined. The theorem
file already warns the two are not interchangeable; that warning becomes
load-bearing here.

**Falsifier for A1:** a ladder that shows `band_0.25` still rising at
`h/s ≪ 1` with the viscous block subtracted. That would mean the reconstruction
is not `h`-independent (a pipeline bug) or not Lipschitz.

**Days:** 3–4 to write A1/A2/A3/A4 to publication standard with the evidence
table of §1.5; 2 for the decimation + 1024² experiments. The experiments should
run **first**, because a decimation result showing insensitivity to `s` would
mean the theory section stands as written and A collapses to a footnote.

---

## 2. TARGET B — a lower bound on certificate width

### 2.1 What the deployed certificate actually is (read off the code, not the prose)

`scripts/functional_audit_gate.py:871-882`. Score `σ_i = ‖R_h(û_i)‖`
(`residual_norm`), target `E_i = |ΔC_{D,i}|`, nonconformity `s_i = E_i/σ_i`,
`c = ` empirical 0.90 quantile of `{s_i}` on a random half, bound
`W_i = c·σ_i`. So this is **normalised (locally-weighted) split conformal**, and
the reported "median bound width" is `median_i(c σ_i)`.

Arithmetic check: `c_median = 0.051759` × `median ‖R(raw_seed0)‖ = 0.1525` =
`0.00789` = `median_bound_width`. ✔ The reading is right.

### 2.2 Proposition B1 — the closed form. It reproduces the number, and it is not a theorem.

> **Proposition B1.** Suppose the scale is affine in the error, `σ_i = f + κE_i`
> with `f, κ > 0` constants. Let `m := median(E)`, `ρ := Q_{1-α}(E)/m` and
> `φ := f/(f + κm)` (the floor's share of the median case's score). Ignoring the
> finite-sample correction,
> ```
> median(W) / median(E)  =  ρ / ( φ + (1-φ) ρ ) ,                        (B1)
> ```
> increasing in `φ` from `1` at `φ = 0` to `ρ` at `φ = 1`.

*Proof.* `s = E/(f+κE)` is increasing in `E`, so `q = Q_{1-α}(s)` is attained at
`E_α := Q_{1-α}(E)`: `q = E_α/(f+κE_α)`. `W = q(f+κE)` is increasing in `E`, so
`median(W) = q(f+κm)`. Write `D := f+κm`, so `f = φD`, `κm = (1-φ)D`, and
`f + κE_α = D(φ + (1-φ)ρ)`. Then
`median(W)/m = [ρm/(D(φ+(1-φ)ρ))]·D/m = ρ/(φ+(1-φ)ρ)`. ∎

**It reproduces the headline number on seed 0.** From
`functional_audit_gate_analysis.json` / §3.0 of the gate report:
`m = 0.0014151`, `Q_{0.9}(E) = 0.02133` ⇒ `ρ = 15.07`; `φ = 0.8636` (the
committed `floor_share_of_typical_score`). (B1) gives **5.16** against a measured
**5.568** — 8%.

**And it fails on the other two seeds.** `ρ₁ = 0.0207/0.0014866 = 13.92` gives
5.04 against a measured 6.73; `ρ₂ = 0.0236/0.0012728 = 18.54` gives 5.46 against
a measured 6.76. Under-predicting by **25% and 19%**. (Caveat: `φ` is committed only for
seed 0, so I reused it; that is part of why the check is partial, and it is also
part of why the check is not strong evidence.)

**Why B1 must not be presented as the theorem the brief wants.** The hypothesis
`σ = f + κE` says *the only defect in the scale is the additive floor*. Set
`φ = 0` and the model makes `σ` an exact oracle (`σ ∝ E`), so the formula
attributes **100% of the inefficiency to the floor by construction**. The real
scale is nothing like that: `Spearman(σ, E) = 0.592–0.626`
(`functional_audit_gate.json` §3), not 1. Agreement to 8% on one number with one
free parameter, and 25–30% disagreement on two more, is not evidence for a
mechanism. A referee who works in conformal prediction will write that sentence
in the first paragraph of the report.

### 2.3 Proposition B2 — the distribution-free version. True, and weak.

> **Proposition B2.** Let `(E_i, σ_i)_{i=1..n+1}` be exchangeable with `σ_i > 0`,
> and let `q = s_{(k)}` be the `k`-th order statistic of the calibration scores
> with `k = ⌈(1-α)(n+1)⌉`. Then
> ```
> q ≥ E_{(k)} / max_i σ_i ,   hence
> median_i(q σ_i) / median_i(E_i)  ≥  ρ_{1-α} / Γ ,    Γ := max_i σ_i / median_i σ_i .   (B2)
> ```

*Proof.* `s_i = E_i/σ_i ≥ E_i/max σ`. The map `i ↦ E_i/max σ` is a monotone
rescaling, so the `k`-th order statistic of `{s_i}` is at least the `k`-th order
statistic of `{E_i/max σ}` = `E_{(k)}/max σ`. Multiply by `median σ` and divide
by `median E`. ∎

**Hypotheses that must be stated:** exchangeability of the calibration/test pairs
(the load-bearing and, on OOD splits, violated assumption — see §2.7); `σ > 0`;
and nothing else. **What it does not say:** anything about validity. Validity is
distribution-free and holds for any score; the floor cannot break it. B2 is
strictly a width statement, which is the right genre.

**Why it is weak.** The floor enters only through `Γ`, and `Γ` is **not**
controlled by the floor in this data, because the floor's own case-to-case
dispersion is as large as the score's: `norm_truth_std = 0.1619`
(`residual_floor_realdata.json`) against a deployed-Transolver score spread of
`0.161` (`residual_floor_theorem.tex`, "Empirical confirmation"). The floor is
not a case-constant pedestal; it is a case-difficulty signal, which is precisely
why the monitor *ranks* at AUROC 0.95. Consistency check: measured 5.568 implies
`Γ ≥ 15.07/5.568 = 2.71`, satisfied and far from tight.

### 2.4 The reframing the paper must adopt whether or not B is attempted

Put three reference points on one axis, all from committed numbers, seed 0:

| scale | median width | × median error |
|---|---:|---:|
| **uninformative** (`σ ≡ const`): the marginal split-conformal interval | `Q_{0.9}(E) = 0.02133` | **15.1×** |
| **measured** (`σ = ‖R_h(û)‖`) | `0.00788` | **5.57×** |
| **oracle** (`σ ∝ E`) | `= E` | **1×** |

**The deployed residual monitor cuts the distribution-free interval by 2.7×
relative to using no monitor at all.** The `5.6–6.8×` is dominated by the fact
that `|ΔC_D|` has a 90th percentile 15× its median (mean 0.01375 against median
0.00142 — extremely heavy-tailed). That is a property of the *model's error
distribution*, not of the residual floor. Any marginally-valid distribution-free
90% interval on this error distribution is ≥ 15× the median error unless the
scale carries information; the monitor carries some, and gets to 5.6×.

Consequences for the manuscript, in `sections/residual_floor_theorem.tex`:

* **Safe as written:** "the certificate is 5.6–6.8× the drag error it certifies
  and 1.6× the median field error" (a measurement); "an interval 6× the median
  drag error is not a design tool" (a usability judgement).
* **Not supported:** the causal clause "*because* 86% of a typical prediction's
  score is floor". Neither B1 nor B2 licenses it, and the 15× marginal baseline
  argues most of the width is the error tail. Either delete the "because", or
  measure it (§2.5).
* **Currently unstated and favourable:** the monitor buys 2.7× over no monitor.
  That belongs in the "what remains deployable" argument and it is free.

### 2.5 The gate that decides whether any version of B survives — half a day

**The right oracle scale is a field difference, not a difference of norms.**
`‖R_h(û)‖ = ‖r* + Le + o(‖e‖)‖`; norms do not decompose additively, and
`‖R(û)‖ − ‖r*‖` can be negative (it is, on 2.5–7.0% of Transolver cases). The
quantity to use is

```
σ'_i := ‖ R_h(û_i) − R_h(u*_i) ‖          (the floor-subtracted monitor, oracle version)
```

**This is computable exactly from committed caches, with no forward passes.**
Prediction fields are cached as `z["raw"]` / `z["corrected"]` in
`data/cache/acceptance_gate/seed{k}/*.npz` and the truth fields in
`data/cache/airfrans_full_test_r128_n200.pkl`
(`scripts/functional_audit_gate.py:481-492`), and `PhysicsChecker` returns the
residual maps. Estimated cost: ~5 minutes of CPU, plus an afternoon of scripting.

Re-run the identical split-conformal protocol with `σ'` in place of `σ` and
compare `bound_over_error`:

| outcome | reading |
|---|---|
| drops to **≲ 2×** | the floor is causally responsible; B has a *measured* impossibility with a mechanism, and B1's model is approximately right. Worth writing up — but as a measurement with a supporting calculation, not as a theorem. |
| stays **≳ 4×** | the floor is not the cause; the width is the error tail plus the monitor's imperfect correlation with error. **Target B is dead**, and one clause of the manuscript must be deleted. |

**My prediction, registered here:** it drops, but to roughly 3–4×, not to 1–2×.
Reason: even a perfectly floor-subtracted `‖Le‖` is a *field* error norm, and the
target is a *drag* error — a functional of the field that the paper's own
functional audit shows is not recoverable from the residual by any signed
projection (AUROC 0.61 against the norm's 0.94). A scale that tracks field error
still has to certify a heavy-tailed drag error. If that prediction holds, B is
not an impossibility result about the floor; it is a statement about functional
observability, which is `sec:drag_observability` territory and already partly
written.

**Two caveats to state in any writeup.**

1. `σ'` uses `r*`, unavailable at deployment. This is an oracle counterfactual —
   exactly the right instrument for an impossibility argument and exactly the
   wrong thing to present as a method.
2. **`σ'` removes more than the floor.** The deployed Transolver's `û` is itself
   rasterised from a point cloud through the same pipeline as `u*`, so the field
   difference `R_h(û) − R_h(u*)` partially cancels *shared* rasterisation noise as
   well as the floor. That is acceptable for an upper bound on what floor removal
   could buy, but it means a large improvement in the gate is **not** evidence
   that the floor alone was responsible. A grid-native backbone (the dropout-FNO
   arm) does not share that pipeline and can be run as the contrast.

### 2.6 Prior art for B — what I searched and what I found

Searched this session (arXiv API and abstract pages). Reported honestly, including
the searches that returned nothing.

| source | relevance |
|---|---|
| Barber, Candès, Ramdas, Tibshirani, *The limits of distribution-free conditional predictive inference* (arXiv:1903.04684; Inf. & Inference 2021) | **The canonical impossibility in this space**: nontrivial distribution-free *conditional* coverage forces infinite expected width. A referee will ask how B differs, and the answer must be in the theorem statement, not the rebuttal: B1/B2 are about *marginal* width with an informative-but-floored score and are not corollaries of it. The family resemblance is close enough that "we already know distribution-free intervals must be wide" is a likely reflex. |
| Lei & Wasserman (2014, JRSS-B); Lei, G'Sell, Rinaldo, Tibshirani, Wasserman (2018, JASA; arXiv:1604.04173) | The **upper** side is settled literature: split-conformal width converges to the oracle width when the base estimator is consistent, and the locally-weighted (normalised-residual) variant is introduced there. B would be the lower side. |
| Vovk, Nouretdinov, Fedorova, Petej, Gammerman, *Criteria of efficiency for conformal prediction* (arXiv:1603.04416, 2016) | Efficiency criteria and optimal conformity measures — but for **classification**, and about probabilistic criteria, not regression interval width. |
| Papadopoulos et al., normalised nonconformity measures | Methodological; empirical efficiency gains from a good `σ`. No lower bound found. |
| Searches returning **zero** results | `abs:"conformal prediction" AND abs:"interval length" AND abs:"lower bound"`; `abs:"conformal" AND abs:"efficiency" AND abs:"oracle band"` |

**Honest risk assessment.** I found no paper stating B1 or B2. But both are
two-line calculations, and "not previously written down" and "novel enough to
survive a desk rejection on originality" are different bars. **B1 and B2 clear
the first and, on their own, almost certainly not the second.** A paper that has
been desk-rejected twice on originality should not stake a month on a two-line
lemma in a field with a well-known impossibility theorem next door. B is
publishable only bundled with §2.5's *measured* causal attribution — that is, as
an empirical result with a supporting calculation, not as a theorem.

### 2.7 A concrete implementation defect in the committed certificate — fix this regardless

`scripts/functional_audit_gate.py:876`:

```python
c = float(np.quantile(err[cal] / np.maximum(score[cal], 1e-30), q))   # q = 0.90
```

This is the **plain empirical 0.90 quantile with linear interpolation**, not the
finite-sample-corrected conformal quantile `⌈(1-α)(n_cal+1)⌉`-th order statistic.
With `n_cal = 100`, `np.quantile(·, 0.90)` returns the value at position
`1 + 0.9×99 = 90.1` of 100, so the expected coverage is `≈ 90.1/101 = 0.892`,
not `≥ 0.900`.

**The measured coverages are 0.891 / 0.8945 / 0.891 / 0.893.** That is not
sampling noise around 0.90 — it is the predicted deficit, to three decimal
places, on all four arms.

Consequences:

1. As computed, **the certificate does not carry the finite-sample split-conformal
   guarantee** `P(|ΔC_D| ≤ c·σ) ≥ 1-α`. The paper says "split conformal returns
   [a valid bound] whatever the score, and ours attains 0.891–0.895 coverage
   against a 0.90 target" — the mechanism sentence is right, the estimator used
   is not the one the theorem is about.
2. The fix is one line, and it should be written as an order statistic rather
   than as a quantile with a `method` argument, because `method="higher"` on the
   rounded level lands on the 92nd order statistic rather than the required 91st
   — valid, but one step conservative:
   ```python
   k = int(np.ceil((n_cal + 1) * 0.90))          # = 91 for n_cal = 100
   c = float(np.sort(ratio)[min(k, n_cal) - 1])  # exact conformal quantile
   ```
   Expected effect: coverage rises to ≥ 0.90. **Do not assert a width effect
   without measuring it** — the ratio distribution is heavy-tailed (`ρ ≈ 15`),
   and moving one or two order statistics in its upper tail can move the width by
   more than the "small correction" intuition suggests. Report the new
   `bound_over_error` from the rerun; it is a one-minute job.
3. **State the exchangeability hypothesis explicitly** wherever the certificate
   is claimed. It is the load-bearing assumption: the guarantee is over a random
   calibration/test exchange from one pooled AirfRANS split, and it says nothing
   about the OOD regime where a certificate is most wanted. The manuscript does
   not currently name it in the "certify" paragraph.

Route: engineering (one-line fix + rerun of `--followup`, ~1 minute of compute),
and one sentence in the manuscript.

### 2.8 Verdict on B

**Not provable as an impossibility result.** B1 is a model computation whose
hypothesis assumes the conclusion; B2 is a genuine distribution-free bound but is
about scale dispersion rather than about the floor, and is loose by a factor ~3
on this data. Neither reproduces the measured numbers on more than one seed. The
framing in the brief — "an impossibility result with its confirming measurement
already in hand" — does not survive contact with the arithmetic, because the
confirming measurement is mostly explained by a heavy-tailed error distribution
that has nothing to do with the floor.

**Days:** 0.5 for the gate of §2.5. If it passes (`≲2×`), 4–6 days to a
defensible writeup — but even then it should be sold as a measurement, not a
theorem. If it fails, 0 days and one deleted clause.

---

## 3. TARGET C — conditions for a consistent cheap monitor

### 3.1 The characterisation exists and it is a tautology

> **Proposition C1.** Let `M_h` be any deployment-time monitor. `M_h` is
> *consistent with the generator* — `M_h(u*_x) = 0` for every case `x` — iff
> `M_h(·) = R_h(·) − r*(x) + N(·)` for some `N` vanishing at `u*_x`, where
> `r*(x) := R_h(u*_x)`. Hence a cheap consistent monitor exists **iff the floor
> field `r*(x)` is computable from the deployment-time inputs at deployment
> cost.**

*Proof.* Immediate from the definition. ∎

That is the whole characterisation, and it is empty until "computable at
deployment cost" is given content. The two known ways to give it content are the
two the paper already names and dismisses: recompute the reference operator
(defect correction, `stetter1978defect`; τ-correction, `brandt2011multigrid`),
which is the cost the surrogate exists to avoid; or use a norm-equivalent
least-squares functional built jointly with the discretisation
(`bochevgunzburger2009book`), which a post-hoc audit forgoes by construction.

### 3.2 The one non-obvious observation, and it cuts against a committed sentence

**`r*(x)` has free labels.** At training time the truth is available on every
training case, so `r*(x) = R_h(u*_x)` is computable at zero marginal cost for the
entire training set. It is therefore an ordinary supervised regression target,
and the design rule writes itself: **predict `r̂*(x)` from `(sdf, mask, BCs)` as
an auxiliary head, and monitor `‖R_h(û) − r̂*‖`.**

This is worth writing down because the theorem file currently says the correction
term is "exactly the object a deployment-time monitor cannot form, since it would
require the fine operator we are trying to avoid running", and — separately —
that `zhang2026phymgn`'s remedy "likewise requires `R_h(u*)`, which a
deployment-time monitor by definition does not have". **Both sentences are too
strong.** It does not require the fine operator; it requires a *prediction* of a
quantity for which supervised labels are free. The honest statement is that the
monitor cannot form it *exactly*, and that its error is bounded below by the
floor-prediction error, which is a different and weaker claim.

### 3.3 Why this is not a contribution

1. **Prior art in kind.** It is `zhang2026phymgn`'s remedy (define the loss
   relative to the residual present in the ground-truth data) moved from training
   time to audit time, and it is defect correction with a learned defect. The
   delta is small and the paper already cites both.
2. **The negatives survive it verbatim.** With `r̂*` in place of `r*`,
   `thm:residual-floor` legs (i)–(iv) hold with `r*` replaced by the residual
   `r* − r̂*`, and `prop:kernel` (K1)–(K4) are untouched — the kernel is a
   property of `L`, and subtracting a constant field changes nothing. So the
   design rule shrinks the floor's magnitude and removes none of the structural
   obstructions. The "descend" negative in particular is unaffected: the
   minimiser set of `½‖R_h(u) − r̂*‖²` is still large and still does not contain
   `u*` unless `r̂* = r*` exactly.
3. **The identifiability obstruction is real but is not a theorem.** `r̂*` must
   be validated on the distribution it is deployed on. That is exactly the OOD
   regime where a certificate is wanted, and exactly where the conformal
   exchangeability hypothesis (§2.7) also fails. **The failure modes of the floor
   predictor and of the certificate coincide**, which is a good paragraph and a
   genuinely useful design warning. It is not a theorem, and attempting to make
   it one would require a distribution-shift assumption that the paper cannot
   justify from one dataset.
4. **The failed goal-oriented attempt is not evidence against C**, and the report
   should not use it as such. `functional_audit_gate.md` §5.1 shows a *signed*
   functional destroys ranking through cancellation; that is a fact about signed
   projections (the unsigned integral recovers AUROC 0.94–0.96 and beats the norm
   on Spearman), not about consistency.

### 3.4 What it costs to answer the useful half of C

**Zero extra work, because §2.5's gate is the same experiment.** Running the
oracle floor-subtracted conformal certificate simultaneously gives (i) B's causal
attribution and (ii) the *upper bound* on what any learned `r̂*` could achieve,
since the oracle `r*` dominates every predictor of it. If the oracle version does
not materially improve the certificate, no learned floor predictor can, and C is
closed for the price of B's gate.

If the oracle version *does* improve it materially, the follow-up is 2 days: fit
`r̂*` from `(sdf, mask, u_in, log_Re)` on the training split, measure the
floor-prediction error against `‖Le‖`, and report the resulting width. That would
be a small constructive addendum — a design rule with a measured ceiling — not a
theorem, and it should be scoped as such.

### 3.5 Verdict on C

**Not provable as an interesting theorem.** The characterisation is a tautology;
the constructive version is prior art in kind; the surviving obstruction is an
identifiability argument that cannot be made distribution-free. **Rank lowest.**
The useful half comes free with B's gate, and the §3.2 observation should be
actioned as a **softening of two over-strong sentences** in the theorem file.

---

## 4. What none of this touches

For the record, so the author does not over-read this report. Nothing above
weakens:

* `thm:consistency-floor` (a), (b), (c). The proof is correct as written; the 2-D
  identity `|T| = √2‖S‖_F|∇ν_t|` and its degeneracy characterisation are right;
  the hypotheses are correctly scoped to the exact continuum solution on a
  compact interior set, and the text already says the measured floor is a
  different object. Prop A1 is a statement about the *rasterised label* and
  brackets it from the other side.
* `thm:residual-floor` (i)–(iv), including the repaired leg (iii) lower bound
  `‖e‖ ≥ ‖Pr*‖/σ_max` holding uniformly over the affine minimiser set, and the
  corrected cone constant `α < ‖Pr*‖`.
* `prop:kernel` (K1)–(K4).
* The measured negatives that carry the paper: 24/24 residual-descent divergence
  from the exact truth; 0/16 decay under refinement; the 5/5-seed W1 ablation;
  the pre-registered functional gate's double failure; AUROC 0.952 on
  worst-decile drag error.

Those are the paper's assets. The theory section is already the right *size*; the
question is whether two of its mechanism attributions are right, and §1 says one
of them probably is not.

---

## 5. Recommendation

**Do not spend a month. Spend one week, and spend it defending rather than
extending.**

| # | task | days | why |
|---:|---|---:|---|
| 1 | **Cloud decimation at fixed `h`** (§1.7) | 1.5 | Decides whether four committed mechanism sentences survive. Highest expected value in the project. |
| 2 | **Oracle floor-subtracted conformal gate** (§2.5) | 0.5 | Decides B *and* the useful half of C. Committed caches, no forward passes. |
| 3 | **Fix the conformal quantile** (§2.7) | 0.1 | The certificate as computed does not carry the guarantee it claims. One line, one rerun. |
| 4 | **Write Prop A1 + A3 + the §1.5 evidence table** into `sec:floor_ladder`, and edit the four mechanism sentences of §1.6 | 3 | Converts "does not decay" from a measurement into a theorem, and pre-empts the "you are differentiating an interpolant" objection instead of inviting it. |
| 5 | Soften the two over-strong sentences of §3.2; add the exchangeability sentence; add the 2.7×-over-no-monitor sentence of §2.4 | 0.5 | All corrections, all cheap. |

**Total ≈ 5.5 days, then submit.**

**Why not the month.** The three targets fail for three different reasons and
none is fixable by effort. A's exponent is not a stable quantity — it is a
crossover slope over `h/s ∈ [2,8]`, and no amount of analysis makes a crossover
slope into a scaling law. B's impossibility dissolves on arithmetic: the measured
inefficiency is mostly a heavy-tailed error distribution, and the two theorems
available are a model computation and a loose dispersion bound, next door to a
famous impossibility theorem that referees will reach for first. C is a tautology
whose constructive form is already cited in the paper's own related-work
paragraph.

**What the exposure actually is, sized honestly.** Four sentences in one section
attribute the ladder's rise to operator provenance, and the evidence in §1.5 says
they are wrong. That is an edit, not a retraction: the refinement leg's
*conclusion* ("refinement does not recover the floor") survives under A1, the
thesis survives under `thm:consistency-floor`, and every measured negative in §4
survives untouched. The reason to spend the week is that a referee who notices
what §1.5(b) notices — that the paper's own pipeline null rises at the fine end,
at the same slope as the data it is a null for — would frame it as a fatal
oversight rather than as four sentences. Finding it internally converts it into a
strength.

**Why the paper is still worth submitting.** Its value was never the theory. It
is a pre-registered, gated, multi-seed diagnostic with unusually disciplined
negative results and a rare willingness to withdraw its own claims. Prop A1 makes
the refinement leg *stronger* than it is now — a proved finite limit is a better
answer to "refine the grid" than a fitted negative exponent, because the fitted
exponent invites a referee to ask what happens at `h < s`, and the current text
has no answer. Tasks 1–5 close that hole. A month of theorem-hunting would not.

**The one thing that would change this verdict.** If task 1 shows the floor is
insensitive to cloud decimation at fixed `h`, then the representation mechanism
is dead, the operator-provenance attribution is vindicated, and the growth
becomes a genuinely unexplained phenomenon worth a month — because then Prop A1
would be in tension with the data and something in the pipeline is not what it
claims to be. That is the branch worth buying with 1.5 days.
