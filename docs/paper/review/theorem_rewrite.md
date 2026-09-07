# Rewrite of `sections/residual_floor_theorem.tex` — what changed and what it costs

Author: theory/formal-guarantees. Date 2026-09-07.
Inputs: `theorem_audit.md`, `floor_resolution_study.md`, `FINDINGS.md` (F3, F4),
`body.tex` contribution list, and the implementing code
(`physics/residuals.py`, `physics/operators.py`, `core/types.py`).

The file was edited concurrently while this task ran. This report describes the
diff against the state on disk at the time of the rewrite, which already carried
a repaired leg (iii), a corrected leg (iv) cone constant, a restated leg (A), and
the classical-relatives paragraph.

---

## 1. What is newly proved

### Theorem 1 (`thm:consistency-floor`) — the consistency floor

This did not exist anywhere in the paper. `body.tex:91-96` asserts it in bold
("provably ... its floor has a nonzero continuum limit ... (H2) stops being an
assumption") and nothing discharged that assertion. It is now discharged.

**Statement.** Let `(u*, p*, nu_t)` solve steady incompressible RANS with a
*linear (Boussinesq) eddy-viscosity closure* — modelled deviatoric Reynolds
stress `-2 nu_t S`, isotropic part absorbed into the modified pressure. Let
`K ⊂⊂ Omega_f` be compact with `u* ∈ C^4(K)`, `nu_t ∈ C^1(K)`. Then

* **(a)** the monitored dimensional residuals satisfy `r_c = O(h^2)` and
  `r_i = T_i + O(h^2)` with `T_i = (∂_j nu_t)(∂_j u_i + ∂_i u_j) = 2(S ∇nu_t)_i`;
  and in 2-D, exactly, `|T| = 2 λ |∇nu_t| = √2 ‖S‖_F |∇nu_t|`.
* **(b)** `T(x) = 0` iff `S(x) = 0` or `∇nu_t(x) = 0`.
* **(c)** if `|{x ∈ K : S ≠ 0 and ∇nu_t ≠ 0}| > 0` then
  `liminf_{h→0} ‖R_h(u*|_h)‖ ≥ (|K|/|Ω|)^{1/2} (ℓ/U²) ‖T‖_{RMS,K} > 0`.

**Hypotheses, all load-bearing.**

| Hypothesis | Why it is needed | Honest status |
|---|---|---|
| Linear eddy-viscosity closure, stress `-2 nu_t S` | fixes the term the monitor omits | AirfRANS uses Spalart–Allmaras, which is of this form; the statement is written to cover k-ω SST too |
| `u* ∈ C^4(K)`, `nu_t ∈ C^1(K)` | `C^3` for the central first difference, `C^4` for the compact second difference, both `O(h^2)`; `C^1` for `∇nu_t` to exist | genuine restriction — see the rasterisation caveat below |
| `K` compactly interior to the fluid domain | uniform consistency constants; excludes the wall | same methodological choice the refinement ladder makes with its fixed physical band; stated as such in the section |
| `S ≠ 0` and `∇nu_t ≠ 0` on positive measure | non-degeneracy | very weak; equivalent to "the flow deforms and the eddy viscosity varies somewhere" |

**Two sharpenings beyond the audit's sketch.**

1. The audit conditioned non-degeneracy on `∇nu_t` "not orthogonal to the strain
   rate". **That hypothesis is removable.** In 2-D, incompressibility makes `S`
   symmetric *and traceless*, so `S = [[a,b],[b,-a]]`, `det S = -(a²+b²)`, and
   both singular values equal `λ = √(a²+b²)`. Hence `|Sx| = λ|x|` for *every*
   `x`, giving the exact identity `|T| = √2 ‖S‖_F |∇nu_t|` and degeneracy only
   when a factor vanishes. There is no orthogonal direction to hide in.
2. The degeneracy set is exactly the uniform freestream (`S = 0`, `∇nu_t = 0`),
   which is precisely the field of leg (i) with an exactly zero residual. The
   theorem predicts its own exception, and the exception is the spurious
   minimum. This is stated in the section as an internal consistency check.

**Corroboration from a committed artifact.** The algebraic identity of (a) is
already verified numerically as a unit test in
`results/certificates/floor_resolution_decomposition.json`:
`stress_divergence_identity_order = 1.93`,
`omit_gradnu_exactly_zero_for_constant_nu = true`,
`omit_gradnu_fires_for_varying_nu = true`. That is cited in the section.

### Proposition 1 (`prop:kernel`) — the undetectable subspace

Replaces a paragraph that offered two examples with four provable structural
facts, none of which needs a measurement:

* **(K1)** `dim ker L ≥ 4|M| - 3|A| = |M| + 3|W| ≥ |M|` — at least a quarter of
  the fluid state, by counting (three equations per cell on four unknowns, plus
  three dimensions per zeroed wall-ring cell). Claimed as a *fraction*; the
  integers `|M|, |W|, |A|` are deliberately not chased, since the fraction is
  what carries the argument.
* **(K2)** Solid degrees of freedom are in `ker R_h` **exactly and nonlinearly**,
  not merely in `ker L`. Requires the hypothesis that the body is at least three
  cells from the crop border (the one-sided edge stencils reach two cells
  inward); satisfied by the 3c×3c crop about a unit chord. Verified against
  `residuals.py:140-154` — the wall ring is a 4-connected dilation, and every
  interior stencil is 4-local.
* **(K3)** `R_h` is **affine** in `nu_t`, so the `nu_t` Jacobian block is
  **diagonal with closed-form entries** `c_k = (ℓ/U²)√((Δ_h u*)² + (Δ_h v*)²)`,
  and the statement is exact rather than first-order. Requires the clip
  `max(nu_eff, 0)` to be inactive, i.e. `nu + nu_t + δ ≥ 0`; stated. Physical
  reading: eddy viscosity is unobservable wherever there is nothing to diffuse.
* **(K4)** The Nyquist checkerboard `p_ε(i,j) = ε(-1)^{i+j}` is annihilated by
  the central difference at **every interior node**; the response is confined to
  the outer border ring, an `O(1/N)` fraction of the active set, and would be
  *exactly* invisible were that ring excluded as the wall ring already is. Named
  as classical odd–even decoupling and cited to Rhie–Chow.

**On (K4) I deliberately did not do what was tempting.** The audit forbids
claiming the checkerboard as an exact null mode (correctly — the one-sided edge
stencils return `2ε/h`); `FINDINGS.md` F3 asks for the Rhie–Chow mode to be
named. Both are satisfied by claiming *interior* annihilation plus confinement,
and by explicitly noting it is not an exact null mode. I also computed and then
**discarded** an RMS gain ratio against a smooth pressure mode (≈0.60 at
N=128): it is weak, it is an artifact of the border rather than the physics, and
it would replace a true exact statement with a soft measured one that invites
"so it is only 40% suppressed".

---

## 2. What was dropped, weakened, or withdrawn

| Item | Action | Reason |
|---|---|---|
| Contribution (a): "the *quantified* operator-specific floor `‖e_inf‖ = ‖L⁺r*‖ ≤ ‖r*‖/σ_min`" | **Withdrawn explicitly in the text** | wrong direction for the thesis and numerically vacuous; the section now says so and says we withdraw it |
| The σ_min two-sided bound in the "good detector" paragraph | **Removed** | it was the same demolished constant, left load-bearing two paragraphs after being purged from leg (iii). Ranking is now claimed as *measured*, with `eq:rf-decomp` offered only as a labelled heuristic |
| "(H2) holds because `R_h` omits the no-slip closure *and* the 128² grid under-resolves the boundary layer" | **Replaced** | refuted by the ladder: the floor *grows* under refinement. (H2) now attributed to operator provenance |
| The closure-omission mechanism as the section's explanation of the floor | **Demoted throughout** | repairing the full stress divergence moves the floor <0.1%; the term is 3.5–4.8% |
| Leg (A) "for every λ ≥ 0" as a resolution-independent closed form | **Scoped to the deployed resolution**, with the full margin collapse `+9.3% → -2.8%` and `λ* ≈ 600` reported as a measured limitation | margin crosses zero by 512²; the ordering was never universal (11/16 across all rungs) |
| The section's opening sentence | **Corrected** | it described `tab:iters` backwards: the residual *rises* (0.113→0.618) while error *falls* (3.92→2.29, then back to 2.58) |
| Legs (iii)/(iv) presented without scope | **Scoped** as linearised/gradient-flow statements, pointed at `sec:descent` for the nonlinear measured version | the audit's "the system never runs residual descent" is now stale — `sec:descent` does exactly that, so (iv) is the mechanism for a measured result rather than a hypothetical |

**Kept, because correct.** Legs (i), (ii), (iii) as repaired, and (iv) with the
corrected constant `α < ‖P_range L r*‖`. Leg (i) now carries the one-line reason
it cannot be a rasterisation artifact: a spatially constant field has identically
zero finite differences at every `h` and for every stencil, one-sided edges
included.

**One strengthening of leg (iii) worth flagging.** Since `L⁺` maps into
`(ker L)^⊥`, every minimiser `e = e_inf + k` satisfies
`‖e‖² = ‖e_inf‖² + ‖k‖² ≥ ‖e_inf‖²`. The lower bound therefore holds
**uniformly over the whole affine minimiser set**, not just at the min-norm
tie-break. This *retires* the audit's objection (c) ("it identifies the wrong
point") rather than conceding it, since descent landing on the projection of the
prediction is still covered.

---

## 3. Honest scope — what the section now claims and does not

* **Proved:** the floor is bounded away from zero in the continuum limit
  (Theorem 1); the truth is neither a minimiser nor generically a stationary
  point, and every minimiser of the linearised objective is displaced by at
  least `‖P r*‖/σ_max` (Theorem 2); the undetectable subspace is at least a
  quarter of the fluid state with three exactly characterised families
  (Proposition 1).
* **Not proved, and stated as not proved:** the *size* of the observed floor.
  Theorem 1's term is ~4% of it. The section says explicitly that the theorem
  closes "refine the grid" *in principle* and the ladder closes it *in
  practice*, that neither explains the other's magnitude, and that the measured
  driver is the Cartesian-stencil / body-fitted-reference mismatch evidenced by
  continuity and momentum rising at the same rate in the same cases.
* **Not quantified, and the word "quantified" is not used:** `σ_max` and
  `‖P r*‖` are unmeasured, so leg (iii) is a strictly-positive-displacement
  result, not a detection limit.
* **Not claimed:** that DWR provably fails here (would need `z` constructed and
  `⟨z, r*⟩` bounded below); that the checkerboard is an exact null mode; that
  the monitor cannot rank.
* **Caveat stated in the section:** the measured `r*` is evaluated on a
  rasterised `u*` — a linear interpolant of an unstructured FV solution — which
  is not `C^4` and carries representation error. So the measured floor mixes the
  theorem's term with that error. The theorem is stated as a limit on the exact
  solution; the magnitude is reported as a measurement; neither is presented as
  evidence for the other.
* **Three-verb alignment:** the section is explicitly organised as certify-no
  (theory), descend-no (theory + `sec:descent`), rank-yes (measured, with the
  0.952 drag AUROC named so no reader can read the section as denying ranking).
  A dedicated subsection, "Ranking survives; certification does not", exists to
  make this impossible to misread.

**A pre-emption I added that was not in the brief.** Our own MMS gate reports
`p = 2.06` and could be read by a numerical analyst as proving the operator
consistent. It cannot: the manufactured field is a potential flow, so
`∇²u ≡ 0`, the monitored viscous term vanishes identically, and the field is an
exact solution of the monitored operator *for any* `nu_t` — the same artifact
records that multiplying `nu_t` by 50 changes the norm by `6.7e-6` relative
(`native_blind_to_nut: true`). The MMS gate measures truncation and is
structurally blind to inconsistency. Better we say this than a reviewer.

---

## 4. Citations

**Added to `refs.bib`, both verified against Crossref this session:**

* `rhiechow1983` — Rhie & Chow, AIAA Journal **21**(11):1525–1532, 1983,
  doi:10.2514/3.8284. Used for (K4).
* `bochevgunzburger2009book` — Bochev & Gunzburger, *Least-Squares Finite
  Element Methods*, Springer, Applied Mathematical Sciences, 2009,
  doi:10.1007/b13382. Used for the norm-equivalence line. (Series volume number
  not asserted — Crossref did not return it.)

**Already present and used:** `stetter1978defect`, `brandt2011multigrid`,
`morin2000data`, `beckerrannacher2001dwr`, `lei2026newtonkrylov`,
`zhang2026phymgn`, `krishnapriyan2021failure`, `wang2022ntk`, `trefethenbau1997`.

**Not used, flagged.** The pre-existing key `bochev2009lsfem` points to *"A
locally conservative least-squares method for Darcy flows"*, CNME 24(2):97–110 —
a real paper, but about local conservation, **not** LSFEM norm-equivalence, so
it was the wrong support for the claim it was attached to. I switched the
citation to the book rather than edit the entry. Its `year = {2009}` also looks
wrong (CNME vol. 24 issue 2 is 2008); worth checking if it is cited elsewhere.

**The novelty sentence, now present.** Dual-weighted-residual and goal-oriented
estimation work for reduced-order models precisely because
`r = A_FOM(u_ROM) - b` vanishes at the truth *by construction*. A
dataset-trained surrogate has no `A_FOM`. The obstruction localises on
**operator availability** — not RANS, not 2-D, not the grid.

---

## 5. Claims elsewhere in the manuscript that now depend on this section

| Location | Depends on | Status |
|---|---|---|
| `body.tex:91-96` contribution 2, bold "provably ... nonzero continuum limit ... (H2) stops being an assumption" | Theorem 1 | **now discharged** |
| `body.tex:81-89` contribution 1, "exact zero at every resolution" | Theorem 2(i) + its new one-line justification | supported |
| `body.tex:340-344` `tab:positioning`, "Nonzero, with a nonzero continuum limit" | Theorem 1 | supported |
| `body.tex:1253-1259` "the inconsistency of the monitored operator gives the floor a nonzero continuum limit" | Theorem 1 | supported |
| `body.tex:243-246` "a nonzero floor at the reference solution and a kernel of undetectable modes" | Theorem 1 + Proposition 1 | supported, and now much stronger on the kernel |
| `body.tex:1285-1292` `sec:descent`, "this is theorem leg (ii) as a measurement" | Theorem 2(ii) | supported; see suggested edit E4 for the now-ambiguous "theorem leg" |
| `body.tex:1330-1332` "its minimiser is displaced from `u*`" | Theorem 2(iii) | supported |

---

## 6. Edits still required in `body.tex` (not made — out of scope per the brief)

Ranked by severity.

* **E1 (required, factual).** `body.tex:266`: "the *residual-floor theorem*
  **quantifying** the operator-specific detection limit and the undetectable
  kernel modes". Nothing quantifies a detection limit — `σ_max` and `‖P r*‖` are
  unmeasured and the section deliberately avoids the word. Suggested: "the
  *residual-floor theorem* establishing a nonzero continuum limit for the floor
  and characterising the undetectable kernel modes".
* **E2 (required, consistency with the proof).** `body.tex:93-94` says the
  operator "uses `ν_eff∇²u` in place of `∇·(ν_eff∇u)`". Three operators are in
  play and this names the weakest gap. The true RANS operator is the full
  symmetric stress divergence `∇·(ν_eff(∇u+∇uᵀ))`, whose omission is the
  symmetric pair `(∂_j ν_t)(∂_j u_i + ∂_i u_j)` — and that is also what the
  resolution study's "repaired" control repaired. Theorem 1 is proved against
  the symmetric form (the `∇·(ν_eff∇u)` comparison follows a fortiori).
  Suggested: replace `∇·(ν_eff∇u)` with `∇·(ν_eff(∇u+∇uᵀ))`.
* **E3 (recommended, audit O2/O5).** `body.tex:225-232`: the theorem "marks the
  boundary between" success and failure, and correction "fails where the
  monitored residual carries a nonzero floor". The theorem contains no statement
  about the success regime, and carrying *some* floor is not sufficient for
  failure — every discrete monitor carries one. The operative quantity is the
  signal-to-floor ratio at the operating point, which the paper has:
  `‖R_h(û)‖/‖r*‖ ≈ 0.59` in the mean, `< 1` on 80% of cases. Suggested: scope to
  "the floor is a necessary condition for the failure mode we observe, and prior
  successes operate where it is absent", and quote the ratio.
* **E4 (minor, now-ambiguous reference).** `body.tex:1288` "This is theorem leg
  (ii) as a measurement". With two theorems in the section this no longer
  resolves. Suggested: "This is \autoref{thm:residual-floor}(ii) as a
  measurement".
* **E5 (minor).** The section now names the monitored scalar explicitly
  (whole-grid RMS of `√(r_c²+r_x²+r_y²)`, per-equation non-dimensionalised, BC
  excluded) and warns that the ladder's band-restricted numbers are a different
  scoring region. If `sec:floor_ladder` does not already say which norm each of
  `0.192` and `0.0624` is, it should, since both appear for `128²`.

---

## 7. Routing to engineering (unchanged from the audit, still open)

* `results/certificates/residual_floor_realdata.json` carries
  `"monitored_residual": "... matching physics_residual_torch"`. **False** — the
  probe calls `PhysicsChecker.diagnose`, which uses the *compact* Laplacian
  (`operators.py:186`), while `physics_residual_torch` builds the *wide* stencil
  `ddx(ddx(·))` (`residuals.py:524-525`), which annihilates every period-2 mode
  and therefore has a strictly larger kernel. Fix the string or reconcile the
  stencils.
* `residuals.py:200` boosts the no-slip penalty on the wall layer;
  `residuals.py:348` then zeroes `bc` on exactly those cells. The boost is dead
  code as far as the monitor is concerned. **Not raised in the section** —
  E2 of the audit never reported, the sign flip is an unconfirmed estimate, and
  the brief did not ask for it. It remains a live risk for the λ-sweep paragraph
  if a reviewer reads `residuals.py:348`.
* Decide and document one monitored scalar: `Diagnostics.residual_norm` is RMS
  over the whole grid; `scripts/probe_residual_floor.py` uses RMS over the
  fluid. They differ by `√(|M|/|Ω_h|) ≈ 0.99` here, so nothing turns on it, but
  the paper should name one — the section now names the whole-grid version.

---

## 8. Build notes

* No LaTeX toolchain was available in this session, so the file was **not
  compiled**. It should be built before commit.
* New environments: a second `theorem` and one `proposition`. Both
  `\newtheorem{theorem}` and `\newtheorem{proposition}` are declared
  (`preamble.tex:65-66`) and both `\theoremautorefname` and
  `\propositionautorefname` are provided (`preamble.tex:58-59`).
* Theorem numbering shifts by one for `thm:residual-floor`. Grepped `body.tex`
  for prose referencing a theorem by number: **no matches**, so nothing breaks.
* `mathtools` is **not** loaded, so `psmallmatrix` was replaced with
  `\left(\begin{smallmatrix}...\end{smallmatrix}\right)` (amsmath).
* **Two symbol collisions resolved.** (i) `L` is the Jacobian in this section, so
  the reference length in the non-dimensionalisation is written `ℓ` throughout.
  (ii) The strain-rate eigenvalue in `eq:2d-identity` is written `λ_S`, because
  the boundary-term paragraph uses `λ` for the no-slip weight (`ρ_λ`, `λ* ≈ 600`,
  `λ = 1`), which matches `bc_weight_sweep.py` and must not be renamed.
* **`|Ω|` pinned to the crop, not the fluid region.** `eq:floor-limit` averages
  over the whole grid, so the region-ratio factor is `(|K|/|Ω_crop|)^{1/2}` with
  `Ω` the full crop domain — now defined explicitly in the setup, with
  `Ω_f ⊂ Ω` the fluid region. Resolving `|Ω|` to the fluid domain would overstate
  the bound. The proof now shows the mean-square step explicitly
  (`MS_Ω ≥ (n_K/n_Ω) MS_K`, discarded cells contributing non-negatively) rather
  than asserting the ratio.
* The proof of (a) now states that `ν_eff ∂_i(∂_j u_j)` vanishes for the
  **exact** solution, and that the discrete divergence of the rasterised field
  (`≈ 0.12 U/L`) is a separate quantity treated later — pre-empting a reader who
  knows that number.
* Labels preserved: `sec:residual_floor`, `thm:residual-floor`, `eq:rf-decomp`,
  `eq:rf-lower`. New: `thm:consistency-floor`, `prop:kernel`, `eq:monitored`,
  `eq:rans`, `eq:omitted-term`, `eq:2d-identity`, `eq:floor-limit`.
* All outbound `\autoref` targets verified to exist: `sec:iters`, `tab:iters`,
  `sec:descent`, `sec:selective`, `sec:floor_ladder`, `sec:indist-ablation`,
  `sec:method`.
