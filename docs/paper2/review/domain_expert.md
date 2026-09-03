# Referee report — Computers & Fluids

**Manuscript:** "A perfect flow field can be a bad initial condition: representation, not
accuracy, decides neural warm starts for RANS" (A. Jabbary, K. Ghanavati)

**Reviewer role:** CFD + neural-operator domain expert.
**Basis of review:** the full draft (`docs/paper2/DRAFT.md`), the supporting code
(`src/neuroforge/solver/{placement,scoring,warmstart,openfoam,cgrid}.py`,
`scripts/{placement_probe,corpus_probe,mesh_native_probe,sequencing_probe}.py`) and the
committed result files in `results/`. Several of the findings below are drawn from the
authors' own committed data rather than from the text.

**Recommendation: major revision.** Score **4/10** as it stands; the underlying experiment
is worth publishing, but one abstract-level claim is contradicted by the repository's own
diagnostics and the central contrast is not the controlled contrast it is described as.

---

## 1. Summary and the technical core as I understand it

The paper asks whether a neural surrogate's field can be used as the initial condition for
a production steady RANS solve (OpenFOAM v2606 `simpleFoam`, SIMPLEC, Spalart–Allmaras,
body-fitted C-grid, 31,700 cells, first cell 1e-5 chord, Re = 3e6, NACA 4-digit sections),
and answers: only if the surrogate's *output representation* samples the wall-normal
direction where the solver's cells are.

The argument has four moving parts.

1. **An accuracy-free demonstration.** Take the converged field, hand it back three ways.
   Mesh-native: +92.0% iteration saving on Cd@1%. Through a wall-fitted 256x64 grid:
   −181.6%. Through a uniform 128^2 raster: −548%.
2. **A closed form for the near-wall damage.** A resampled field gives every cell inside
   the representation's first station that station's value, so the first-cell wall gradient
   is overestimated by `G = u+(y1+)/u+(yc+)` — the law of the wall, no fitted parameter,
   measured as an upper bound (over-predicting by 1.3–2.6x) across a fifty-fold range of
   stations and Re 1e3 to 3e6. Consequence: placement, not budget (a 512^2 raster fails;
   an 8,192-value graded grid passes).
3. **A self-falsification.** Grading the representation so its first station lies inside
   the first cell takes the wall-gradient error from 1219% to 1.8% — and convergence gets
   *worse* (−266.8%, 0/5). Wall-function repair and smoothing of the repair likewise move
   convergence by 0.1 and 0.6 points. The near-wall state is therefore declared not the
   mediator, and the damage is relocated to the **pressure** field on the grounds that every
   projection preserves viscous drag (+69 to +86%) while destroying total drag (Cd_p
   −183.9%).
4. **A recipe and its accounting.** Mesh-native, boundary-layer-only, velocity + eddy
   viscosity together: +18.4% on Cd_v across 13 cases, 13/13, p = 0.0002; a K = 25 probe
   acceptance gate bounding the worst case at (1 + K/N) x cold; a charged grid-sequencing
   baseline that makes a better seed but is said to lose on price.

I have checked the physics of (2), the implementation of the metric, the provenance of
every headline number in (1) and (3), and the fairness of the baseline in (4).

---

## 2. Technical strengths (genuinely correct and well executed)

These are not courtesies; each is something I checked and expected to find wrong.

- **The law-of-the-wall implementation is right, including the part reviewers assume is
  wrong.** `placement.u_plus` blends `u+ = y+` (y+ < 5) into `u+ = ln(y+)/0.41 + 5`
  (y+ > 30) linearly in `log y+`. Evaluated against Spalding's law: at y+ = 10 the blend
  gives 8.21 (Spalding ≈ 8.2); at y+ = 15 it gives 10.08 (Spalding ≈ 10.05). For a
  quantity that is a *ratio* of two `u+` values this is more than adequate, and the
  docstring's justification for it is correct.
- **`friction_velocity` is correct** for kinematic fields (`u_tau = sqrt(nu du/dy|_w)`),
  and recovering `u_tau` from the converged solution's own first-cell gradient is sound
  *here* precisely because the mesh has y+_c ≈ 0.7: the first cell sits in the linear
  sublayer, so `u_c/y_c` approximates `du/dy|_w` to O(y+^2). This is one of the few
  configurations in which that shortcut is defensible, and the paper is in it.
- **`iterations_to_force_band` implements what it claims.** The walk-back-from-the-end
  over the maximal in-band run is the correct implementation of "enters *and stays*", not
  the far commoner (and wrong) first-crossing. `bounded_saving` censoring non-finishers at
  budget rather than dropping them is the right call and the reported before/after
  (−199.4% vs −31.2%) shows it mattered.
- **§4 is the most transferable part of the paper.** The eight rules — thresholds read as a
  multiple of the residual floor, one external shared reference, reference restricted to
  settled arms, per-`Time`-block parsing under `nNonOrthogonalCorrectors`, "a converged run
  is finished, not truncated" — are correct, non-obvious, and each is demonstrated to have
  changed a sign on this project's own data. I would accept a shorter paper built around
  §4 alone.
- **The under-relaxation diagnosis is real physics, not tuning.** Identifying SIMPLEC at
  0.9 as the residual-floor cause (1.1e-5 → 3.5e-7 at U 0.7 / nuTilda 0.4) and ruling out
  the 2e5-aspect-ratio cells is the correct order of investigation and is documented in
  `openfoam.py` with the alternatives that were tried and failed.
- **The falsified predictions are reported as falsified.** §7.2 (pressure seeding is inert
  because SIMPLE recomputes p from continuity — physically correct), §7.3 (an oracle wake
  seed is worth +0.5% here), §5.3 (the extrapolation and divergence explanations both
  falsified), and the withdrawal of the earlier 13%-accuracy claim after the
  nearest-vertex wall-distance bug. This is unusually honest practice and should be
  preserved through revision.
- **Controls are present and used as gates.** A converged-field oracle in every tree, read
  before anything else, with the rule that a failing control voids the experiment.

---

## 3. Technical weaknesses, ranked

### F1 — **FATAL (as written).** The paper's central twist rests on a description of a seed that the authors' own committed data contradicts

The abstract, Highlight 4, Contribution 4, §5.2.1, §6.7 and §11 all turn on this sentence:

> "The arm reproducing the near-wall state to **1.8%** and its roughness exactly converges
> at −266.8%, winning 0 of 5."

The 1.8% is **one scalar** — the first-cell wall-normal gradient. The same experiment's
`round_trip_change_pct` diagnostic, written by `scripts/placement_probe.py` and committed
in `results/placement_naca*.json`, records the relative L2 change the round trip inflicts
on the field **inside the boundary layer**. For the oracle family:

| case | arm | u | v | nut |
|---|---|---:|---:|---:|
| 0012@4 | `or_proj_coarse` | 26.5% | 30.8% | 30.5% |
| 0012@4 | `or_proj_fine`   | **28.3%** | 32.1% | **40.0%** |
| 0012@0 | coarse / fine | 26.6 / **28.4** | 28.8 / 28.1 | 30.5 / **40.6** |
| 0015@6 | coarse / fine | 26.0 / **29.1** | 23.1 / **31.0** | 30.1 / **37.5** |
| 2412@2 | coarse / fine | 26.4 / **28.5** | 26.1 / **27.4** | 31.4 / **39.3** |
| 2415@5 | coarse / fine | 27.4 / **28.8** | 25.9 / **28.3** | 31.0 / **45.0** |

So `or_proj_fine` is **not** "a near-perfect reproduction of the converged boundary layer".
It is ~28% wrong in `u` and ~40% wrong in `nut` inside the layer, and it is worse than
`or_proj_coarse` on `u` in 5/5 cases and on `nut` in 5/5 cases. The only quantity on which
it is better is the single wall-gradient scalar.

Three consequences, and they are not cosmetic:

1. The abstract's factual claim is false as stated and must be corrected.
2. The falsification is much narrower than advertised. What §5.2.1 demonstrates is that
   *the first-cell wall-gradient scalar is not a sufficient summary of what a projection
   does to a seed* — not that "the near-wall state is not the mediator". The near-wall
   *state* (u, v, nut in the layer) was never restored by any arm in this study.
3. The paper's own condition 3 (§5.4: `nut` is the least forgiving channel, an eddy
   viscosity inconsistent with its velocity field destroys Cd_p) supplies a parsimonious
   candidate explanation for the fine-vs-coarse ordering that the authors had in their
   committed results and never tested. In fairness it is not a clean explanation either:
   `or_proj_half` carries the *largest* `nut` damage (40–46%) and converges like `coarse`
   (−183.3% vs −181.6%), not like `fine`. So `nut` damage is not monotone with convergence
   — but neither is the wall gradient, and the paper only reports the latter.

**Required fix:** restate the 1.8% as the wall-gradient scalar it is; report the L2
round-trip damage per field per arm in the same table as the convergence numbers (the data
already exist); and rewrite Contribution 4, §6.7 and §11 to the claim the data support.

Related calibration point: the fine-vs-coarse gap is 85 points *between two catastrophic
arms* (−182 vs −267 on a scale where cold = 0). Per-case values in
`results/depth_placement.json` (Cd@0.01) show fine worse than coarse in 5/5 **paired**
cases, so the ordering is real and would be better supported by a paired test than by the
unpaired bootstrap intervals currently quoted — but "converges worst of all" and "winning
0 of 5" is rhetoric that outruns an 85-point difference between two failures. The
defensible statement is the null: *restoring the first-cell gradient does not recover the
solve*.

### M1 — **MAJOR.** §5.2's headline table is a two-variable contrast presented as a one-variable one

§5.2 states: "one field handed to the solver through different representations, with
everything else held fixed ... the arms differ only in how that field was stored and read
back." From `scripts/placement_probe.py`:

- `oracle_mesh` = `truth` — the whole converged field, **all four channels including p**,
  over the **entire 20-chord domain**.
- `or_proj_*` = `bl_only(project(truth))` — projected, then boundary-layer masked; and
  `warmstart.masked_seed` blends only `u, v, nut`, so **`p` is not seeded at all** in these
  arms, and everything outside ~3 delta is freestream.
- `cartesian_128` (−548.4%) comes from a different script and a **different tree**
  (`corpus_probe.py` / `depth_corpus`), i.e. a different settled reference and arm set —
  which the paper's own rule ("a number must name the arm set it was computed over")
  forbids mixing into one table without saying so.

So the "274-point swing from representation alone" is a swing across representation +
seeded region + seeded channel set + arm set. Every one of those is a variable the paper
elsewhere shows is worth hundreds of points (§5.3: whole-field vs BL is −568% vs +34%;
§5.4: channel splits swing +42% to −293%; §7.3: an inconsistent region boundary is −242%).

I do not accept the alternative pairing either: `fitted_256x64` (§5.3, −172.6%) is
region-matched to `oracle_mesh` only superficially, because `clustered_seed` runs with
`n_max = 1.0` and fills freestream beyond one chord of wall distance — deleting the wake and
the outer circulation field, which for a lifting section is an O(Gamma/2 pi r) ≈ several
per cent velocity error over the whole outer domain plus a discontinuity at r = 1 chord.
`cartesian_128` crops to `case.domain` similarly.

**There is no clean representation contrast in the study.** The missing arm is cheap and
decisive: **the exact converged field, mesh-native, masked to exactly the region the
projected arms cover** (`bl_only(truth)`), five solves in the existing tree. It also answers
the question your data raises most sharply and never addresses: `or_proj_fine` (exact BL,
projected, cold outside) reads −266.8% while `nf_bl` (a *much less accurate* network BL,
mesh-native, cold outside) reads +34.1%. Either representation is doing that, or handing
the solver an exact boundary layer with a cold outer field is itself harmful — and the
paper's thesis depends on which.

### M2 — **MAJOR.** The harm numbers live on a quantity that is a rate measurement in the five-case tree and demonstrably not one on the corpus, and the paper never reconciles the two

To be fair to the authors first: I checked, and **the Cd@1% row in the placement tree is
readable by the paper's own rule** — `results/depth_placement.json` records
`settled_spread = 0.00212` against a 0.005 limit and `"readable": true`, with the oracle
flat at +92.0 / +91.9 / +91.2 across the three bands. The readability objection does *not*
apply to §5.2. Two other objections do.

**(a) Two of the three metrics in that same tree disagree in sign with the headline.**

| arm | Cd@1% | **Cd_v@1%** | residual 5e-6 |
|---|---:|---:|---:|
| `or_proj_coarse` | −181.6% | **+86.1%** | +40.6% |
| `or_proj_fine`   | −266.8% | **+69.1%** | +36.6% |
| `nf_bl`          | +34.1%  | +14.6% | −80.5% |

§5.6 argues the residual is the wrong objective, and I accept that. It does not address the
fact that **Cd_v — the metric §5.1 elevates precisely because it is monotone and rate-like,
and which the paper calls 60–84% of the drag — says the projected oracle seeds are strongly
beneficial.** The paper cannot both make Cd_v the headline metric at n = 13 and dismiss what
it says about the projections at n = 5.

**(b) `iterations_to_force_band` is a last-exit statistic and the paper never separates
first entry from last exit.** A seed that starts near the answer, is pushed out of a ±1%
band by the solver's own transient, and returns at iteration 3000 scores −270% while never
being far from the answer; a cold start approaching monotonically from far away scores 0%.
That is exactly the signature of "residual excellent, force-band terrible", and Cd is the
quantity most prone to it here because Cd_p is small, oscillatory and slow. The flat oracle
does not rebut this: an oracle *starts* at the answer, so its last exit is early at any
band, which says nothing about arms that are perturbed and re-enter late. **Please report
first-entry alongside last-exit for every arm in the §5.2 table, and show the Cd(iteration)
traces for cold, `or_proj_coarse` and `oracle_mesh` on one case.** I am not predicting the
answer; the reader currently cannot tell, and the whole headline depends on it.

**(c) The generalisation gap is unreconciled.** Cd@1% is readable in the 5-case placement
tree and is *not* readable on the 13-case corpus (the oracle control there swings
+49.7% / −42.6% / +12.8%; every total-drag band is rejected). So the paper headlines the
tree where Cd works and silently drops Cd where it does not. Worse, on the corpus's only
readable metric the 128^2 raster seed of the exact field is a **null** (+3.4%, p = 0.27) and
is used as evidence that the pipeline does not manufacture savings — while the abstract
says the same seed costs 548%. Both statements are true on their own metrics and their own
case sets; the paper must state them together, in the abstract, rather than deploying
whichever suits the paragraph.

### M3 — **MAJOR.** The grid-sequencing baseline is decided by an accounting choice the paper itself calls pessimistic

From `results/sequencing_naca00124.json`: the coarse solve ran `"iterations": 6000`,
`"converged": false`, and is charged
`fine_equivalent_iterations = 6000 x 7850/31700 = 1486` against a cold run of 696. That is
the entire basis for "the learned seed's advantage is price, not quality."

**No practitioner runs the coarse level to a 6000-iteration non-converged budget.** Grid
sequencing stops the coarse solve when its own forces settle. The authors have the coarse
run's force history, so re-charging at the coarse solve's own Cd_v force-band iteration
costs **zero new solves**. If the coarse solve settles in the same ballpark as the fine one
(~800 iterations), the charge is ~198 fine-equivalent, and `sequenced_bl` (+75.9% on Cd_v,
i.e. ~168 fine iterations) totals ~366 against a cold 696 — a saving several times the
learned seed's +14.6%, and the conclusion reverses. I am not asserting that grid sequencing
wins; I am asserting that **the paper's comparison does not establish that it loses**, and
that the re-score which would settle it is free. The authors' own caveat box concedes both
corrections "point the same way"; the bold conclusion above it is not adjusted for that
concession.

Second prong, stated as a request rather than a number: §9 argues that seconds are the unit
an engineer cares about, and §10 correctly forbids quoting wall-clock from the sequencing
tree because it was run concurrently. So the seconds-versus-iterations question for the
classical baseline is currently **unanswered**, not answered unfavourably. Since the
learned seed's advantage is claimed to be price, and price is seconds, **re-measure the
coarse solve exclusively and charge grid sequencing in seconds** in the same accounting
frame as §9. Charging the classical baseline in iteration-equivalents while advertising the
learned seed's ~11 s of inference is not a like-for-like price comparison.

### M4 — **MAJOR.** The pressure localisation is inference by subtraction, not a measurement of the pressure field

§5.2.2, §6.7 and §11 conclude "the damage is to the pressure field" from: Cd_v is preserved,
Cd is destroyed, therefore Cd_p. Since Cd = Cd_p + Cd_v, that is arithmetic, and it is a
*relabelling* of the force decomposition rather than an independent finding. Three things
make it weaker than the paper presents:

- **No pressure is handed over in the arms concerned.** `masked_seed` blends only u, v, nut;
  `or_proj_*` and `nf_bl` both start from p = 0. So "a projection corrupts the pressure
  field" cannot be a statement about the seeded p; it must mean the solver's *subsequent*
  pressure development is delayed. That is plausible, but it is a different claim and needs
  different evidence.
- **The band on Cd_p is absolutely tighter than the band on Cd.** Cd_p is 16–40% of Cd, so
  "1% of Cd_p" is 0.16–0.4x the absolute tolerance of "1% of Cd". "Pressure drag converges
  three times slower than total drag" is therefore partly a statement about the tolerance,
  not about a rate. The paper notes the rows are not additive in `PLANS.md` but the draft
  presents 1850 vs 700 iterations as a physical fact.
- **§5.3's divergence evidence is internally implausible and cuts against the mechanism.**
  "Every seed but the converged-field oracle sits at ~1.4e-6" first continuity error, with
  the whole-field arm *cleanest* at 1.19e-6. Seeds differing by ~30% in boundary-layer
  velocity cannot plausibly produce the same continuity error unless what is being read is
  a **normalised** initial residual for p rather than the unnormalised ∫|∇·u| dV. The
  unnormalised quantity is the actual source term of the pressure correction and is the
  first field-level evidence §5.2.2 needs.

**What would establish the claim, mostly from data already on disk:** for each seed report
(i) the pressure-equation source ∫|∇·u_seed| dV on the fine mesh, unnormalised;
(ii) ||Cp_seed(x) − Cp_conv(x)|| along the surface after K = 1, 10, 100 iterations;
(iii) the circulation / effective displacement-thickness error of the seed. Any one turns
"locating the damage in the pressure field" from arithmetic into physics.

Physically, the pressure story *is* plausible for SIMPLE/SIMPLEC and I would credit it if
measured: near-wall velocity error is high-wavenumber and is annihilated by a few
under-relaxed momentum sweeps (hence Cd_v recovers immediately), while a displacement /
circulation error is the smooth, globally elliptic mode that SIMPLE removes slowest (hence
Cd_p is the laggard). That is the classical multigrid smoothing argument, and it is the
right frame for this paper (see §4 below).

### M5 — **MAJOR.** The closed-form validation has two degenerate rows and is evaluated at a different `y_c` from the rest of the paper

`results/closed_form_validation.json` records `"probe": 4e-06`, i.e. the measured gradient is
taken at 4e-6 chord, while §3, §6.5 and `scripts/preflight.py` all use the mesh's first cell
centre = 5e-6. This is visible in the arithmetic: the sublayer rows predict 2.50x for a
1e-5 station and 1.25x for a 5e-6 station, which are exactly 1e-5/4e-6 and 5e-6/4e-6, not
1e-5/5e-6 = 2.0 and 1.0. Table 5 and Table 6 are therefore not evaluated against the same
mesh quantity, and Table 5's caption ("against the measured first-cell gradient") is not
accurate.

Worse, in those same two rows `measured G = 1.000000` with `ratio_std ≈ 1e-16` across five
cases. A measurement with exactly zero variance across five different airfoils is an
identity, not a measurement: it arises because `clustered_seed` populates the grid by
**nearest-neighbour donor from the mesh**, so once the grid's first station is at or below
the mesh's first cell the round trip is the identity near the wall. The paper half-says this
("the implementation clips a query below its first station"), but the consequence is not
drawn: **"it bounds the damage above in every row" rests on three non-degenerate rows**
(ratios 1.63, 2.17, 2.55), not five, and the "1219% → 1.8%" achievement is largely the
statement that a grid finer than the mesh reproduces the mesh — not evidence that a graded
*surrogate output* would carry the near-wall state, since a real surrogate emits u(y1) at
y1 rather than the donor cell's value.

### M6 — **MAJOR.** In the saturated regime the criterion is not the law of the wall

`amplification` caps `u+` at `u_inf/u_tau` when the log law exceeds freestream. For every
uniform-raster row in Table 6 (y+ = 420–1700) the cap binds, so `G = (u_inf/u_tau)/u+(y_c+)`
— a quantity containing no kappa, no B and no log law, equivalent to "the seed puts
freestream velocity in the first cell". Check: 21.0/0.716 = 29.3x, exactly Table 6's entry
for both 128^2 and 256^2. The law-of-the-wall content is active only in the wall-fitted rows
with y+ ≈ 3–40. The paper's repeated framing ("this is the law of the wall and nothing
else; it contains no fitted parameter") is therefore true but, for the rasters that carry
the headline, vacuous — the same verdict follows from "the first station is outside the
boundary layer".

Also: the same 128^2 format is quoted at **29.3x** in Table 6 and **36.6x** in §7.1. The
difference is the case's `u_tau` (0.0477 vs ≈0.0427); the paper does not say which case each
table uses. Table 7 (Re transfer) is a third set of `u_tau` values again.

### M7 — **MAJOR/MINOR.** The repair's negative result is weaker than "tested to destruction"

§10 concedes the two limitations but §5.5/§11 lean on the falsification hard.

- Inverting the equilibrium law at the coarse arm's station (2.5e-4 chord) presumes the
  station is inside a turbulent equilibrium layer. At Re = 3e6 the boundary layer near the
  leading edge is O(1e-5)–O(1e-4) chord thick; over a substantial fraction of the surface
  the station lies **outside** the layer, so `invert_u_tau` is fed a near-freestream value
  and returns a `u_tau` that is meaningless (and large). The paper should report the
  fraction of surface stations with h1 > delta99(x); if it is large, the repair was never
  in its domain of validity and the negative result is about the implementation, not about
  the physics.
- `nut` is rebuilt as `kappa u_tau d` with van Driest damping, in which y+ is formed from
  the **reconstructed** u_tau — so an error in u_tau propagates nonlinearly into nut, and
  §5.4 says nut is the least forgiving channel. §10 admits this cannot be excluded as the
  cause; given §5.4, I would say it is the leading candidate, and the paper should say so.

### Minor points

- **The placement ladder changes more than the grading.** Going from `first = 2.5e-4` to
  `5e-6` at fixed `n_n = 64` changes the geometric ratio from 1.140 to 1.214, i.e. it
  re-samples the *entire* wall-normal direction, not just the near wall. "Only the grading
  of the grid changes" is true but is not "one variable" in any physically meaningful sense
  — as the 5/5 increase in BL L2 error (F1) demonstrates.
- **`n_s = 256` is held fixed in every projected arm, and that is where I would look next.**
  256 stations over a ~2.05-chord surface is Δs ≈ 0.008 chord, which smears the leading-edge
  suction peak — the most plausible *representational* route to a corrupted pressure field,
  and the one thing common to every projected arm and therefore invisible to the placement
  ladder. The decisive experiment at equal budget is 512x32 versus 128x128. I would want it
  before believing that "storing the field on a grid at all corrupts something the boundary
  layer does not contain" is a statement about grids rather than about tangential
  resolution.
- **Re-transfer table (§6.6).** At Re <= 1e5 with 42–62% of the surface carrying near-zero
  wall shear, `u_tau` is not a friction velocity and `y+` is not a wall unit; the row is a
  formal evaluation of an expression outside its domain. §10 nearly says this; §1's
  contribution 3 ("the criterion holds across Reynolds number, checked from Re 1e3") does
  not. Limit the claim to Re >= 1e6 and present the low-Re rows as an extrapolation
  demonstration.
- **`cell_centre = min(wall_distance)`.** Using the minimum over all cells as "the first
  cell centre" is a lower bound (point-to-segment distance on a curved surface undercuts the
  true normal distance); it makes G larger. Report the median and the min.
- **Report the achieved y+ distribution**, not the design value. "y+ < 1" is a mesh-design
  statement; with `u_tau` varying along the chord the peak y+ near the leading edge is the
  number a referee wants.
- **§5.6 point 2** ("the Ux residual measures how much the field changes per iteration, so
  it rewards smoothness") should state OpenFOAM's residual normalisation explicitly, since
  the normalisation factor is itself field-dependent and is part of why the two metrics
  disagree.
- **Highlights 1 and 5** should not stand unqualified given §5.1's null on Cd_v.

---

## 4. Novelty assessment against named prior art

**Honest position: this is a new *diagnostic framing* plus a well-executed set of negative
results and a scoring protocol. It is not a new capability, and on the capability axis it is
behind the closest prior art.**

- **Zhou et al., JCP 529 (2025) 113871 / arXiv:2312.11842 (ref 12)** is the closest work and
  the paper positions itself carefully against it. I could not verify from the abstract the
  specific figures §2.1 attributes to it (11x to 1% force error, 16x to 5%, Re = 6e6,
  `rhoSimpleFoam` + SA on wall-resolved unstructured meshes) — the abstract says "at least
  two-fold". **Please cite section/table numbers for each of those claims.** The stake is
  high: if [12] does reach 11–16x on a force-error metric with a region-to-point (hence
  mesh-native) seed on a comparable wall-resolved configuration, then NeuroForge's +18.4%
  on viscous drag *with total drag destroyed* is strictly weaker on the same axis, and the
  differentiator reduces to the criterion, the negative results and the protocol. That is
  still publishable — but it must be stated that way rather than as "consistent with, not
  contrary to, what we measure", which reads as parity.
- **Fuchi et al. (ref 8)** is fairly handled: the wake-bound measurement in §7.3 (+0.5%) is
  the right way to defuse a 26.3x that concerns a different region, and it is a measurement
  rather than a rebuttal. Good.
- **"Resampling destroys near-wall gradients" is not new to CFD.** Conservative and
  consistent solution transfer between non-matching meshes (`mapFields`, AMR prolongation,
  FSI/overset interpolation) has known this for decades; the AirfRANS paper (ref 1) reports
  the same overestimate for predictions. What *is* new is quantifying it in wall units as a
  parameter-free pre-flight test on a *surrogate output format*, before any solve. That is a
  modest but real contribution and I would credit it — provided M5 and M6 are fixed so the
  test is not oversold.
- **The paper is missing its own best theoretical frame: classical multigrid.** The core
  observation — near-wall (high-wavenumber) seed error is annihilated in O(1) relaxations
  so Cd_v recovers immediately, while the smooth global pressure / circulation error
  persists and dominates the iteration count — is the standard smoothing/aliasing argument
  (Brandt, *Math. Comp.* 31 (1977) 333; Trottenberg, Oosterlee & Schüller, *Multigrid*,
  2001). Framed that way, three of the paper's "surprises" stop being surprises and become
  confirmations, which *strengthens* the work: (i) why restoring the wall gradient cannot
  help; (ii) why the residual and the force metric disagree in sign; (iii) why grid
  sequencing makes a good seed. It also supplies the predictive tool §6.7 and §10 say they
  lack: the right pre-flight quantity is not the wall-gradient ratio but *how much of the
  seed error survives K smoothing sweeps* — which is precisely what the K = 25 probe in §8
  already measures. For the same reason the paper should **own** the connection to Oh et al.
  (ref 5) rather than distancing from it ("theirs is a spectral property of the Jacobian ...
  ours is a geometric defect of the representation"): the paper's own results say the
  geometric defect is not what matters and the spectral property is.
- **§4's protocol** may be the most durable contribution and is, as far as I know, not
  written down anywhere in the ML-for-CFD warm-start literature in this form.

Verdict: **a new combination and a useful diagnostic, not a new primitive**, and one whose
headline capability number sits below the closest cited baseline.

---

## 5. Questions to the authors

1. **F1.** Given `round_trip_change_pct` in `results/placement_naca*.json` (u ≈ 28%,
   nut ≈ 40% inside the BL for `or_proj_fine`, worse than `or_proj_coarse` in 5/5 cases),
   on what basis does the abstract describe that arm as "reproducing the near-wall state to
   1.8%"? Does the falsification survive restatement as "the first-cell gradient scalar is
   not a sufficient summary of the projection's damage"?
2. **M1.** Please run and report `bl_only(truth)` — the exact converged field, mesh-native,
   masked to the projected arms' region — in the placement tree. Without it, is
   `oracle_mesh` (+92.0%, whole field, 4 channels) versus `or_proj_coarse` (−181.6%, BL only,
   3 channels, projected) a representation contrast at all?
3. **M2(a).** Cd_v is the metric §5.1 elevates as the trustworthy one, and in the placement
   tree it reads +86.1% / +69.1% for the very arms the abstract calls catastrophic. How
   should a reader weigh those two rows against each other?
4. **M2(b).** For one case, what are the Cd(iteration) traces for cold, `or_proj_coarse` and
   `oracle_mesh`? What is the **first-entry** saving alongside the last-exit saving for every
   arm in the §5.2 table?
5. **M2(c).** How do you reconcile the abstract's "a 128^2 raster costs 548%" with §5.1's use
   of that same seed as a null control (+3.4%, p = 0.27) on the corpus's only readable
   metric? Both belong in the abstract.
6. **M3.** What is the coarse solve's own Cd_v force-band iteration, and what does §5.7 read
   when the coarse solve is charged at that rather than at its 6000-iteration non-converged
   budget? Will you re-measure the sequencing tree exclusively so the classical baseline can
   be charged in seconds, in the same frame as §9?
7. **M4.** What is the unnormalised ∫|∇·u_seed| dV on the fine mesh for each seed? Is the
   "~1.4e-6 first continuity error" a normalised initial residual? What is
   ||Cp_seed − Cp_conv|| along the surface after 1, 10 and 100 iterations?
8. **M5.** Why is the closed form validated at a 4e-6 probe when the mesh's first cell centre
   is quoted as 5e-6 throughout? What does Table 5 read at 5e-6? And what does the bound
   claim become when the two rows with measured G = 1.000 (sigma = 0) are removed as the
   identities they are?
9. **M6.** Which case's `u_tau` is used in Table 6 (29.3x) versus §7.1 (36.6x) for the same
   128^2 format? In the saturated rows, what does the law of the wall contribute beyond
   `u_inf/u_tau`?
10. **Next experiment.** What happens at fixed 16,384 values with `n_s = 512, n_n = 32`
    versus `n_s = 128, n_n = 128`? If tangential resolution at the suction peak is what
    destroys Cd_p, the entire wall-normal narrative is measuring the wrong axis.
11. **M7.** Over what fraction of the surface does the coarse arm's first station (2.5e-4
    chord) lie outside delta99(x)? Where it does, what does `invert_u_tau` return?
12. **Novelty.** Section/table citations in ref 12 for the 11x/16x force-error figures, the
    Reynolds number, the solver and the mesh type.
13. Do the authors agree that the multigrid smoothing argument (high-frequency error damped
    in O(1) sweeps, smooth error persistent) explains §5.2.1, §5.5, §5.6 and §5.7
    simultaneously, and that it supplies the forward-looking criterion §6.7 says is missing?

---

## 6. Score and recommendation

**Score: 4/10. Major revision.**

The experimental apparatus is good, the scoring discipline is better than most papers in
this literature, and the willingness to publish falsifications of the authors' own
pre-registered mechanism is exactly what the field needs. I want this work to appear.

**The biggest technical risk to the claims**, stated once: the paper's most quoted sentence —
that a seed reproducing the converged boundary layer to 1.8% converges worst of all — is
contradicted by the authors' own committed diagnostics, and the contrast that carries the
title is not a controlled contrast. Fix those and the paper's real result stands and is
worth publishing: **on this configuration the first-cell wall gradient is computable in
closed form, is destroyed by every rasterised output format, and is nevertheless not what
decides a warm start — the slow, smooth, pressure-carrying modes are.** That is a clean
multigrid-consistent statement, it is supported by controls, and it is more interesting than
the representation headline currently on the front of the paper.

Conditions for acceptance:

1. Correct the 1.8% characterisation and rewrite Contribution 4, §6.7 and §11 accordingly
   (F1). **Blocking.**
2. Run and report `bl_only(truth)`, or withdraw the "representation alone" framing of §5.2
   (M1). **Blocking.**
3. Report first-entry alongside last-exit, with traces, for every headline harm number, and
   reconcile the Cd/Cd_v disagreement and the 5-case/13-case gap in the abstract (M2).
   **Blocking.**
4. Re-charge grid sequencing at the coarse solve's own force-band convergence (zero new
   solves), and re-measure that tree exclusively so it can also be charged in seconds;
   adjust the "price, not quality" conclusion to whatever the re-score says (M3).
5. Provide at least one field-level measurement supporting the pressure localisation (M4).
6. Fix the y_c inconsistency, remove the degenerate rows from the "bounds in every row"
   claim, and state the saturated-regime degeneracy (M5, M6).
7. Add the multigrid framing and the Oh et al. connection; obtain section-level citations for
   ref 12 and restate the novelty position honestly (§4 above).
