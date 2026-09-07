# Drag observability — the raster round-trip control, and what it costs the claim

Date: 2026-09-07. Answers `whitespace.md` **A0**, the control that blocks G1 and
the A1 ladder. Producers: `scripts/drag_observability_roundtrip.py` →
`results/control/drag_observability_roundtrip.json`;
`scripts/drag_covariate_control.py` → `results/control/drag_covariate_control.json`.
Decision rule committed in **`9e21823`, before the run**; results in `da23297`.

**Wall clock: 46 min**, 200 cases × 3 levels × 4 arms, CPU-only, single process
(`OMP_NUM_THREADS=4`, `CUDA_VISIBLE_DEVICES=""`). No GPU, no training cache
touched. Covariate control: 2 s on already-committed JSON.

---

## 0. Verdict, in one paragraph

**The raster owns the loss — and `ρ_D = 0.839` was never a ceiling.** The
pre-registered **R-WORSE** branch obtains: putting the exact AirfRANS truth on a
128² raster and scattering it back to the mesh nodes, then integrating with
**AirfRANS's own force integrator**, gives `ρ_D = 0.700` — *below* our 0.839.
Faithful integration of the rasterised field does worse than our scheme, so our
0.839 cannot have been a surface integral of that information.

**And a second control, which the brief did not ask for, kills the headline
sentence outright.** Angle of attack is in the simulation name. A three-parameter
regression on `(U, α, α²)` that never touches a flow field ranks official drag at
**ρ = 0.874** — beating *every* integrator arm on ground-truth fields (0.611–0.839)
and every prediction (0.828–0.845). The sentence *"no near-wall scheme recovers
the official drag ranking beyond ρ_D ≈ 0.84"* must be **withdrawn as stated**: it
reports a number below the do-nothing baseline and presents it as a ceiling.

**What survives, and is stronger than what it replaces:** the lift/drag
observability asymmetry, now measured *inside a single fixed integrator
formulation* instead of confounded with a near-field/far-field scheme change, and
now covariate-controlled.

| from the same 128² raster | drag | lift |
|---|---:|---:|
| ρ vs official, AirfRANS's own integrator | **0.700** | **0.9967** |
| partial ρ, controlling `(U, α)` | **0.133** | **0.976** |
| covariate-only baseline | 0.874 | 0.934 |
| median relative error | 133% | **12.3%** |

Lift clears its covariate baseline by a wide margin and retains almost all of its
rank information after `(U, α)` is removed. Drag does not clear its baseline at
all, and retains **13%**. That is the finding, and it is now hostile-reviewer-proof
in a way the 0.839 number never was.

---

## 1. Method — and why this is the control, not a variant of the old measurement

`Simulation.force_coefficient(reference=True)` reads `internal.point_data['p']`
and `['U']` directly, so on the untouched cloud it returns the label by
construction and gives a trivial 1.0. That is the trap, and it is why the arm was
skipped. The control has to make rasterisation loss present while holding
integrator formulation fixed at AirfRANS's own:

```
native cloud --rasterise N²--> raster --scatter back to mesh nodes-->
    overwrite Simulation.velocity / .pressure --> force_coefficient(reference=False)
```

`reference=False` reads the **mutable attributes** `self.velocity` (M,2) and
`self.pressure` (M,1) and integrates them with AirfRANS's own formulation: VTK
`compute_derivative` for wall shear on the body-fitted mesh, `reorganize` onto the
airfoil polyline, `ptc`, cell-`Length` quadrature. Rasterisation is then the only
variable.

**Alignment is exact, not approximate.** `airfrans.dataset.load` builds its
per-case array by concatenating `Simulation.position / .velocity / .pressure /
.nu_t`. The point set our rasteriser consumes *is* `sim.position`, node for node,
so no matching step exists to go wrong.

**Arms.** `rt_pipeline` (u,v,p round-tripped *with* the loader's solid-mask
zeroing — exactly what `load_airfrans` feeds the model), `rt_raw` (bare raster, no
mask policy), `rt_p_only` (pressure only; velocity left at reference), `rt_u_only`
(velocity only; pressure left at reference).

**Gates — all four exactly `0.0`.**

| gate | what it would have caught | value |
|---|---|---:|
| identity | attributes set to reference must reproduce the label | `0.0` |
| far-field refill | nodes outside the crop keep reference values; perturbing everything with `sdf > 0.5c` must not move the force | `0.0` |
| `rt_p_only` cdv | velocity untouched ⇒ viscous drag must return exactly | `0.0` |
| `rt_u_only` cdp | pressure untouched ⇒ pressure drag must return exactly | `0.0` |

The last two are the ones that matter: a global identity gate would not catch a
`reference=True` slip inside a single arm. They are exact at all three levels.

---

## 2. The round-trip table (n = 200, `task='full'` test split)

`ρ` is Spearman against the official label component. `cdv ratio` is the median of
round-trip cdv over label cdv — how much of the viscous drag *magnitude* survives.

| N | arm | ρ(cd) | ρ(cdp) | ρ(cdv) | ρ(cl) | cdv ratio | cdp factor |
|---:|---|---:|---:|---:|---:|---:|---:|
| 128 | `rt_pipeline` | **0.700** | 0.767 | 0.666 | **0.9967** | **0.0055** | 2.00 |
| 128 | `rt_raw` | 0.700 | 0.767 | 0.659 | 0.9967 | 0.0056 | 2.00 |
| 128 | `rt_p_only` | 0.716 | 0.767 | 1.0 | 0.9967 | 1.0 | 2.00 |
| 128 | `rt_u_only` | 0.984 | 1.0 | 0.659 | 1.0000 | 0.0056 | 1.00 |
| 256 | `rt_pipeline` | 0.710 | 0.777 | 0.666 | 0.9986 | 0.0105 | 1.44 |
| 512 | `rt_pipeline` | 0.762 | 0.831 | 0.668 | **0.9995** | 0.0191 | 1.24 |

Median relative error, `rt_pipeline`: cd 133% / 64% / 78%; **cdv 99.4% / 99.0% /
98.1%**; cl **12.3% / 6.0% / 3.2%**.

Three things fall out.

**(a) `rt_pipeline` and `rt_raw` agree to three decimals.** The loader's
solid-mask zeroing is not the culprit. That retires "your mask policy did it".

**(b) Viscous drag is annihilated, and stays annihilated.** The round trip returns
**0.55% / 1.05% / 1.91%** of the true cdv at 128/256/512 — a ~99% loss at every
level, improving only linearly in N. Viscous drag is 68% of total drag magnitude
in this dataset. This is the mechanism.

**(c) Lift converges cleanly and drag does not.** Lift's relative error halves at
each refinement (12.3 → 6.0 → 3.2%, first order) and its ρ goes 0.9967 → 0.9986 →
0.9995. Drag's ρ crawls 0.700 → 0.710 → 0.762 and its magnitude error does not
even move monotonically.

---

## 3. Why — and the scale argument that predicts it independently

Wall shear is set by the viscous sublayer. At `Re = U·c/ν` with `c_f ≈ 0.003`
(`u_τ/U ≈ 0.039`), `y⁺ = 5` sits at

```
y/c = 5 · (ν/Uc) / (u_τ/U)
```

which over the dataset's `U ∈ [31.8, 93.2]` m/s (`Re ∈ [2.0, 6.0]×10⁶`) gives
`y/c ∈ [2.2, 6.3]×10⁻⁵`. On a 3-chord crop with `h = 3/N` that needs
**N ≈ 4.7×10⁴ – 1.4×10⁵** to difference faithfully.

**The pre-registered consistency check passes.** Measured median nearest-neighbour
spacing of the source cloud within `sdf < 0.01c`, over 6 cases: **3.31×10⁻⁵ chord**
— i.e. `y⁺ = 5.1`. AirfRANS resolves to exactly the sublayer edge the scale
argument predicts, agreement to **1.03×**, far inside the pre-registered 3×
tolerance.

**Two further independent estimates of the same number.** Extrapolating the
measured `cdv` ratio (`ratio ∝ N^0.893`) to 1 gives **N ≈ 4.3×10⁴**. Extrapolating
`ρ(cd)` log-linearly to the cdp-implied ceiling of 0.983 gives **N ≈ 8.6×10⁴**.
Three routes — a boundary-layer scaling, a magnitude extrapolation and a rank
extrapolation — land within a factor of 2 of each other.

Surface pressure, by contrast, varies on the **chord** scale and carries no
sublayer requirement, which is why lift has no such barrier. That single sentence
is what makes the asymmetry a mechanism rather than two facts sitting next to each
other.

---

## 4. The covariate control — the part that costs us the headline

Cd is largely a smooth function of case covariates, and AirfRANS hands them over
in the simulation **name**: `airFoil2D_SST_<U>_<α>_<NACA digits>`. `Simulation.reset()`
itself parses fields 2 and 3 as `inlet_velocity` and `angle_of_attack`. So any
quantity monotone in α rank-correlates with cd while containing no integrated-force
information whatsoever.

| baseline, no flow field touched | ρ vs official cd |
|---|---:|
| α alone | 0.800 |
| \|α\| alone | 0.818 |
| U alone | −0.195 |
| OLS `(U, α)` | 0.830 |
| **OLS `(U, α, α²)`** | **0.874** |

Against every arm ever reported on this benchmark:

| arm | marginal ρ_D | partial ρ_D given `(U,α)` | marginal ρ_L | partial ρ_L |
|---|---:|---:|---:|---:|
| near-field offset, GT field | 0.839 | 0.647 | 0.876 | 0.855 |
| near-field extrapolated, GT | 0.800 | 0.558 | 0.882 | 0.852 |
| far-field CV, GT | 0.611 | 0.531 | **1.0000** | **0.9999** |
| Transolver seed 0 / 1 / 2 | 0.845 / 0.828 / 0.843 | 0.653 / 0.625 / 0.642 | 0.880 | 0.857 |
| **round trip, AirfRANS's own integrator, 128²** | **0.700** | **0.133** | **0.9967** | **0.976** |

**Not one drag arm reaches the 0.874 covariate baseline.** The covariate baseline
is also a *lower* bound on what case identity buys, because the NACA digits are in
the name too and are deliberately not controlled for.

Lift behaves in the opposite way: the far-field CV arm reaches partial
ρ_L = 0.9999 against a 0.934 covariate baseline for lift, and the round trip
reaches 0.976. Lift's number is real force information. Drag's is mostly α.

---

## 5. The claim the paper may now make

**Delete** (G1 fact 2, and the §4 second sentence of `whitespace.md`):

> "no near-wall integration scheme recovers the official drag ranking from the
> exact ground-truth field beyond ρ_D ≈ 0.84 across an 18-variant design sweep"

It is below a three-parameter baseline that ignores the flow field, and the
round trip shows it is not a ceiling on anything.

**Replace with** (every number measured, gated, and covariate-controlled):

> On AirfRANS, rasterising the exact ground-truth field onto the 128² Cartesian
> crop a surrogate consumes and integrating it with **the dataset's own force
> integrator** recovers lift at ρ_L = 0.997 (12.3% median error) but drag at only
> ρ_D = 0.700 (133% median error), and the drag figure does not survive controlling
> for the case parameters that AirfRANS encodes in the simulation name: partial
> ρ_D = 0.133 against partial ρ_L = 0.976, where a regression on `(U, α, α²)` alone
> already ranks drag at 0.874. The mechanism is that 68% of the drag is viscous and
> wall shear lives in a viscous sublayer at y⁺ ≈ 5, i.e. `y/c ≈ 2–6×10⁻⁵`; the round
> trip returns 0.55% of the true viscous drag at 128² and 1.91% at 512², and three
> independent routes put the raster resolution required for recovery at
> N ≈ 4×10⁴–1.4×10⁵ — a 10⁵–10⁶× increase in cell count over the deployed grid.
> Surface pressure varies on the chord scale and carries no such requirement, which
> is why lift converges at first order over the same ladder. **On a raster
> representation, lift is observable and drag is not; and a reported drag rank
> correlation on this benchmark must be published against its covariate baseline,
> because the standard one is below it.**

Two scope concessions to state in the paper, not to be extracted by a reviewer:

- **This is a claim about raster/voxel-output surrogates** (DeepCFD, U-Net, FNO
  families), not about mesh-native models. Do not compare our numbers to
  mesh-native ones as if they were the same measurement.
- **The round trip is a consistent interpolation**, so it converges to the identity
  as `h → 0`. "Drag is not observable from a raster" is therefore **not**
  resolution-independent and we do not claim it is. The claim is a *quantified
  resolution requirement*, which is why the ladder is the substance and not an
  optional extra.

---

## 6. Positioning against DD-RNO — and a correction to our own first reading

**DD-RNO, arXiv:2608.13490** (Mehta, Bhati, Akolekar; IIT Jodhpur; 13 Aug 2026),
verified against the arXiv abstract page this session. On AirfRANS: *"LCQ reduces
drag MSE by 7.5× relative to conventional pressure integration and raises drag rank
correlation from ρ = 0.250 to ρ = 0.997."* LCQ is a "learned canonical quadrature"
that "replaces unstable pressure integration with flow-conditioned, learned
integration weights that predict lift and drag directly from surface pressure."

**A reading we tested and had to discard.** Viscous drag is 68% of cd, and surface
pressure cannot contain it — so predicting drag from surface pressure at ρ = 0.997
looked prima facie impossible without label access. **That inference is wrong**, and
our own data refutes it: in the official labels,

| | ρ vs official cd |
|---|---:|
| official **cdp** (pressure drag) | **0.983** |
| official **cdv** (viscous drag) | **−0.204** |

cdv is the larger *part* of cd but carries almost none of its *ranking* — it is
roughly constant across cases while cdp varies with α, and the two are mildly
anti-correlated (−0.327). So a perfect surface-pressure integrator can legitimately
rank total drag at 0.983 with no viscous information at all. DD-RNO's 0.997 is
above that, which is worth a raised eyebrow, but it is **not** evidence of
regression and we must not imply it is. Recorded in the JSON under
`label_composition` so nobody re-derives the bad argument.

**What actually distinguishes us, stated precisely.** Three things, none of which
is "their number is wrong":

1. **They measure the symptom; we measure the instrument.** Their 0.250 is what
   conventional integration achieves *on their predictions*. Neither number is
   accompanied by the ground-truth control — what the same integrator achieves on
   the *exact* field. Without it, 0.250 → 0.997 cannot be decomposed into "our
   quadrature integrates better" versus "our quadrature regresses drag from a
   pressure-shaped input". We ran that control; it is the asset they lack.
2. **Neither of us can be read without a covariate baseline, and we supply one.**
   0.874 from `(U, α, α²)` on the case name. Their 0.997 clears it; their 0.250 does
   not, and neither does our 0.839. That baseline is the first thing a referee
   should ask any AirfRANS force paper for, and as far as we can find nobody reports
   it.
3. **Different representations — do not merge the numbers.** Their 0.250 is on a
   mesh-native representation, ours is a 128² raster. `0.250 ≠ 0.700 ≠ 0.839` are
   three different measurements and the report says so.

**Scoop exposure is real and asymmetric.** They are actively working the AirfRANS
force problem and are one control away. Our differentiator is the ground-truth
control plus the covariate baseline, both now committed and timestamped.

---

## 7. What this does to `whitespace.md`

| item | status |
|---|---|
| **A0** | **Done.** R-WORSE branch. The raster owns the loss; 0.839 is not a ceiling. |
| **A1** (ladder) | **Done in the same run** — 128/256/512, and it is the substance, not a gated extra. |
| G1 fact 1 ("no measurable headroom", weak form) | Still fine, but now uninteresting: the round trip shows the shared number is not a ceiling. |
| G1 fact 2 ("18-variant sweep does not escape it") | **Withdraw the ceiling reading.** The sweep spans 0.796–0.841, all below the 0.874 covariate baseline. |
| G1 fact 3 (lift/drag asymmetry) | **Strengthened, and de-confounded.** Previously lift's 0.99999 came from the far-field arm while drag's 0.839 came from a near-field arm — different schemes. Now both come from one fixed integrator on one raster. |
| §4 second headline sentence | **Rewrite** per §5 above. |
| `pred_af_skipped` in `force_vs_official.json` meta | **Superseded** — the arm it declines to run is this report. |

---

## 8. Rebuttal lines

> **Reviewer:** *your ρ_D = 0.84 mixes rasterisation loss with your own
> integrator's formulation, so it says nothing about the representation.*
> **We did:** rasterised the exact AirfRANS truth at 128²/256²/512², scattered it
> back to the original mesh nodes, and integrated with **AirfRANS's own**
> `force_coefficient`, holding formulation fixed so rasterisation was the only
> variable. Four gates at exactly 0.0, including two per-arm negative gates.
> **Evidence:** ρ_D = 0.700 / 0.710 / 0.762 against ρ_L = 0.9967 / 0.9986 / 0.9995.
> The raster owns the loss. We also found our 0.839 was *not* a ceiling and have
> withdrawn that wording.

> **Reviewer:** *a Spearman against official cd mostly measures angle of attack,
> which is in the file name.*
> **We did:** exactly that regression, and reported it as the baseline.
> **Evidence:** `(U, α, α²)` ranks official cd at 0.874, above every integrator arm
> ever reported here including on ground-truth fields. We report partial
> correlations throughout: drag 0.133, lift 0.976 on the round trip. The
> asymmetry is what survives the control; the ceiling claim is not.

---

## 9. Honest limits

- **Extrapolations are extrapolations.** `N ≈ 4×10⁴–1.4×10⁵` comes from a
  boundary-layer scaling plus two log-linear fits over a 4× measured range. It is
  an order of magnitude, not a prediction, and the JSON says so.
- **The covariate baseline is a lower bound.** Geometry (NACA digits) is in the
  name and is not controlled for. Controlling for it would raise 0.874.
- **Partial Spearman is linear-in-ranks.** A nonlinear dependence on `(U, α)` would
  leave residual case identity in the partial numbers, biasing them *upward* — so
  drag's 0.133 is, if anything, generous.
- **Out-of-crop nodes keep reference values.** This is generous to the round trip;
  the far-field gate shows it cannot move the force (exactly 0.0).
- **We measure agreement with a reference, not with truth.** The official
  coefficients are themselves an OpenFOAM post-process carrying their own error.
- **512² is near the interpolation limit.** Far-field source spacing is ~4.6×10⁻³
  chord against `h = 5.9×10⁻³` at 512², the same caveat `floor_resolution_study.md`
  §1a states. The 128²→256² leg alone already shows the asymmetry.
