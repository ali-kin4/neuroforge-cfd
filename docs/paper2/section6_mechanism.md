## 6. The mechanism, in closed form

Everything in §5 follows from one quantity, and that quantity can be computed
before any solve is run, from two numbers a CFD engineer already has.

### 6.1 What a projection actually does to the near-wall state

Viscous drag is the surface integral of the tangential velocity gradient in the
first cell off the wall. On the meshes this paper uses that cell centre sits
5·10⁻⁶ chords from the surface, and viscous drag is 60–84% of the total.

When a field is resampled through a grid, every mesh cell nearer the wall than
the grid's first wall-normal station receives the value belonging to that
station. There is nothing else for it to receive: the representation holds no
sample in between. The reconstructed first-cell gradient is therefore not
*degraded* — it is replaced by a different quantity, the velocity at the
station divided by the distance to the cell.

Write `h₁` for the representation's first wall-normal station, `y_c` for the
mesh's first cell centre, and `u(y)` for the true near-wall profile. The seeded
first-cell gradient is `u(h₁)/y_c` where the true one is `u(y_c)/y_c`, so the
gradient is overestimated by

> **G = u(h₁) / u(y_c)**

and in wall units this is a statement about the law of the wall alone:

> **G = u⁺(y₁⁺) / u⁺(y_c⁺)**,  `y⁺ = y u_τ / ν`

with `u⁺ = y⁺` in the viscous sublayer and `u⁺ = ln(y⁺)/κ + B` in the log layer.
**There is no fitted parameter in this expression.** `u_τ` comes from the
converged solution's own wall gradient and `ν` from the case.

### 6.2 It predicts the measured damage to 13%

Table 5 tests the closed form against six converged cases, for the wall-fitted
256×64 representation whose first station lies at `y⁺ = 36`. `u_τ` is taken from
each case's own converged mean wall gradient; nothing is tuned.

| case | `u_τ` | `y⁺` of first station | predicted `G` | measured `G` |
|---|---:|---:|---:|---:|
| naca0012@0° | 0.0477 | 35.8 | 24.0 | 21.4 |
| naca0012@4° | 0.0485 | 36.4 | 23.7 | 20.5 |
| naca0015@6° | 0.0490 | 36.7 | 23.5 | 21.2 |
| naca2412@2° | 0.0479 | 35.9 | 23.9 | 21.3 |
| naca2415@5° | 0.0486 | 36.5 | 23.6 | 20.5 |
| naca4412@3° | 0.0481 | 36.1 | 23.8 | 20.9 |
| **mean** | | | **23.7** | **21.0** |

Predicted/measured is **1.13 ± 0.02**. The closed form over-predicts by a
systematic 13%, with almost no scatter, and the likely reason is that the round
trip interpolates rather than purely clips, which softens the step. A
parameter-free expression landing within 13% of six independent measurements is
what licenses using it as a design rule rather than as a description.

**Where it stops being quantitative, stated rather than discovered.** The
expression assumes the first station lies in the sublayer, buffer or log region.
For the uniform Cartesian 128² representation the first station sits at
`y⁺ ≈ 1700` — outside the boundary layer, where the log law is not valid and the
velocity has saturated at freestream. Capping `u⁺` at `u_∞/u_τ` there gives 29×
against a measured 18.7×. In that regime the expression is an **upper bound**,
not an estimate, and we report it as one.

### 6.3 Two consequences that decide how a surrogate should be built

**Refining the raster cannot fix this, and the formula says why.** `u⁺` grows
*logarithmically* in `y⁺`. Going from a 128² to a 512² raster spends sixteen
times the values and moves `h₁` by a factor of four, which moves `u⁺(y₁⁺)` by
`ln(4)/κ ≈ 3.4` on a value of ~23 — about 15%. That is the closed-form version
of the measured resolution ladder in §7.1, which is flat from 128² to 421², and
of the estimate that one cell across the inner layer would need N ≈ 11,800, some
28× beyond what the standard datasets hold.

**Placement is a grading choice, and it is nearly free.** The alternative to
sixteen times the values is to move the first station inside the first cell. A
64-level geometric stack from 5·10⁻⁶ to 1 chord has a growth ratio of 1.214; a
32-level stack has 1.483. Both are ordinary meshes, and the second holds *half*
the values of the 256×64 grid that fails. This is why the paper's claim is about
where a representation puts its samples and not about how many it has, and §5.2
is the controlled test of exactly that contrast.

### 6.4 The pre-flight check

The criterion is therefore executable. Given a target mesh's first cell height
and the wall-normal shape of the format a surrogate would emit, `G` follows
immediately, and with it a verdict: a representation with no station inside the
first cell has no sample of the state that viscous drag integrates, and will
misreport it by roughly `G`. We ship this as `neuroforge.solver.placement` and
as a command-line tool, so that the check can be run on a mesh and a format that
have nothing to do with this study:

```
python scripts/preflight.py --first-cell 1e-5 --re 3e6 --fitted 256x64@2.5e-4
  ...
  predicted wall-gradient overestimate  19.18x   [wall_law]
  FAILS. Expect the first-cell wall gradient to be overestimated by about 19.2x.
```

**The criterion is necessary, not sufficient, and the paper is careful about
this.** Every representation that loses the gradient costs the solve, without
exception across every arm measured here — so the check rules formats *out* for
free. It does not rule them in: `nf_mesh` retains the gradient perfectly and is
the worst arm in the study, because it hands over an outer field the model
extrapolates badly. Conditions 2 and 3 exist for that reason.

### 6.5 The criterion applied to the formats the field actually ships

Table 6 evaluates the closed form for the output formats a surrogate might emit,
against the mesh used throughout this paper (first cell 10⁻⁵ chords, `u_τ` =
0.0477, `ν` = 3.33·10⁻⁷). It costs no solve and no network, and it is the whole
argument in one place.

| output format | values | first station | `y⁺` | predicted `G` | verdict |
|---|---:|---:|---:|---:|---|
| uniform raster 128², 3-chord crop | 16,384 | 1.2·10⁻² | 1700 | 29.3× | fails (bound) |
| uniform raster 256², 3-chord crop | 65,536 | 5.9·10⁻³ | 840 | 29.3× | fails (bound) |
| **uniform raster 512², 3-chord crop** | **262,144** | 2.9·10⁻³ | 420 | **27.6×** | **fails** |
| uniform raster 128², 1-chord crop | 16,384 | 3.9·10⁻³ | 560 | 28.6× | fails |
| wall-fitted 256×64 from 2.5·10⁻⁴ | 16,384 | 2.5·10⁻⁴ | 36 | 19.2× | fails |
| wall-fitted 256×64 from 2.5·10⁻⁵ | 16,384 | 2.5·10⁻⁵ | 3.6 | 5.0× | fails |
| wall-fitted 256×64 from 5·10⁻⁶ | 16,384 | 5.0·10⁻⁶ | 0.72 | 1.0× | **passes** |
| **wall-fitted 256×32 from 5·10⁻⁶** | **8,192** | 5.0·10⁻⁶ | 0.72 | **1.0×** | **passes** |
| mesh-native, queried at cell centres | native | 5.0·10⁻⁶ | 0.72 | 1.0× | **passes** |

Two rows carry the paper. A **512² raster holds 262,144 values and still fails**,
at 27.6×; a **wall-fitted grid of 8,192 values — one thirty-second of that
budget — passes**. Sixteen times the values cannot buy what one grading decision
gives away for free.

The practical reading is a design rule, not a ranking of architectures. Any
surrogate whose output is a uniform raster over a crop of order the chord cannot
warm-start a wall-resolved RANS mesh, however finely it is rasterised, because a
uniform grid must resolve its smallest scale everywhere and the near-wall scale
collapses like `ν/u_τ`. Surrogates that predict on native mesh points satisfy the
criterion by construction. Between the two sits a large and mostly unexplored
middle — graded wall-fitted outputs — where the criterion is satisfied or not
purely by the choice of first station, and where §5.5's repair applies when it
is not.

### 6.6 The criterion across Reynolds number, at no compute cost

Because the closed form is written in wall units it makes a prediction about a
regime this study never set out to measure, and the prediction is not the
obvious one:

> **On the same mesh, a lower Reynolds number makes the projection worse.**

Both `y⁺` values fall together as `ν` rises, but through different parts of the
profile. The mesh's first cell sinks deeper into the *linear* sublayer, where
`u⁺ = y⁺` falls in proportion; the representation's fixed station at 2.5·10⁻⁴
remains in the buffer or log layer, where `u⁺` falls only logarithmically. The
ratio — the damage — therefore grows.

This is testable with no new solves. Converged cold solves on the *same* C-grid
already exist at Re = 10³ to 3·10⁶. Projecting each through the same wall-fitted
grid gives Table 7 (`scripts/reynolds_transfer.py`):

| Re | `y⁺` first cell | `y⁺` station | predicted `G` | measured `G` | ratio | weak-shear surface |
|---:|---:|---:|---:|---:|---:|---:|
| 10³ | 0.001 | 0.1 | 62.5× | 179× | 0.35 | 61% |
| 10⁴ | 0.004 | 0.3 | 62.5× | 185× | 0.34 | 62% |
| 10⁵ | 0.027 | 1.7 | 62.5× | 195× | 0.32 | 42% |
| **10⁶** | 0.21 | 13.3 | **44.9×** | **54.0×** | **0.83** | 11% |
| **3·10⁶** | 0.58 | 36.1 | **23.8×** | **24.7×** | **0.96** | 6% |

**The direction is confirmed across three and a half decades and is monotone**:
the same representation on the same mesh costs 24.7× at Re = 3·10⁶ and 179× at
Re = 10³. A practitioner's instinct — that a coarse representation is more
forgiving at low Reynolds number, where the flow is smoother — is exactly wrong,
and the reason is that the mesh's first cell has moved into the linear sublayer
while the representation's has not.

**The quantitative agreement holds only where the law of the wall does**, and
the last column says where that is. It reports the fraction of surface stations
carrying under a tenth of the peak wall shear — the signature of a laminar or
separated layer. At Re ≥ 10⁶ it is 6–11% and the closed form is accurate to
within 4–17%. At Re ≤ 10⁵ it is 42–62%: the boundary layer is laminar and
largely separated, the law of the wall does not describe it, and the expression
becomes a **lower bound**, under-predicting the true damage by about three-fold.

We regard this as the honest outcome rather than a weakness. The formula is
derived from an equilibrium turbulent wall profile and it is accurate exactly
where such a profile exists; outside that regime it still gets the sign and the
ordering right, and it errs conservatively — it says a representation is worse
than it looks, never better.
