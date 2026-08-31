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
