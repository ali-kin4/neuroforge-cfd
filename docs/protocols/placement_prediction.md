> **SUPERSEDED 2026-09-01.** The measurements this file was validated against
> were produced by a `clustered_seed` that measured wall distance to the
> nearest surface *vertex*, overestimating the first ring by a median 1147x.
> The 1.13 +- 0.02 agreement below is an artifact of that bug. With it fixed
> the closed form is an **upper bound** over-predicting by 1.3-2.6x
> (`results/closed_form_validation.json`). **The predictions in this file were
> registered before the data existed and the file is kept unedited below as the
> record of that, not as a statement of what is true.**

---

# Pre-registered prediction — the placement probe

**Written 2026-08-31 at 11:58, while `scripts/placement_probe.py` was still on
case 1 of 5 and no arm had returned a number.** Committed before the data
exists, because §3.7 of `docs/PLANS.md` says a prediction made after seeing the
result is not a prediction. If the numbers come back against this file, the file
stays and the paper reports that it was wrong.

## The model

The projection destroys the wall gradient by a mechanism that is arithmetic, not
statistical. `clustered_seed` clips every query below its first station
(`np.clip(d, first, n_max)`), so a mesh cell nearer the wall than the
representation's first station receives **the velocity belonging to that
station**. Viscous drag integrates `u_t / y` in the first cell, so the seeded
first-cell gradient is overestimated by exactly

$$ G \;=\; \frac{u(h_1)}{u(y_c)} \;=\; \frac{u^{+}(y_1^{+})}{u^{+}(y_c^{+})} $$

where `h_1` is the representation's first wall-normal station, `y_c` the mesh's
first cell centre (here the diagnostic's probe station, 4e-6), and `u+(y+)` is
the law of the wall — `u+ = y+` below `y+ = 5`, `u+ = ln(y+)/0.41 + 5.0` above
30, blended in between.

**Nothing in this is fitted.** `u_tau` comes from the converged solution's own
mean wall gradient, `nu` from the case. There is no free parameter.

## What it already predicts, on data that exists

Against `results/seed_gradient.json`, six cases, for the wall-fitted 256x64
projection at `first = 2.5e-4` (`y+ = 36`):

| | predicted | measured |
|---|---:|---:|
| first-cell gradient overestimate | **23.7x** | **21.0x** |

Ratio predicted/measured **1.13 +- 0.02** across all six cases. The 13% is a
systematic overestimate, and the likely reason is that the round trip
interpolates rather than purely clips, which softens the step.

**Known limit of the model, stated before it is tested.** It holds where the
representation's first station lies in the log layer. For the uniform Cartesian
128^2 arm the first station sits at `y+ ~ 1700`, past the edge of the boundary
layer, where the log law is invalid and the velocity has saturated at
freestream; there the closed form over-predicts (about 33x against a measured
18.7x) and should be read as a bound, not an estimate.

## The prediction for the run now in flight

`scripts/placement_probe.py` moves the first station and holds everything else.
With `u_tau ~ 0.0477` and `nu = 3.33e-7`:

| arm | `first` | values | `y+` of first station | **predicted gradient overestimate** |
|---|---:|---:|---:|---:|
| `*_coarse` | 2.5e-4 | 16,384 | 35.8 | **~24x** (reproduces today's -58.8% / -206% arms) |
| `*_fine` | 5.0e-6 | 16,384 | 0.72 | **~1.25x** |
| `*_half` | 5.0e-6 | **8,192** | 0.72 | **~1.25x** |

So, stated so it can fail:

1. **`*_fine` and `*_half` recover most of the gradient**, landing near `nf_bl`'s
   54% error rather than the projections' ~1900%.
2. **`*_half` is not materially worse than `*_fine`**, despite having half the
   values. The model has no term for the value budget, only for the first
   station.
3. **Both go positive on drag convergence**, where `*_coarse` is strongly
   negative.
4. **`or_proj_half` (the exact converged field, half budget, correct placement)
   is positive too** — the accuracy-free version of the same statement.

If (1)-(3) hold, the paper's mechanism is **station placement**, the value
budget is close to irrelevant over this range, and the criterion is computable
in closed form from `y+` alone before any solve is run.

If instead the fine-placement arms stay negative, then the round trip is
damaging the field by some route other than the near-wall clip, the closed form
above is wrong or incomplete, and the paper reports the projection failure
without the causal story it currently tells.

## What must change in the paper either way

The sentence "no 16,384-value grid has a station near the first cell" is
**false** and comes out regardless of the outcome. A 64-level geometric stack
from 5e-6 to 1.0 has growth ratio 1.214, which is an ordinary mesh. What is true
is narrower and more useful: *the output formats this field actually ships put
their first station in the log layer or beyond, and that is what costs the
solve.*
