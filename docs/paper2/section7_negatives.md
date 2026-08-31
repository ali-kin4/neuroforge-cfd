## 7. What does not work, and why that matters

Three predictions a reader would reasonably make are false here, and each was
tested rather than argued away. They are in the paper because a recipe that only
ever confirms itself is not evidence, and because each failure is explained by
the same closed form as the successes.

### 7.1 Refining the raster does not help, and the amount by which it does not is predicted

The obvious response to §5.2 is to spend more values. We measured it —
`scripts/resolution_ladder.py`, uniform Cartesian rasters from 128² to 421² of
the **exact converged field**, so surrogate accuracy is not in the comparison —
and the saving is flat and negative throughout.

§6 says why, quantitatively. The predicted first-cell gradient overestimate
across that ladder is

| raster | values | first station | `y⁺` | predicted `G` |
|---|---:|---:|---:|---:|
| 128² | 16,384 | 1.17·10⁻² | 1677 | 36.6× |
| 181² | 32,761 | 8.29·10⁻³ | 1186 | 36.6× |
| 256² | 65,536 | 5.86·10⁻³ | 839 | 36.6× |
| 362² | 131,044 | 4.14·10⁻³ | 593 | 35.9× |
| 421² | 177,241 | 3.56·10⁻³ | 510 | 35.3× |

**A 10.8-fold increase in stored values moves the predicted damage from 36.6× to
35.3×.** Two things make it flat. `u⁺` grows logarithmically, so quadrupling the
resolution buys `ln(4)/κ ≈ 3.4` on a value in the twenties; and above the
boundary-layer edge the velocity has saturated at freestream, so it buys nothing
at all. The measured ladder is flat for the same reason, and the closed form
turns "we tried and it did not help" into "here is the factor by which it cannot".

The scale of the gap is worth stating plainly. Resolving one cell across the
inner layer on a uniform grid would need N ≈ 11,800 — about 28× beyond what the
standard datasets hold and roughly 10⁸ stored values. **The uniform-grid route
is not expensive; it is closed.**

### 7.2 Seeding what the cold solver is slowest at makes it slower

Decomposing the cold solve by quantity suggests an obvious strategy. Iterations
to settle within 1% of converged, cold against a seed of the exact field:

| quantity | cold | oracle seed | share of `C_d` |
|---|---:|---:|---:|
| viscous drag `C_d,v` | ~700 | ~53 | 60–84% |
| lift `C_l` | ~950 | 1 | — |
| pressure drag `C_d,p` | ~1850 | 1–2 | 16–40% |

A cold solver is slow at pressure and fast at the near-wall velocity gradient; a
surrogate is the reverse. The inference — hand over the pressure, keep the
near-wall velocity — is what we pursued, and it is **false**:

- `fitted_p`, pressure only, is **inert**: +0.2% on drag, +0.1% elsewhere, at
  every depth.
- `composite`, potential-flow pressure plus a surrogate boundary layer, is
  **−305.4%**.
- `potentialFoam` alone — the free classical alternative — is inert on drag
  (+0.6% on `C_d`@1%) and mildly positive on lift (+3.3%).
- The arm that wins hands over velocity and eddy viscosity inside the boundary
  layer and **no pressure at all**.

The reason is SIMPLE's structure. Pressure is recomputed from continuity given
the velocity field, so a pressure seed inconsistent with `U` is overwritten
within a few iterations; only fields entering the momentum and turbulence
transport carry information forward. We keep this section because a falsified
prediction from a measured decomposition is stronger evidence than an
unfalsified one, and because it is the reason the recipe is not obvious.

### 7.3 The wake is worth half a per cent here

The largest acceleration reported in this literature — 26.3× iterations, 16.4×
wall-clock [8] — comes from initialising the far wake, and every seed in this
paper deliberately does the opposite, cutting off at 3.5 chords and handing the
wake back to the solver. That looks like a limitation, so we bounded it rather
than defending it.

`scripts/wake_probe.py` seeds the **exact converged field** across the whole
downstream region — 37.5% of the cells, 21.6% of them fully — which bounds what
*any* wake model could buy on these cases. Five cases, Re = 3·10⁶:

| metric | oracle wake seed | 95% CI | per case |
|---|---:|---|---|
| **`C_d,v`@1%** | **+0.5%** | [+0.4, +0.7] | +0, +0, +1, +1, +1 |
| `C_d,v`@0.5% | +1.3% | [+1.0, +1.5] | +1, +1, +1, +1, +2 |
| `C_d`@1% | −242.1% | [−676, −0.3] | −1098, −103, −19, −4, +13 |
| `C_l`@1% | −22.0% | [−68, +1.6] | −92, +0, +2, +2 |

**A perfect wake seed is worth half a per cent.** On a 2-D attached-flow airfoil
at Re = 3·10⁶ on a 20-chord C-grid, the solver is not spending its time
developing the wake; it is spending it on the near-wall state. So the two
results are about different regimes, and §2.1 notes that the 26.3× is itself
reported as conditional on an accurate near-body field being supplied
separately. Our restriction to the boundary layer is a **finding**, not a
compromise, and their result and ours compose rather than compete.

It also repeats §5.4's lesson at a different scale: the wake seed is *harmful*
on total drag, because handing over a downstream field while leaving the
boundary layer cold is another inconsistent pair. Consistency is not a detail of
the recipe — it is most of it.

### 7.4 What the criterion does not do

It is necessary, not sufficient, and the study contains its own counterexample.
`nf_mesh` hands over the network's whole-field prediction at the solver's cell
centres: it satisfies the placement criterion perfectly, retains the wall
gradient, and is the **worst arm in the study** at below −568% on total drag,
because the model's training `sdf` distribution is centred on 0.23 chords while
the C-grid reaches 20, so the outer field is extrapolation. A representation
that fails the criterion can be ruled out for free; one that passes it still has
to satisfy conditions 2 and 3.
