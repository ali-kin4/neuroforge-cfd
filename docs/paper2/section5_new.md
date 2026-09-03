### 5.2 Condition 1 — the representation must resolve the first cell

This is the condition the closed form of §6 predicts, and it is tested here by
moving one thing: **where the representation puts its first wall-normal
station.** Every arm below is the same network prediction, restricted to the same
boundary layer, sent through the same wall-fitted round trip. Only the grading of
that grid changes.

| arm | values | first station | `y⁺` | stations inside cell 1 |
|---|---:|---:|---:|---:|
| `*_proj_coarse` | 16,384 | 2.5·10⁻⁴ | 36 | 0 |
| `*_proj_fine` | 16,384 | 5·10⁻⁶ | 0.72 | 1 |
| **`*_proj_half`** | **8,192** | 5·10⁻⁶ | 0.72 | 1 |

`*_proj_half` is the arm that decides it: correct placement at **half** the value
budget, against wrong placement at double. If budget mattered it should lose.

**Result** (`Cd_v`@1%, five cases, cold = 696 iterations, oracle control +92.5%;
bootstrap 95% CI, and the per-case values because the mean is not the claim):

| arm | Cd_v@1% | 95% CI | wins | per case |
|---|---:|---|---:|---|
| `nf_proj_coarse` | **+7.7%** | [+5.2, +9.3] | 5/5 | +3 +8 +9 +9 +10 |
| `nf_proj_fine` | +16.5% | [+12.2, +20.6] | 5/5 | +9 +14 +17 +20 +23 |
| **`nf_proj_half`** | **+19.9%** | **[+16.1, +23.7]** | 5/5 | +14 +16 +21 +23 +26 |
| `nf_bl` (mesh-native) | +14.6% | | 5/5 | |

**Halving the budget and moving the first station inside the first cell nearly
triples the saving**, from +7.7% to +19.9%, and the two confidence intervals do
not overlap. The correctly-placed grid also beats mesh-native evaluation
(+14.6%), which is what the criterion says it should: mesh-native is one way to
satisfy the condition, not the condition itself.

The same contrast on the **exact converged field**, which removes the network
from the question entirely, gives the same ordering: `or_proj_coarse` +41.7%
[+36.7, +48.1], `or_proj_fine` +62.2% [+51.8, +73.3], `or_proj_half` +47.6%
[+43.7, +51.5]. So this is a property of the representation, not of our model.

> **What may not be read from this tree.** Every *total-drag* row here is
> unreadable: the settled arms disagree about the converged `C_d` by 2.88% on a
> 1% band, well outside the readability rule of §4. Eight arms spanning a wide
> range of seed quality is what makes the reference spread that large. We report
> `C_d,v`, which is readable, monotone across bands (+19.9% / +15.6% / +13.1%),
> and is 60–84% of the drag — and we do not quote a total-drag number from this
> experiment at all.

### 5.5 Restoring the wall gradient is not sufficient — a falsified prediction

Section 6 shows the damage a projection does is available in closed form. A
factor that is known can be divided back out, so we built the repair the
mechanism implies: invert the law of the wall at the representation's own first
station to recover `u_τ`, using **only what the representation already carries**,
and re-evaluate the profile at each cell's own wall distance (§6, and
`solver/placement.py`).

**It works on the gradient.** Measured on the seeds exactly as the solver
received them:

| arm | first-cell wall-gradient error | roughness (× converged) |
|---|---:|---:|
| `nf_bl` (mesh-native) | 53.7% | 4.2 |
| `nf_proj` | 1583% | 7.2 |
| **`nf_proj_fix`** | **42.5%** | **11.1** |
| `or_proj` | 1881% | 5.9 |
| `or_proj_fix` | 46.6% | 8.5 |

The repair takes a projection from 1583% error to 42.5% — **better than the
mesh-native prediction's 53.7%**, on the quantity viscous drag integrates.

**And it does not recover the solve.**

| arm | Cd_v@1% | Cd@1% |
|---|---:|---:|
| `nf_bl` (mesh-native) | +14.6% | **+34.2%** |
| `nf_proj` | +7.7% | −47.5% |
| **`nf_proj_fix`** | +4.9% | **−45.2%** |
| `or_proj` | +41.7% | −187.4% |
| `or_proj_fix` | +28.4% | −74.4% (3/3) |

`nf_proj_fix` carries a **better wall gradient than `nf_bl`** and converges 79
percentage points worse on total drag. On viscous drag the repair is not merely
neutral but slightly harmful (+4.9% against the unrepaired +7.7%). The one place
it helps materially is the oracle projection's total drag, −187.4% → −74.4%, and
that remains far worse than starting cold.

**We report this as a falsification, because that is what it is.** We predicted
the repair would work, pre-registered the reasoning, built it, and it did not.
What it establishes is stronger than what it was meant to show: **the first-cell
wall gradient is necessary and demonstrably not sufficient.** No other arm in
this study separates the two so cleanly, because no other arm has a correct wall
gradient and a bad outcome at the same time.

One measurable difference survives and we name it without claiming it: the
repaired seed's wall gradient is **11.1× rougher** along the surface than the
converged field, against 4.2× for the mesh-native seed. The repair reconstructs
the profile's magnitude station by station and does nothing to make neighbouring
stations agree. Whether that tangential roughness is what costs the solve is
**not established here**, and it is the obvious next experiment.

### 5.7 The classical baseline: grid sequencing

The comparator that matters is not a uniform freestream. Production aerodynamics
warm starts by **grid sequencing** — solve on a coarsened mesh, map the result up,
continue on the fine one — and a learned initialisation measured only against a
cold start has not answered the question a practitioner asks. We run it as an
arm: the same C-grid family coarsened by two (7,850 cells against 31,700, first
cell 2·10⁻⁵ against 10⁻⁵), mapped with a nearest-cell map in body-fitted
coordinates, and its coarse solve **charged**.

| row | cold | oracle | `nf_bl` | `sequenced_bl` | `sequenced_vel` | `sequenced_nut` | `sequenced` |
|---|---:|---:|---:|---:|---:|---:|---:|
| Cd@1% | 802 | +92.1% | +33.9% | −302.4% | −4.2% | — | −375.4% |
| Cd_v@1% | 696 | +92.4% | +14.6% | **+75.9%** | +2.2% | +34.9% | +63.1% |
| Cl@1% | 944 | +99.9% | +10.1% | +49.4% | −0.1% | −59.7% | +33.1% |

**Three things follow, and none of them is "ours is better".**

**The classical seed is the better seed.** On viscous drag grid sequencing reads
+75.9% against our +14.6% — five times the saving. That is exactly what §6
predicts for it: a coarsened body-fitted mesh keeps its stations clustered at the
wall, so it satisfies condition 1 by construction. **The criterion correctly
predicts the behaviour of a method that contains no network, no training data and
no surrogate, and from which it was not derived.**

**And it still is not worth running here.** The coarse solve is a real cost.
Converted at the cell-count ratio — the conservative direction, since a coarse
cell is cheaper — it charges **1486 fine-equivalent iterations against a cold run
of 696**. The saving cannot pay for that. Our seed's advantage is therefore not
quality but **price**: ~11 s of inference against a second solve. That is the
honest comparison, and it is a different claim from the one the warm-start
literature usually makes.

**The channel condition holds for it too, and total drag is destroyed.** The
split mirrors §5.4's exactly: the saving rides on the eddy viscosity (+34.9%
alone) while velocity alone is inert (+2.2%). Unlike `nf_bl`, however, handing
both channels over does **not** rescue total drag (−302.4%). The consistency
condition of §5.4 is evidently about consistency *at the fine mesh's resolution*,
which a coarse velocity field does not have — it carries a `nut` generated by a
strain field the fine mesh does not reproduce.

> **Caveats, stated rather than buried.** The coarse solve ran its full 6000
> iterations, so the charge above is pessimistic; a practitioner would stop it
> earlier. And the mapper is ours, not a production `mapFields`: it leaves a
> first-cell gradient overestimate of order 6–10× where the coarse mesh's
> placement alone would permit about 2×, so this arm is a **lower bound** on what
> grid sequencing can do. Both caveats point the same way — grid sequencing
> would look better, not worse, with more care — and neither changes the
> conclusion that its seed is good and its price is a second solve.
