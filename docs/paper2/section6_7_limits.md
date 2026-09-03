### 6.7 What the criterion does not do: it does not predict the solve

The sections above establish what a representation does to the near-wall state,
and they establish it firmly: the damage is computable, bounded, ordered
correctly across a fifty-fold range of first stations and three and a half
decades of Reynolds number, and controlled by station placement rather than by
value budget. **It does not follow, and we could not show, that this damage is
what makes a seed slow.** This section reports the attempt and its failure,
because the distinction decides how the criterion should be used.

**The test.** Section 5.5 repairs a projected seed's first-cell wall gradient by
inverting a wall function at the representation's own first station. It works on
the gradient: 1254% error becomes ~55%, against 53.7% for the mesh-native seed
that converges well. If the gradient were the mediator, the repaired seed should
converge like the mesh-native one. It does not — it converges like the
unrepaired projection, within 0.7 percentage points.

**The second test.** One difference survived: the repaired seed's wall gradient
was 18.1× rougher along the surface than the converged field, against 4.2× for
the mesh-native seed, because each station is inverted independently. Smoothing
the reconstructed `u_τ` along the wall brings that to 3.9×, matching the working
seed on *both* diagnostics. It changes the primary metric by 0.6 points.

| arm | grad error | roughness | `C_d`@1% | `C_d,v`@1% |
|---|---:|---:|---:|---:|
| `nf_bl` mesh-native | 53.7% | 4.2× | **+34.3%** | +14.6% |
| `nf_proj` | 1254% | 7.2× | −32.1% | +14.5% |
| `nf_proj_fix` | ~55% | 18.1× | −32.0% | +11.7% |
| `nf_proj_smooth` | ~55% | **3.9×** | −31.4% | +11.5% |

Three seeds spanning **1254% to 55%** in gradient error and **18.1× to 3.9×** in
roughness all land between −31% and −32% on total drag, where the mesh-native
seed reads +34.3%. And `nf_proj` already matches `nf_bl` on viscous drag
(+14.5% against +14.6%) while carrying twenty-three times its gradient error.

**The conclusion we are forced to.** In this configuration the first-cell wall
gradient — in magnitude and in smoothness — does not predict convergence. Both
candidate mediators are eliminated by direct measurement, and we do not have a
third. Whatever the projection destroys that costs the solve, it is not the
quantity §6.1 identifies.

**So the criterion must be used for what it measures.** It is a statement about
representations: it says, before any solve, how badly a given output format will
misreport the near-wall state, and it is accurate enough to order formats and
conservative enough never to flatter one. It is **not** a predictor of
convergence, and §6.4's pre-flight tool should be read as reporting a
representation's fidelity, not a forecast of a speedup.

**One signal points at where to look next.** On *pressure* drag — a secondary
quantity here, converging three times slower than total drag, and with its 1%
row unreadable — the smoothed repair is the only seed in the study that is
positive: +19.8% at the 0.5% band and +57.8% at 0.2%, against −115% for the
recommended seed. A rough wall-shear distribution driving a spurious pressure
response is a coherent reading of that, and it fits §5.4's finding that pressure
is where inconsistent seeds do their damage. We report it as an observation on a
secondary quantity rather than a result, because it was found after the fact and
the protocol of §4 exists to stop us doing otherwise. The experiment that would
settle it is a smoothed *mesh-native* seed, which this study does not contain.
