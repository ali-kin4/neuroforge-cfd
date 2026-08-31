## 2. Related work

### 2.1 Warm-starting a flow solver with a learned field

Using a network's prediction as a solver's initial condition is an active line
with a wide spread of reported gains, from about 2× to 26×. The spread is
usually read as a difference in method quality. **We read it as a difference in
representation and in which region is seeded, and §6 gives the criterion that
sorts it.**

Zhou et al. [12] map a low-fidelity potential-flow solution to a RANS field with
an equivariant vector-cloud operator and use the result to start
`rhoSimpleFoam` with Spalart–Allmaras on wall-resolved unstructured meshes at
Re = 6·10⁶ — a configuration close to ours. They report about 2× on the residual
and, on the metric this paper also uses, **11× to reach 1% force error and 16×
to reach 5%**. Their operator is a *region-to-point* map evaluated at a target
point, so it is mesh-native by construction, and the criterion in §6 says a
mesh-native seed retains the wall gradient. Their result is the largest
near-wall warm-start gain in the literature and it is consistent with, not
contrary to, what we measure.

Fuchi et al. [8] report the largest number in this literature — **26.3× fewer
iterations and 16.4× wall-clock** — from a convolutional wake-extension model.
It is worth being precise about what that result contains, because at first
reading it dwarfs everything here. Their method divides the domain into
near-body, wake and off-body regions; the network predicts the *wake*, and the
acceleration is reported to be achieved "when combined with an accurate flow
prediction in the near-body region". The near-wall state is therefore supplied
already correct, and the network's contribution is the region where no near-wall
gradient exists to lose. §7.3 measures the complementary bound in our
configuration: an **oracle** seed of the exact converged field across the entire
downstream region is worth **+0.5%** on viscous drag here. The two results are
about different regimes and compose rather than compete.

Sharpe et al. [7] initialise transient URANS with a point-based model combined
with potential flow and report ~2×, using a drag-band metric close to ours; we
measure `potentialFoam` alone as an arm and find it inert on drag (+0.6%). Hu et
al. [13] predict at cell and wall-face centroids — mesh-native again — for a
submarine hull and report 3.5× at a residual threshold of 5·10⁻⁶, with
cross-mesh generalisation. **Their validation is residual-based throughout and
reports no force coefficient.** That is the common practice in this literature,
and §5.6 is the reason we departed from it: on our configuration one seed is
**+22.1% on the residual and −58.8% on total drag**. We are not claiming their
result is wrong; we are reporting that on our cases the two metrics can disagree
in sign, so we fixed a force metric before running the arms.

At the level of the linear algebra rather than the field, NOWS [4] supplies
learned initial guesses to Krylov solvers and reports up to 90% time reduction;
Oh et al. [5] constrain learned corrections so Newton convergence is preserved,
reporting 5.4× at 6.4M DOF; Khodak et al. [6] select preconditioners online.
These are complementary to and composable with an outer-field seed. Oh et al.
are the closest prior art to our §6 in *spirit* — both say that L² accuracy is
not what makes a seed good — and it is worth stating the difference plainly:
theirs is a spectral property of the Jacobian, established for Newton solvers;
ours is a named, measured, geometric defect of the *representation* (the
first-cell wall gradient), with a closed form and a pre-flight test that needs no
solver internals.

### 2.2 The near-wall region is independently known to be where these models fail

The mechanism we measure is not a surprise to the prediction community; what is
new is its consequence for warm starting.

The AirfRANS benchmark paper [1] reports that models "have difficulties
predicting wall shear stresses as velocity values at the closest nodes from the
geometry are often largely overestimated", and identifies this as what damages
the drag coefficient. §6 measures the same overestimate — a factor of ~20 —
arising from the *representation alone*, with the exact converged field in place
of any prediction.

DD-RNO [14] argues that "a single neural architecture cannot simultaneously
resolve sharp near-wall boundary layers and smooth far-field potential flow" and
routes query points to separate inviscid, boundary-layer and wake decoders by
wall distance. Zhang et al. [15] replace isotropic proximity modelling with
explicit tangential–normal structure for the same reason. Both are motivated by
prediction accuracy. That an independent line arrives at a *region split by wall
distance* is convergent evidence for condition 2 of §5.3, reached from the other
direction.

### 2.3 Mesh-native surrogates

Transolver [2], PCNO [3] and their successors [16, 17] predict on native mesh
points rather than on a raster, and the capability is presented as an accuracy or
memory convenience. This paper's claim is that it is not a convenience: it is the
difference between a seed that accelerates a solve and one that costs more than
starting cold. A surrogate stored as a 128² image cannot be used for warm
starting at all, and no amount of raster refinement recovers it (§7.1).

### 2.4 Classical initialisation, which is what a practitioner actually uses

The comparator that matters is not a uniform freestream. Production aerodynamics
warm starts by **grid sequencing**: solve on a coarsened mesh, map the result up,
continue on the fine mesh — shipped in OpenFOAM as `mapFields`, and closely
related to full-multigrid initialisation, which is standard in structured
compressible codes. A learned initialisation measured only against a cold start
has not answered the question a practitioner asks.

We therefore run grid sequencing as an arm (§5.7), with the coarse solve charged
in fine-mesh-equivalent iterations. It is also the criterion's out-of-sample
test: a coarsened body-fitted mesh keeps its wall-normal stations clustered at
the wall, so §6 **predicts** that grid sequencing preserves the first-cell
gradient and helps — a prediction about a method containing no network, no
training data and no surrogate, and one the criterion was not derived from.

### 2.5 Fallbacks and worst-case guarantees

Preserving worst-case behaviour by falling back to the classical method when a
learned component is untrustworthy is an established pattern; Yavlovich et al.
[9] apply it to linear assignment with a dual warm start and retain baseline
runtime even at 100% fallback, and Schmidtobreick et al. [18] warm-start
active-set solvers with a GNN while retaining convergence guarantees. **We cite
this as prior art for the pattern.** What is ours is its instantiation for a PDE
solver's initial *field*: a decision rule that reads only a short probe of the
solve it is about to commit to, leave-one-case-out calibration, and a measured
capture-versus-cost curve showing that longer probes are monotonically worse
(§8). We note that Zhou et al. [12] report exactly the failure such a rule
exists to catch — in their extrapolation case the initialisation gives no clear
advantage and the residual sharply increases — and have no test for it.
