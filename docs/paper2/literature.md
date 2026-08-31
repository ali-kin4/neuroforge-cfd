# Verified literature notes — Paper 2

Working file, **not** part of the paper. Every arXiv id below was verified
against the arXiv API on 2026-08-31 (`title`, `authors`, `published` all
returned); journal placements were verified against the publisher page or a
search result that names volume and pages. Anything not verified is marked
⚠ and must not be cited until it is.

The point of this file is the table in §3: **the literature's spread in reported
speedups is organised by representation and by which region is seeded**, and
that organisation is what the paper's criterion predicts. It is the cheapest
strong evidence in the paper, because it costs no compute and it is checkable by
any reader.

---

## 1. Warm-starting a solver with a learned field

| id | authors, date | what it seeds | representation | metric | headline |
|---|---|---|---|---|---|
| [2312.11842](https://arxiv.org/abs/2312.11842) | Zhou, Han, Zafar, Wolf, Schrock, Roy, Xiao — 2023-12-19, **published J. Comput. Phys. 2025** (S0021999125001548) | potential flow → RANS field ("super-fidelity") | **mesh-native**: VCNN-e is a region-to-point map evaluated at a target point | residual 1e-6/1e-8 **and force error** | ~2× on residual; **11× to reach 1% force error, 16× at 5%** |
| [2501.14699](https://arxiv.org/abs/2501.14699) | Fuchi, Wolf, Schrock, Beran — 2025-01-24 | **wake only**; near-body supplied separately | CNN on a grid | time steps + wall-clock | **26.3× iterations, 16.4× wall-clock** — but stated to be effective *"when combined with an accurate flow prediction in the near-body region"* |
| [2503.15766](https://arxiv.org/abs/2503.15766) | Sharpe, Ranade, Tangsali et al. (NVIDIA) — 2025-03-20 | transient URANS initial field; ML + potential flow hybrid | point-based (DoMINO) | drag band | ~2×, 50% reduction in time-to-convergence |
| [2601.02693](https://arxiv.org/abs/2601.02693) | Hu, Wu, Ding, Wang, Yang, Wang — 2026-01-06 | full-domain initial field, SUBOFF submarine | **mesh-native**: predicts at cell and wall-face centroids from geometric features | **residual only** — no force coefficients reported | 3.5× at 5e-6, 2.0× at 5e-8; ~2× cross-mesh |
| [2511.02481](https://arxiv.org/abs/2511.02481) | Eshaghi, Anitescu, Valizadeh et al. — 2025-11-04, **published CMAME 458 (2026) 118989** | inner Krylov initial guesses | operator-level, discretisation-agnostic | solver time | up to 90% time |
| [2606.21828](https://arxiv.org/abs/2606.21828) | Oh, Lee, Darbon et al. — 2026-06-20 | Newton initial guesses, spectrally constrained | — | Newton iterations | 5.4× at 6.4M DOF |
| [2509.08765](https://arxiv.org/abs/2509.08765) | Khodak, Jung, Wynne et al. — 2025-09-10 | preconditioner choice, online | — | total + linear solve | 1.5× total, 4× linear |
| [2605.09382](https://arxiv.org/abs/2605.09382) | Yavlovich, Agbaria, Mhamed et al. — 2026-05-10 | linear-assignment dual warm start **with a fallback** | — | runtime | never worse than baseline even at 100% fallback |
| [2511.13174](https://arxiv.org/abs/2511.13174) | Schmidtobreick, Arnström, Häusner et al. — 2025-11-17 | active-set solver warm start | GNN | iterations | retains convergence guarantees |

**The pattern, and it is the paper's cheapest argument.** Every large reported
gain comes from one of exactly two situations:

1. the prediction is evaluated **natively at the solver's own points**, so the
   near-wall state survives (Zhou et al., 11–16×; Hu et al., 3.5×); or
2. the seeded region **contains no near-wall state at all** — the wake and
   off-body — with the near-body supplied by something already accurate
   (Fuchi et al., 26.3×).

No published study reports a large gain from a grid-projected seed of the
near-wall region. That is exactly what the criterion in this paper says is
impossible, and it is a retrodiction over work none of which was designed to
test it.

**Two further observations worth a sentence each in the paper.**

- **Hu et al. (2026) report residual speedups and no force coefficients.** They
  are explicit that only field-level L2 (~3.3%) and iteration counts are given.
  Our §5.6 measures a seed that is **+22.1% on the residual and −58.8% on total
  drag** — the same sign disagreement, on the same class of metric. This is not
  a criticism of their result; it is the reason our protocol fixed the force
  metric before the arms were run.
- **Zhou et al. (2025) observed the failure our acceptance test bounds.** In
  their extrapolation case the initialisation gives "no clear advantages", the
  residual "sharply increase[s]", and they rely on the solver to recover
  accuracy. They have no acceptance test. §8 is precisely a rule for that case.

## 2. Near-wall prediction is independently known to be the hard part

This is the strongest support for the mechanism, and none of it is ours.

- **AirfRANS itself** [2212.07564] reports that models "have difficulties
  predicting wall shear stresses as velocity values at the closest nodes from
  the geometry are often largely overestimated", and that this is what damages
  the drag coefficient. Our measurement — a 16k-value grid projection
  *overestimates the wall shear by ~20×* — is the same failure, isolated to the
  representation and reproduced with the exact converged field.
- **DD-RNO** [2608.13490] (Mehta, Bhati, Akolekar, 2026-08-13) argues "a single
  neural architecture cannot simultaneously resolve sharp near-wall boundary
  layers and smooth far-field potential flow" and routes query points to
  separate inviscid / boundary-layer / wake decoders by wall distance. Their
  motivation is prediction accuracy; ours is warm-start viability. **Convergent
  evidence for condition 2 from an independent direction**, three weeks old.
- **Geometry-aware anisotropic boundary correction** [2606.09963] (Zhang, Huang,
  Jiang et al., 2026-06-08) replaces isotropic proximity modelling with explicit
  tangential–normal structure "to distinguish boundary-aligned propagation from
  wall-normal gradient variations".
- **Evaluation of SOTA architectures for aerodynamic prediction** [2607.13866]
  (Scherz, Hines, Bekemeyer, 2026-07-15) and the **ML4CFD competition
  retrospective** [2506.08516] (Yagoubi, Danan, Leyli-Abadi et al.) are the
  field's own audits of where these models stand.

## 3. Mesh-native surrogates

[2402.02366] Transolver; [2501.14475] PCNO; [2604.03582] low-rank spatial
attention; [2512.23192] PGOT; [2605.30375] multigrid-hierarchical full-field
prediction for 3-D aircraft. This line argues mesh-native prediction is better
*for accuracy*. Nobody in it has shown it is the difference between a warm start
that works and one that costs more than starting cold.

## 4. Classical convergence acceleration — the baseline we owe the reader

Grid sequencing / full multigrid initialisation is the standard industrial warm
start and appeared **nowhere** in this draft before 2026-08-31. `potentialFoam`
was measured (inert, +0.6% on Cd@1%) but it is not what a practitioner reaches
for.

- Full multigrid initialisation obtains lift to within 1% of converged after a
  few cycles on the final level (NASA/Swanson–Turkel line of work on multistage
  time stepping; see also the mode-multigrid survey [1802.08962]).
- **`scripts/sequencing_probe.py` now runs it as an arm**, with the coarse solve
  charged in fine-mesh-equivalent iterations. ⚠ Numbers pending.

Note the relationship, which is the reason this arm is more than fairness: a
coarsened body-fitted mesh keeps its stations clustered at the wall, so the
placement criterion **predicts grid sequencing helps** — a prediction about a
method with no network in it.

## 5. Also in the target journal

Worth citing because the venue is *Computers & Fluids* and its reviewers will
know these:

- **Sousa, Afonso, Veiga Rodrigues**, "Surrogate-based pressure–velocity
  coupling: accelerating incompressible CFD flow solvers with machine learning",
  *Computers & Fluids*, July 2026 (S0045793026002586). Surrogate inside the
  PISO pressure–velocity loop; introduces a **solver-intrinsic,
  hardware-independent** effort metric that also charges surrogate overhead —
  the same accounting concern as our §9.
- "Towards scalable surrogate models based on neural fields for large scale
  aerodynamic simulations", *Computers & Fluids* 2025 (S0045793025003895).

## 6. Verified ids, for the bibliography

All returned a matching record from the arXiv API on 2026-08-31:

```
2212.07564 2402.02366 2501.14475 2511.02481 2606.21828 2509.08765
2503.15766 2501.14699 2605.09382 2312.11842 2601.02693 2608.13490
2606.09963 2506.08516 2607.13866 2604.03582 2512.23192 2511.13174
2605.30375 2501.14870
```

Non-arXiv, verified separately: Spalart & Allmaras (1994); Weller, Tabor, Jasak
& Fureby (1998); NOWS = CMAME **458** (2026) 118989; super-fidelity = *J.
Comput. Phys.* 2025; the two *Computers & Fluids* entries in §5.
