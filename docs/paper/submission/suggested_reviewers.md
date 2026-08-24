# Suggested Reviewers — JCP (NeuroForge)

**Revised 2026-08-24 after checking the JCP editorial board.**

> ⚠️ **Removed: George Em Karniadakis.** He is an **Associate Editor of JCP**
> ("Physics-informed machine learning, Neural operators, Stochastic modeling,
> Uncertainty quantification, Theory of deep learning"). Suggesting a sitting AE as a
> reviewer is a category error and he is a plausible *handling editor* for this paper.
> Also on the board and therefore off-limits as reviewers: **Dongbin Xiu** (Editor-in-Chief;
> UQ + machine learning), **Charbel Farhat**, **George Biros**.
>
> Checked and confirmed *not* on the board, so still valid: Perdikaris, Kovachki,
> Gallinari, Goswami, Chakraborty.

**Selection principle (revised).** Fame is not the goal — *assignability* is. Editors
preferentially pick reviewers already in the journal's database, and very senior people
mostly decline or delegate. So the list is now **2 senior + 3 mid-career**, and two
candidates were chosen specifically because they **published this exact genre in JCP
within the last 18 months**: suggesting them both reinforces the scope argument and gives
the editor names already in the system.

**Conflict screen (2026-07-12, both authors; re-affirmed 2026-08-24):** none shares an
institution with **Ali Jabbary** (Urmia University) or **Kasra Ghanavati** (University of
Greenwich), and none appears among either author's co-authors (Ali's: Ghasabehi, Shams,
Jafarmadar, Pourmahmoud, Rosen, Abdollahi, Ahmadi, Samanipour). All are arms-length.

> **Affiliations and emails must be verified before entry** — do not enter from memory.

## Recommended set

### Already published this genre in JCP (highest assignability)
1. **Souvik Chakraborty** — co-author of the deep-ensemble UQ paper in *JCP* 534:114012
   (2025), cited in our related work. Directly qualified on the ensemble-spread arm of the
   trust signal, and demonstrably in JCP's reviewer pool.
2. **Yangshuai Wang** — senior author of the conformal-prediction PINN paper in *JCP*
   561:114979 (2026), cited in our related work. The closest published match to our
   split-conformal certificate; ideal for the calibration and exchangeability scoping.

### Neural operators / physics-informed ML (senior)
3. **Paris Perdikaris** — physics-informed ML and operator learning. Covers the
   residual-as-physics angle and would scrutinize the dissociation claim rigorously.
4. **Nikola Kovachki** — neural operator theory; strongest fit for the rigour of the
   residual-floor theorem and its operator-theoretic framing.

### UQ / error estimation for surrogates (mid-career)
5. **Vignesh Gopakumar** — conformal prediction for PDE surrogates (cited); mid-career and
   likely to actually accept.

## Judgment calls — decide knowingly, not by default
- **Patrick Gallinari** (AirfRANS co-author). Previously listed as a domain reviewer. Note
  the stake the earlier screen missed: our paper repairs the force-measurement pipeline and
  decomposes drag error into *measurement-limited vs model-limited* — that is a claim about
  **his benchmark's** metric. Best-informed possible reviewer, and the one with a stake in
  the finding. Include only if you want that argument tested at full strength.
- **Somdatta Goswami** (ANCHOR, `roy2025anchor` — the closest prior art we position
  against). Rigorous and expert, with a plausible competitive stake in the novelty framing.

## Do NOT suggest
- Any sitting JCP editor (see box above).
- Authors of the Transolver-lineage leaderboard race (LinearNO/PGOT/LRSA) — competitive
  proximity on the same benchmark.
- Anyone either author has collaborated with.
