# Journal shortlist — the field nobody has evaluated yet

**Date of all verification: 2026-09-12** unless a row says otherwise.
**Status: COMPLETE.** Written incrementally, section by section, against power-cut risk.
**Headline: the incumbent recommendation (Computers & Fluids) is withdrawn on an
editorial conflict (§4); the new primary recommendation is the Journal of Computational
Science (§8).**

> **Note on commits.** This session has no shell tool, so I could not run `git`. The file is
> written incrementally instead (skeleton first, then one section per pass) so that a power
> cut loses at most one section. **The parent agent must commit `docs/paper/review/journal_shortlist.md`
> explicitly by path** — `git status` shows an untracked `psh_ls.err` and
> `results/interpolation/point_space_oracle_ls.json`; do not `git add -A`.

---

## 0. What is being placed, and the one number that had to be resolved first

**The paper as of 2026-09-12, after the point-space head-to-head closed.** It is no longer
"a trivial baseline beats surrogates". Four findings, in the order that now matters:

1. **What a surrogate buys is near-wall representation.** At native AirfRANS cloud resolution
   (200 test cases, 35.8M nodes, no raster in the comparison path) the parameter interpolator
   loses on every channel — `u` 144x, `v` 134x, `p` 4.85x, `nut` 18.3x — and an *oracle*
   least-squares combination of all 800 training fields is still **21.3x** worse than
   Transolver inside the first 0.005 chord. The whole interpolation family is closed at the
   wall; the gap is representational, not a tuning deficit.
2. **The published field-MSE ranking is decided by a weighting nobody states.** The identical
   two predictions on the identical 200 cases rank `mse_p` at **8.39x** one way (area-uniform
   128² raster), **0.44x** under the dataset's own node measure, and **0.21x** per node on the
   native cloud.
3. **The scoring raster's own error exceeds the model's.** The r128 round-trip error at the
   native nodes is **413x** the surrogate's near the wall (418x pooled in the earlier
   accounting). The protocol cannot adjudicate near-wall skill at all.
4. **A regression on published metadata outranks four of five published baselines** on the
   force coefficients (lift 0.9821, drag 0.9318 on the official labels), and the `reynolds`
   split does not test Reynolds generalisation.

**Claim 1–3 are, in the literal verification-and-validation sense, a statement that reported
model error is dominated by uncommitted numerical (interpolation/quadrature) error.** That
reframing is what makes a set of numerical-analysis and V&V venues eligible that a pure
ML-sociology framing would score at zero appetite. It is the reason this document runs *two*
keyword sweeps per candidate, not one (§1).

### The word-count discrepancy, resolved and stated once

| Source | Figure |
|---|---|
| Task brief to this agent (2026-09-12) | **~11,000 words, 19 exhibits** |
| `venue_taste.md` §0 (2026-09-12) | **~14,700 words, 10 figures + 9 tables** |

The exhibit counts agree (19). The word counts do not. The ~11,000 figure is consistent with
`r2_round2.md` §2.6's instruction to **cut the trust/conformal/selective-prediction layer
(~510 lines)**; the ~14,700 figure is the manuscript *before* that cut. **This document uses
11,000 words as the planning figure for the post-cut manuscript and 14,700 as the
worst-case figure if the trust layer is retained**, and every length row below states which
one it is being scored against. Where a cap sits between the two (e.g. a 13,000-word cap),
the row says so explicitly and treats the cut as a precondition rather than a detail.

---

## 1. Method, provenance, and the two sweeps

Provenance marks are inherited verbatim from `venue_plan.md` §0 / `venue_taste.md` §0:

- **[P]** primary fetch — the page or API endpoint itself was retrieved and read.
- **[S]** search-extraction — the primary page's content as surfaced by a search engine.
- **[V]** verified in a browser session (inherited rows only, dated).

**ScienceDirect returns HTTP 403 to every automated fetch** (re-confirmed today). Published-record
evidence is therefore from the **OpenAlex API** (`api.openalex.org`), queried per journal source
id. Crossref and publisher landing pages (Springer Link, Wiley Online Library, ASME Digital
Collection, SIAM, ACM DL) are used where they do not block.

### The two keyword sweeps, and why there are two

`venue_taste.md` swept on **Set A**, terms calibrated to the *old* headline (an ML-sociology
claim):

> `overoptimism OR "weak baseline" OR "weak baselines" OR cautionary OR "critical assessment" OR "reporting bias" OR "reality check"`

That is no longer the lead. **Set B** is calibrated to the *current* headline (a
discretisation-and-error-measurement claim):

> `"error metric" OR "evaluation metric" OR "interpolation error" OR "discretisation error" OR "grid convergence" OR quadrature OR "mesh resolution" OR "a posteriori error" OR verification OR validation`, scoped to benchmark/surrogate/machine-learning contexts

Both counts are reported per candidate so the ranking is auditable against whichever framing
the cover letter finally takes. **A venue that scores 0 on Set A and high on Set B is a venue
the previous sweep would have wrongly discarded.**

---

## 2. Triage table — all candidates, cheapest disqualifier first

Gate order: **(1) genuine no-APC route → (2) JCR impact factor exists → (3) genre sweep →
(4) editorial-board conflict → (5) length / review model.**

### 2.1 OpenAlex source ids resolved [P]

| Venue | OpenAlex id | ISSN-L | Publisher | works indexed | `is_oa` |
|---|---|---|---|---|---|
| Int. J. Numerical Methods in Fluids | S155225502 | 0271-2091 | Wiley | 8,360 | false |
| Computers & Mathematics with Applications | S4210205691 | 0898-1221 | Elsevier | 16,775 | false |
| Journal of Computational Science | S192071280 | 1877-7503 | Elsevier | 2,438 | false |
| Engineering with Computers | S89333158 | 0177-0667 | Springer | 2,953 | false |
| Advances in Engineering Software | S16540516 | 0965-9978 | Elsevier | 4,323 | false |
| Structural and Multidisciplinary Optimization | S48050435 | 1615-147X | Springer | 5,669 | false |
| Archives of Computational Methods in Engineering | S65486112 | 1134-3060 | Springer | 1,848 | false |
| ASME J. Verification, Validation and Uncertainty Quantification | S4210223704 | 2377-2158 | ASME | 244 | false |

All eight are `is_oa: false` — i.e. subscription titles with an optional hybrid OA route.
**G1 (no-APC) passes for all eight by declining the OA option**, the same mechanism already
verified in-browser for Computers & Fluids (`venue_plan.md` §12 [V]).

### 2.2 Set B sweep — the ML-evaluation-genre count [P]

Query, run 2026-09-12 against each source id, `from_publication_date:2019-01-01`:

> `title_and_abstract.search: ("machine learning" OR neural OR surrogate) AND (benchmark OR "error metric" OR "evaluation metric" OR "fair comparison" OR "critical assessment" OR pitfalls)`

**The raw count is not the finding — most hits are method papers that merely use the word
"benchmark". The finding is whether any hit is a paper whose *object* is the evaluation.**

| Venue | Set B count | Genuine evaluation/limitations papers among the hits |
|---|---:|---|
| **Journal of Computational Science** | **22** | **Three, and they are the real thing:** *Accuracy vs Efficiency: Benchmarking Graph Neural Networks on edge GPU hardware* (2026-07-30); *Benchmarking atom-level explainability against pharmacophore-computed labels in molecular machine learning* (2026-07-08); ***Exploring the limitations of transformer models for metocean forecasting*** (2026-06-03) |
| **Engineering with Computers** | **48** | One: *Assessment of physics-informed neural networks for beam bending problems in structural mechanics* (2026-07-01). Remainder are new-architecture papers (PISaNN, ST-PINN, MIFE-ONet…) |
| **Structural and Multidisciplinary Optimization** | **57** | None found in the top 15 — all surrogate-method and emulator papers |
| **Computers & Mathematics with Applications** | **13** | One, weakly: *A comprehensive analysis of physics-informed neural networks for solving one-way coupled problems* (2026-04-12) |
| **Int. J. Numerical Methods in Fluids** | **2** | **None.** One is a journal-issue record; the other is an FD-PINN method paper. Confirms `venue_plan.md` §3's OUT ruling under the *new* framing too |
| **ASME JVVUQ** | 1 (2022→) | **One, and it is near-exact genre:** *Application of SciML-Adapted PCMM to Deep Neural Network Surrogate Model Used for Aerodynamic Coefficient Prediction* (2025-12-01) |

**The single most decision-relevant row is Journal of Computational Science.** `venue_plan.md`
§5 held it only as an interchangeable "scope-safe landing spot… no taste evidence gathered."
That was wrong by omission: JOCS has published, within the last four months, a
*Benchmarking-X-against-Y* paper, an *explainability-benchmark-against-computed-labels* paper,
and a **limitations-of-a-model-class** paper in a geophysical-fluids forecasting domain. That
is demonstrated appetite for this genre, from the published record, in the sense
`venue_taste.md` demanded — and it is the opposite of the C&F finding (zero in seven years).

**The second is ASME JVVUQ.** It is the only venue found whose *subject matter* is the
current headline: whether a computational result's reported error is credible, and how much
of it is uncommitted numerical error. Its one ML paper is a V&V maturity assessment of a
neural aerodynamic-coefficient surrogate. **But see §3 for the gate it fails.**

### 2.3 Title-only sweep — the discriminating one for ML venues

The Set B abstract sweep is **useless at ML journals**: "benchmark" appears in nearly every
abstract (Neural Networks returns 1,092, Machine Learning 282, DMKD 123). A **title-only**
sweep is the discriminating instrument, because a venue only lets the word into a *title* when
the evaluation *is* the contribution. Query, `from_publication_date:2019-01-01` [P]:

> `title.search: (benchmark OR benchmarking OR "fair comparison" OR "critical assessment" OR pitfalls OR "reality check" OR reproducibility OR "empirical comparison" OR "experimental evaluation" OR "bake off")`

| Venue | Title-hits | Named examples — and whether the evaluation *is* the contribution |
|---|---:|---|
| **Journal of Computational Science** | **7** | **The best evidence found anywhere in this sweep.** ***When simpler models win: A large-scale computational benchmark of lexical and transformer NLP pipelines for predicting medication effectiveness*** (**2026-08-27**); ***Exploring the limitations of transformer models for metocean forecasting*** (2026-06-03); *Accuracy vs Efficiency: Benchmarking Graph Neural Networks on edge GPU hardware* (2026-07-30); *Benchmarking atom-level explainability against pharmacophore-computed labels* (2026-07-08). **Four of the seven are evaluation-as-contribution, all within the last five months.** |
| **Data Mining and Knowledge Discovery** | **14** | A *standing* evaluation lane: ***Bake off redux: a review and experimental evaluation of recent time series classification algorithms*** (2024-04-19, plus a 2024-07-04 Correction); ***Optimal selection of benchmarking datasets for unbiased machine learning algorithm evaluation*** (2023-10-20); *Deep unsupervised domain adaptation for time series classification: a benchmark* (2025-06-09) |
| **Machine Learning (Springer)** | **19** | Real, and method-agnostic: ***On the Evaluation of Machine Unlearning Methods: A Multi-domain Classification Benchmark*** (2026-06-29); ***Counterfactual Explanation Bake-Off: A Review and Experimental Evaluation for Time Series Classification*** (2026-05-01); ***Novel applications of item response theory for analysing data set complexity and benchmark selection*** (2025-08-29) |
| **Neural Networks** | 19 | Mostly LLM-benchmark releases — but one is domain-relevant: ***Benchmarking autoregressive conditional diffusion models for turbulent flow simulation*** (2026-01-24) |
| **Engineering with Computers** | 38 (looser query incl. `assessment`) | Only one is an ML assessment: *Assessment of physics-informed neural networks for beam bending problems in structural mechanics* (2026-07-01). The rest are engineering reliability/qualification assessments |
| **Knowledge-Based Systems** | 22 | **None are critiques.** All are *"we release a new benchmark dataset"* (HuddsTrafficFL, KidMind, SecReEvalBench…). Same application-firehose pattern `venue_taste.md` §4 found at EAAI |

### 2.4 The Set A control — and the finding it produces

The **inherited** `venue_taste.md` critique query, run on Journal of Computational Science
[P]: **meta.count = 0.**

**That is the point.** JOCS scores **zero on Set A** and **four genuine evaluation papers on
the title sweep**. Under the sweep `venue_taste.md` ran, JOCS would have been discarded as
having no appetite for this genre. It has more recent, more on-point appetite than any
domain-adjacent venue examined in that document, including the incumbent. **The previous
sweep's terms were measuring the old headline.**

---

## 3. Triage verdicts — every candidate, with the gate it died at

Gate order: **G1 no-APC → G2 JCR IF → G3 genre → G4 conflict → G5 length/review.**

### 3.1 Dismissed in one line each

| Candidate | Died at | Reason |
|---|---|---|
| **Scientific Reports** | **G1** | Fully gold OA (APC ~USD 2,290), not hybrid — there is no free route. The task's phrasing invited a check; the answer is no. OUT. |
| **Archives of Computational Methods in Engineering** | **G3** | IF **12.9** and its genre *is* the critical assessment — but it *"exclusively publishes extended state-of-the-art reviews"* [S]. This is primary research, not a review, and is not invited. OUT. |
| **Knowledge-Based Systems** | **G3** | IF ~7.2, but 22 title-hits and **not one is a critique** — all are new-benchmark-dataset releases. EAAI's failure mode at a different journal. OUT. |
| **Int. J. for Numerical Methods in Fluids** | **G3** | Set B = **2**, both method papers; Set-B-numerical = 7, all discretisation papers. Zero ML-evaluation appetite under *either* framing. Confirms `venue_plan.md` §3. OUT. |
| **Flow Turbulence and Combustion** | **G3** | Set B = **0**. OUT. |
| **Theoretical and Computational Fluid Dynamics** | **G3** | Bar is a fluid-dynamics theoretical result; no evaluation genre. Unchanged from `venue_plan.md` §3. OUT. |
| **Journal of Fluids Engineering (ASME)** | **G3** | Set B = **3** in seven years, none an evaluation critique (one is a special-issue front-matter record). IF ~2. OUT. |
| **Structural and Multidisciplinary Optimization** | **G3** | Set B = 57 but **zero** evaluation-as-contribution papers in the top 15 — all surrogate-*method* and emulator papers. The RSM-home-ground intuition is not borne out by the record. OUT. |
| **SIAM J. Scientific Computing / J. Scientific Computing** | **G3** | Both set a new-numerical-analysis bar with a theorem; neither has an empirical-evaluation lane. Same failure mode as CMAME. OUT. |
| **Computers & Mathematics with Applications** | **G3** | Set B = 13, one weak analysis paper; the journal is a PINN-method outlet. OUT. |
| **Advances in Engineering Software** | **G3** | IF **6.3** (Q1, 28/179 Eng. Multidisciplinary) is attractive, but Set B = 26 with zero evaluation-as-contribution; it is a methods-and-tools outlet. Held as a distant fallback only. |
| **JMLR** | **G2/G3** | Diamond OA (free to publish) **and** carries a JCR IF — the TMLR quota ruling does not transfer. But its genre is theory and method; a CFD benchmark audit has no lane. Not recommended, listed for completeness. |
| **ACM TOMS** | **G3** | Subscription + IF, but its object is *mathematical software*. Redundant with the CPC finding in `venue_taste.md` §5 — it would take the harness, not the finding. |
| **Reliability Engineering & System Safety** | **status change** | `venue_plan.md` §2 ranked it #3 at IF 13.7. **That ranking is now stale**: it rested entirely on the conformal/selective-prediction trust layer, which `r2_round2.md` §2.6 instructs be **cut**. With the trust layer gone the paper has no reliability object at all. Demote from "highest-reward branch" to OUT unless the trust layer is reinstated. |

### 3.2 Surviving to a deep profile

**Journal of Computational Science · Machine Learning (Springer) · Data Mining and Knowledge
Discovery · ASME JVVUQ · Neural Networks · Engineering with Computers**, plus the incumbent
**Computers & Fluids** carried forward from `venue_taste.md` as the benchmark to beat.

---

## 4. CONFLICT — the finding that changes the incumbent recommendation

> **An author of AirfRANS holds an editorial role at Computers & Fluids, and co-signed its
> 2026 machine-learning special-issue editorial alongside the author of three more of the
> benchmarks this paper audits.**

**Paola Cinnella.** The load-bearing evidence is the *publication record*, not a bio page:

| Fact | Source | Prov. |
|---|---|---|
| AirfRANS authors, in order: **Florent Bonnet, Jocelyn Mazari, Paola Cinnella, Patrick Gallinari** (NeurIPS 2022 Datasets & Benchmarks; arXiv 2212.07564) | OpenAlex works API, `title.search:AirfRANS` | **[P]** |
| **The C&F editorial *"Fusing data and physics: Machine learning for computational fluid dynamics"* (2026-07-14) is authored by *Neil Ashton, Richard P. Dwight and **Paola Cinnella***** | OpenAlex works API, `source:S195914795, type:editorial` | **[P]** |
| *"Editor-in-Chief: Computers & Fluids"*; *"Associate Editor: International Journal of Heat and Fluid Flow"*; credited with **AirfRANS** and **LearnFluidS** | `neilashton.co.uk/podcasts/s4-e4-prof-paola-cinnella-on-ai-for-science-and-fluid-mechanics/` — a third-party guest bio, **not a masthead** | **[P, weak source]** |
| *"P. Cinnella is the Editor-in-Chief of Computers and Fluids"* | search extraction; note the `shop.elsevier.com` C&F page was fetched and **lists no editors at all** | **[S]** |
| Editorial board member, **Flow, Turbulence and Combustion** | search extraction | **[S]** |

**Read the provenance carefully, because the two claims have different strengths:**

- **That Cinnella holds an editorial role at C&F on the ML stream is [P] and settled.** She
  co-signed the journal's 2026 ML special-issue editorial. Signing a journal editorial is an
  editorial act; it is recorded in the journal's own pages.
- **That her specific title is "Editor-in-Chief" is [S] / [P, weak].** No Elsevier masthead was
  retrievable (ScienceDirect 403s; the Elsevier shop page lists no editors). **It is entered in
  §10 as must-eyeball item 0.**

**The conflict conclusion does not depend on the title, and that is the point.** Whether
Cinnella is Editor-in-Chief or a third guest editor of the ML special issue, she is
editorially active at Computers & Fluids, on the machine-learning stream specifically, and she
is an author of AirfRANS.

> **Correction to `venue_taste.md` §1.C, recorded rather than silently applied.** That document
> states [S] that *"the guest editors are Dr Neil Ashton (NVIDIA) and Prof. Richard Dwight (TU
> Delft)."* **There is a third signatory and it is Paola Cinnella** [P]. The correction cuts
> against C&F: the editorial trio for the live ML special issue consists of an author of
> AhmedML, WindsorML and DrivAerML; a standing Associate Editor; and an author of AirfRANS.
> **That is four of the five benchmarks this paper audits, represented in one editorial.**

### Why this is disqualifying rather than manageable

`venue_taste.md` §1.C identified two conflicts at C&F — **Neil Ashton** as ML special-issue
guest editor (author of AhmedML, WindsorML, DrivAerML) and **R. P. Dwight** as a standing
Associate Editor — and proposed a mitigation: name Ashton in Editorial Manager's "opposed
reviewers" field, do not oppose Dwight, submit to the regular issue. **That mitigation does
not survive this finding, for three reasons:**

1. **The opposed-reviewers field cannot reach an editor.** It governs *reviewers*, not the
   handling editor. Naming Ashton there — `venue_taste.md`'s proposed mitigation — does nothing
   about an AirfRANS author sitting in the editorial chain, and if her title is Editor-in-Chief
   it does nothing about the top of the masthead either. There is no submission-side control
   that addresses this.
2. **The conflict is with the object of the paper, not an adjacent interest.** Dwight's
   exposure was topical (data-driven turbulence modelling). Cinnella's is direct: the paper's
   central claims are that AirfRANS's **r128 scoring protocol cannot adjudicate near-wall
   skill** (its own round-trip error exceeds the surrogate's by 413x), that **its published
   field-MSE ranking is an artifact of an unstated weighting**, that **its `reynolds` split
   does not test Reynolds generalisation**, and that **a regression on its case names outranks
   four of five of its published baselines**. Those are four criticisms of a dataset and
   protocol the handling editor co-authored.
3. **It compounds, it does not replace.** C&F now carries an EiC who authored the audited
   benchmark, an Associate Editor adjacent to the field, and a live ML special issue
   guest-edited by the author of three more of the audited benchmarks. That is the entire
   editorial chain.

**Consequence: Computers & Fluids is withdrawn as the primary recommendation.** This does not
mean the paper would be handled unfairly — Cinnella's public record is that of a serious
methodologist, and a scrupulous editor may well recuse. It means the **expected value is
wrong**: a third desk rejection here costs the project its best-fit-by-domain venue *and*
burns the one remaining Elsevier CFD title, and the probability of that outcome cannot be
estimated as low when the screening editor's own dataset is the subject.

**Also conflicted, and therefore out on conflict independently of genre:**

| Venue | Conflict |
|---|---|
| **International Journal of Heat and Fluid Flow** | Cinnella, Associate Editor **[P]** |
| **Flow, Turbulence and Combustion** | Cinnella, editorial board **[S]** (already OUT on genre, §3.1) |
| **Computers & Fluids** | Cinnella EiC **[P/S]**; Dwight AE **[S]**; Ashton SI guest editor **[S]** |

**Ashton:** no editorial-board appointment at any journal was found in a targeted search
**[S, negative]**. `venue_taste.md` §9's open item — whether Ashton holds a C&F board role —
remains unresolved and is now *moot*, because the EiC conflict dominates it.

---

## 5. The shortlist — verified tables

### 5.1 Journal of Computational Science (Elsevier) — **PRIMARY RECOMMENDATION**

| | Value | Prov. |
|---|---|---|
| **Cost model** | Hybrid. OA APC **USD 2,810** — decline it. Subscription route carries no fee. | [S] |
| **Truly free?** | **Yes**, via the subscription licence, same mechanism verified in-browser for C&F | [S] |
| **JCR Impact Factor** | **4.0**, JCR released **2026-06-17** on 2025 citation data; **Q2**; ranked 239th of the Q2 Engineering set | [S] |
| **Length limit** | **Up to 12,000 words for a full-length article.** Abstract ≤250 words. Highlights *encouraged*, not mandatory | [S] |
| **Review model** | **Single anonymized** | [S] |
| **Time to first decision** | Not published on the guide page. **UNVERIFIED** | — |
| **Publisher / portal** | Elsevier, Editorial Manager — existing account `AJabbary-884` reusable | [P, repo] |
| **Comparable papers published** | **4 evaluation-as-contribution papers in the last 5 months** (§2.3) | [P] |

**Scope quote** [S]:

> "…an international platform to exchange novel research results in **simulation-based science
> across all scientific disciplines**… Computational science typically unifies three distinct
> elements: **Modeling, Algorithms and Simulations**…; **Software developed to solve science**
> …engineering… problems; and **Computer and information science**…"

**Named comparable papers, with dates** [P]:

1. ***When simpler models win: A large-scale computational benchmark of lexical and transformer
   NLP pipelines for predicting medication effectiveness*** — **2026-08-27.** A "the simple
   baseline wins" benchmark paper, published **sixteen days ago**.
2. ***Exploring the limitations of transformer models for metocean forecasting*** — 2026-06-03.
   A limitations-of-a-model-class paper in a geophysical-fluids forecasting domain.
3. *Accuracy vs Efficiency: Benchmarking Graph Neural Networks on edge GPU hardware* — 2026-07-30.
4. *Benchmarking atom-level explainability against pharmacophore-computed labels in molecular
   machine learning* — 2026-07-08. Evaluating a model class against a reference it was not
   scored on — structurally the nearest analogue to this paper's node-measure re-scoring.

**Domain competence is present but applied:** a fluids sweep from 2024 returns 39 works [P],
including *Scalable CFD simulations in multi-billion voxel micro-CT images of porous materials
using OpenFOAM on ARCHER2* and *An uncertainty visualization framework for large-scale
cardiovascular flow simulations*. JOCS readers know CFD; they are not boundary-layer
specialists. **That is an advantage here** — §6 explains why.

**The one binding constraint: 12,000 words — and it is tighter than it looks.**

It is tempting to write "~11,000 fits, with 1,000 to spare." That is wrong, for three reasons,
and this row is the **single biggest execution risk** at this venue:

1. **The 11,000 figure is the post-cut estimate, not a measured count.** It comes from the task
   brief and corresponds to the manuscript *after* the `r2_round2.md` §2.6 trust-layer removal
   (~510 lines). Nobody has counted the post-cut manuscript. `venue_taste.md` measures the
   *current* file at ~14,700.
2. **Elsevier word counts normally include captions**, and this paper carries **19 exhibits**
   with dense captions. The `tab:native` caption mandated by `point_space_headtohead.md` §6.3
   alone is five clauses long (both arms are the published rows, G1/G2 tolerances, the in-body
   convention, per-seed ratios, the win counts, and the two memo rows).
3. **The manuscript is gaining content as well as losing it.** Mandatory additions already on
   the books: a new `tab:native` table and its caption (§6.3), a rewritten abstract (§6.2), a
   rewritten limitations bullet (§6.1), two corrected claim paragraphs (§6.4, §6.5), plus
   `naming_and_positioning.md` §6's drop-in related-work paragraph and **eight lineage
   citations**.

**So the precondition is not "make the cut" but "make the cut absorb both the trust layer and
the new native-node material, and then count."** Concretely: the cut has to remove roughly
**3,700 words** from the current manuscript *and* fund the additions above, against a hard cap.
**Run `scripts/check_submission.py` word count on the post-cut build before filing** — if it
lands above ~11,500 the submission is not ready, and this is a go/no-go rather than a detail.

---

### 5.2 ASME Journal of Verification, Validation and Uncertainty Quantification — **the best subject fit on the list, and it fails on identity**

| | Value | Prov. |
|---|---|---|
| **Cost model** | **"There are no fees for submitting or publishing a standard research paper in ASME journals."** OA optional. | [S] |
| **Truly free?** | **Yes — but with a trap.** ASME assesses **excess page charges beyond 12 printed journal pages**; a waiver may be requested from the journal editor. A 19-exhibit paper will exceed 12 pages. **This is exactly the "no APC ≠ no charges" trap `venue_plan.md` §7 item 5 flagged for Begell.** Must be cleared with the editor *before* filing. | [S] |
| **JCR Impact Factor** | **0.9** (stated on the journal's own ASME Digital Collection page); CiteScore 2.9; SJR Q3 | [S] |
| **Length limit** | 12 printed pages before excess-page charges | [S] |
| **Review model** | ASME single-anonymized (standard). **UNVERIFIED** | — |
| **Volume** | **244 works since 2015**, 164 since 2019 — roughly 23 papers/year | [P] |
| **Comparable papers** | ***Application of SciML-Adapted PCMM to Deep Neural Network Surrogate Model Used for Aerodynamic Coefficient Prediction*** (2025-12-01) — a V&V maturity assessment of a **neural aerodynamic-coefficient surrogate**; *On the Numerical Accuracy of the Discontinuous Galerkin Method in the Presence of Right-Hand Side Discontinuities* (2026-03-01); *Embracing the Epistemic: A Practical VVUQ Framework* (2026-03-01) | [P] |

**Scope quote** [P], from the ASME journal page:

> "Areas of interest including, but not limited to: Code verification; **Solution verification**;
> Validation; Uncertainty quantification; Model prediction; **Model adequacy**; **Model
> accuracy**; Predictive capacity; Model maturity; … **Model discrepancy**; Sensitivity
> analysis; **Model fidelity**; Intended use; Context of use; …"

**"Solution verification" is the literal name, in this journal's own vocabulary, for what this
paper measured.** Solution verification is the estimation of *discretisation error in a
computed result* — and finding 3 is that the scoring raster's discretisation error exceeds the
model error it is being used to measure, by 413x near the wall. No other venue on any list
names the paper's object in its own scope statement.

**Why it is not the primary recommendation: IF 0.9.** The author's stated requirement is
journal identity with a JCR impact factor (`venue_plan.md` §3). JVVUQ has one, so it clears
the gate literally — but at 0.9 it is below every other candidate, below the incumbent's 3.0,
and below the 3.9 of the journal that desk-rejected the paper. Filing the project's strongest
result at IF 0.9 is a real cost and the author should make that call knowingly, not by default.

---

### 5.3 Machine Learning (Springer)

| | Value | Prov. |
|---|---|---|
| Cost model | Hybrid. OA APC **USD 2,990–3,290** — decline. `is_oa: false` | [P] |
| Truly free? | **Yes**, subscription route | [P] |
| **JCR IF** | **4.9**, JCR 2026-06-17 on 2025 data; **Q2** | [S] |
| Length limit | Springer Nature default: **no manuscript size limit**; abstract ≤350 words. Journal-specific override **UNVERIFIED** (IdP redirect loop blocks the guidelines page) | [S] |
| Review model | **UNVERIFIED** — Springer ML is historically single-anonymized but the page is inaccessible | — |
| Time to first decision | *"in most cases the decision is made in less than three months"* | [S] |
| Comparable papers | 19 title-hits; *On the Evaluation of Machine Unlearning Methods: A Multi-domain Classification Benchmark* (2026-06-29); *Counterfactual Explanation Bake-Off: A Review and Experimental Evaluation for Time Series Classification* (2026-05-01); *Novel applications of item response theory for analysing data set complexity and benchmark selection* (2025-08-29) | [P] |

**The trade:** the highest IF among genuine genre-fits, a review culture that judges rigor
rather than novelty, **and a non-Elsevier editorial pool** — which matters after two Elsevier
desk rejections. **Against it:** zero CFD domain competence. The near-wall physics that is now
the paper's headline (finding 1) has no reviewer here, and `venue_taste.md` §2.C recorded the
same weakness at NeurIPS ED, where the chairs themselves named domain coverage as a problem.

---

### 5.4 Data Mining and Knowledge Discovery (Springer)

| | Value | Prov. |
|---|---|---|
| Cost model | Hybrid. OA APC **USD 3,190** — decline. `is_oa: false` | [P] |
| Truly free? | **Yes**, subscription route | [P] |
| **JCR IF** | **5.5**, JCR 2026-06-17, **Q1**. *(A second journalmetrics record shows 4.3; the 5.5/Q1 figure is the 2026-release page. **Discrepancy noted, not resolved** — verify before relying on it.)* | [S] |
| Review model / length | **UNVERIFIED** (IdP redirect loop) | — |
| Comparable papers | **The strongest standing evaluation lane found anywhere**: ***Bake off redux: a review and experimental evaluation of recent time series classification algorithms*** (2024-04-19, + Correction 2024-07-04); ***Optimal selection of benchmarking datasets for unbiased machine learning algorithm evaluation*** (2023-10-20); *Deep unsupervised domain adaptation for time series classification: a benchmark* (2025-06-09) | [P] |

DMKD is the venue that most reliably publishes "we re-ran everybody under one protocol and here
is what changed." The *Bake off* lineage is the direct ancestor of this paper's method. **But
the domain gap is total** — no physics, no CFD, no simulation — and the paper's lead finding is
now a boundary-layer statement. Ranked below Machine Learning only because ML (Springer)'s
readership at least tolerates scientific-ML.

---

### 5.5 Neural Networks (Elsevier) and Engineering with Computers (Springer) — held

| | Neural Networks | Engineering with Computers |
|---|---|---|
| Cost | Hybrid, APC USD 3,110 — decline; free by subscription [P] | Hybrid, APC ~USD 3,290 — decline; free by subscription [S] |
| JCR IF | ~6 (**UNVERIFIED** this cycle) | **4.1**, Q1, JCR 2026-06-17 [S] |
| Genre | 19 title-hits, mostly LLM-benchmark releases — but one is on-domain: ***Benchmarking autoregressive conditional diffusion models for turbulent flow simulation*** (2026-01-24) [P] | 38 hits on a looser query; only *Assessment of physics-informed neural networks for beam bending problems* (2026-07-01) is an ML assessment [P] |
| Verdict | **Held.** The turbulence-benchmarking precedent is real but isolated; the journal's bar is a contribution to neural-network science, as `venue_plan.md` §3 already noted | **Held.** Genuine simulation-engineering scope and Q1, but the evaluation genre is thin and the record is dominated by new PINN architectures |

---

## 6. Editorial-board conflict check — every name, including the clean ones

Names checked on every shortlisted board: **Bonnet, Mazari, Cinnella, Gallinari** (AirfRANS);
**Ashton** (DrivAerML / AhmedML / WindsorML); **Elrefaie, Ahmed** (DrivAerNet++); plus
**Dwight** carried over from `venue_taste.md`.

| Venue | Named leadership found | Conflict | Prov. |
|---|---|---|---|
| **Computers & Fluids** | **Paola Cinnella, Editor-in-Chief**; R. P. Dwight, Associate Editor; Ashton, ML special-issue guest editor | **DISQUALIFYING** — see §4 | [P]+[S] |
| **Int. J. Heat and Fluid Flow** | **Paola Cinnella, Associate Editor** | **CONFLICT** | [P] |
| **Flow, Turbulence and Combustion** | **Paola Cinnella, editorial board** | **CONFLICT** (already OUT on genre) | [S] |
| **Journal of Computational Science** | EiCs **Sarika Jalan** and **Valeria Krzhizhanovskaya**; board of **69 editors across 23 countries** | **None of the eight names found.** **PARTIAL CHECK — see caveat** | [S] |
| **Machine Learning (Springer)** | EiC **Michelangelo Ceci** | **None of the eight names found** | [S, negative] |
| **Data Mining and Knowledge Discovery** | EiC **Eyke Hüllermeier** | **None of the eight names found** | [S, negative] |
| **ASME JVVUQ** | Masthead not retrievable (ASME journaltool returned only the portal index) | **UNVERIFIED — must be eyeballed** | — |

**Per-person findings, stated even where clean:**

- **Paola Cinnella** — EiC of Computers & Fluids, AE of Int. J. Heat and Fluid Flow, board
  member of Flow Turbulence and Combustion. **Three conflicted venues, all in fluids.**
- **Patrick Gallinari** (AirfRANS author #4, ISIR/Sorbonne + Criteo AI Lab) — **no editorial
  board appointment found** at Machine Learning, DMKD or Neural Networks [S, negative]. He is
  the name most likely to appear on an ML-journal board and he does not; this is the check that
  makes the Springer-ML branch viable.
- **Florent Bonnet / Jocelyn Ahmed Mazari** (AirfRANS #1, #2; now Ansys SimAI) — **no editorial
  roles found anywhere** [S, negative].
- **Neil Ashton** (NVIDIA) — **no editorial-board appointment found at any journal** [S,
  negative]. `venue_taste.md` §9's open question about a C&F board role is now moot: the EiC
  conflict dominates it.
- **Mohamed Elrefaie / Faez Ahmed** (MIT DeCoDE, DrivAerNet++) — **no editorial roles found**
  [S, negative]. This clears Engineering with Computers and Advances in Engineering Software,
  the design/engineering-computing titles where they were most likely to appear.

> **Caveat, stated rather than glossed.** The JOCS board has **69 members** and
> ScienceDirect returns **HTTP 403** to every automated fetch of the editorial-board page, so
> the eight names were checked against the *leadership and search-surfaced* board only, not
> against an enumerated list of all 69. **This is a must-eyeball item before filing** — open
> `sciencedirect.com/journal/journal-of-computational-science/about/editorial-board` in a
> browser and Ctrl-F the eight names. Same for the ASME JVVUQ masthead.

---

## 7. Desk-screen read for the top three — what the editor sees in thirty seconds

Two editors have already said no in that window, and both said the same thing: **CMAME**
("no new computational methodology within scope") and **JCP** ("originality with respect to
other published papers is too questionable"). Both screens asked *novelty of method*. So the
only question that matters for each candidate is: **does this editor's screen ask for a novel
method, or for a novel result?**

### 7.1 Journal of Computational Science — **SURVIVES**

**What the editor sees:** title, first two sentences of the abstract, and the figure list.
Editors-in-Chief are Sarika Jalan (complex systems / network dynamics) and Valeria
Krzhizhanovskaya (computational science, University of Amsterdam, long-running ICCS chair).
Neither is a CFD specialist; both are computational-science generalists.

**The screen question is "is this simulation-based science?"** — the scope's operative phrase
is *"novel research **results** in simulation-based science across all scientific
disciplines."* **The word "novel" attaches to results, not to method.** That is the precise
difference from CMAME's "new computational methodology" and it is the whole argument for this
venue. The paper has novel results in quantity; it has no new method and never will.

**What kills it in thirty seconds:**
- If the title reads as *a CFD paper*, a generalist EiC bounces it as belonging in a fluids
  journal. The word "airfoil" must not be the first noun.
- If it reads as *a complaint about how a community reports results*, it bounces as not being
  simulation-based science at all. This is the same failure mode `venue_taste.md` §1.D
  identified at C&F and the fix is the same: lead with the discretisation.

**What saves it:** the scope's three named elements map cleanly, and unusually the *third*
one helps. **"Software developed to solve science… problems" is one of only three constituents
of the journal's own definition of computational science** — so the NullBench harness is
*in-scope contribution*, not back-matter engineering. That is the opposite of its status at
C&F (`venue_taste.md` §1.D: "demote to a data-availability statement"). At JOCS it can stay in
the body.

**Residual risk: MEDIUM-LOW.** The honest exposure is that JOCS publishes a lot of
domain-application computational science and a generalist editor may not see why a
raster-resolution result about airfoil surrogates matters to the journal's readership. The
answer — and it must be in the first paragraph of the cover letter — is that the mechanism is
not airfoil-specific: **any** simulation-based ML benchmark that scores predictions on a
resampled uniform grid inherits the same defect, and the harness is what makes the check
portable.

### 7.2 ASME JVVUQ — **SURVIVES, almost certainly**

**The screen question is "is this verification, validation or uncertainty quantification?"**
and the answer is in the journal's own keyword list: **solution verification**. An editor here
sees a paper reporting that a computational result's *reported* error is dominated by the
discretisation error of the instrument measuring it — which is the discipline's founding
concern, applied to a new object.

**Residual desk risk: LOW.** The paper's problem at JVVUQ is not the desk. It is IF 0.9, a
~23-paper-per-year readership, an unverified masthead, and the excess-page-charge exposure
beyond 12 printed pages (§5.2). **Those are post-acceptance costs, not screening costs.**

### 7.3 Machine Learning (Springer) — **COIN FLIP, and the bounce is predictable**

**The screen question is "is this a machine learning contribution?"** The bake-off and
benchmark-evaluation lane is real and recent (§2.3), and the review culture judges rigor. But
the standard bounce for domain-specific evaluation work at a general ML journal is *"this is an
application paper; it belongs in a domain venue"* — and this paper's **lead finding is now a
boundary-layer statement**, which is the most domain-specific thing in it.

**Residual desk risk: MEDIUM-HIGH**, and it *rose* when the point-space head-to-head landed.
Before 2026-09-12 the headline was "a trivial baseline beats surrogates," a general-ML claim
legible to this board. Today the headline is "what a surrogate buys is near-wall
representation," which is not. **The reframing that strengthened the paper scientifically
weakened it for this venue.** That is worth stating plainly because it is counterintuitive.

---

## 8. Primary recommendation

### Submit to the **Journal of Computational Science**.

**The reason in one paragraph.** It is the only candidate that clears all five gates at once:
the subscription route is genuinely free, it carries a **JCR Impact Factor of 4.0 (Q2, 2026
release)** which exceeds the withdrawn incumbent's 3.0, review is **single anonymized** so the
live arXiv preprint and the named GitHub/Zenodo links cost nothing, the **12,000-word** limit
accommodates the post-cut manuscript, **no editorial-board conflict was found**, and — the
finding that actually decides it — **its published record contains four evaluation-as-
contribution papers in the last five months**, including one titled *"When simpler models win"*
published sixteen days ago. Against C&F's **zero ML-evaluation critiques in seven years**, that
is not a marginal difference in taste; it is the difference between a venue that has the
category and one that does not.

### The framing the cover letter must lead with

**Not** "published CFD-surrogate benchmarks are mis-scored." **Lead with the numerical
statement, in the journal's own vocabulary:**

> We submit this work under the journal's interest in *novel research results in
> simulation-based science*. Machine-learning surrogates for simulation are almost universally
> scored by resampling their predictions onto a uniform grid and computing a field error. We
> measure what that resampling costs. On a standard external-aerodynamics benchmark the
> scoring raster's own round-trip error, with no model in the loop, exceeds the surrogate's
> error by a factor of 413 near the wall — so the protocol cannot adjudicate the region it is
> most often used to adjudicate. Re-scoring identical predictions under three different and
> equally defensible measures moves the published ranking of two methods from 8.39x one way to
> 0.21x the other. We release a tested harness that computes both diagnostics for any
> benchmark of this form.

Three things that paragraph does deliberately: it **names the object as a discretisation**, it
**puts a number before any claim about the literature**, and it **puts the software in the lead**
because software is one of the journal's own three constituents of computational science.

**Paragraph 2 — the positive result, which is what makes this not a complaint.** The paper does
not conclude that surrogates are worthless; it concludes the opposite, and says what they buy:
at native cloud resolution, with no rasterisation anywhere in the comparison path, the learned
surrogate beats parameter interpolation on every channel (144x / 134x / 4.85x / 18.3x) and an
*oracle* combination of all 800 training fields is still 21.3x worse inside the first 0.005
chord. **A surrogate earns the boundary layer.** That sentence should appear in the cover
letter, because it converts the paper from an attack into a measurement — the same structural
move `naming_and_positioning.md` §2.1 identified as what made Errica et al. land.

**Paragraph 3 — pre-registration and reproducibility**, which is this project's genuine
differentiator: decision rules committed at `a54cb75` before any number existed, one recorded
amendment that touched a gate tolerance only, five gates that abort the run on failure, G2
reproducing the deployed backbone at relative error 0.00e+00, and a Zenodo DOI.

**Paragraph 4 — disclose the arXiv preprint plainly; omit the rejection history.** The
reasoning in `venue_plan.md` §8 is unchanged and still correct: desk rejections elsewhere are
not discoverable, carry no disclosure obligation, and hand a third editor a pre-authorised
reason. Keep the positioning sentence — *the contribution is an empirical finding about how an
existing class of methods is measured, not a new numerical scheme* — which delivers the useful
half without the ammunition.

### The abstract surgery this recommendation requires — do not file without it

**`docs/paper/abstract.tex` as it stands today cannot be the abstract of the JOCS
submission**, and this is the one place where the §9 two-paper split stops being strategy and
becomes an edit. The current abstract's **last four sentences** are the covariate-null and
`reynolds`-split material:

> *"…a train-fit regression on the case name ranks official lift at 0.9821 and drag at 0.9318,
> above all four published lift entries… Nor does the `reynolds` split test Reynolds
> generalisation…"*

That material is **the spine of the NullBench paper destined for NeurIPS ED 2027** (§9). If it
ships inside the JOCS manuscript, it is journal-published, and ED 2027 is closed to it. Three
consequences, in order:

1. **Cut the covariate-null and force-coefficient block from the JOCS abstract**, replacing it
   with the native-node result from `point_space_headtohead.md` §6.2 — which is in any case the
   stronger lead now, and which that document already drafted.
2. **Decide, before filing, how much of the covariate-null material stays in the JOCS body.**
   A one-paragraph mention that cites the forthcoming work is survivable; a results section is
   not. The safe line is: the JOCS paper is about *measure dependence and representation
   limits*, full stop.
3. **The `reynolds`-split finding is the genuinely ambiguous one.** It is AirfRANS-specific, so
   it belongs with the measurement paper; but it is also a benchmark-protocol defect, so it
   strengthens NullBench. **Assign it deliberately to one paper and record the decision** —
   this is exactly the kind of item that ends up in both by accident.

### Title

Abandon any coined system name and any claim about the field. The house style at JOCS is a
descriptive results-first phrase. Something in the shape of:

> *Measure dependence and representation limits in the field-error scoring of machine-learning
> flow surrogates*

"Benchmark" should not be the first noun and "airfoil" should not appear in the title at all —
it belongs in the abstract's second sentence, where it reads as the instance rather than the
subject.

### Fallback ladder

| # | Venue | Condition | Cost model | IF |
|---|---|---|---|---|
| **1** | **Journal of Computational Science** | File next. Requires the trust-layer cut to clear 12,000 words, the retitle, and a browser check of the 69-member board | Free (subscription) | **4.0** |
| **2** | **ASME JVVUQ** | Only after the author accepts IF 0.9 **and** the excess-page-charge waiver is agreed with the editor **in advance** | Free, page charges TBC | 0.9 |
| **3** | **Machine Learning (Springer)** | Requires re-leading with the evaluation-methodology claim over the boundary-layer claim — i.e. reverting to the pre-2026-09-12 emphasis. Non-Elsevier pool | Free (subscription) | **4.9** |
| **4** | **Data Mining and Knowledge Discovery** | Same reframe as #3, deeper. Strongest genre lane, zero domain fit | Free (subscription) | 5.5 (Q1) |
| **5** | **Engineering with Computers** / **Neural Networks** | Scope-safe landings; thin evaluation genre | Free (subscription) | 4.1 / ~6 |
| — | **Computers & Fluids** | **WITHDRAWN on conflict (§4)** — not a fallback | — | 3.0 |
| — | **RESS** | **WITHDRAWN** — its case rested on the trust layer, which is being cut (§3.1) | — | 13.7 |

**Deliberate portfolio note.** #1 is Elsevier, but #3 and #4 are Springer and #2 is ASME.
After two Elsevier computational-title desk rejections on identical grounds, the ladder must
not be four more draws from the same deck. It is not.

---

## 9. What the best available option costs, relative to the venue that is closed until 2027

The ideal venue is **NeurIPS Evaluations & Datasets**, whose 2026 cycle closed 2026-05-06 and
whose next intake is ~May 2027 (`venue_taste.md` §2). Measured honestly:

| | NeurIPS ED 2027 | Journal of Computational Science |
|---|---|---|
| **Acceptance criteria** | *"Originality does **not** necessarily require introducing an entirely new method."* *"**Beating a baseline is not required.**"* *"Analyze strengths, limitations, or failure modes of existing benchmarks or evaluation practices"* — solicited verbatim | *"novel research results in simulation-based science"* — novelty attaches to results, but nothing solicits critique |
| **Is the negative framing survivable?** | It is **the solicited category** | Tolerated, not invited. Must be written as a measurement throughout |
| **The harness** | First-class contribution; code release mandatory | **In-scope contribution** — software is one of the journal's three named constituents. Better than at any Elsevier fluids title |
| **Audience** | The people who built AirfRANS, DrivAerML, WindsorML — i.e. the people who could change the protocol | Simulation scientists broadly; few of them will change a CFD benchmark |
| **Journal identity / IF** | **None.** Not a journal | **IF 4.0, Q2, JCR** — satisfies the author's stated requirement |
| **Timing** | Submit ~May 2027, decision ~Dec 2027 | **Submit now** |
| **Format cost** | 9 pages + appendix — a ~60% cut and a structural rewrite | 12,000 words — fits after the trust-layer cut already planned |
| **Anonymity** | Double-blind against a live preprint, a coined name and a username-bearing URL — ~half a day plus an anonymous mirror | **Single anonymized — zero cost** |

**The three things JOCS costs you, stated without softening:**

1. **The solicited-category framing.** At ED the paper's shape is what the call asks for; at
   JOCS every sentence has to be engineered so that a negative result reads as a measurement.
   That engineering is exactly where an unsupported sentence gets introduced — `venue_plan.md`
   §8's standing rule applies with full force.
2. **The audience that could act on it.** A protocol criticism published where the protocol's
   authors and users do not read it may be correct and inert. This is the real loss and it is
   not recoverable by impact factor.
3. **ED 2027 itself.** NeurIPS does not accept work already published in a journal. **Filing at
   JOCS closes the ED 2027 door for this manuscript.** That must be a conscious decision.

**The resolution, and it is already implicit in the repository.** These are two papers built
from one evidence base, and `venue_taste.md` §2.D and §8.2 already sketched the split:

- **The AirfRANS measurement paper** — measure dependence, the 413x representation ceiling, the
  native-node head-to-head, the oracle floor, the band ladder — goes to **JOCS now**. It is a
  single-benchmark, physics-heavy result and it is finished.
- **NullBench** — the five-benchmark metadata-null control, the R² 0.10–0.97 spread, the
  WindsorML counterexample, the pre-registered rule, the released harness, the corrected
  leaderboards — goes to **NeurIPS ED 2027** as a nine-page artifact paper with
  `null_travels.md` as its spine.

Split that way, the eight-month wait costs nothing, because the thing that waits is the thing
that needs the ED audience, and the thing that files now is the thing a journal can judge.
**If instead the covariate-null material stays inside the JOCS submission, ED 2027 is closed to
it too — so the split is a decision to take before filing, not after.**

---

## 10. Must-eyeball list before filing

ScienceDirect (403), Springer Link (IdP redirect loop) and the ASME masthead tool all blocked
automated retrieval. These need a human browser session:

0. **The Computers & Fluids masthead — Cinnella's exact title.** §4's *conflict* is [P]-settled
   (she co-signed the 2026 ML editorial), but whether she is **Editor-in-Chief** or a guest
   editor is [S]. This does not change the withdrawal, but it should be on the record correctly.
   <https://www.sciencedirect.com/journal/computers-and-fluids/about/editorial-board>
1. **JOCS editorial board, all 69 members** — Ctrl-F for Bonnet, Mazari, Cinnella, Gallinari,
   Ashton, Elrefaie, Ahmed, Dwight.
   <https://www.sciencedirect.com/journal/journal-of-computational-science/about/editorial-board>
2. **JOCS guide for authors** — confirm the **12,000-word limit**, single-anonymized review,
   the 250-word abstract cap, and that highlights are encouraged rather than mandatory.
   <https://www.sciencedirect.com/journal/journal-of-computational-science/publish/guide-for-authors>
3. **JOCS open-access options** — confirm the subscription licence is offered at acceptance
   with no fee. <https://www.sciencedirect.com/journal/journal-of-computational-science/publish/open-access-options>
4. **ASME JVVUQ masthead** — editor-in-chief and associate editors, same eight names.
5. **ASME excess-page-charge policy** — confirm the per-page rate beyond 12 printed pages and
   whether a waiver is realistically granted, *before* filing anywhere at ASME.
6. **Machine Learning (Springer) and DMKD** — review model (single vs double anonymous) and any
   journal-specific length override; both pages are behind the IdP loop.
7. **DMKD impact factor** — two journalmetrics records disagree (**5.5 Q1** vs **4.3 Q2**).
   Resolve before quoting.

---

## 11. Confidence and caveats

- **Strong [P]:** every OpenAlex `meta.count`, every source id, every named example title and
  date, the AirfRANS author list, **the authorship of the 2026-07-14 C&F machine-learning
  editorial (Ashton, Dwight, Cinnella)**, the JVVUQ scope keyword list, and the JVVUQ
  publication record.
- **[S] only:** all JCR impact factors; the JOCS 12,000-word limit, single-anonymized review
  model and scope quote; ASME's "no fees for standard research papers" and the excess-page
  charge; the Machine Learning and DMKD editors-in-chief; **Cinnella's specific *title* at C&F
  (the editorial role itself is [P]; "Editor-in-Chief" is not)**.
- **Status of the C&F withdrawal, stated precisely.** It rests on the [P] fact that an AirfRANS
  author co-signed the journal's live ML special-issue editorial, not on the [S] title. **The
  withdrawal stands as written**; only the masthead wording in §4 is pending item 0.
- **UNVERIFIED and material:** the full 69-name JOCS board; the ASME JVVUQ masthead; review
  models and length limits at Machine Learning and DMKD; Neural Networks' current IF; time to
  first decision at every candidate.
- **A finding stated as a finding:** across the candidates swept here, **the venue with the
  most recent and most on-point appetite for this paper's genre — Journal of Computational
  Science — scored ZERO on the keyword set `venue_taste.md` used.** Sweep terms calibrated to
  an ML-sociology headline cannot find venues whose appetite is for evaluation-as-measurement.
  Any future venue sweep in this project should run both term sets.
- **The single most consequential fact in this document** is that the Editor-in-Chief of the
  previously recommended venue is an author of the benchmark the paper audits. It was not found
  by a scope search or a genre sweep; it was found by checking a named person against a board.
  **Do the conflict check early, not last.**
