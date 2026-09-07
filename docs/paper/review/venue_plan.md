# Venue plan after two desk rejections

**Manuscript:** "NeuroForge: Self-Auditing Neural CFD Surrogates with Calibrated
Physics-Residual Trust" — `docs/paper/body.tex` + `docs/paper/abstract.tex`,
~14,700 words, 31 pp (46 pp in `elsarticle`), 10 figures, 9 tables.

**Rejection history**

| Venue | Date | Stage | Stated reason |
|---|---|---|---|
| CMAME | 2026-08-02 | desk, no review | "no new computational methodology within scope" |
| Journal of Computational Physics | 2026-09-07 | desk, no review | "originality with respect to other published papers is too questionable to justify that your submission is sent to reviewers"; will not reconsider |

**All verification in this document was done on 2026-09-07** unless a row says otherwise.

---

## 0. How to read the tables — verification provenance

ScienceDirect returned **HTTP 403** to every automated fetch of a guide-for-authors
page, a text-extraction proxy returned **451**, and AIP Publishing returned **403**.
So the rows below carry one of two provenance marks:

- **[P] primary-fetch** — the page itself was retrieved and read.
- **[S] search-extraction** — the content of the primary page as surfaced by a search
  engine reading that same URL. Good enough to *rank* on. **Not good enough to file
  on.**

There is a short must-eyeball list in §7. Do not submit before clearing it.

---

## 1. The diagnosis that drives the ranking

Both rejections were on **novelty of method**, at desk, without review. Neither was
about rigor, scope-mismatch, or presentation. That is a single, specific failure mode:

> The venue's acceptance criterion includes methodological novelty, and this paper's
> contribution is an empirical finding + a falsification + a calibration layer built
> from known parts.

So the discriminating question for every candidate is **not** "does this paper fit the
topic?" It is **"does the venue's own scope text make novelty of method an acceptance
criterion, or does it make rigor the criterion?"** A venue that publishes ML-CFD papers
but screens on novelty will produce desk rejection #3 on identical grounds.

A second, near-equal criterion emerged during verification: **review model**. A public
arXiv preprint (2607.10333 v3), a coined system name that runs through the whole
manuscript, and a code URL containing the author's GitHub username make double-anonymized
venues materially more expensive and slightly less honest. A single-anonymized venue
makes that entire problem disappear. See §5.

This ranking therefore optimises **P(reaches review) first, impact factor second**. After
two desk rejections that is the correct order.

---

## 2. Ranked shortlist

### #1 — Computers & Fluids (Elsevier) — **PRIMARY RECOMMENDATION**

| | | prov. |
|---|---|---|
| Cost model | **Hybrid.** Subscription route costs nothing. OA route optional and to be declined. | [S] |
| Truly free? | **Yes** via the subscription licence at the licensing step. | [S] |
| Length limit | **No stated article length limit.** Abstract "concise and factual… does not exceed 250 words" — the current abstract is 249. | [S] |
| Review model | **Single anonymized.** | [S] |
| IF / CiteScore | **IF 3.0** (JCR released 2026-06-17, on 2025 data); **CiteScore 5.9**. | [S] |
| Time to first decision | Not published on the guide page. Historically ~6–10 weeks for CFD papers; treat as unverified. | — |
| Publisher / portal | Elsevier, Editorial Manager. Existing account `AJabbary-884` works. | [P] |
| Desk-screen risk | **LOW–MEDIUM** | |

**Scope quote (the reason this is #1):**

> "The development of numerical methods relevant to fluid flow computations, computational
> analysis of flow physics and fluid interactions and novel applications to flow systems and
> to design are pertinent to Computers & Fluids. The journal also accepts papers dealing with
> **uncertainty quantification in fluid flow simulations, reduced-order and surrogate models
> for fluid flows**, optimization and control. **Papers dealing with machine learning approaches
> applied to fluid flow modeling are welcome, provided they show excellent scientific
> character.** In particular, the authors are encouraged to perform comparisons with traditional
> numerical reconstruction methods, to provide a clear presentation of training vs validation
> cases, together with sufficient diversity in these cases, to analyze the physical
> consistency/theoretical analysis of the ML model, and **to discuss the limitations of the
> method as well as its merits.**"

Read that sentence against the failure mode. The bar it sets for an ML paper is
**"excellent scientific character"** — a *rigor* bar — followed by a list of encouraged
practices. It never asks for a new numerical method. And the enumerated practices are, almost
line for line, an inventory of what this manuscript already contains:

| C&F encouraged practice | What the manuscript has |
|---|---|
| UQ in fluid flow simulations | Split-conformal coverage layer, multi-split, bootstrap CIs |
| Surrogate models for fluid flows | Three architecturally distinct backbones + a competitive Transolver-class SOTA baseline |
| Clear training vs validation presentation, sufficient diversity | AirfRANS + DeepCFD, hash-manifested splits, 5 seeds |
| Physical consistency / theoretical analysis of the ML model | The discretised RANS residual audit **and** the residual-floor theorem |
| Discuss limitations as well as merits | The falsification (residual fails as a correction objective), the honest-baseline subsection, the explicit limitations list |

This is the only candidate where the venue's own scope text can be quoted back as a
compliance checklist rather than argued around.

**Why it beats the higher-IF options:** single-anonymized review means the arXiv preprint
stays up, the "NeuroForge" name stays in the text, the GitHub and Zenodo links stay live, and
no anonymized mirror is needed. Zero rework. The submission is the existing
`neuroforge_cfd_elsevier.pdf` build plus a retargeted cover letter.

**The honest downside:** IF 3.0 is below JCP's 3.9 and well below EAAI and RESS. And the
review-stage scrutiny will be the toughest of any candidate, because the readership is CFD
practitioners — see §6.

---

### #2 — Engineering Applications of Artificial Intelligence (EAAI, Elsevier)

| | | prov. |
|---|---|---|
| Cost model | **Hybrid.** OA APC **USD 3,040** (ex. tax) — to be declined; subscription route free. | [S] |
| Truly free? | **Yes** via subscription licence. | [S] |
| Length limit | **None** in the guide; only the 250-word abstract cap. **Weakly sourced** — this is the 2026-08-21 reading of the guide page, and today's searches of that same (blocked) page never surfaced the length clause at all. One source, cited twice, is not two sources. Given the manuscript is ~14,700 words, treat this with the same seriousness as the RESS cap: **must be eyeballed** (§7). | [S], single source |
| Review model | **Double anonymized** — separate title page + anonymized manuscript, mandatory. | [S] |
| IF / CiteScore | **IF 9.0** (JCR 2026-06-17); **CiteScore 11**; Q1. | [S] |
| Time to first decision | Not published. High submission volume; expect a fast desk decision either way. | — |
| Desk-screen risk | **MEDIUM–HIGH** | |

**Scope quote:**

> "…rapid publication of work describing **the practical application of AI methods in all
> branches of engineering**. Submitted papers should report **novel aspects of AI used for a
> real-world engineering application** and also **validated using public data sets for easy
> replicability of the research results**."

**Honest assessment — does it survive a desk screen?** The task asked for a judgement, so:
**probably, but it is a coin flip, and it is not the safest next move.** Two clauses cut
against the paper and one strongly for it.

- *Against — "real-world engineering application."* This is a benchmark methods paper. AirfRANS
  and DeepCFD are public research benchmarks, not a deployed engineering system. There is no
  industrial partner, no design study that shipped, no field data. An EAAI screening editor
  looking for the application will not find one. The aerodynamic-design framing (a trust layer
  that decides when a surrogate's answer may be used in a design loop) is *plausible* but it is
  a narrative wrapper, not a demonstrated application.
- *Against — "novel aspects of AI."* The word "novel" is right there in the scope, which is the
  exact word that killed the paper twice. The counter is that "novel aspects of AI" is a weaker
  bar than CMAME's "new computational methodology" — a novel *use* of conformal prediction as a
  physics-residual trust layer plausibly clears it — but it is not a bar the paper avoids.
- *For — "validated using public data sets for easy replicability."* Very few submissions comply
  with this as thoroughly as this one does: two public datasets, hash manifest, Zenodo DOI,
  `REPRODUCE.md`, committed seeds. This is a genuine, quotable differentiator and it should be
  the *second* paragraph of the cover letter.

**What the cover letter must lead with at EAAI:** not the dissociation, and not the theorem.
Lead with the **decision problem an engineer actually has**: a fast neural surrogate is being
used inside an aerodynamic design loop, and there is currently no way to know per-case whether
its answer may be trusted; this paper supplies a calibrated, distribution-free trust layer that
answers that question, with a guaranteed no-harm acceptance gate, validated on public data.
Only then, in the results paragraph, the two-way dissociation as the finding that makes the
trust layer honest.

**Cost of complying with double-anonymized review honestly:** see §5. It is real work — roughly
a half-day — but it is tractable and it is *not* a reason to avoid EAAI. It is a reason not to
do it first when a single-anonymized venue of adequate quality is available.

---

### #3 — Reliability Engineering & System Safety (RESS, Elsevier)

| | | prov. |
|---|---|---|
| Cost model | **Hybrid.** Subscription route free; OA route carries an APC — decline it. | [S] |
| Truly free? | **Yes** via subscription licence. | [S] |
| Length limit | **"only accepts manuscripts with a maximum 13,000 words"** — manuscript is ~14,700. **GO/NO-GO.** | [S] — **must be eyeballed** |
| Review model | Anonymized; single vs double not confirmed. | [S] |
| IF / CiteScore | **IF 13.7** (JCR 2026-06-17); **CiteScore 10.2**; Q1. Highest of any candidate. | [S] |
| Time to first decision | Not published. | — |
| Desk-screen risk | **MEDIUM–HIGH** | |

The bar here is genuinely the right one — reliability, uncertainty, trust in model
predictions — and RESS publishes a great deal on surrogate models and UQ. Conformal
coverage, selective prediction, and a monotone no-harm gate are RESS-flavoured objects.

**Two things must be true before this is viable:**

1. **The 13,000-word cap.** ~1,700 words must come out, and the cut has to be real, not
   appendix-shuffling. Verify the cap in a browser first — if it is 13,000, this is a
   substantial editing job; if it turns out to be advisory, the calculus changes.
2. **A genuine reliability reframe, not a bolted-on one.** RESS papers estimate failure
   probabilities of engineered systems. This paper has no system, no failure probability, and
   no risk decision. It would need to be honestly recast as *reliability of the surrogate as a
   prediction instrument*: the trust score as a screening statistic, the conformal layer as a
   coverage guarantee, the acceptance gate as a no-harm safeguard, and the falsification as
   evidence that a widely-assumed reliability indicator does not do the job it is assumed to do.
   That last framing is actually strong for RESS — "a commonly assumed physics-based reliability
   indicator is diagnostic but not corrective" is a risk-relevant negative result. But it is a
   rewrite of the framing, not a new cover letter.

Highest reward on the list. Keep it as the deliberate, funded next move if C&F declines and
there is appetite for a rewrite — not as a quick resubmission.

---

### #4 — Computer Physics Communications (CPC, Elsevier)

| | | prov. |
|---|---|---|
| Cost model | **Hybrid.** "No publication fee is charged to authors"; OA APC **USD 4,680** — decline. | [S] |
| Truly free? | **Yes** via subscription route. | [S] |
| Length limit | None stated. | [S] |
| Review model | Not confirmed; Elsevier default single anonymized. | — |
| IF | **IF 3.9** (JCR 2026-06-17); Q1. | [S] |
| Article types | **CPiP** (Computer Programs in Physics — program archived in the CPC Program Library on Mendeley Data, must carry an approved open-source licence) and **CP** (Computational Physics papers; "normally include software implementation and performance details… available via GitHub, Zenodo or an institutional repository"). | [S] |
| Desk-screen risk | **MEDIUM–HIGH** | |

The attraction was that the CPiP track judges the **program**, not the novelty of the
numerics, which would sidestep the exact failure mode. The repository is MIT-licensed
(`D:\Codes\Github\neuroforge-cfd\LICENSE`), on GitHub, with a Zenodo DOI — mechanically eligible.

**But the scope text does not actually give the orthogonality it promised:**

> "The focus of CPC is on contemporary computational methods and techniques and their
> implementation, the effectiveness of which will normally be evidenced by the author(s)
> **within the context of a substantive problem in physics**."

and, for CPiP:

> "Papers and associated computer programs that address **a problem of contemporary interest in
> physics that cannot be solved by current software** are particularly encouraged."

That is an effectiveness-in-physics bar. A PyTorch trust-and-calibration layer for 2-D airfoil
RANS surrogates is not obviously a physics problem to that editorial board, and CPC sits in the
same Elsevier computational-physics family as the journal that just rejected the paper on
originality. Additionally, filing as CPiP means restructuring the manuscript around the program
(mandatory Program Summary, program-library deposit, code as the object of the paper) — that is
a different paper, not a resubmission.

Keep it, rank it fourth, and only reach for it if the paper is going to be rewritten anyway.

---

### #5 — Safety net: Journal of Computational Science, or Engineering with Computers

Two interchangeable scope-safe landing spots. Both hybrid, both free by the subscription route,
both around IF 4, both with no length problem, both with a low novelty bar.

| | Journal of Computational Science (Elsevier) | Engineering with Computers (Springer) | prov. |
|---|---|---|---|
| Cost | Hybrid; OA APC USD 2,810 — decline | Hybrid; OA APC £2,290 / USD 3,290 — decline | [S] |
| Truly free? | Yes, subscription route | Yes, subscription route ("published articles are made available to institutions and individuals who subscribe") | [S] |
| IF | **4.0** (2026-06-17), Q2 | **4.1** (2026-06-17), Q1 | [S] |
| Scope | "international platform to exchange novel research results in **simulation-based science across all scientific disciplines**… Modeling, Algorithms and Simulations; **Software developed to solve science… problems**" | "technologies supporting **simulation-based engineering**… includes adaptive simulation techniques, **data-driven and hybrid modeling methods**, design through analysis workflows, **engineering software development**, simulation-based optimization" | [S] |
| Length | None stated | None stated | [S] |
| Desk risk | **LOW** | **LOW–MEDIUM** | |

Neither is a step up in prestige. Both are near-certain to at least reach review. Use one of
these if C&F and EAAI both decline and there is no appetite for the RESS rewrite.

---

### #6 — Backstop: Journal of Machine Learning for Modeling and Computing (Begell House)

| | | prov. |
|---|---|---|
| Cost model | Begell subscription title — no APC. **Confirm no page charges before filing.** | [S] |
| Scope | "machine learning methods for **modeling and scientific computing**… the use of machine learning techniques to model real-world problems, the development of novel numerical strategies in conjunction with machine learning methods, and fundamental mathematical and numerical analysis for understanding machine learning methods." | [S] |
| Metrics | **Scopus-indexed since 2024; accepted into Web of Science ESCI.** CiteScore 5.8. A publisher news page advertises "Impact Factor 3.5" — **treat that as marketing**: ESCI titles do not carry a JCR Impact Factor. | [S] |
| Desk risk | **VERY LOW** | |

Scope fit is close to perfect and cost is zero, but it fails the author's stated wish for
"journal identity and an impact factor" — there is no JCR IF. Guaranteed-landing backstop only.

---

## 3. Explicitly considered and ruled out

| Venue | Verdict | Reason |
|---|---|---|
| **CMAME** | OUT | Already desk-rejected 2026-08-02. |
| **Journal of Computational Physics** | OUT | Desk-rejected 2026-09-07, will not reconsider. |
| **TMLR / JMLR** | OUT | Author's call — wants journal identity and an impact factor. |
| **Machine Learning: Science and Technology (IOP)** | OUT | Fully OA, £2,500; and an 8,500-word cap against ~14,700. |
| **Data-Centric Engineering (Cambridge)** | OUT | Fully open access. Violates the no-APC constraint outright. |
| **Physics of Fluids (AIP)** | OUT — **changed since 2026-08-21** | Hybrid and genuinely free (AIP charges no page or colour fees; the USD 3,800 Author Select fee is optional). But **two 2026 editorials from the new co-Editors-in-Chief establish a "Physics First" policy**: "the central contribution of a publication must contribute to an advancement in physical understanding," assessed by "What is the physical question? What mechanism or principle is revealed?" A trust-calibration and falsification paper answers neither. This is a live, dated scope narrowing and it converts PoF from plausible to high desk risk. |
| **International Journal for Numerical Methods in Fluids** | OUT | Same new-numerics bar that produced the CMAME rejection. |
| **Theoretical and Computational Fluid Dynamics** | OUT | Bar is a fluid-dynamics theoretical result. |
| **Journal of Fluid Mechanics** | OUT | Bar is a fluid-mechanics insight. Near-certain desk rejection. |
| **AIAA Journal** | OUT | Bar is an aerospace result; additionally a US professional society in an export-control-sensitive field — the one place on this list where the Iranian affiliation adds real friction. Not worth it when Elsevier options exist. |
| **Aerospace Science and Technology** | Low priority | Free and hybrid, but the bar is an aerospace application — the same problem as EAAI without EAAI's public-dataset clause working in your favour. |
| **Neural Networks (Elsevier)** | Low priority | Hybrid and free, but the bar is a contribution to neural-network science; a CFD trust layer reads as an application to that board. |
| **Probabilistic Engineering Mechanics** | Low priority | Right conceptual bar (UQ), but a narrower and more stochastic-mechanics-flavoured readership than RESS, at much lower IF. Only if RESS declines and the reframe already exists. |

---

## 4. Does a "welcomes negative results" venue change the calculus? — No, and here is why

This was searched directly. **The category is structurally incompatible with the no-APC
constraint.** Venues that explicitly solicit negative, null, or falsification results are
almost entirely gold open access (PLOS ONE, Royal Society Open Science, Data-Centric
Engineering, the various "Journal of Negative Results in X") or are conference tracks
(ICSME 2026 and EASE 2026 both run explicit negative-results tracks — but they are software-
engineering conferences, wrong field, and give no journal identity). No subscription or hybrid
journal with an impact factor was found that advertises negative results as a welcomed category.

**The practical substitute, and it is a good one:** Computers & Fluids' scope text says ML papers
should "**discuss the limitations of the method as well as its merits**." That is the closest
thing to an explicit invitation for a falsification available inside the constraint, and it can
be quoted in the cover letter. It reframes the negative result from a weakness the paper has to
excuse into a requirement the paper over-satisfies.

So: the falsification is still the paper's strongest content, but it should be **positioned as
rigor, not marketed as a genre**. That positioning is venue-specific — see §8.

---

## 5. The anonymization problem, per venue

**The good news first:** Elsevier's preprint policy permits preprints and does not treat an
existing preprint as breaking double-anonymized review. The standard to meet is **good-faith
anonymization of the submitted files**, not unlinkability. **Do not take arXiv 2607.10333 down**,
and do not rename the package.

| Venue | Review model | Obstacle | Work required |
|---|---|---|---|
| **Computers & Fluids** | Single anonymized | **None.** | **Zero.** Names, arXiv link, GitHub URL, Zenodo DOI, and the "NeuroForge" name all stay exactly as they are. |
| **EAAI** | Double anonymized | Real but tractable | See below |
| **RESS** | Anonymized, model unconfirmed | Assume double until verified | Same as EAAI |
| **CPC / JOCS / EwC** | Assume single | Likely none | Verify before filing |

**What honest compliance costs at a double-anonymized venue:**

1. **Separate title page file** with authors, affiliations, ORCIDs, acknowledgements, funding
   statement, and CRediT — none of which may appear in the manuscript file.
2. **Anonymized manuscript**: strip acknowledgements, funding, and the CRediT block from the
   back matter of the review build only.
3. **De-anonymizing links.** The back matter cites `github.com/ali-kin4/neuroforge-cfd` — the URL
   *is* the author's username — plus the Zenodo DOI. For review, replace both with an
   `anonymous.4open.science` mirror or a "repository link withheld for double-anonymized review;
   available to the editor on request" note. Restore at acceptance.
4. **The coined system name is the real leak, and the checklist does not mention it.** "NeuroForge"
   plus the title googles straight to the preprint in one query. Because both PDF builds share
   `abstract.tex` and `body.tex`, the clean fix is a **single system-name macro in `preamble.tex`**
   that expands to `NeuroForge` in the arXiv/camera-ready build and to a generic descriptor (e.g.
   "the released package") in the review build. Cheap, reversible, and it is the difference between
   anonymized in form and anonymized in fact.
5. **Keywords 7 → 6** at EAAI (per the 2026-08-21 check; re-verify).

**This is a genuine reason to prefer C&F for the immediate resubmission.** It is not a reason to
rule EAAI out.

---

## 6. Review-stage risk at C&F — know this before you file

Desk risk is low; **review** risk is the highest of any candidate, because the readership is CFD
practitioners rather than an AI audience. Three exposures, in order:

1. **"Comparisons with traditional numerical reconstruction methods" is in the very scope
   sentence being quoted as the fit argument** — so a reviewer will hold the paper to it. What the
   manuscript actually has: force coefficients evaluated against the official OpenFOAM AirfRANS
   labels (`body.tex:771`, `body.tex:1419`), i.e. the reference data *is* a traditional solver. What
   it does **not** have: a controlled head-to-head against a classical solver. `body.tex:1358–1368`
   states plainly that the ~286x figure "indicates the order of the gap, not a controlled speed-up
   measurement," and `body.tex:1636–1638` states the OpenFOAM/SU2 fallback backends "are not
   implemented." Both disclosures are honest and should stay. **Pre-empt this in the cover letter**
   in one sentence: the classical anchor is the AirfRANS reference solution itself, and the
   uncontrolled timing comparison is labelled as such in the text.
2. **The uniform 128² grid and near-wall resolution.** A CFD reviewer will press on whether the
   discrete residual is meaningful in the boundary layer. The residual-floor theorem is the right
   defence and it is already in the paper (`sections/residual_floor_theorem.tex`); make sure it is
   signposted in the introduction, not only in its own section.
3. **The turbulence closure dependence** — `body.tex:1630–1635` already concedes that the audit
   operator is only as good as the predicted nu_t (R² 0.996 on the grid backbone but 0.67–0.70 on
   MeshGraphNet). Leave it in; a CFD reviewer who finds it stated is reassured, one who finds it
   hidden is not.

None of these change the recommendation. All three are better raised by the authors than
discovered by a reviewer.

---

## 7. Must-eyeball list before filing (the [S] rows that matter)

ScienceDirect and AIP both blocked automated retrieval, so these need a human browser session:

1. **C&F — confirm there is no article length limit, and confirm single-anonymized review.**
   <https://www.sciencedirect.com/journal/computers-and-fluids/publish/guide-for-authors>
2. **C&F — confirm hybrid status and that the subscription licence is offered at acceptance.**
3. **RESS — confirm the 13,000-word cap.** This is a go/no-go and it is currently sourced from a
   search snippet. <https://www.sciencedirect.com/journal/reliability-engineering-and-system-safety/publish/guide-for-authors>
4. **EAAI — confirm that there is genuinely no article length limit** (single-sourced from the
   2026-08-21 read; a ~14,700-word manuscript makes this a go/no-go, exactly as at RESS). Also
   confirm the keyword limit (1–6 as of 2026-08-21), that **double-anonymized review is still
   mandatory rather than author-choice**, and that the "validated using public data sets" sentence
   is still in the current scope text.
   <https://www.sciencedirect.com/journal/engineering-applications-of-artificial-intelligence/publish/guide-for-authors>
5. **Any Begell filing — confirm there are no page charges**, since "no APC" and "no charges" are
   not the same thing at society/small-publisher titles.

---

## 8. Primary recommendation

### Submit to **Computers & Fluids**, next.

**The reason, in one paragraph.** Two desk rejections in five weeks on the same ground mean the
binding constraint is no longer prestige — it is reaching review at all. C&F is the only candidate
whose published scope text (a) names uncertainty quantification and surrogate models for fluid
flows as in-scope topics, (b) sets the bar for ML papers at "excellent scientific character"
rather than methodological novelty, and (c) enumerates a list of encouraged practices that this
manuscript can be mapped onto line by line, *including* discussing limitations as well as merits.
It is hybrid and genuinely free by the subscription route, it has no length limit against a
14,700-word manuscript, the 249-word abstract already clears its 250-word cap, the existing
`elsarticle` build and Editorial Manager account are reused unchanged, and — decisively —
**single-anonymized review makes the arXiv preprint, the GitHub URL, and the "NeuroForge" name
into non-issues**. The cost is an IF of 3.0 against JCP's 3.9. That is the right trade after two
desk rejections.

### What the cover letter and abstract must lead with

**Do not lead with the falsification at C&F.** That framing is calibrated to a novelty screen;
C&F's screen is scope-and-rigor. Invert the order: lead with what the residual *is* good for,
then present the falsification as the limitation analysis their guide explicitly asks for.

**Cover letter, paragraph 1 — name the box, quoting their scope:**
> We submit this work under the journal's stated interest in *uncertainty quantification in fluid
> flow simulations* and *surrogate models for fluid flows*. The paper asks whether the discretised
> steady-RANS residual of a neural surrogate can be used to decide, per case and without ground
> truth, whether that surrogate's prediction may be trusted — that is, whether a surrogate's
> answer may be used in an aerodynamic design loop or must be referred back to a solver.

That last clause is doing specific work. The same scope sentence also covers "**novel applications
to flow systems and to design**," and that is the phrase a C&F editor uses to decide whether this is
a fluids paper or an ML paper that merely mentions fluids. One line tying the trust layer to a
design decision closes the only remaining desk-screen gap. Keep it to one line — at C&F the design
framing is a qualifier, not the lead. (At EAAI it becomes the lead; see §2.)

**Cover letter, paragraph 2 — map onto their encouraged practices, in their order.** Comparison
against the traditional solver (AirfRANS OpenFOAM reference solutions, with the uncontrolled
timing caveat stated up front, §6.1); clear training/validation presentation with diversity
(three architecturally distinct backbones, two datasets — AirfRANS and DeepCFD, five seeds,
hash-manifested splits); physical consistency and theoretical analysis (the residual audit plus
the residual-floor theorem, which proves the discrete residual does not vanish at the true
solution); and limitations as well as merits.

**Cover letter, paragraph 3 — the falsification, framed as rigor:**
> In the spirit of the journal's request that authors discuss limitations as well as merits, we
> report a negative result as a first-class finding: the same residual that works as a trust
> signal fails as a correction objective — reducing it does not reduce field error — and a
> controlled ablation shows the gain from our learned corrector does not come from conditioning
> on the residual. We regard this dissociation, and the theorem that explains it, as the paper's
> most durable content.

**Paragraph 4 — reproducibility, and one disclosure decision the author must make.**
Open-source MIT package, Zenodo DOI, `REPRODUCE.md`, hash manifest. **Disclose the preprint**
(arXiv:2607.10333 v3) plainly: it is discoverable, and non-disclosure of a discoverable preprint
looks evasive. Then state the positioning sentence, which is the part that does the real work:

> The contribution is an empirical and theoretical finding about an existing class of methods,
> not a new numerical scheme.

**Author decision — do NOT treat this as an instruction.** Whether to also disclose the two prior
desk rejections is a genuine judgment call, and the balance is against it. Arguments each way:

- *For disclosing:* it is candid, it pre-empts an editor who happens to know, and it explains the
  positioning sentence.
- *Against, and this is the stronger case:* desk rejections at other journals are **not
  discoverable**, carry **no disclosure obligation**, and are not standard to volunteer in
  Editorial Manager. Telling a third Elsevier editor that two Elsevier journals declined the paper
  on originality hands them a ready-made, pre-authorised reason to do the same. The positioning
  sentence above delivers the entire useful half of the disclosure without supplying the ammunition.

**Recommendation: disclose the preprint, omit the rejection history, keep the positioning
sentence.** If the author prefers full candour, the cost is real and should be accepted knowingly.

**Abstract change — one edit, worth making.** The abstract currently opens on a *framework*
("we make the surrogate audit itself"), which is the framing that read as "engineering artifact"
to two previous screens, and the dissociation does not appear until sentence three. Move the
two-way dissociation to sentence **two** and demote the self-auditing framework to the
contribution list at the end. That is a reordering, not a rewrite, and it keeps the 249-word count
(re-count after editing; the JCP-era note in `submission/SUBMISSION_CHECKLIST.md` about
`$\approx 0.9$` and the em-dash inflating naive counters still applies).

**Title.** Leaving it unchanged is defensible at C&F, because a single-anonymized CFD journal will
not penalise a named package. But note honestly: the title still leads with a coined system name,
which is what invited "no new computational methodology" twice. If there is appetite for one
change, leading the title with the finding rather than the system is the highest-leverage edit
available. It is optional at C&F and close to mandatory at EAAI and RESS.

### Fallback order

1. **Computers & Fluids** — now. No rework. (Free, IF 3.0, single anonymized.)
2. **EAAI** — if C&F declines. Requires the double-anonymization package (§5) and an
   application-led cover letter (§2). (Free, IF 9.0.)
3. **RESS** — only with budget for a real reframe and a cut to ≤13,000 words, and only after the
   word cap is verified in a browser. (Free, IF 13.7 — the highest-reward branch.)
4. **CPC** — only if the paper is being restructured as a program paper anyway. (Free, IF 3.9.)
5. **Journal of Computational Science** or **Engineering with Computers** — scope-safe landing.
   (Free, IF 4.0 / 4.1.)
6. **JMLMC (Begell)** — guaranteed landing, no JCR impact factor. (Free, Scopus + ESCI.)

### One standing rule

Do not file at #2 or later until an experiment-rigor pass confirms the claims are still supported
after any reframing edits, and the reviewer-side objections in §6 are answered in the text rather
than in the cover letter alone. Reframing a paper across venues is exactly where an unsupported
sentence gets introduced.

---

## 9. Sanctions / Iran affiliation — settled, for the Elsevier candidates

Verified **[P]** at <https://www.elsevier.com/about/policies-and-standards/trade-sanctions>
(2026-09-07). The policy states that "authors publishing in their personal capacity who are
employed by an academic or research institution that are controlled by the government … are not
deemed to be publishing on behalf of the government," and further that "if authors are publishing
in their personal capacity, they are not disqualified from publishing even if their research
institution itself is sanctioned." The binding conditions are that the authors are not designated
SDNs and are publishing in a personal capacity. A corresponding author at Urmia University meets
this. **Empirically confirmed:** both CMAME and JCP processed the submission and decided it on
the merits.

Springer (Engineering with Computers) and Begell present no known additional barrier. The single
flag worth keeping is US professional societies — AIAA in particular, in an export-control-
sensitive field — which is one more reason AIAA Journal is out in §3.

---

## 10. Sources verified, with access date

All accessed **2026-09-07** unless noted.

**Primary-fetch [P]:**
- Elsevier trade sanctions policy — <https://www.elsevier.com/about/policies-and-standards/trade-sanctions>
- Repository licence (MIT) — `D:\Codes\Github\neuroforge-cfd\LICENSE`

**Guide/scope pages read via search-engine extraction [S]** (all returned HTTP 403 to direct
automated fetch):
- Computers & Fluids guide for authors — <https://www.sciencedirect.com/journal/computers-and-fluids/publish/guide-for-authors>
- Computers & Fluids journal page (aims & scope) — <https://www.sciencedirect.com/journal/computers-and-fluids>
- EAAI guide for authors — <https://www.sciencedirect.com/journal/engineering-applications-of-artificial-intelligence/publish/guide-for-authors>
- EAAI journal page (aims & scope) — <https://www.sciencedirect.com/journal/engineering-applications-of-artificial-intelligence>
- RESS guide for authors — <https://www.sciencedirect.com/journal/reliability-engineering-and-system-safety/publish/guide-for-authors>
- CPC guide for authors — <https://www.sciencedirect.com/journal/computer-physics-communications/publish/guide-for-authors>
- Journal of Computational Science guide for authors — <https://www.sciencedirect.com/journal/journal-of-computational-science/publish/guide-for-authors>
- Engineering with Computers, aims & scope and publishing options — <https://link.springer.com/journal/366/aims-and-scope> and <https://link.springer.com/journal/366/how-to-publish-with-us>
- JMLMC (Begell) — <https://www.begellhouse.com/journals/journal-of-machine-learning-for-modeling-and-computing.html> and <https://www.begellhouse.com/news/6>
- Physics of Fluids "Physics First" editorials — <https://pubs.aip.org/aip/pof/article/38/3/030401/3383053/Physics-first-Faster-and-fairer-A-renewed-vision> and <https://pubs.aip.org/aip/pof/article/38/8/080401/3402803/Physics-First-What-we-mean-by-Physics-First>
- Physics of Fluids charges — <https://pubs.aip.org/aip/pof/pages/charges>

**Metrics (JCR 2026, released 2026-06-17, on 2025 citation data)** — <https://www.journalmetrics.org/>
per-journal pages for Computers & Fluids, EAAI, RESS, Computer Physics Communications,
Journal of Computational Science, Engineering with Computers.

---

## 11. Changes since the 2026-08-21 fallback check in `submission/SUBMISSION_CHECKLIST.md`

| Item | Then (2026-08-21) | Now (2026-09-07) |
|---|---|---|
| EAAI cost / length / IF | Hybrid, free; no length limit; IF 9.0 | **Unchanged.** OA APC now quantified at USD 3,040 (declined). |
| EAAI review model | Double anonymized | **Unchanged** — but re-verify whether it is mandatory or author-choice. |
| Computers & Fluids | Listed only as "third option… scope-safe but lower metrics"; IF 3.0, CiteScore 5.6 | **IF 3.0 confirmed. CiteScore: discrepancy, not a change** — this search returned 5.9 against the checklist's 5.6. CiteScore is an annual metric and does not move in seventeen days, so one of the two figures is simply wrong; do not report either as movement. Resolve it on the browser pass (§7). More importantly the checklist **understated the journal**: it did not record that the scope text explicitly names UQ and surrogate models, sets a rigor rather than novelty bar for ML papers, asks for limitations to be discussed, and uses **single-anonymized** review. On the criteria that matter after two originality desk rejections, it outranks EAAI. |
| Ordering | JCP → EAAI → C&F | **JCP is dead. New order: C&F → EAAI → RESS → CPC → JOCS/EwC → JMLMC.** |
| Physics of Fluids | Not considered | **Considered and ruled out** — the 2026 "Physics First" policy is a live scope narrowing. |
| Negative-results venues | Not considered | **Considered and ruled out as a category** — structurally gold-OA. |
