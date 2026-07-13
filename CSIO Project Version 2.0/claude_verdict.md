# Peer Review
**Manuscript:** *Resource-Aware Staged Feature Acquisition for Retrospective Heart Disease Triage*
**Author:** Sparsh Chopra (Chandigarh College of Engineering and Technology)
**Reviewer perspective:** Critical peer review, as would be conducted for an IEEE health-informatics conference or a clinical-AI methods journal (e.g., IEEE JBHI, JAMIA, or an MDPI applied-AI title).

---

## 1. Executive Summary

This is a methodologically careful but empirically modest paper. It reframes heart-disease prediction as a two-tier, cost-aware triage simulation (basic intake features vs. a full 37-feature diagnostic profile) and audits a fixed-threshold routing policy using cross-validation, a held-out split, and — notably — leave-one-site-out (LOSO) testing on the four constituent UCI sites.

Its strongest asset is scientific honesty: it reports its own safety failure (4 true positives among 45 low-risk "bypass" patients) rather than hiding it, and it explicitly declines to claim clinical validity. That kind of self-auditing is genuinely uncommon and works in the paper's favor.

Its weakest points are (1) thin methodological novelty relative to the cost-sensitive/cascade-classification literature it cites, (2) no comparison against any learned acquisition policy or strong tabular baseline (e.g., gradient boosting), and (3) a leave-one-site-out result that is worse than the authors' own framing suggests — at two of the four sites, raw accuracy falls *below* a trivial majority-class baseline, which is a stronger and more concerning finding than "substantial site shift."

**Bottom line:** Not ready for a top ML venue in its current form (novelty gap). Plausible for a clinical-informatics/applied-health-AI venue after major revision, on the strength of its auditing rigor rather than its algorithmic contribution.

---

## 2. Overall Publication Score

| Category | Score /10 | Rationale |
|---|---|---|
| Novelty | 4 | Fixed two-tier cascade; core techniques (cascades, cost-sensitive acquisition) pre-exist and are cited, not extended |
| Technical Quality | 7 | Sound pipeline, correct split-before-preprocessing discipline, reasonable model choices |
| Scientific Rigor | 8 | CV + held-out + LOSO + calibration + decision-curve + bypass-safety audits is an unusually complete evaluation stack |
| Methodology | 6 | Thresholds (0.30/0.70) are heuristic, not derived from calibrated risk or a utility function |
| Experiments | 6 | Broad but shallow — no baseline against SOTA tabular models or existing budgeted-acquisition methods |
| Validation | 7 | LOSO is a genuine strength most papers using this dataset skip |
| Statistics | 6 | Good bootstrap CIs; McNemar test on 1 discordant case is close to uninformative and the paper somewhat undersells this |
| Writing | 7 | Clear, well-hedged, but dense and repetitive in places (limitations restate results) |
| Presentation | 6 | Tables are thorough but numerous; some could be merged |
| Literature Review | 5 | Misses the "learning to defer" / selective-prediction literature directly relevant to two-stage routing |
| Reproducibility | 6 | Software versions and dataset source given; no code repository link; one sentence about rotating API keys is a yellow flag (see §6) |
| Clinical Relevance | 3 | Authors themselves state this is not deployment-ready; bypass branch is demonstrably unsafe |
| Overall Impact | 5 | Useful as a workflow-auditing case study; unlikely to be cited as a methods contribution |

**Overall Publication Score: ≈ 58 / 100**

---

## 3. Recommendation

> **If submitted today: Major Revision** (clinical-informatics / applied-AI venue) — bordering on **Reject** for a methods-focused ML venue.

**Why not lower:** The evaluation discipline (LOSO, calibration, decision curves, bypass-safety accounting) is above the bar for most papers using this dataset, and the paper's own limitations section is unusually candid.

**Why not higher:** No novel algorithm, no comparison to any competing acquisition method, and one of the two headline claims (staged acquisition preserves accuracy while saving Tier-2 cost) is undercut by the paper's own safety and site-shift results.

### Acceptance probability by venue (qualitative estimate)

| Venue type | Estimate | Why |
|---|---|---|
| IEEE regional/applied conference | Moderate | Fits practice-oriented, workflow-simulation scope |
| Clinical informatics journal (JAMIA-style) | Moderate, w/ major revision | Rigor and TRIPOD/PROBAST framing are valued there; novelty bar is lower than core ML venues |
| MDPI applied-AI/health journal | Moderate–high | Comparable in scope to the Raman et al. paper the authors cite (same journal family) |
| Elsevier general ML/health journal | Low–moderate | Reviewers likely to request stronger baselines |
| Top-tier ML conference (NeurIPS/ICML/KDD) | Low | Novelty framed by the authors themselves as "not algorithmic" |

---

## 4. What Works (Strengths)

- **Honest safety reporting.** Table IV (4 positives among 45 bypassed patients, NPV 91.1%) is reported prominently, not buried — and the paper correctly concludes this makes the gate unsuitable for discharge decisions rather than spinning it as a success.
- **Leave-one-site-out testing.** Almost no paper using the combined UCI heart-disease file reports LOSO. This is the paper's most valuable empirical contribution, independent of its staged-acquisition framing.
- **Split-before-preprocessing discipline** (§III-C): imputation and scaling fit only on the training partition — a basic but frequently violated correctness requirement.
- **Calibration + decision-curve analysis** go beyond accuracy/AUROC, which is appropriate for a paper making a workflow-utility argument.
- **Clear scoping of the LLM explanation layer** as an unevaluated prototype rather than a validated contribution — this is the right way to include a speculative feature without overclaiming it.

---

## 5. Major Issues

### M1 — Accuracy falls below the trivial majority-class baseline at two of four LOSO sites
**Evidence:** Table VII: Switzerland — prevalence 0.935, reported accuracy 0.813; VA Long Beach — prevalence 0.745, reported accuracy 0.730.
**Why this matters:** A model that simply predicted "positive" for every patient at these two sites would score 93.5% and 74.5% accuracy respectively — both *higher* than what the trained cascade achieves. The paper reports this as "substantial site shift," but the more precise finding is that the model is actively worse than a content-free rule at these two sites. This is a stronger and more damaging result than the framing suggests.
**Severity:** Critical
**Probability this causes rejection/major-revision request:** High
**Fix:** Report the majority-class baseline explicitly alongside Table VII for every site; discuss why training-set prevalence (~55%) diverges so far from Switzerland/VA prevalence, and whether per-site or prevalence-adjusted threshold recalibration would close the gap.
**Effort:** Low (one extra column + a paragraph); does not require new experiments, since majority-baseline accuracy is derivable from the reported prevalence figures already in Table VII.

### M2 — Novelty gap versus cited literature
**Evidence:** §II-B explicitly states "the novelty is not algorithmic," while citing cascade classifiers (Viola & Jones), budgeted feature acquisition (Nan et al.), and cost-sensitive diagnosis (Kachuee et al.) as prior art, without comparing against any of them empirically.
**Why this matters:** A reviewer will ask: what does a fixed, hand-set two-threshold gate offer over a learned budgeted-acquisition policy, and is it better, worse, or just simpler? Without that comparison, the paper cannot support a claim of practical advantage — only descriptive feasibility.
**Severity:** High
**Probability this causes rejection:** High at a methods venue; Low–Medium at an applied/clinical venue.
**Fix:** Either (a) add one learned baseline (e.g., a simple cost-weighted classifier or reject-option / learning-to-defer model) as a comparison point, or (b) reframe the contribution explicitly as a *reporting/auditing methodology* rather than a *system*, and narrow claims accordingly.
**Effort:** Medium (a) requires new experiments; (b) is a rewrite, no new experiments.

### M3 — Thresholds are heuristic, not derived from calibrated risk or utility
**Evidence:** §III-F: "θL = 0.30, θH = 0.70... pragmatic prototype bands, not calibrated clinical cutoffs" (authors' own admission).
**Why this matters:** The entire cost/accuracy trade-off reported in the abstract depends on where these two numbers are set. Table V shows the trade-off is highly threshold-sensitive (0 bypassed positives at θL=0.05–0.15, rising to 15 bypassed positives at θL=0.60). Headline numbers are therefore an arbitrary operating point, not a Pareto-optimal one.
**Severity:** Medium–High
**Probability this causes rejection:** Medium
**Fix:** Report the full ROC-style trade-off curve (bypass rate vs. bypassed-positive rate) as a primary result rather than a single table, and pick the reported operating point via an explicit utility/cost function (even a hypothetical one) rather than round numbers.
**Effort:** Low–Medium (largely a re-presentation of data already computed for Table V).

### M4 — No strong tabular-ML or existing-method baseline
**Evidence:** Tier 2 model is RF + RF + LR + KNN soft-voting (§III-E); no gradient-boosted tree baseline (XGBoost/LightGBM/CatBoost) despite these being the de facto standard for structured clinical tabular data, and despite the compared Raman et al. paper (ref. [5]) using SHAP-based interpretation on similar data.
**Why this matters:** Reviewers will want to know whether 90.2% held-out Tier-2 accuracy is a strong result or simply typical for this well-studied dataset. Without a standard baseline, the number is hard to contextualize.
**Severity:** Medium
**Probability this causes rejection:** Medium
**Fix:** Add one gradient-boosting baseline trained identically; report head-to-head.
**Effort:** Low (single additional model, same pipeline).

### M5 — McNemar test is close to uninformative but is presented as a formal statistical result
**Evidence:** "Exact McNemar testing found no significant paired accuracy difference... (p = 1.000), but this comparison had only one discordant case" (§IV-A).
**Why this matters:** A test with one discordant pair has essentially no power; reporting p = 1.000 without heavy caveats risks being read as "no difference was found" rather than "no difference could possibly have been detected."
**Severity:** Medium
**Fix:** Either drop the test or replace/supplement it with a paired test computed across the 5 CV folds (much larger effective N), and state the achieved power of the held-out McNemar test explicitly.
**Effort:** Low (CV predictions already exist).

### M6 — Related work omits the closest adjacent literature: learning to defer / selective prediction
**Evidence:** §II covers cost-sensitive learning and cascades but not the "reject option" / "learning to defer" line of work, which is the more precise formal analogue of a two-stage human-facing routing gate.
**Why this matters:** This is the single closest body of related work to the paper's actual contribution, and its absence will be noticed by any reviewer familiar with selective classification.
**Severity:** Medium
**Fix:** Add 2–4 citations and a short paragraph distinguishing the fixed-threshold approach here from learned deferral/reject-option classifiers.
**Effort:** Low.

---

## 6. Reproducibility Flag (should be fixed before any submission)

Section V-B states: *"Any public release should remove and rotate all API keys before publication."*

This sentence should not appear in a submitted manuscript at all — its presence suggests a draft artifact (system prompt, code comment, or internal note) may not have been fully scrubbed before this version was assembled, and it raises a legitimate question for reviewers/editors about whether credentials were ever exposed in a linked repository or supplementary file. **Recommend removing this line and independently verifying no live keys are referenced anywhere in code or supplementary material before submission** — this is a manuscript-hygiene issue, not just a style note.

---

## 7. Minor Issues

- No count given for how many cholesterol values were zero/invalid before imputation (§III-C) — a single number would strengthen the preprocessing description.
- Table VIII shows isotonic calibration *worsening* Brier score and AUROC while modestly improving ECE — worth one sentence explaining why (small held-out N makes isotonic calibration noisy) rather than leaving the result unexplained.
- Female (N=38) and >65-year (N=20) subgroup counts in Table IX are explicitly called "inadequate," which is appropriate, but the subgroup section could be shortened or moved to an appendix given the authors' own caveat that no claim is being made from it.
- The LLM/explanation layer (§III-G) occupies significant manuscript space relative to its evaluated content ("not evaluated in this paper"); consider trimming to a short paragraph plus a note in Future Work.
- Abstract and Results are numerically consistent (spot-checked: CV cascade 84.5%±2.2%/0.898±0.034 AUROC/25.1%±3.3% avoidance matches Table VI; held-out 89.7%/24.5% matches Table II) — no discrepancies found here, which is good practice worth preserving in revision.

---

## 8. Section-by-Section Notes

| Section | Assessment |
|---|---|
| Abstract | Accurate, appropriately hedged, numbers match body — no inflation detected |
| Introduction | Clear motivation; five contributions listed are reasonable but contribution 1 ("formalizes a two-tier triage simulation") oversells novelty given cascades are pre-existing |
| Related Work | Solid but missing learning-to-defer/selective-prediction literature (M6) |
| Methods | Rigorous, correct train/test discipline; threshold choice under-justified (M3) |
| Results | Comprehensive; majority-baseline comparison missing at site level (M1) |
| Discussion | Honest, appropriately cautious; correctly foregrounds LOSO over the more flattering held-out numbers |
| Limitations | Unusually thorough (10 enumerated limitations) — a genuine strength |
| Reproducibility | Adequate versions/citations; missing code repo link; API-key sentence needs removal (§6) |
| References | Well-chosen and relevant, though could add reject-option/selective-classification citations (M6) |

---

## 9. Novelty Assessment

The paper itself concedes the core idea is not algorithmically new (§II-B), and this review agrees with that self-assessment. What is genuinely being offered is: (1) an explicit, auditable two-tier split on a specific dataset, and (2) an unusually complete post-hoc safety/calibration audit of that split. Item (2) is the paper's real contribution — but it is a *methodology-for-reporting* contribution, not a modeling contribution, and the paper would likely be better received if it were framed and positioned that way from the title onward (e.g., emphasizing "a safety-auditing case study for staged clinical AI" rather than "Smart Clinic Assistant... triage").

---

## 10. Final Verdict

**Major Revision**, contingent on:
1. Adding the majority-baseline comparison to the LOSO table (M1) — required, low effort.
2. Either adding a comparative baseline (learned acquisition or gradient-boosted model) or explicitly reframing the contribution as an audit methodology rather than a system (M2, M4) — required, choose one path.
3. Removing the API-key sentence and confirming no credentials appear anywhere in associated materials (§6) — required, trivial effort, but important.
4. Adding learning-to-defer/selective-prediction citations (M6) — recommended, low effort.

With these changes, this becomes a solid, honest applied-AI paper suitable for a clinical-informatics or applied-health-AI venue. Without them, reviewers at any venue are likely to flag the novelty gap and the unexamined site-level accuracy failure as grounds for rejection.