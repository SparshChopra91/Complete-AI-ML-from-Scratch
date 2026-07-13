# Phase 1 Manuscript Draft: IMRAD Structure

Working manuscript for the Smart Clinic Assistant project.

This file is written for publication planning, not marketing. The strongest publishable angle is not "another heart-disease classifier." The stronger claim is a resource-aware clinical decision-support architecture that combines staged feature acquisition, explainable ML, and clinician-facing language generation.

## Submission-Readiness Corrections

These corrections must be handled before journal submission.

1. Do not say the project uses the exact same dataset as Raman et al. Raman et al. used the Cleveland subset after complete-case filtering. The current project file `heart_disease_uci.csv` contains 920 records across Cleveland, Hungary, Switzerland, and VA Long Beach. The paper should say "multi-site UCI Heart Disease data" unless the experiments are rerun on the Cleveland-only subset.

2. Decide which routing policy is the primary contribution. The current `app.py` is conservative: Tier 2 is requested for gray-zone and high-risk Tier 1 cases, and only low-risk cases bypass Tier 2. The stronger "uncertainty gate" paper claim is symmetric: Tier 2 is requested only when 0.30 < p1 < 0.70, while low-risk cases are discharged for review and high-risk cases are escalated without waiting for additional model inputs. Both policies are reported below, but the final paper and app should match.

3. Do not publish `.env`. The folder contains an OpenRouter API key. Exclude it from any repository, rotate the key before public release, and describe the LLM service only as an environment-variable-configured API dependency.

4. Avoid overclaiming clinical deployment. This is a retrospective decision-support simulation on public data. The correct claim is workflow feasibility and economic simulation, not validated clinical safety.

5. Add confidence intervals before submission. Raman et al. explicitly noted the weakness of a single held-out split. Our current metrics are also single-split estimates and should be strengthened with repeated stratified cross-validation or bootstrap confidence intervals.

## Local Project Audit

| File | Role in the paper | Publishability note |
| --- | --- | --- |
| `epidemiologia-07-00075.pdf` | Primary reference paper by Raman et al. | Use for literature gap: strong SHAP-based accuracy, but no staged feature acquisition or clinician-facing translation layer. |
| `gemini_publishing_verdict.md` | Narrative strategy | Useful for gap framing, but language should be softened for academic writing. Say "operational gap," not "fatal flaw." |
| `heart_disease_uci.csv` | Study dataset | 920 records: Cleveland 304, Hungary 293, VA Long Beach 200, Switzerland 123. Target was binarized as target > 0. |
| `Data_Processing_final.py` | Preprocessing pipeline | Stratified 80/20 split, one-hot encoding, imputation, scaling, Tier 1/Tier 2 slicing. |
| `Model_handler_final.py` | Model training/loading | Tier 1 Random Forest; Tier 2 soft VotingClassifier with two Random Forests, Logistic Regression, and KNN. |
| `Ai_explainer_shap.py` | Prototype SHAP-to-LLM explainer | Shows the original idea of forwarding SHAP values to an LLM; `app.py` now has the integrated version. |
| `app.py` | Deployable Streamlit interface | Implements the two-tier workflow, SHAP charts, and OpenRouter/Gemini clinical note generation. |
| `image.png` | UI result screenshot | Useful as a figure placeholder, but regenerate after SHAP fix so the figure does not show "SHAP unavailable." |
| `readme.md`, `prompt.md` | Design and system notes | Useful for internal documentation; do not cite these as scientific evidence. |
| `Tier_1_model.pkl`, `Tier_2_model.pkl` | Frozen trained models | Use for reproducibility; report model classes and evaluation metrics. |

## Raman et al. Reference Analysis

Raman et al. (2026), "Machine Learning for Coronary Heart Disease Prediction: Comparative Analysis of Framingham and Cleveland Subset of the UCI Dataset with SHAP-Based Interpretability," was published in *Epidemiologia* on 1 June 2026. The paper used a leakage-controlled ML workflow, compared multiple algorithms on Framingham and the Cleveland subset of UCI Heart Disease, and used SHAP to explain influential features.

Key details from the Raman paper:

- Cleveland subset: 297 records after complete-case filtering, with 237 training samples and 60 held-out test samples.
- Best Cleveland threshold metrics: Logistic Regression accuracy 0.8667, precision 0.8571, recall 0.8571, F1 0.8571, MCC 0.7321, AUROC 0.9475, Brier score 0.2096.
- Best Cleveland AUROC/calibration: KNN AUROC 0.9531 and Brier score 0.0942.
- Cleveland SHAP factors: number of major vessels (`ca`), chest pain type (`cp`), thallium stress-test result (`thal`), and age.
- Major limitation they acknowledged: a single held-out split and only 60 Cleveland test samples after complete-case cleaning, limiting stability and generalizability.

How to use Raman et al. respectfully:

Raman et al. should be cited as evidence that traditional ML plus SHAP can reach strong predictive performance and interpretable feature rankings on coronary heart disease data. The gap is not that Raman et al. were wrong. The gap is that their full-vector workflow does not explicitly optimize the timing or cost of acquiring expensive downstream features, and their explanation layer remains a model-interpretability artifact rather than a workflow-ready clinical note.

## Literature Positioning

| Research area | Reference direction | How it supports this paper |
| --- | --- | --- |
| Cardiovascular importance | WHO cardiovascular disease fact sheet | Establishes clinical burden and motivation. |
| Dataset provenance | UCI Heart Disease repository | Establishes public data provenance and feature vocabulary. |
| Baseline SHAP cardiac ML | Raman et al. 2026 | Main contrast paper: accuracy and SHAP, but no staged acquisition or LLM translation. |
| SHAP theory | Lundberg and Lee 2017 | Justifies additive local feature attribution. |
| Cost-sensitive feature acquisition | Kachuee et al. 2019; Vivar et al. 2020 | Supports the idea that diagnostic features have acquisition costs and should not be assumed free. |
| Clinician-centered explainability | Tonekaboni et al. 2019; Stiglic et al. 2020 | Supports the claim that explanations must fit clinical context and workflow. |
| Medical LLMs | Singhal et al. 2023 and related medical LLM work | Supports careful use of LLMs to translate structured evidence into text, with human oversight. |
| Regulatory framing | FDA Clinical Decision Support guidance | Supports framing this as clinician-reviewed decision support, not autonomous diagnosis. |

# Manuscript Draft

## Title

Smart Clinic Assistant: A Cost-Aware Two-Tier Machine Learning and Generative Explanation Architecture for Heart Disease Triage on Multi-Site UCI Data

## Abstract

Background: Machine learning models for coronary heart disease prediction commonly assume that all diagnostic variables are available before inference, even when those variables require downstream testing or specialist resources. Recent SHAP-based work by Raman et al. demonstrated strong predictive performance and interpretable feature rankings on the Cleveland subset of the UCI Heart Disease dataset, but did not explicitly model feature-acquisition cost or convert attribution outputs into clinician-facing language. Objective: We developed Smart Clinic Assistant, a two-tier decision-support architecture that first screens patients using low-cost intake variables and requests a full diagnostic profile only when the routing policy requires additional evidence. Methods: A 920-record multi-site UCI-style heart disease dataset was split using stratified 80/20 partitioning. Tier 1 used age, sex, chest pain type, and resting blood pressure. Tier 2 used the full encoded diagnostic profile. A Random Forest gatekeeper estimated initial risk, while a soft-voting Tier 2 ensemble produced final risk. SHAP values from the Tier 2 model were forwarded to an OpenRouter-hosted Gemini model to generate a short patient-facing clinical note. Results: On 184 held-out records, the full Tier 2 model achieved 90.2% accuracy and 0.9198 AUROC. The implemented conservative cascade achieved 89.7% accuracy while avoiding 24.5% of Tier 2 profiles. A strict symmetric uncertainty-gate simulation avoided 66.8% of Tier 2 profiles while maintaining 86.4% accuracy. Conclusion: The proposed architecture reframes heart disease ML from a static all-features classifier into a resource-aware triage workflow that links predictive accuracy, test stewardship, and explanation usability.

Keywords: heart disease prediction; clinical decision support; SHAP; feature acquisition cost; medical informatics; generative AI; UCI Heart Disease; triage

## 1. Introduction

Cardiovascular disease remains a leading global cause of mortality, making early risk identification a continuing priority for clinical decision support. Public datasets such as the UCI Heart Disease repository have therefore become common benchmarks for evaluating machine learning approaches to coronary heart disease prediction. These datasets are useful for algorithmic comparison, but they also risk encouraging a simplified assumption: that every clinical feature is available simultaneously and at no operational cost.

Raman et al. recently provided a strong reference point for this field. Their 2026 study compared machine learning models across Framingham and the Cleveland subset of the UCI Heart Disease dataset using a leakage-controlled pipeline and SHAP-based interpretability. On the Cleveland subset, Logistic Regression achieved 0.8667 accuracy and KNN achieved 0.9531 AUROC. SHAP analysis identified clinically plausible predictors, including number of major vessels, chest pain type, thallium stress-test result, and age.

However, high static accuracy does not by itself solve the deployment problem faced by resource-constrained clinics. A full-vector classifier treats low-cost intake information and downstream diagnostic information as equally available. In practice, variables such as major vessels visualized by fluoroscopy or thallium stress-test results are not equivalent to age, sex, chest pain description, or resting blood pressure. They require clinical workflow time, specialized infrastructure, and financial cost. A model that requires these values for every patient may perform well mathematically while remaining inefficient as a first-contact triage tool.

A second limitation is cognitive usability. SHAP improves transparency by assigning local feature contributions, but raw attribution values and beeswarm plots are not automatically actionable during a short clinical encounter. Clinicians require explanations that are concise, context-aware, and tied to the patient in front of them. Therefore, a useful decision-support system should not only compute feature importance, but also translate the main risk-raising and risk-lowering factors into readable clinical language while preserving clinician oversight.

To address these gaps, we propose Smart Clinic Assistant, a two-tier machine learning and generative explanation architecture for heart disease triage. The system first estimates risk from low-cost intake variables. It then uses a threshold-based routing policy to determine whether the full Tier 2 diagnostic profile is required. Finally, it converts local SHAP attributions into a concise clinical note using a large language model configured through an external API. The contribution is not a new classifier alone; it is an operational architecture that joins staged diagnostic acquisition, risk prediction, interpretable attribution, and language-level explanation.

## 2. Methods

### 2.1 Study Design

This was a retrospective machine learning study using a public, de-identified heart disease dataset. No patient interaction occurred, and no identifiable personal data were collected. The current project should be described as a decision-support simulation, not as a deployed medical device or a clinically validated diagnostic system.

The study compared four evaluation modes:

1. Tier 1 only: low-cost intake features only.
2. Tier 2 full-profile model: all processed diagnostic features available for every patient.
3. Implemented conservative cascade: low-risk Tier 1 cases bypass Tier 2; gray-zone and high-risk cases proceed to Tier 2.
4. Strict symmetric uncertainty gate: low-risk and high-risk Tier 1 cases are routed immediately; only gray-zone cases proceed to Tier 2.

The final submission should choose one primary routing policy. The conservative cascade better matches the current app. The symmetric uncertainty gate better matches the strongest cost-saving manuscript claim.

### 2.2 Dataset

The local dataset file `heart_disease_uci.csv` contains 920 records and 16 columns. The source-site distribution is:

| Site | Records |
| --- | ---: |
| Cleveland | 304 |
| Hungary | 293 |
| VA Long Beach | 200 |
| Switzerland | 123 |

The binary target was defined as:

```
y = 1 if target > 0
y = 0 if target == 0
```

The full local dataset contains 509 positive cases and 411 negative cases. The 80/20 stratified split produced 736 training records and 184 held-out test records. The held-out positive rate was 55.4%.

Reviewer warning: because Raman et al. used the Cleveland subset after complete-case filtering, this manuscript should not claim direct dataset identity. The safer framing is: "we extend the Raman-style UCI/SHAP line of work from a full-vector Cleveland subset analysis to a multi-site staged-acquisition triage simulation."

### 2.3 Preprocessing

The preprocessing pipeline is implemented in `Data_Processing_final.py`. The raw clinical variables were mapped, imputed, one-hot encoded, and scaled after a stratified train-test split.

Continuous variables:

- age
- resting blood pressure (`trestbps`)
- cholesterol (`chol`)
- maximum heart rate (`thalch`)
- ST depression (`oldpeak`)

Categorical or binary variables:

- sex
- chest pain type (`cp`)
- fasting blood sugar (`fbs`)
- resting electrocardiogram (`restecg`)
- exercise-induced angina (`exang`)
- slope
- number of major vessels (`ca`)
- thalassemia/stress-test result (`thal`)

Missingness indicators were created for clinically important absent values such as `ca_missing`, `chol_missing_or_zero`, and `oldpeak_missing`. One-hot encoding expanded the final Tier 2 feature matrix to 37 features.

### 2.4 Two-Tier Feature Slicing

The model intentionally separates low-cost intake features from the complete diagnostic profile.

Tier 1 feature set:

```
X1 = {age, sex, chest pain type, resting blood pressure}
```

In encoded form, Tier 1 contains eight columns:

```
age, gender_Female, gender_Male,
cp_asymptomatic, cp_atypical angina, cp_non-anginal, cp_typical angina,
trestbps
```

Tier 2 feature set:

```
X2 = X1 + {chol, fbs, restecg, thalch, exang, oldpeak, slope, ca, thal, missingness indicators}
```

This feature split is the central engineering contribution. It allows the system to ask: "Can the patient be routed using low-cost intake data, or does the model require downstream diagnostic information?"

### 2.5 Model Architecture

Tier 1 model:

- RandomForestClassifier
- `n_estimators = 600`
- `max_depth = 4`
- `random_state = 43`

Tier 2 model:

- Soft VotingClassifier
- Random Forest, `n_estimators = 200`, `max_depth = 12`, `random_state = 7`
- Random Forest, `n_estimators = 1000`, `max_depth = 5`, `random_state = 43`
- Logistic Regression, `max_iter = 5000`
- KNN, `n_neighbors = 7`

Each model returns a posterior probability estimate:

```
p1 = f1(X1)
p2 = f2(X2)
```

where p1 is the Tier 1 estimated probability of heart disease and p2 is the Tier 2 estimated probability after the full diagnostic profile is available.

### 2.6 Routing Policy

The uncertainty gate uses two thresholds:

```
theta_L = 0.30
theta_H = 0.70
```

Strict symmetric uncertainty-gate policy:

```
if p1 <= theta_L:
    route = low risk, no Tier 2 profile requested
elif p1 >= theta_H:
    route = high risk, immediate escalation without waiting for Tier 2 profile
else:
    route = gray zone, request Tier 2 profile
```

Current implemented conservative app policy:

```
if p1 <= theta_L:
    route = low risk, no Tier 2 profile requested
else:
    route = request Tier 2 profile
```

The conservative policy is safer for a demo because high-risk cases receive the full diagnostic panel. The symmetric policy is stronger for the financial-efficiency thesis because both confidently low-risk and confidently high-risk cases bypass the full Tier 2 model.

### 2.7 SHAP-to-Generative Explanation Layer

The SHAP layer converts Tier 2 model behavior into feature-level local contributions. Because the Tier 2 model is a scikit-learn VotingClassifier, the app explains the positive-class `predict_proba` output using a model-agnostic SHAP permutation explainer. The top local factors are sorted by absolute contribution:

```
impact_i = abs(phi_i)
```

The top factors are forwarded to an OpenRouter-hosted Gemini model with instructions to produce one short patient-facing paragraph. The prompt forbids mention of SHAP, algorithms, AI, or machine learning and asks for warm clinical language. This creates a translation layer:

```
raw model attribution -> ranked clinical factors -> concise clinical note
```

The LLM output should be presented as draft decision-support text requiring clinician review. It should not be framed as autonomous medical advice.

### 2.8 Evaluation Metrics

Predictive performance was evaluated using:

- accuracy
- precision
- recall/sensitivity
- specificity
- F1-score
- Matthews correlation coefficient
- AUROC
- Brier score
- confusion matrix

Financial efficiency was evaluated as Tier 2 profile avoidance:

```
FER = (N - N_T2) / N
```

where `N` is the number of evaluated patients and `N_T2` is the number of patients requiring Tier 2 acquisition under the routing policy. If a local institution assigns a monetary cost `C_T2` to a Tier 2 diagnostic profile, estimated savings are:

```
Savings = (N - N_T2) * C_T2
```

For publication, report the financial metric as "Tier 2 profiles avoided" unless validated local dollar costs are available.

## 3. Results

### 3.1 Dataset Split

The 920-record dataset was split into 736 training records and 184 held-out test records. The training positive rate was 55.3%; the held-out positive rate was 55.4%.

### 3.2 Predictive Performance

| Model/policy | Accuracy | Precision | Recall | Specificity | F1 | MCC | AUROC | Brier |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Tier 1 only | 0.7880 | 0.7838 | 0.8529 | 0.7073 | 0.8169 | 0.5692 | 0.8442 | 0.1533 |
| Tier 2 full profile | 0.9022 | 0.8750 | 0.9608 | 0.8293 | 0.9159 | 0.8046 | 0.9198 | 0.1074 |
| Conservative cascade, current app | 0.8967 | 0.8739 | 0.9510 | 0.8293 | 0.9108 | 0.7927 | 0.9222 | 0.1068 |
| Symmetric uncertainty gate | 0.8641 | 0.8291 | 0.9510 | 0.7561 | 0.8858 | 0.7304 | 0.8740 | 0.1238 |

Interpretation:

The full Tier 2 model produced the highest raw accuracy at 90.2%. The conservative cascade retained nearly all Tier 2 performance, dropping only 0.5 percentage points of accuracy while avoiding Tier 2 acquisition for low-risk patients. The stricter symmetric uncertainty gate produced lower accuracy but a much stronger resource-saving profile.

### 3.3 Confusion Matrices

Rows represent true labels and columns represent predicted labels: `[[TN, FP], [FN, TP]]`.

Tier 1 only:

```
[[58, 24],
 [15, 87]]
```

Tier 2 full profile:

```
[[68, 14],
 [ 4, 98]]
```

Conservative cascade, current app:

```
[[68, 14],
 [ 5, 97]]
```

Symmetric uncertainty gate:

```
[[62, 20],
 [ 5, 97]]
```

Figure instruction for manuscript:

Do not paste these matrices as raw code in the final paper. Convert them into a 2x2 heatmap figure with sensitivity, specificity, and false-negative count annotated. Reviewers will focus heavily on false negatives because this is a triage use case.

### 3.4 Routing and Financial Efficiency

Tier 1 threshold routing on the 184 held-out test records:

| Tier 1 route | Count | Percent |
| --- | ---: | ---: |
| Low risk, p1 <= 0.30 | 45 | 24.5% |
| Gray zone, 0.30 < p1 < 0.70 | 61 | 33.2% |
| High risk, p1 >= 0.70 | 78 | 42.4% |

Conservative current app:

```
Tier 2 profiles requested = 139 / 184
Tier 2 profiles avoided = 45 / 184
Financial efficiency rate = 24.5%
```

Strict symmetric uncertainty gate:

```
Tier 2 profiles requested = 61 / 184
Tier 2 profiles avoided = 123 / 184
Financial efficiency rate = 66.8%
```

Recommended paper framing:

The conservative cascade is a safety-first deployment policy. The symmetric uncertainty gate is a cost-first triage policy. The manuscript can report both as an ablation study:

- "Safety-preserving cascade": 89.7% accuracy with 24.5% Tier 2 reduction.
- "Cost-optimized uncertainty gate": 86.4% accuracy with 66.8% Tier 2 reduction.

This is stronger than reporting only one number because it shows an explicit accuracy-cost trade-off.

### 3.5 Explainability and Clinical Note Generation

After the SHAP fix in `app.py`, the Tier 2 explainer returns 37 local feature rows for a patient profile. In one tested profile with 31.7% final Tier 2 risk, the highest local drivers included major-vessel encoding, asymptomatic chest pain, thalassemia/stress-test category, slope, and exercise-induced angina indicators.

The publishable claim should be:

"The system does not expose raw SHAP values as the final clinical artifact. Instead, it uses SHAP internally to identify the highest-impact local factors and forwards those structured factors to a generative model for a concise clinical-note draft."

The paper should include:

- one SHAP bar chart for a patient-level explanation,
- one risk/protective contribution balance chart,
- one example generated clinical note after de-identification and clinician review,
- the exact prompt template in an appendix.

## 4. Discussion

This study extends conventional heart disease machine learning work by reframing prediction as a staged clinical workflow. Raman et al. showed that leakage-controlled ML models with SHAP can reach strong accuracy on the Cleveland subset and produce clinically plausible feature rankings. Our work accepts that foundation but shifts the research question from "Which model is most accurate when all features are present?" to "How much diagnostic information must be acquired before a useful triage decision can be made?"

The results suggest that staged acquisition can preserve much of the accuracy of a full-profile model while reducing downstream feature requirements. Under the conservative policy currently implemented in the app, the cascade achieved 89.7% accuracy and avoided 24.5% of Tier 2 profiles. This policy sacrifices little predictive performance relative to the full Tier 2 model. Under the strict uncertainty-gate simulation, the system avoided 66.8% of Tier 2 profiles and still achieved 86.4% accuracy, approximating the accuracy reported by Raman et al. on the smaller Cleveland subset while using a staged routing design.

The second contribution is explanation delivery. SHAP improves model transparency, but raw attributions are not always workflow-ready. Smart Clinic Assistant uses SHAP as an internal evidence layer and then generates a short plain-language note. This design is aligned with clinician-centered explainability: the output should make the key drivers easier to inspect, not require the clinician to interpret a dense attribution plot during time-limited care.

The study also has important limitations. First, the current dataset is public, retrospective, and heterogeneous across source sites. Second, the evaluation uses a single stratified split; repeated cross-validation, bootstrap confidence intervals, and site-stratified validation are needed. Third, the financial metric is a profile-avoidance proxy, not a verified hospital cost analysis. Fourth, LLM-generated explanations may be fluent but clinically imperfect; all generated notes require clinician review, prompt auditing, and safety evaluation. Fifth, the current app policy and the strongest uncertainty-gate claim differ for high-risk patients, and this must be resolved before submission.

Despite these limitations, the architecture is publishable if positioned carefully. The novelty is not that the model beats every prior classifier. The novelty is the integrated system design: low-cost intake triage, explicit routing thresholds, full-profile escalation only when policy requires it, local explainability, and generative clinical-note translation.

## 5. Conclusion

Smart Clinic Assistant bridges a practical gap between machine learning accuracy and real-world clinical workflow. On a 920-record multi-site UCI-style heart disease dataset, the full-profile Tier 2 model achieved 90.2% accuracy. A conservative two-tier cascade preserved 89.7% accuracy while avoiding 24.5% of Tier 2 profiles, and a stricter uncertainty-gate simulation avoided 66.8% of Tier 2 profiles while maintaining 86.4% accuracy. These results support the feasibility of resource-aware staged triage as an alternative to all-at-once heart disease prediction models. Before clinical use, the system requires external validation, uncertainty calibration, prospective evaluation, and clinician review of generated explanations.

## Figure and Table Package for Submission

The final manuscript should include these items.

| Item | Content | Source |
| --- | --- | --- |
| Figure 1 | Two-tier architecture diagram: Tier 1 intake, gate, Tier 2 escalation, SHAP, LLM note | Draw from `app.py` workflow |
| Figure 2 | Routing distribution bar chart: low, gray, high counts | Metrics script results |
| Figure 3 | System-wide confusion matrix heatmap | Conservative and/or symmetric cascade |
| Figure 4 | Accuracy vs Tier 2 profiles avoided trade-off | Compare Tier 2 full, conservative cascade, symmetric gate |
| Figure 5 | Patient-level SHAP contribution chart | `app.py` SHAP output |
| Figure 6 | Screenshot of final Streamlit dashboard | Regenerate after SHAP fix |
| Table 1 | Dataset and preprocessing summary | `heart_disease_uci.csv`, `Data_Processing_final.py` |
| Table 2 | Model performance metrics | Current held-out test evaluation |
| Table 3 | Comparison against Raman et al. | Raman PDF plus our metrics |

## Recommended Paper Title Variants

1. Cost-Aware Two-Tier Heart Disease Triage Using Staged Feature Acquisition and Generative Explanation
2. From SHAP Attribution to Clinical Notes: A Two-Tier Resource-Aware Architecture for Heart Disease Decision Support
3. Smart Clinic Assistant: A Staged Diagnostic Machine Learning Workflow for Resource-Aware Cardiac Triage

Best option for publication:

"Cost-Aware Two-Tier Heart Disease Triage Using Staged Feature Acquisition and Generative Explanation"

This title is stronger than "Smart Clinic Assistant" alone because it states the scientific contribution directly.

## Target Venue Fit

Potential journal categories:

- medical informatics
- digital health
- explainable AI in healthcare
- clinical decision support systems
- applied machine learning in cardiology

Best-fit paper type:

- retrospective machine learning methods paper,
- decision-support prototype paper,
- workflow-aware XAI paper.

Avoid submitting as:

- clinical validation paper,
- diagnostic accuracy study claiming readiness for deployment,
- generative AI safety paper without clinician evaluation.

## References

[1] World Health Organization. Cardiovascular diseases fact sheet. https://www.who.int/news-room/fact-sheets/detail/cardiovascular-diseases-%28cvds%29

[2] UCI Machine Learning Repository. Heart Disease dataset. https://archive.ics.uci.edu/dataset/45/heart+disease

[3] Raman, S.; Thakkar, D.; Calixte, J.; et al. Machine Learning for Coronary Heart Disease Prediction: Comparative Analysis of Framingham and Cleveland Subset of the UCI Dataset with SHAP-Based Interpretability. *Epidemiologia*. 2026;7:75. DOI: https://doi.org/10.3390/epidemiologia7030075. Local file: `epidemiologia-07-00075.pdf`.

[4] Lundberg, S. M.; Lee, S.-I. A Unified Approach to Interpreting Model Predictions. NeurIPS 2017. https://arxiv.org/abs/1705.07874

[5] Kachuee, M.; Karkkainen, K.; Goldstein, O.; et al. Cost-Sensitive Diagnosis and Learning Leveraging Public Health Data. https://arxiv.org/abs/1902.07102

[6] Vivar, G.; Zwienenberg, J.; Akker, H. van den; et al. Personalized Diagnosis via Cost-Sensitive Bayesian Optimization. https://arxiv.org/abs/2003.14127

[7] Tonekaboni, S.; Joshi, S.; McCradden, M. D.; Goldenberg, A. What Clinicians Want: Contextualizing Explainable Machine Learning for Clinical End Use. MLHC 2019. https://proceedings.mlr.press/v106/tonekaboni19a.html

[8] Stiglic, G.; Kocbek, P.; Fijacko, N.; et al. Interpretability of machine learning-based prediction models in healthcare. *WIREs Data Mining and Knowledge Discovery*. 2020. https://arxiv.org/abs/2002.08596

[9] Singhal, K.; Azizi, S.; Tu, T.; et al. Large Language Models Encode Clinical Knowledge. https://arxiv.org/abs/2212.13138

[10] U.S. Food and Drug Administration. Clinical Decision Support Software Guidance. https://www.fda.gov/regulatory-information/search-fda-guidance-documents/clinical-decision-support-software

[11] Collins, G. S.; Moons, K. G. M.; et al. TRIPOD+AI reporting guidance for prediction models using artificial intelligence. Use as reporting checklist before submission. https://www.tripod-statement.org/

## Appendix A: Exact Current Held-Out Metrics

Dataset:

```
records = 920
train = 736
test = 184
train positive rate = 0.5530
test positive rate = 0.5543
```

Routing counts:

```
low: 45
gray: 61
high: 78
```

Tier 2 profile avoidance:

```
conservative app policy = 45 / 184 = 24.5%
symmetric uncertainty gate = 123 / 184 = 66.8%
```

Primary model results:

```
Tier 1 accuracy = 0.7880
Tier 2 full-profile accuracy = 0.9022
Conservative cascade accuracy = 0.8967
Symmetric uncertainty-gate accuracy = 0.8641
```

## Appendix B: Final Pre-Submission Checklist

- Align `app.py` behavior with the selected primary routing policy.
- Regenerate the UI screenshot after SHAP charts display correctly.
- Add bootstrap confidence intervals for all metrics.
- Add threshold sensitivity analysis for theta_L and theta_H.
- Add calibration plots for Tier 1 and Tier 2.
- Add site-stratified or leave-one-site-out validation.
- Add a clinician-reviewed sample of generated notes.
- Remove and rotate all API keys.
- Move all prompts and model settings into a reproducibility appendix.
- State clearly: "This is clinical decision support only and requires clinician review."
