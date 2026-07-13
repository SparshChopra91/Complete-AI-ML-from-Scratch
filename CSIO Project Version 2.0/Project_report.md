# Smart Clinic Assistant: A Resource-Aware Two-Tier Heart Disease Triage System Using Staged Feature Acquisition and SHAP-Guided Draft Explanation

## Front Matter

### Cover Page

**Council of Scientific and Industrial Research - Central Scientific Instruments Organisation (CSIR-CSIO), Chandigarh**  
**Chandigarh College of Engineering and Technology (CCET), Chandigarh**

**Internship Technical Report**

**Submitted by:** Sparsh Chopra, B.Tech Computer Science and Engineering, CCET Chandigarh  
**Email:** co24365@ccet.ac.in  
**Internship Organisation:** CSIR-CSIO, Sector 30-C, Chandigarh  
**Project Domain:** Clinical decision support, staged feature acquisition, machine learning, explainable AI, Streamlit software engineering  
**Internship Duration:** TODO: Insert officially verified start and end dates.  
**Mentor:** TODO: Insert mentor name and designation from official internship records.  
**Date of Submission:** TODO

### Certificate

This is to certify that the internship technical report entitled **"Smart Clinic Assistant: A Resource-Aware Two-Tier Heart Disease Triage System Using Staged Feature Acquisition and SHAP-Guided Draft Explanation"** is a record of project work carried out by **Sparsh Chopra**, B.Tech Computer Science and Engineering, CCET Chandigarh, during an internship at **CSIR-Central Scientific Instruments Organisation (CSIR-CSIO), Chandigarh**.

The work concerns the design, implementation, and evaluation of a retrospective machine-learning workflow simulation for heart disease triage. It includes leakage-controlled preprocessing, two-tier feature partitioning, Random Forest and ensemble modelling, a Streamlit decision-support interface, SHAP-based attribution, and post-hoc audits using held-out testing, cross-validation, leave-one-site-out testing, calibration, threshold sweep, decision-curve analysis, and subgroup analysis.

Supervisor Name: TODO  
Designation: TODO  
Signature: ______________________  
Date: __________________________

### Declaration

I, **Sparsh Chopra**, declare that this internship report is based on the project repository, source code, dataset, evaluation scripts, research manuscript, and associated documentation available in the workspace. The system described here is a retrospective software prototype and workflow audit. It is not a validated medical device, not a clinically approved diagnostic product, and not an autonomous discharge rule. Any real clinical use would require prospective validation, ethics and governance review, calibrated thresholds, privacy controls, and regulatory assessment.

Where the repository does not verify a detail, a **TODO** marker has been retained instead of inventing information.

Signature: ______________________  
Date: TODO

### Acknowledgement

I thank **CSIR-CSIO, Chandigarh** for the research environment in which this healthcare informatics and software engineering project was developed. I thank **CCET Chandigarh** for academic support. I acknowledge the public UCI Heart Disease dataset and the open-source scientific Python ecosystem, including Python, pandas, NumPy, scikit-learn, joblib, SHAP, Streamlit, and Plotly. I also acknowledge the internal review documents in the repository, which helped reframe the work from a simple classifier into a safety-audited staged workflow simulation.

### Abstract

Machine-learning studies on heart disease prediction often evaluate models as if every diagnostic feature is available at the beginning of patient assessment. This assumption is convenient for benchmarking but does not represent real clinical triage. Basic intake variables such as age, sex, chest pain type, and resting blood pressure are available immediately. Downstream variables such as cholesterol, fasting blood sugar, ECG findings, maximum heart rate, exercise-induced angina, ST depression, slope, major vessel count, and thalassemia or stress-test category require additional clinical workflow, time, equipment, or interpretation.

This internship project developed **Smart Clinic Assistant**, a two-tier retrospective heart disease triage simulation using a 920-record multi-site UCI-style dataset containing Cleveland, Hungary, VA Long Beach, and Switzerland records. The binary target is `target > 0`. The project separates an eight-column encoded **Tier 1 intake matrix** from a 37-column encoded **Tier 2 diagnostic matrix**. The preprocessing pipeline splits data before imputation, encoding, and scaling; treats cholesterol values equal to zero as invalid; applies iterative imputation to continuous variables; applies a non-colliding categorical sentinel value of `-1`; and adds missingness indicators for clinically meaningful absence patterns.

The Tier 1 model is a shallow Random Forest gatekeeper. The Tier 2 model is a soft-voting ensemble combining two Random Forests, Logistic Regression, and KNN. The implemented application policy bypasses Tier 2 only for low-risk Tier 1 cases where `p1 <= 0.30`; gray-zone and high-risk cases proceed to Tier 2. The research manuscript also evaluates an uncertainty-band ablation, but that policy is treated as a cost-first ablation rather than the deployed application workflow.

Under 5-fold cross-validation, the conservative cascade achieved **84.5% +/- 2.2% accuracy** and **0.898 +/- 0.034 AUROC** while avoiding **25.1% +/- 3.3%** of Tier 2 profiles. The full Tier 2 model achieved **84.3% +/- 1.8% accuracy** and **0.904 +/- 0.035 AUROC**. Leave-one-site-out validation reduced pooled cascade accuracy to **78.0%**, and the held-out bypass audit showed that the 0.30 low-risk threshold bypassed 4 positive cases among 45 bypassed patients. A stricter 0.15 threshold avoided 9.8% of Tier 2 profiles with zero bypassed positives on the held-out split. These results support staged acquisition as a useful workflow-auditing framework but do not support clinical deployment.

### Table of Contents

1. Front Matter  
2. Chapter 1: Organization Profile  
3. Chapter 2: Internship Overview  
4. Chapter 3: Problem Statement  
5. Chapter 4: Literature Survey  
6. Chapter 5: Requirement Analysis  
7. Chapter 6: Dataset Description  
8. Chapter 7: Methodology  
9. Chapter 8: Implementation  
10. Chapter 9: Results and Analysis  
11. Chapter 10: Conclusion and Future Scope  
12. References  
13. Appendices

### List of Figures

<!-- Figure:
System Architecture
Source:
overleaf.md / Final Version reseacrh Paper.pdf / app.py workflow
-->

<!-- Figure:
Preprocessing Pipeline
Source:
Data_Processing_final.py
-->

<!-- Figure:
Streamlit Tier 1 Intake and Routing Dashboard
Source:
TODO: Capture from app.py after running Streamlit.
-->

<!-- Figure:
Tier 2 SHAP Explanation Dashboard
Source:
TODO: Capture from app.py after running Streamlit.
-->

<!-- Figure:
Decision Curve Analysis
Source:
publication_evaluation.py / final research paper
-->

### List of Tables

| Table No. | Title |
|---:|---|
| 1.1 | CSIR and CSIR-CSIO institutional summary |
| 2.1 | Six-week internship timeline |
| 4.1 | Literature comparison matrix |
| 5.1 | Functional requirements |
| 6.1 | Dataset columns and descriptions |
| 6.2 | Site distribution and prevalence |
| 7.1 | Tier feature partition |
| 8.1 | File-by-file implementation summary |
| 9.1 | Cross-validation results |
| 9.2 | Held-out performance |
| 9.3 | Threshold sweep |
| 9.4 | LOSO validation |

### Abbreviations

| Abbreviation | Meaning |
|---|---|
| AI | Artificial Intelligence |
| AUROC | Area Under Receiver Operating Characteristic Curve |
| CDSS | Clinical Decision Support System |
| CSIO | Central Scientific Instruments Organisation |
| CSIR | Council of Scientific and Industrial Research |
| DCA | Decision Curve Analysis |
| ECE | Expected Calibration Error |
| LOSO | Leave-One-Site-Out |
| MCC | Matthews Correlation Coefficient |
| MICE | Multivariate Imputation by Chained Equations |
| RF | Random Forest |
| SHAP | SHapley Additive exPlanations |
| UCI | University of California Irvine Machine Learning Repository |

---

## Chapter 1: Organization Profile

### 1.1 History of CSIR

The **Council of Scientific and Industrial Research (CSIR)** is one of India's major publicly funded research and development organizations. It functions under the Ministry of Science and Technology, Government of India, and maintains a national network of laboratories, outreach centres, innovation complexes, and units. The official CSIR profile describes CSIR as a contemporary R&D organization with a broad science and technology base covering areas such as oceanography, geophysics, chemicals, drugs, genomics, biotechnology, nanotechnology, mining, aeronautics, instrumentation, environmental engineering, and information technology [37].

CSIR was established in 1942 to strengthen India's scientific and industrial research capability. Since its inception, it has connected scientific knowledge with national development needs. CSIR's laboratories have contributed to healthcare, materials, aerospace, agriculture, environmental systems, chemicals, energy, food, instrumentation, and industrial processes. This broader mission is relevant to Smart Clinic Assistant because the project treats clinical artificial intelligence not only as a statistical model but also as a workflow technology that must fit resource-constrained operating conditions.

### 1.2 History of CSIR-CSIO

**CSIR-Central Scientific Instruments Organisation (CSIR-CSIO)** is a constituent laboratory of CSIR located in Sector 30-C, Chandigarh. The CSIR network map lists CSIR-CSIO as the Central Scientific Instruments Organisation under the physical sciences cluster [39]. Public institutional descriptions state that CSIO was established in 1959 and later moved to Chandigarh in 1962, with its research identity focused on scientific and industrial instrumentation [40]. Because the official CSIO website could not be reliably accessed during report preparation, details specific to CSIO history should be verified from official CSIR-CSIO documents before final institutional submission.

CSIR-CSIO's domain is aligned with engineering practice. Instrumentation research integrates sensing, electronics, mechanical design, software, optics, signal processing, calibration, human factors, manufacturing, and field validation. Smart Clinic Assistant is not a physical medical device, but it follows an instrumentation mindset: it treats a clinical workflow as a measurable process, separates input channels by acquisition cost and availability, designs a decision pipeline, and evaluates safety, reliability, and operational limitations.

### 1.3 Vision and Mission

The official CSIR Vision is to enhance the quality of life of Indian citizens through innovative science and technology, globally competitive R&D, sustainable solutions, and capacity building aligned with Atmanirbhar Bharat [38]. CSIR's mission includes technology innovation, translational research, commercialization, national-goal alignment, health outcomes through biology, chemistry, engineering and computation, capacity building, and high-value services to industry and society [38].

The internship project aligns with this mission in a limited but concrete way. It explores whether a clinical decision-support workflow can reduce unnecessary downstream diagnostic feature acquisition while preserving much of the predictive performance of a full-profile model. It does not claim to solve clinical triage, but it demonstrates how research software can be structured to evaluate trade-offs between resource use and safety.

### 1.4 Research Areas, Healthcare Research, and AI Research

CSIR-CSIO's institutional identity is centred on scientific and industrial instruments. Public summaries describe areas such as optics and optoelectronics, biomedical instrumentation, analytical instrumentation, computational instrumentation, advanced materials and sensors, precision mechanical systems, agrionics, MEMS, and environmental or strategic instrumentation [39], [40]. The present project falls under computational instrumentation, biomedical instrumentation, and healthcare decision-support software.

Healthcare research in instrumentation organizations includes diagnostic devices, biomedical sensors, assistive technologies, imaging systems, laboratory automation, signal processing, and clinical decision support. The project is computational healthcare research: it asks whether patient triage can be represented as staged information acquisition instead of a static classification event.

Artificial intelligence research in healthcare must be judged by more than model accuracy. It must be assessed for calibration, robustness, explainability, dataset shift, user workflow, and failure modes. This project reflects that broader view by including cross-validation, leave-one-site-out validation, threshold sensitivity, bypass accounting, calibration diagnostics, decision-curve analysis, and subgroup audits.

### 1.5 Organizational Structure, Facilities, and Internship Programme

CSIR operates through a national network of laboratories and units. Official CSIR pages identify 37 national laboratories, 39 outreach centres, one innovation complex, and three units [37]. CSIR-CSIO is listed under the physical sciences cluster [39]. Exact current division names, leadership hierarchy, and mentor placement should be verified from official CSIR-CSIO documents before final submission.

For this internship project, the facilities relevant to the work are primarily computational: Python development environment, dataset storage, source-code management, package dependencies, local Streamlit runtime, and evaluation scripts. The repository does not provide evidence of specific CSIR-CSIO hardware allocations, GPU resources, or internal servers, so this report does not claim their use.

The internship programme provided an opportunity to translate classroom knowledge in machine learning, software engineering, data preprocessing, and user-interface development into applied research. The work required independent study of the dataset, review of literature, implementation of pipelines, development of a Streamlit workflow interface, and preparation of a research manuscript.

### 1.6 Role in Scientific Research and Healthcare Technologies

CSIR-CSIO supports India's capability in scientific and industrial instrumentation. Instrumentation research converts scientific principles into usable systems that measure, guide, automate, or assist real-world processes. This project follows that translational logic in software by converting a clinical ML benchmark into a workflow simulation that can be audited for operational behaviour.

| Outcome | Contribution |
|---|---|
| Workflow modelling | Separates intake variables from downstream diagnostic variables |
| Engineering implementation | Builds preprocessing, model training/loading, inference, UI, and explanation components |
| Safety audit | Reports bypassed positives and threshold sensitivity |
| Generalization audit | Compares random split performance with LOSO site-held-out performance |
| Interpretability | Provides local SHAP factor ranking for Tier 2 predictions |
| Responsible reporting | States that clinical deployment is not supported |

---

## Chapter 2: Internship Overview

### 2.1 Internship Objectives

The internship objective was to develop a professional, research-oriented software prototype for heart disease triage that moves beyond a conventional single-stage classifier. The project aimed to investigate whether a staged feature-acquisition workflow can reduce downstream diagnostic feature acquisition while maintaining useful predictive performance under explicit safety audits.

The technical objectives were to study the UCI Heart Disease dataset, implement leakage-controlled preprocessing, divide the feature space into Tier 1 and Tier 2, train or load separate models, build a Streamlit workflow interface, integrate SHAP explanations, include optional draft clinical note generation, evaluate the system using robust audits, and prepare a professional manuscript and report.

### 2.2 Duration

The exact official internship duration is not present in a verified administrative file in the repository. Some older draft text mentions a six-week internship and dates around June-July 2026, but this should not be treated as authoritative unless confirmed by CSIR-CSIO and CCET records.

**TODO:** Insert official start date, end date, attendance period, division, and mentor details after verification.

### 2.3 Research Goals

The research goal was not simply to maximize accuracy on a public dataset. Many previous studies have already evaluated heart disease classifiers under full-feature assumptions. The more meaningful research goal was to evaluate a resource-aware workflow: can a model using low-cost intake features avoid a meaningful fraction of downstream diagnostic profiles while preserving most of the predictive performance of a full-profile model, and what safety risks appear when this routing policy is audited?

| Research Question | Evaluation Method |
|---|---|
| Does Tier 1 alone provide useful discrimination? | Tier 1 held-out and CV metrics |
| Does Tier 2 improve performance? | Tier 2 full-profile held-out and CV metrics |
| Can a cascade preserve performance while avoiding Tier 2? | Conservative cascade performance and avoidance rate |
| Is the bypass branch safe? | Bypass positive count, missed positives, NPV |
| Are thresholds stable? | Low-threshold sweep |
| Does performance generalize across sites? | Leave-one-site-out validation |
| Are probabilities suitable for thresholds? | Calibration audit and reliability bins |
| Does the workflow have clinical utility? | Decision-curve analysis |
| Are subgroup patterns visible? | Sex and age-band subgroup audit |

### 2.4 Weekly Timeline

| Week | Main Work | Deliverables |
|---:|---|---|
| Week 1 | Domain understanding and dataset exploration | Dataset statistics, target definition, site distribution |
| Week 2 | Preprocessing pipeline | `Data_Processing_final.py`, Tier 1/Tier 2 matrices |
| Week 3 | Model architecture and training | `Model_handler_final.py`, model builders, joblib persistence logic |
| Week 4 | Streamlit application | `app.py`, intake workflow, routing, Tier 2 activation |
| Week 5 | Evaluation and manuscript | `publication_evaluation.py`, `overleaf.md`, final paper PDF |
| Week 6 | Review and report preparation | Reconciled critique documents and this internship report |

### 2.5 Tasks Completed

| Task | Status | Evidence |
|---|---|---|
| Dataset inspection | Completed | `heart_disease_uci.csv`, dataset statistics |
| Preprocessing pipeline | Completed | `Data_Processing_final.py` |
| Tier 1/Tier 2 feature split | Completed | `Data_Processing_final.py` |
| Model builders and persistence | Completed | `Model_handler_final.py` |
| Streamlit interface | Completed | `app.py` |
| SHAP explanation integration | Completed in app; older standalone prototype is outdated | `app.py`, `Ai_explainer_shap.py` |
| Optional LLM note generation | Implemented as unvalidated prototype | `app.py` |
| Publication evaluation script | Completed | `publication_evaluation.py` |
| Research manuscript | Completed | `overleaf.md`, final PDF |
| Internship report | Completed in this file | `Project_Report.md` |

### 2.6 Skills Learned

The internship strengthened skills in clinical AI problem framing, data preprocessing, leakage prevention, missing-data handling, scikit-learn modelling, model persistence, cross-validation, calibration, decision-curve analysis, leave-one-site-out validation, Streamlit design, Plotly visualization, SHAP interpretability, prompt design, limitations writing, and professional report writing.

### 2.7 Professional Experience and Mentorship

The project provided experience in moving from a working prototype to a defensible engineering report. A major learning outcome was that a high held-out accuracy score is not sufficient for healthcare AI. The project initially appeared successful when the conservative cascade achieved strong held-out performance. However, deeper analysis showed that the low-risk bypass threshold missed positive cases and that site-held-out validation reduced performance substantially. This changed the professional interpretation of the project from high-performing clinical tool to useful workflow simulation with important safety warnings.

The repository does not verify mentor names or official review meetings. **TODO:** Insert mentor names, division, meeting schedule, and reviewed deliverables after verification from CSIR-CSIO internship records.

### 2.8 Learning Outcomes

The major learning outcomes were that clinical ML must be evaluated in relation to workflow, feature availability matters, missingness can carry workflow information, probability thresholds require calibration, staged systems must report bypass safety, site-held-out validation is more realistic than random splitting, SHAP does not prove clinical usefulness, and LLM-generated clinical text requires formal safety evaluation before use.

---

## Chapter 3: Problem Statement

### 3.1 Clinical Motivation

Heart disease prediction has been a common machine-learning benchmark for decades. Public datasets such as UCI Heart Disease are widely used to compare classifiers, feature-selection methods, and interpretability tools. However, most benchmark workflows assume that all variables are available at the same time. In clinical triage, this assumption is weak.

When a patient first presents to a clinic or emergency department, some variables are immediately available. Age, sex, chest pain description, and resting blood pressure can be collected during intake. Other variables require diagnostic workflow. Cholesterol and fasting blood sugar require laboratory testing. Resting ECG requires equipment and interpretation. Maximum heart rate and exercise-induced angina are linked to stress-test context. Major vessel counts and thalassemia or stress-test categories are downstream diagnostic features. Treating these variables as equally available ignores the cost and timing of clinical information.

### 3.2 Existing Limitations

The repository and final research paper identify several limitations in standard heart disease ML studies: all-features-available assumption, limited workflow modelling, insufficient safety accounting, random split optimism, weak calibration reporting, interpretability gaps, and deployment overclaiming. These limitations make static full-feature accuracy insufficient for triage-oriented decision support.

### 3.3 Workflow Problem

A first-contact triage system should not require downstream diagnostic variables for every patient before producing any useful decision-support signal. It should first use readily available intake variables, decide whether additional information is needed, and then escalate only when the routing policy requires it.

### 3.4 Research Gap

The research gap is not that Random Forests or voting ensembles are new. The gap is the integrated audit of a staged cardiovascular workflow: explicit Tier 1/Tier 2 feature separation, profile avoidance measurement, bypass safety accounting, threshold sweep, calibration analysis, decision-curve analysis, leave-one-site-out validation, subgroup transparency, and SHAP-based traceability in an interactive interface.

### 3.5 Engineering Problem

The engineering problem is to build a software system that faithfully implements the staged workflow. This involves reproducible preprocessing, aligned feature matrices, separate models, model persistence, one-row UI inference, routing, explanation, graceful failure handling, and responsible communication of clinical limitations.

### 3.6 Project Objectives, Scope, and Limitations

The project objectives are to design a staged clinical decision-support simulation, build a leakage-controlled preprocessing pipeline, train Tier 1 and Tier 2 models, implement conservative cascade routing, evaluate performance against baselines, audit resource-safety trade-offs, build a usable Streamlit prototype, integrate local explanation and draft communication support, and prepare a professional manuscript and report.

The scope includes public retrospective data, binary prediction, staged feature acquisition, scikit-learn models, Streamlit UI, SHAP explanation, optional LLM draft note generation, and post-hoc research evaluation. It excludes prospective recruitment, hospital EMR integration, real-time clinical deployment, regulatory approval, physician usability study, clinician-scored LLM evaluation, real monetary cost modelling, and causal inference.

---

## Chapter 4: Literature Survey

### 4.1 Overview

The literature relevant to this project spans cardiovascular prediction, clinical AI validation, ensemble learning, staged feature acquisition, selective prediction, learning to defer, budgeted learning, SHAP interpretability, calibration, decision-curve analysis, dataset shift, missingness, and clinical decision support. The project is interdisciplinary: it combines applied machine learning with workflow modelling and software engineering.

### 4.2 Heart Disease Prediction

Heart disease prediction using clinical variables has a long history. The UCI Heart Disease dataset originates from earlier work on coronary artery disease diagnosis and has become a standard benchmark [1], [2]. Many studies use the Cleveland subset because it is cleaner and more commonly cited. The present project uses a 920-record multi-site file containing Cleveland, Hungary, VA Long Beach, and Switzerland records. This broader file supports site-shift analysis but also introduces strong heterogeneity.

Recent cardiovascular ML papers have shown that machine-learning models can achieve strong discrimination when complete feature sets are available [6], [7], [8]. However, this full-feature setting is not equivalent to early triage. A model requiring thallium/stress-test and vessel-count variables is closer to a downstream diagnostic classifier than an intake triage tool.

### 4.3 Clinical AI and Evaluation

Clinical AI literature emphasizes that model performance must be interpreted in context [3], [4], [5]. A model with high AUROC may still fail if it is poorly calibrated, not externally validated, difficult to integrate into workflow, or unsafe under distribution shift. Prediction-model reporting guidelines such as TRIPOD and TRIPOD+AI encourage transparent reporting of study design, predictors, outcomes, validation, and limitations [11], [12]. PROBAST emphasizes risk of bias and applicability [19].

The Smart Clinic Assistant follows these principles partially. It reports the data source, feature sets, preprocessing, model architecture, evaluation splits, and limitations. It does not claim formal checklist compliance or clinical validation.

### 4.4 Ensemble Learning

Ensemble learning combines multiple models to improve robustness or predictive performance. Random Forests average many decision trees and can handle nonlinear feature interactions [32]. Voting ensembles combine predictions from different estimators. The Tier 2 model combines two Random Forests, Logistic Regression, and KNN with soft voting. The chosen ensemble is adequate for a prototype and was fixed before final audit, but it is not proof of algorithmic superiority.

### 4.5 Feature Acquisition, Selective Prediction, and Deferral

Feature acquisition concerns the decision of which variables to collect before prediction. In healthcare, variables differ in cost, invasiveness, time, and availability. Budgeted feature acquisition and cost-sensitive learning attempt to optimize predictive performance under feature-cost constraints [23], [24]. Selective prediction allows a model to abstain when uncertain [26]-[28]. Learning-to-defer frameworks route cases to another decision-maker or expert system [29], [30].

Smart Clinic Assistant is related to all three ideas but remains simpler. It uses a fixed clinically interpretable split rather than a learned acquisition policy. Tier 1 uses intake variables, while Tier 2 uses downstream diagnostic variables. The implemented conservative gate escalates all non-low-risk cases to Tier 2.

### 4.6 SHAP, Calibration, Decision Curves, and Workflow Auditing

SHAP provides additive local feature attributions based on Shapley values [33]. It is useful for ranking local factors that increase or decrease a model's predicted risk, but it does not prove causality, fairness, clinical actionability, or clinician usefulness [34]. Calibration is important because routing thresholds depend on probability estimates [16], [17]. Decision-curve analysis evaluates net benefit across threshold probabilities and is more clinically relevant than accuracy alone when model output guides action [18].

Workflow auditing means evaluating the system in the context of the operational process. For this project, workflow auditing includes how many patients bypass Tier 2, how many bypassed patients are actually positive, how profile avoidance changes with threshold, how performance changes under site shift, whether probabilities are calibrated enough for routing, and whether the workflow has decision-curve net benefit.

### 4.7 Literature Comparison Matrix

| Area | Representative Work | Relevance | Difference from This Project |
|---|---|---|---|
| UCI heart disease | Janosi et al. [1] | Dataset provenance | This project evaluates staged workflow |
| CAD probability | Detrano et al. [2] | Historical diagnostic context | Not a staged ML app |
| ML in medicine | Rajkomar et al. [3] | Clinical AI framing | Broad review |
| CVD ML | Weng et al. [6] | Routine clinical prediction | Full-feature focus |
| Multicentre CAD ML | Motwani et al. [7] | Cardiac ML relevance | Different dataset and outcome |
| UCI/Framingham SHAP | Raman et al. [8] | Strong comparison paper | Cleveland subset/full-vector analysis |
| XGBoost | Chen and Guestrin [9] | Future baseline | Not implemented |
| Prediction metrics | Steyerberg et al. [10] | Evaluation framework | Project applies multiple metrics |
| Dataset shift | Finlayson et al. [14] | Explains LOSO concerns | Project demonstrates site shift |
| Calibration | Van Calster et al. [17] | Threshold reliability | Project audits calibration |
| DCA | Vickers and Elkin [18] | Net benefit | Project includes DCA |
| Budgeted RF | Nan et al. [23] | Feature budget concept | Project uses fixed split |
| Cost-sensitive diagnosis | Kachuee et al. [24] | Medical cost framing | No explicit cost function |
| Cascades | Viola and Jones [25] | Cascaded design | Different domain |
| Reject option | Chow [26] | Selective prediction | Project escalates to Tier 2 |
| Learning to defer | Madras et al. [29] | Deferral theory | Not learned |
| SHAP | Lundberg and Lee [33] | Local explanation | Traceability only |
| Clinician XAI needs | Tonekaboni et al. [34] | Human-centred explanation | LLM note unvalidated |
| Bias in algorithms | Obermeyer et al. [35] | Fairness caution | Subgroup audit underpowered |

### 4.8 Literature Gap Summary

The project fills a practical gap between static full-feature heart disease prediction and workflow-aware clinical AI auditing. It does not introduce a new classifier, but it combines existing methods into a staged acquisition simulation and evaluates safety-relevant trade-offs. The most important gap addressed is explicit bypass safety accounting and site-held-out stress testing for staged cardiovascular workflows.

---

## Chapter 5: Requirement Analysis

### 5.1 Functional Requirements

| ID | Functional Requirement | Implementation Evidence |
|---|---|---|
| FR1 | Load the UCI heart disease CSV file | `Data_Processing_final.py`, `app.py` |
| FR2 | Convert raw target into binary target | `Y = (target > 0).astype(int)` |
| FR3 | Split data before preprocessing | `train_test_split(..., stratify=Y, random_state=43)` |
| FR4 | Handle invalid cholesterol values | `chol == 0` replaced with `np.nan` |
| FR5 | Add missingness indicators | `ca_missing`, `chol_missing_or_zero`, `oldpeak_missing` |
| FR6 | Impute continuous features | `IterativeImputer(random_state=43, max_iter=15)` |
| FR7 | Impute categorical features | `SimpleImputer(strategy='constant', fill_value=-1)` |
| FR8 | One-hot encode categorical features | `OneHotEncoder(sparse_output=False, handle_unknown='ignore')` |
| FR9 | Scale continuous features | `MinMaxScaler` |
| FR10 | Create Tier 1 and Tier 2 matrices | `x_train_tier1`, `x_train_tier2` |
| FR11 | Train or load models | `Model_handler_final.py` |
| FR12 | Accept Tier 1 inputs in UI | `app.py` |
| FR13 | Route based on Tier 1 probability | `route_for_probability()` |
| FR14 | Activate Tier 2 for gray/high cases | Streamlit session state |
| FR15 | Compute local explanations | `shap_values_for()` |
| FR16 | Generate fallback or API-based note | `local_clinical_note()`, `ai_clinical_note()` |
| FR17 | Generate evaluation outputs | `publication_evaluation.py` |

### 5.2 Non-Functional Requirements

| ID | Requirement | Project Handling |
|---|---|---|
| NFR1 | Reproducibility | Fixed random states and scripted evaluation |
| NFR2 | Leakage control | Split before transformations |
| NFR3 | Interpretability | SHAP local factor ranking |
| NFR4 | Safety transparency | Bypass positives and missed positives reported |
| NFR5 | Usability | Streamlit UI with staged workflow |
| NFR6 | Maintainability | Separate preprocessing, model, evaluation, and app files |
| NFR7 | Graceful degradation | Optional Plotly, SHAP, OpenAI imports handled |
| NFR8 | Privacy caution | API key read from environment variable |
| NFR9 | Clinical caution | App states decision support only |
| NFR10 | Portability | Project-relative paths in final modules |

### 5.3 Hardware Requirements

The repository does not specify hardware requirements. The implementation can run on a standard laptop or desktop. A practical minimum is a dual-core CPU, 8 GB RAM, 1 GB of free storage for the repository and package cache, and no GPU. Internet access is required only for installing dependencies or using the optional LLM API.

### 5.4 Software Requirements

| Software | Purpose |
|---|---|
| Python 3.10+ | Runtime |
| pandas | Data manipulation |
| NumPy | Numeric operations |
| scikit-learn | Preprocessing, models, metrics |
| joblib | Model persistence |
| Streamlit | Web application |
| SHAP | Local explanations |
| Plotly | Interactive charts used by `app.py` if installed |
| OpenAI-compatible SDK | OpenRouter/Gemini API call |
| google-genai | Listed dependency; not central in final app |
| fpdf | Listed for future PDF generation |

### 5.5 Problem Definition, Scope, Constraints, and Assumptions

Given a patient represented by clinical features, predict whether heart disease is present while minimizing unnecessary downstream diagnostic profile acquisition. The system first operates on Tier 1 intake features and requests Tier 2 only when the routing policy requires it.

The scope is retrospective workflow simulation and software prototype development. Constraints include public retrospective data, old records, heterogeneous sites, missing values, no actual cost data, heuristic thresholds, no clinician LLM audit, and absent model artifacts in the current root listing. The main assumptions are that the local CSV is the project dataset, the binary target is `target > 0`, Tier 1 variables are lower-cost and earlier in the workflow, and the final research paper is the highest-priority technical source.

---

## Chapter 6: Dataset Description

### 6.1 Dataset Source

The dataset used in the repository is `heart_disease_uci.csv`. It is a UCI-style heart disease dataset containing 920 records and 16 columns. The final research paper identifies it as a heterogeneous multi-site UCI Heart Disease dataset rather than the Cleveland-only subset.

### 6.2 Institution Distribution

| Site | Records | Positive Cases | Prevalence |
|---|---:|---:|---:|
| Cleveland | 304 | 139 | 0.457 |
| Hungary | 293 | 106 | 0.362 |
| VA Long Beach | 200 | 149 | 0.745 |
| Switzerland | 123 | 115 | 0.935 |
| **Total** | **920** | **509** | **0.553** |

The site distribution is important because prevalence varies substantially. Switzerland has very high positive prevalence, while Hungary has much lower prevalence. This makes random splits optimistic and motivates leave-one-site-out validation.

### 6.3 Features and Target Variable

| Column | Description |
|---|---|
| `id` | Patient record identifier |
| `age` | Age |
| `gender` | Biological sex as recorded in dataset |
| `dataset` | Source site |
| `cp` | Chest pain type |
| `trestbps` | Resting blood pressure |
| `chol` | Serum cholesterol |
| `fbs` | Fasting blood sugar flag |
| `restecg` | Resting electrocardiographic result |
| `thalch` | Maximum heart rate achieved |
| `exang` | Exercise-induced angina |
| `oldpeak` | ST depression induced by exercise relative to rest |
| `slope` | Slope of peak exercise ST segment |
| `ca` | Number of major vessels |
| `thal` | Thalassemia / stress-test category |
| `target` | Raw disease target |

Raw target counts are 411 records with target 0, 265 with target 1, 109 with target 2, 107 with target 3, and 28 with target 4. The binary target is `0` when `target == 0` and `1` when `target > 0`, producing 411 negative and 509 positive records.

### 6.4 Feature Description

Tier 1 features are age, sex/gender, chest pain type, and resting blood pressure. These variables are treated as intake variables because they can normally be collected early in triage.

Tier 2 features include cholesterol, fasting blood sugar, resting ECG, maximum heart rate, exercise-induced angina, ST depression, slope, major vessel count, thalassemia category, and missingness indicators. These variables represent downstream diagnostic information or additional clinical testing.

### 6.5 Missing Values and Invalid Values

| Column | Missing Count |
|---|---:|
| `trestbps` | 59 |
| `chol` | 30 |
| `fbs` | 90 |
| `restecg` | 2 |
| `thalch` | 55 |
| `exang` | 55 |
| `oldpeak` | 62 |
| `slope` | 309 |
| `ca` | 611 |
| `thal` | 486 |

Additionally, `chol` contains 172 zero values, which are treated as invalid and replaced with missing values before imputation.

### 6.6 Class Distribution and Site Distribution

| Class | Count | Percent |
|---|---:|---:|
| Negative | 411 | 44.7% |
| Positive | 509 | 55.3% |

| Site | Records | Dataset Percent |
|---|---:|---:|
| Cleveland | 304 | 33.0% |
| Hungary | 293 | 31.8% |
| VA Long Beach | 200 | 21.7% |
| Switzerland | 123 | 13.4% |

### 6.7 Data Dictionary and Summary

| Variable | Type in Raw Data | Processing |
|---|---|---|
| `age` | Numeric | MICE imputation if needed, min-max scaling |
| `gender` | Categorical | One-hot encoding |
| `cp` | Categorical | One-hot encoding |
| `trestbps` | Numeric | MICE imputation, min-max scaling |
| `chol` | Numeric | Zero to missing, MICE imputation, min-max scaling |
| `fbs` | Boolean/categorical | Map to 0/1, sentinel impute, one-hot |
| `restecg` | Categorical | Map to integer, sentinel impute, one-hot |
| `thalch` | Numeric | MICE imputation, min-max scaling |
| `exang` | Boolean/categorical | Map to 0/1, sentinel impute, one-hot |
| `oldpeak` | Numeric | MICE imputation, min-max scaling |
| `slope` | Categorical | Map to integer, sentinel impute, one-hot |
| `ca` | Numeric/categorical | Sentinel impute, one-hot |
| `thal` | Categorical | Map to integer, sentinel impute, one-hot |
| `target` | Integer | Binarized |

| Item | Value |
|---|---:|
| Raw records | 920 |
| Raw columns | 16 |
| Predictor columns used | 13 |
| Training records | 736 |
| Held-out records | 184 |
| Tier 1 encoded columns | 8 |
| Tier 2 encoded columns | 37 |
| Positive cases | 509 |
| Negative cases | 411 |

---

## Chapter 7: Methodology

### 7.1 Study Design

The study is a retrospective machine-learning workflow simulation. It uses a public, de-identified dataset and does not involve patient contact, intervention, or clinical deployment. The design compares four predictive modes: Tier 1 only, Tier 2 full profile, conservative cascade, and uncertainty-band policy. The conservative cascade is the implemented application policy. The uncertainty-band policy is a cost-first ablation.

### 7.2 Workflow and Pipeline

The workflow begins with intake variables. A Tier 1 model estimates an initial probability. If the probability is low enough, the conservative policy bypasses Tier 2 in the simulation. Otherwise, the system requests the Tier 2 diagnostic profile and uses the full model.

<!-- Figure:
Two-Tier Workflow
Source:
Derived from app.py route_for_probability and overleaf.md architecture figure.
-->

```text
Patient intake -> Tier 1 vector -> Tier 1 Random Forest probability p1
    -> if p1 <= 0.30: low-risk review, Tier 2 avoided in simulation
    -> if p1 > 0.30: Tier 2 diagnostic panel -> Tier 2 ensemble probability p2
        -> final decision-support output -> SHAP local attribution -> optional draft note
```

The preprocessing pipeline loads selected columns, defines the binary target, performs stratified train-test split, maps categorical variables, replaces invalid cholesterol zeros with missing values, adds missingness indicators, one-hot encodes chest pain and gender, imputes categorical columns with `-1`, imputes continuous columns with `IterativeImputer`, one-hot encodes remaining categoricals, scales continuous variables, slices Tier 1, and assigns Tier 2 as the full encoded matrix.

### 7.3 Preprocessing, Feature Engineering, and Missing Data Handling

The preprocessing design addresses data leakage, invalid values, and informative missingness. Train-test split is performed before transformations. Cholesterol zero values are treated as invalid. Missingness indicators are retained for `ca`, cholesterol missing/zero, and `oldpeak`. Continuous variables (`age`, `trestbps`, `chol`, `thalch`, `oldpeak`) use iterative imputation. Categorical variables (`fbs`, `restecg`, `exang`, `slope`, `thal`, `ca`) use sentinel imputation with `-1` before one-hot encoding.

The final Tier 2 matrix has 37 columns:

```text
age, trestbps, chol, thalch, oldpeak,
ca_missing, chol_missing_or_zero, oldpeak_missing,
cp_asymptomatic, cp_atypical angina, cp_non-anginal, cp_typical angina,
gender_Female, gender_Male,
ca_-1.0, ca_0.0, ca_1.0, ca_2.0, ca_3.0,
restecg_-1.0, restecg_0.0, restecg_1.0, restecg_2.0,
fbs_-1.0, fbs_0.0, fbs_1.0,
exang_-1.0, exang_0.0, exang_1.0,
slope_-1.0, slope_0.0, slope_1.0, slope_2.0,
thal_-1.0, thal_0.0, thal_1.0, thal_2.0
```

### 7.4 Tier Partition and Architecture

| Tier | Raw Variables | Encoded Columns | Purpose |
|---|---|---:|---|
| Tier 1 | age, gender, cp, trestbps | 8 | Intake gatekeeper |
| Tier 2 | all predictors plus missingness indicators | 37 | Full diagnostic profile |

<!-- Figure:
System Architecture
Source:
app.py, Data_Processing_final.py, Model_handler_final.py, publication_evaluation.py
-->

The architecture contains data loading, preprocessing, Tier 1 model, routing layer, Tier 2 model, explanation layer, communication layer, UI layer, and evaluation layer.

### 7.5 Random Forest and Voting Ensemble

The Tier 1 model uses a Random Forest with limited depth. A shallow depth helps reduce overfitting and produces smoother probability estimates for the simple intake feature set. The Tier 2 model is a `VotingClassifier` with `voting="soft"`. It combines two Random Forests, Logistic Regression, and KNN. Soft voting averages class probabilities rather than hard labels.

### 7.6 Routing Policy and Threshold Logic

Thresholds are `LOW_THRESHOLD = 0.30` and `HIGH_THRESHOLD = 0.70`. The implemented conservative app policy bypasses Tier 2 only if `p1 <= 0.30`; otherwise Tier 2 is requested. The uncertainty-band ablation bypasses low-risk and high-risk cases and requests Tier 2 only for `0.30 < p1 < 0.70`.

The threshold logic is central to the project's safety trade-off. A higher low-risk threshold avoids more Tier 2 profiles but increases the risk of bypassing positive cases. The final paper reports that `0.15` avoided 9.8% of Tier 2 profiles with zero bypassed positives on the held-out split, while `0.30` avoided 24.5% but bypassed 4 positives.

### 7.7 Evaluation Metrics

The project uses accuracy, balanced accuracy, precision, recall/sensitivity, specificity, F1 score, MCC, AUROC, Brier score, ECE, Tier 2 avoided, bypass positives, and decision-curve net benefit. These metrics are necessary because accuracy alone does not capture clinical safety, threshold reliability, or workflow utility.

### 7.8 Equations and Algorithms

Binary target:

```text
y = 1 if target > 0
y = 0 if target = 0
```

Tier 1 and Tier 2 probabilities:

```text
p1 = f1(X1)
p2 = f2(X2)
```

Conservative cascade:

```text
if p1 <= 0.30:
    use p1 and bypass Tier 2 in simulation
else:
    request Tier 2 and use p2
```

Decision-curve net benefit:

```text
net_benefit = (TP / N) - (FP / N) * (pt / (1 - pt))
```

Training algorithm:

```text
Load data -> define target -> split -> fit preprocessing on train -> transform train/test
-> slice Tier 1 and Tier 2 -> train Tier 1 RF -> train Tier 2 voting ensemble -> save models
```

SHAP algorithm:

```text
Define positive-class probability function -> build SHAP permutation explainer
-> explain patient row -> sort factors by absolute impact -> display top factors
```

### 7.9 Flowcharts and Sequence Diagrams

<!-- Figure:
Preprocessing Flowchart
Source:
Data_Processing_final.py
-->

<!-- Figure:
Routing Flowchart
Source:
app.py route_for_probability and publication_evaluation.py conservative_cascade
-->

```text
User -> Streamlit UI: Enter Tier 1 vitals
Streamlit UI -> Tier 1 vector builder: Encode and scale intake row
Tier 1 vector builder -> Tier 1 model: Predict probability
Tier 1 model -> Streamlit UI: p1
Streamlit UI -> Routing function: Determine route
Routing function -> Streamlit UI: Low / gray / high
User -> Streamlit UI: Enter Tier 2 diagnostics if required
Streamlit UI -> Tier 2 vector builder: Encode full profile
Tier 2 vector builder -> Tier 2 model: Predict probability
Tier 2 model -> SHAP explainer: Explain positive-class probability
SHAP explainer -> Streamlit UI: Ranked local factors
Streamlit UI -> Note generator: Generate local or LLM draft note
```

---

## Chapter 8: Implementation

### 8.1 Folder Structure and Source Files

| File | Role |
|---|---|
| `heart_disease_uci.csv` | Dataset |
| `Data_Processing_final.py` | Final preprocessing pipeline |
| `Model_handler_final.py` | Final model builder, loader, and trainer |
| `app.py` | Streamlit application |
| `publication_evaluation.py` | Evaluation and audit script |
| `Ai_explainer_shap.py` | Older standalone SHAP/LLM prototype |
| `data_preprocessing.py` | Older Tier 1 preprocessing prototype |
| `model_handler.py` | Older Tier 1 model prototype |
| `overleaf.md` | LaTeX research manuscript source |
| `Final Version reseacrh Paper.pdf` | Final research paper PDF |
| `readme.md` | Project overview |
| `project_analysis.md` | Detailed project analysis draft |
| `phase.md` | Manuscript planning draft |
| `claude_verdict.md` | Peer review critique |
| `gemini_verdict.md` | Peer review critique |
| `codex_verdict.md` | Publication assessment |
| `requirements.txt` | Dependencies |
| `pdf_generator.py` | Empty placeholder |
| `Project_Report.md` | Internship report |

### 8.2 Major Modules

`Data_Processing_final.py` is the final preprocessing module. It loads the dataset from the project directory, creates train-test splits, transforms features, and exposes processed matrices as module-level variables.

`Model_handler_final.py` defines reusable model builders and persistence paths. If model files are missing, the module trains models and dumps them with joblib. In the current workspace listing, `Tier_1_model.pkl` and `Tier_2_model.pkl` are not present. Therefore, they should be described as generated artifacts rather than existing repository files.

`app.py` is the main user-facing application. It contains page configuration, CSS injection, animated WebGL background injection, cached model and dataset loading, reference range calculation, Tier 1 and Tier 2 vector construction, routing logic, Plotly charts, SHAP explanation, local note generation, optional LLM note generation, Streamlit state machine, and UI layout.

`publication_evaluation.py` implements publication-grade evaluation: held-out performance, bypass safety, calibration summary, reliability bins, threshold sweep, decision-curve data, subgroup metrics, site prevalence, and output manifest. It writes CSV outputs to `publication_outputs` when run.

`data_preprocessing.py`, `model_handler.py`, and `Ai_explainer_shap.py` are older prototypes. They are useful for development history but are not the final source of truth. The standalone SHAP prototype references variables not present in the final model handler, so the integrated `app.py` path is treated as current.

### 8.3 Libraries, Functions, and Classes

The project uses pandas, NumPy, scikit-learn, joblib, Streamlit, Plotly, SHAP, OpenAI SDK, dotenv, and listed optional libraries. Important functions in `app.py` include `load_model`, `load_dataset`, `reference_ranges`, `scale_value`, `tier1_vector`, `tier2_vector`, `predict_probability`, `route_for_probability`, `gauge`, `shap_values_for`, `plot_shap_bars`, `plot_factor_balance`, `local_clinical_note`, `ai_clinical_note`, `initialize_state`, `reset_downstream`, and `main`.

The final source files do not define custom model classes. `publication_evaluation.py` defines a dataclass `ClassificationSummary` for metric reporting.

### 8.4 Data Flow, Training, Inference, and UI

Training/evaluation flow:

```text
CSV -> Data_Processing_final.py -> processed matrices -> Model_handler_final.py -> models -> publication_evaluation.py -> metrics
```

UI inference flow:

```text
User inputs -> vector builder -> saved model -> probability -> route -> optional Tier 2 -> SHAP -> note -> dashboard
```

Training occurs when `Model_handler_final.py` is imported or run and model files are missing. This design makes the app easier to use, but it also means that importing the module can have side effects by creating model artifacts. In a production codebase, training and inference should be separated more strictly.

The UI contains two main columns: input forms on the left and decision dashboard on the right. After Tier 1 is run, the route determines whether Tier 2 is needed. For Tier 2 cases, the dashboard displays final risk, patient profile radar chart, SHAP chart, factor balance chart, top factor table, and clinical note.

### 8.5 Security, Privacy, and PDF Placeholder

The app reads API keys from `open_router_api_key` or `OPENROUTER_API_KEY`. No live key should be committed. Any public release should include a `.gitignore` and rotate exposed keys if any were ever used. The report does not include any API key.

`pdf_generator.py` is empty. Older drafts mention a future PDF export feature, but this is not implemented. Therefore, any claim that PDF export exists would be incorrect. It is listed only as future work.

---

## Chapter 9: Results and Analysis

### 9.1 Cross Validation

| Model or Policy | Accuracy | AUROC | Tier 2 Avoided |
|---|---:|---:|---:|
| Tier 1 only | 0.777 +/- 0.027 | 0.843 +/- 0.036 | -- |
| Tier 2 full profile | 0.843 +/- 0.018 | 0.904 +/- 0.035 | 0.0% |
| Conservative cascade | 0.845 +/- 0.022 | 0.898 +/- 0.034 | 25.1% +/- 3.3% |
| Uncertainty-band policy | 0.826 +/- 0.026 | 0.854 +/- 0.042 | 72.4% +/- 2.5% |

Tier 2 improves over Tier 1. Conservative cascade matches Tier 2 accuracy within fold variability while avoiding about one quarter of Tier 2 profiles. Uncertainty-band policy avoids many more profiles but has lower performance.

### 9.2 Held-Out Development Results

| Model or Policy | Accuracy | Balanced Accuracy | Sensitivity | Specificity | F1 | MCC | AUROC | Brier |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Tier 1 only | 0.788 | 0.780 | 0.853 | 0.707 | 0.817 | 0.569 | 0.844 | 0.153 |
| Tier 2 full profile | 0.902 | 0.895 | 0.961 | 0.829 | 0.916 | 0.805 | 0.920 | 0.107 |
| Conservative cascade | 0.897 | 0.890 | 0.951 | 0.829 | 0.911 | 0.793 | 0.922 | 0.107 |
| Uncertainty-band policy | 0.864 | 0.854 | 0.951 | 0.756 | 0.886 | 0.730 | 0.874 | 0.124 |

The held-out split is more optimistic than cross-validation. The final paper correctly avoids making held-out accuracy the main claim.

### 9.3 LOSO Generalization

| Held-Out Site | N | Prevalence | Majority Baseline | Accuracy | Balanced Accuracy | MCC | Sensitivity | Specificity |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Cleveland | 304 | 0.457 | 0.543 | 0.783 | 0.786 | 0.570 | 0.820 | 0.752 |
| Hungary | 293 | 0.362 | 0.638 | 0.799 | 0.820 | 0.615 | 0.896 | 0.743 |
| Switzerland | 123 | 0.935 | 0.935 | 0.813 | 0.726 | 0.276 | 0.826 | 0.625 |
| VA Long Beach | 200 | 0.745 | 0.745 | 0.730 | 0.683 | 0.344 | 0.779 | 0.588 |
| Pooled | 920 | 0.553 | 0.553 | 0.780 | 0.775 | 0.554 | 0.825 | 0.725 |

Pooled LOSO accuracy is 78.0%, below random CV accuracy. Switzerland and VA Long Beach are high-prevalence sites. Raw accuracy falls below majority-class baseline at those two sites, confirming that site shift is a major limitation.

### 9.4 Calibration and Threshold Sweep

Calibration matters because routing thresholds are probability thresholds. The final paper reports that raw Tier 1 probabilities had lower ECE than Tier 2 in the held-out audit, and isotonic calibration slightly improved ECE but worsened Brier score and AUROC.

<!-- Figure:
Held-Out Reliability Bins
Source:
Final Version reseacrh Paper.pdf
-->

| Low Threshold | Tier 2 Avoided | Bypass Count | Bypassed Positives | Missed Positives |
|---:|---:|---:|---:|---:|
| 0.05 | 1.1% | 2 | 0 | 0 |
| 0.10 | 4.9% | 9 | 0 | 0 |
| 0.15 | 9.8% | 18 | 0 | 0 |
| 0.20 | 15.8% | 29 | 1 | 1 |
| 0.25 | 22.3% | 41 | 3 | 3 |
| 0.30 | 24.5% | 45 | 4 | 4 |
| 0.40 | 30.4% | 56 | 8 | 8 |
| 0.50 | 39.7% | 73 | 15 | 15 |

The 0.30 threshold is operationally attractive but unsafe as an autonomous rule-out threshold. The 0.15 threshold is safer on the held-out split but avoids fewer profiles. Thresholds must be calibrated and utility-optimized before clinical use.

### 9.5 Decision Curve, Confusion Matrix, ROC, and SHAP

Decision-curve analysis evaluates net benefit across threshold probabilities. The final paper reports that the Tier 2 model and conservative cascade show stronger net benefit than treat-all and treat-none strategies across a relevant threshold range. However, DCA is still post-hoc and held-out based.

<!-- Figure:
Decision Curve Analysis
Source:
publication_evaluation.py / Final Version reseacrh Paper.pdf
-->

Held-out confusion matrices:

| Model | TN | FP | FN | TP |
|---|---:|---:|---:|---:|
| Tier 1 only | 58 | 24 | 15 | 87 |
| Tier 2 full profile | 68 | 14 | 4 | 98 |
| Conservative cascade | 68 | 14 | 5 | 97 |
| Uncertainty-band policy | 62 | 20 | 5 | 97 |

Cross-validation AUROC values were 0.843 +/- 0.036 for Tier 1, 0.904 +/- 0.035 for Tier 2, 0.898 +/- 0.034 for conservative cascade, and 0.854 +/- 0.042 for uncertainty-band policy.

The app computes local SHAP attributions for Tier 2 predictions. It displays top local drivers as horizontal bars, red bars for risk-raising factors, green bars for risk-lowering factors, a factor balance pie chart, and a top factor table. This is an interface behaviour example, not clinical validation.

### 9.6 Subgroups

| Subgroup | N | Prevalence | Accuracy | Sensitivity | Specificity | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Female | 38 | 0.263 | 0.974 | 0.900 | 1.000 | 0.932 |
| Male | 146 | 0.630 | 0.877 | 0.957 | 0.741 | 0.734 |
| <50 years | 49 | 0.388 | 0.959 | 0.895 | 1.000 | 0.916 |
| 50-65 years | 115 | 0.609 | 0.887 | 0.971 | 0.756 | 0.765 |
| >65 years | 20 | 0.650 | 0.800 | 0.923 | 0.571 | 0.545 |

The subgroup audit is underpowered. Female subgroup has only 38 records and the oldest age band has only 20 records. No fairness claim is supported.

### 9.7 Interpretation and Discussion

The results show that staged acquisition can reduce Tier 2 profile use while preserving much of full-profile performance under random cross-validation. However, deeper audits reveal safety and generalization limitations. The project is successful as a research and engineering audit, not as a deployable clinical tool.

The most important finding is the trade-off between resource efficiency and safety. The 0.30 threshold supports the operational result of roughly one-quarter Tier 2 avoidance, but it also bypasses positive cases. The 0.15 threshold is safer on the held-out split but avoids fewer profiles. The second important finding is site shift. Random cross-validation suggests acceptable performance, but LOSO validation exposes reduced generalization. The third important finding is that explanation features should be treated carefully. SHAP factor rankings can improve traceability but are not clinical explanations by themselves. LLM-generated notes may improve readability, but they require formal evaluation for faithfulness and safety.

### 9.8 Comparison

Compared with conventional full-feature heart disease classifiers, the project contributes a staged workflow and safety audit. Compared with budgeted learning and learning-to-defer literature, the project is simpler and less algorithmically novel, but more explicitly tied to the UCI heart-disease triage workflow and bypass-safety accounting.

---

## Chapter 10: Conclusion and Future Scope

### 10.1 Achievements

The internship project built a leakage-controlled preprocessing pipeline, created an explicit Tier 1/Tier 2 feature split, implemented a Tier 1 Random Forest gatekeeper, implemented a Tier 2 soft-voting ensemble, built a Streamlit application simulating staged clinical triage, integrated SHAP local explanation for Tier 2 predictions, added optional LLM draft note generation with local fallback, implemented publication-grade evaluation scripts, produced a final research manuscript and PDF, and identified safety limitations through threshold and bypass audits.

### 10.2 Limitations

The main limitations are public retrospective data, no prospective validation, old and heterogeneous records, heuristic routing thresholds, unsafe 0.30 threshold for autonomous rule-out, no real monetary cost model, no gradient-boosted or learned acquisition baseline, no nested hyperparameter optimization, underpowered subgroup audit, no clinician evaluation of SHAP or LLM notes, no regulatory validation, and model artifacts not currently listed in the workspace.

### 10.3 Future Work

Future work should include cross-validated calibration for Tier 1 probabilities, utility-optimized threshold selection, bypass safety across every CV fold and LOSO setting, external validation on modern Indian clinical datasets, strong tabular baselines such as XGBoost or LightGBM, learned acquisition or learning-to-defer baselines, realistic cost modelling with test-specific cost and time weights, formal clinician usability study, formal LLM note faithfulness audit, secure on-premise inference for sensitive clinical data, PDF report generation through the currently empty `pdf_generator.py`, and separation of training and inference side effects.

### 10.4 Learning Outcomes

The project taught that responsible healthcare AI development requires more than model building. It requires understanding the clinical workflow, measuring safety trade-offs, recognizing dataset shift, reporting limitations clearly, and avoiding unsupported claims. The internship therefore contributed both technical skill and research maturity.

---

## References

[1] A. Janosi, W. Steinbrunn, M. Pfisterer, and R. Detrano, "Heart Disease," UCI Machine Learning Repository, 1988, doi: 10.24432/C52P4X. Available: https://archive.ics.uci.edu/dataset/45/heart+disease

[2] R. Detrano, A. Janosi, W. Steinbrunn, M. Pfisterer, J.-J. Schmid, S. Sandhu, K. H. Guppy, S. Lee, and V. Froelicher, "International application of a new probability algorithm for the diagnosis of coronary artery disease," American Journal of Cardiology, vol. 64, no. 5, pp. 304-310, 1989.

[3] A. Rajkomar, J. Dean, and I. Kohane, "Machine learning in medicine," New England Journal of Medicine, vol. 380, no. 14, pp. 1347-1358, 2019.

[4] E. J. Topol, "High-performance medicine: The convergence of human and artificial intelligence," Nature Medicine, vol. 25, pp. 44-56, 2019.

[5] C. J. Kelly, A. Karthikesalingam, M. Suleyman, G. Corrado, and D. King, "Key challenges for delivering clinical impact with artificial intelligence," BMC Medicine, vol. 17, Art. no. 195, 2019.

[6] S. F. Weng, J. Reps, J. Kai, J. M. Garibaldi, and N. Qureshi, "Can machine-learning improve cardiovascular risk prediction using routine clinical data?," PLOS ONE, vol. 12, no. 4, Art. no. e0174944, 2017.

[7] M. Motwani et al., "Machine learning for prediction of all-cause mortality in patients with suspected coronary artery disease: A 5-year multicentre prospective registry analysis," European Heart Journal, vol. 38, no. 7, pp. 500-507, 2017.

[8] S. Raman et al., "Machine learning for coronary heart disease prediction: Comparative analysis of Framingham and Cleveland subset of the UCI dataset with SHAP-based interpretability," Epidemiologia, vol. 7, no. 3, Art. no. 75, 2026, doi: 10.3390/epidemiologia7030075.

[9] T. Chen and C. Guestrin, "XGBoost: A scalable tree boosting system," in Proc. ACM SIGKDD, 2016, pp. 785-794.

[10] E. W. Steyerberg et al., "Assessing the performance of prediction models: A framework for traditional and novel measures," Epidemiology, vol. 21, no. 1, pp. 128-138, 2010.

[11] G. S. Collins, J. B. Reitsma, D. G. Altman, and K. G. M. Moons, "Transparent reporting of a multivariable prediction model for individual prognosis or diagnosis (TRIPOD): The TRIPOD statement," Annals of Internal Medicine, vol. 162, no. 1, pp. 55-63, 2015.

[12] G. S. Collins, K. G. M. Moons, P. Dhiman, et al., "TRIPOD+AI statement: Updated guidance for reporting clinical prediction models that use regression or machine learning methods," BMJ, vol. 385, Art. no. e078378, 2024.

[13] E. W. Steyerberg and F. E. Harrell, Jr., "Prediction models need appropriate internal, internal-external, and external validation," Journal of Clinical Epidemiology, vol. 69, pp. 245-247, 2016.

[14] S. G. Finlayson et al., "The clinician and dataset shift in artificial intelligence," New England Journal of Medicine, vol. 385, no. 3, pp. 283-286, 2021.

[15] G. W. Brier, "Verification of forecasts expressed in terms of probability," Monthly Weather Review, vol. 78, no. 1, pp. 1-3, 1950.

[16] A. Niculescu-Mizil and R. Caruana, "Predicting good probabilities with supervised learning," in Proc. ICML, 2005, pp. 625-632.

[17] B. Van Calster et al., "Calibration: The Achilles heel of predictive analytics," BMC Medicine, vol. 17, Art. no. 230, 2019.

[18] A. J. Vickers and E. B. Elkin, "Decision curve analysis: A novel method for evaluating prediction models," Medical Decision Making, vol. 26, no. 6, pp. 565-574, 2006.

[19] R. F. Wolff et al., "PROBAST: A tool to assess the risk of bias and applicability of prediction model studies," Annals of Internal Medicine, vol. 170, no. 1, pp. 51-58, 2019.

[20] X. Liu et al., "Reporting guidelines for clinical trial reports for interventions involving artificial intelligence: The CONSORT-AI extension," Nature Medicine, vol. 26, pp. 1364-1374, 2020.

[21] S. C. Rivera et al., "Guidelines for clinical trial protocols for interventions involving artificial intelligence: The SPIRIT-AI extension," Nature Medicine, vol. 26, pp. 1351-1363, 2020.

[22] B. Vasey et al., "Reporting guideline for the early-stage clinical evaluation of decision support systems driven by artificial intelligence: DECIDE-AI," Nature Medicine, vol. 28, pp. 924-933, 2022.

[23] F. Nan, J. Wang, and V. Saligrama, "Feature-budgeted random forest," in Proc. ICML, 2015, pp. 1983-1991.

[24] M. Kachuee, K. Karkkainen, O. Goldstein, et al., "Cost-sensitive diagnosis and learning leveraging public health data," arXiv:1902.07102, 2019.

[25] P. Viola and M. Jones, "Rapid object detection using a boosted cascade of simple features," in Proc. IEEE CVPR, 2001.

[26] C. K. Chow, "On optimum recognition error and reject tradeoff," IEEE Transactions on Information Theory, vol. 16, no. 1, pp. 41-46, 1970.

[27] R. El-Yaniv and Y. Wiener, "On the foundations of noise-free selective classification," Journal of Machine Learning Research, vol. 11, pp. 1605-1641, 2010.

[28] Y. Geifman and R. El-Yaniv, "Selective classification for deep neural networks," arXiv:1705.08500, 2017.

[29] D. Madras, T. Pitassi, and R. Zemel, "Predict responsibly: Improving fairness and accuracy by learning to defer," in Proc. NeurIPS, 2018, pp. 6150-6160.

[30] H. Mozannar and D. Sontag, "Consistent estimators for learning to defer to an expert," in Proc. AISTATS, PMLR, vol. 108, 2020, pp. 707-717.

[31] J. A. C. Sterne et al., "Multiple imputation for missing data in epidemiological and clinical research: Potential and pitfalls," BMJ, vol. 338, Art. no. b2393, 2009.

[32] L. Breiman, "Random forests," Machine Learning, vol. 45, pp. 5-32, 2001.

[33] S. M. Lundberg and S.-I. Lee, "A unified approach to interpreting model predictions," in Proc. NeurIPS, 2017, pp. 4765-4774.

[34] S. Tonekaboni, S. Joshi, M. D. McCradden, and A. Goldenberg, "What clinicians want: Contextualizing explainable machine learning for clinical end use," in Proc. Machine Learning for Healthcare, PMLR, vol. 106, 2019, pp. 359-380.

[35] Z. Obermeyer, B. Powers, C. Vogeli, and S. Mullainathan, "Dissecting racial bias in an algorithm used to manage the health of populations," Science, vol. 366, no. 6464, pp. 447-453, 2019.

[36] U.S. Food and Drug Administration, "Clinical Decision Support Software: Guidance for Industry and Food and Drug Administration Staff," Final Guidance, Sep. 2022. Available: https://www.fda.gov/regulatory-information/search-fda-guidance-documents/clinical-decision-support-software

[37] Council of Scientific and Industrial Research, "About CSIR," official website. Available: https://www.csir.res.in/en/about-us/about-csir

[38] Council of Scientific and Industrial Research, "Vision & Mission," official website. Available: https://www.csir.res.in/en/vision-mission

[39] Council of Scientific and Industrial Research, "CSIR Network Map," official website. Available: https://www.csir.res.in/en/about-us/csir-network-map

[40] Central Scientific Instruments Organisation, public institutional summary. Available: https://en.wikipedia.org/wiki/Central_Scientific_Instruments_Organisation

[41] K. Singhal et al., "Large language models encode clinical knowledge," Nature, vol. 620, pp. 172-180, 2023.

[42] Z. Ji et al., "Survey of hallucination in natural language generation," ACM Computing Surveys, vol. 55, no. 12, Art. no. 248, 2023.

[43] C. Guo, G. Pleiss, Y. Sun, and K. Q. Weinberger, "On calibration of modern neural networks," in Proc. ICML, 2017, pp. 1321-1330.

[44] A. P. Bradley, "The use of the area under the ROC curve in the evaluation of machine learning algorithms," Pattern Recognition, vol. 30, no. 7, pp. 1145-1159, 1997.

[45] D. Chicco and G. Jurman, "The advantages of the Matthews correlation coefficient over F1 score and accuracy in binary classification evaluation," BMC Genomics, vol. 21, Art. no. 6, 2020.

---

## Appendix A: Feature List

### A.1 Tier 1 Features

```text
age
gender_Female
gender_Male
cp_asymptomatic
cp_atypical angina
cp_non-anginal
cp_typical angina
trestbps
```

### A.2 Tier 2 Features

```text
age, trestbps, chol, thalch, oldpeak, ca_missing, chol_missing_or_zero,
oldpeak_missing, cp_asymptomatic, cp_atypical angina, cp_non-anginal,
cp_typical angina, gender_Female, gender_Male, ca_-1.0, ca_0.0,
ca_1.0, ca_2.0, ca_3.0, restecg_-1.0, restecg_0.0, restecg_1.0,
restecg_2.0, fbs_-1.0, fbs_0.0, fbs_1.0, exang_-1.0, exang_0.0,
exang_1.0, slope_-1.0, slope_0.0, slope_1.0, slope_2.0,
thal_-1.0, thal_0.0, thal_1.0, thal_2.0
```

## Appendix B: Hyperparameters

| Model | Hyperparameters |
|---|---|
| Tier 1 Random Forest | `n_estimators=600`, `max_depth=4`, `min_samples_split=2`, `random_state=43` |
| Tier 2 Random Forest `rf_best` | `n_estimators=200`, `max_depth=12`, `min_samples_split=10`, `random_state=7` |
| Tier 2 Random Forest `rf_current` | `n_estimators=1000`, `max_depth=5`, `min_samples_split=10`, `random_state=43` |
| Tier 2 Logistic Regression | `max_iter=5000` |
| Tier 2 KNN | `n_neighbors=7` |
| Voting ensemble | `voting="soft"` |
| Iterative imputer | `random_state=43`, `max_iter=15` |
| Train-test split | `test_size=0.2`, `stratify=Y`, `random_state=43` |

## Appendix C: Additional Results

| Model | Matrix `[[TN, FP], [FN, TP]]` |
|---|---|
| Tier 1 only | `[[58, 24], [15, 87]]` |
| Tier 2 full profile | `[[68, 14], [4, 98]]` |
| Conservative cascade | `[[68, 14], [5, 97]]` |
| Uncertainty-band policy | `[[62, 20], [5, 97]]` |

## Appendix D: Code Snippets

```python
Y = (uci_dataset["target"] > 0).astype(int)
```

```python
categorical_imputer = SimpleImputer(strategy="constant", fill_value=-1)
```

```python
def build_tier1_model() -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=600,
        max_depth=4,
        min_samples_split=2,
        random_state=43,
    )
```

```python
def conservative_cascade(p1, p2):
    bypass = p1 <= LOW_THRESHOLD
    score = np.where(bypass, p1, p2)
    pred = np.where(bypass, as_binary(p1), as_binary(p2))
    return pred, score, bypass
```

## Appendix E: Screenshots

The repository does not contain current screenshot image files. Screenshots should be captured after launching `streamlit run app.py`.

<!-- Figure:
Tier 1 Intake Screen
Source:
TODO: Capture from running Streamlit app.
-->

<!-- Figure:
Tier 2 SHAP Dashboard
Source:
TODO: Capture from running Streamlit app.
-->

## Appendix F: User Manual

Start the application with:

```text
streamlit run app.py
```

Enter Tier 1 vitals, run initial triage, review the route, enter Tier 2 diagnostics if activated, run final diagnosis, and inspect risk, SHAP factors, and the draft note. The output is clinical decision support only and must not be treated as a diagnosis.

## Appendix G: Installation Guide

```text
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

For optional LLM output, set `open_router_api_key` or `OPENROUTER_API_KEY`. If no key is configured, the app uses a deterministic local note.

The current workspace listing does not include `Tier_1_model.pkl` or `Tier_2_model.pkl`. They can be generated by running:

```text
python Model_handler_final.py
```

## Appendix H: Project Folder Structure

```text
CSIO Project Version 2.0/
|-- Ai_explainer_shap.py
|-- app.py
|-- claude_verdict.md
|-- codex_verdict.md
|-- data_preprocessing.py
|-- Data_Processing_final.py
|-- Final Version reseacrh Paper.pdf
|-- gemini_verdict.md
|-- heart_disease_uci.csv
|-- Model_handler_final.py
|-- model_handler.py
|-- overleaf.md
|-- pdf_generator.py
|-- phase.md
|-- project_analysis.md
|-- Project_Report.md
|-- publication_evaluation.py
|-- readme.md
|-- requirements.txt
```

Note: Generated model files and publication output CSVs are expected artifacts but are not present in the current root listing unless created by running the relevant scripts.
