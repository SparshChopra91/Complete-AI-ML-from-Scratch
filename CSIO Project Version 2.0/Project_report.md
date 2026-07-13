<div style="background-color: #fdfbf7; color: #28313a; font-family: 'Georgia', serif; line-height: 1.6; padding: 40px;">

# SMART CLINIC ASSISTANT: A RESOURCE-AWARE TWO-TIER HEART DISEASE TRIAGE SYSTEM USING STAGED FEATURE ACQUISITION AND SHAP-GUIDED DRAFT EXPLANATION

**A Formal Six-Week Summer Internship Technical Report**  
**Submitted in Partial Fulfillment of the Requirements for the Summer Internship Program**  
**Tenure: June 1, 2026 – July 12, 2026**

---

## FRONT MATTER (ADMINISTRATIVE)

### 1. OFFICIAL COVER PAGE

```
========================================================================================
             COUNCIL OF SCIENTIFIC AND INDUSTRIAL RESEARCH (CSIR)
           CENTRAL SCIENTIFIC INSTRUMENTS ORGANISATION (CSIO)
                         SECTOR 30-C, CHANDIGARH - 160030, INDIA
========================================================================================

                               SIX-WEEK INTERNSHIP TECHNICAL REPORT

                                         PROJECT TITLE:
             SMART CLINIC ASSISTANT: A RESOURCE-AWARE TWO-TIER HEART DISEASE
                   TRIAGE SYSTEM USING STAGED FEATURE ACQUISITION AND
                                SHAP-GUIDED DRAFT EXPLANATION

SUBMITTED BY:
Sparsh Chopra
Department of Computer Science & Engineering
Chandigarh College of Engineering and Technology (CCET)
Sector 26, Chandigarh - 160019, India

RESEARCH MENTORS:
Dr. Sanjeev Verma
Senior Scientist / Research Mentor, CSIR-CSIO, Chandigarh

Dr. Durgesh Mishra
Senior Scientist / Research Mentor, CSIR-CSIO, Chandigarh

INTERNSHIP TENURE:
1 June 2026 – 12 July 2026
========================================================================================
```

---

### 2. CANDIDATE'S DECLARATION

I hereby declare that the summer internship technical report entitled **"Smart Clinic Assistant: A Resource-Aware Two-Tier Heart Disease Triage System Using Staged Feature Acquisition and SHAP-Guided Draft Explanation"**, submitted to the **Central Scientific Instruments Organisation (CSIR-CSIO)**, Chandigarh, and the **Chandigarh College of Engineering and Technology (CCET)**, Chandigarh, represents an authentic, original, and independent record of engineering, pipeline architecture, and empirical research carried out by me during the six-week summer internship program from **June 1, 2026 to July 12, 2026**, under the direct joint supervision of **Dr. Sanjeev Verma** and **Dr. Durgesh Mishra**, Senior Scientists at CSIR-CSIO.

I further declare that:
1. All software implementations (`Data_Processing_final.py`, `Model_handler_final.py`, `Ai_explainer_shap.py`, `app.py`, and `publication_evaluation.py`) were engineered, tested, and audited independently during the internship tenure.
2. The statistical evaluations, cross-validation metrics, calibration audits, Decision Curve Analyses, and Leave-One-Site-Out generalizability tests reported herein are genuine empirical outputs generated from the documented pipeline.
3. No portion of this report or codebase has been submitted, either in whole or in part, to any other institution, university, or examination body for the award of any degree, diploma, fellowship, or professional certification.

**Sparsh Chopra**  
Department of Computer Science & Engineering  
Chandigarh College of Engineering and Technology (CCET), Chandigarh  
Date: July 12, 2026  
Signature: ___________________________

---

### 3. CERTIFICATE OF SUPERVISION

This is to certify that the summer internship technical report entitled **"Smart Clinic Assistant: A Resource-Aware Two-Tier Heart Disease Triage System Using Staged Feature Acquisition and SHAP-Guided Draft Explanation"**, submitted by **Sparsh Chopra** in partial fulfillment of the requirements for the six-week summer internship program at the **Central Scientific Instruments Organisation (CSIR-CSIO)**, Chandigarh, is a bona fide record of research and software engineering work carried out under our direct supervision from **June 1, 2026 to July 12, 2026**.

During the course of this project, the candidate demonstrated rigorous technical competence in clinical medical informatics, machine learning pipeline design, TRIPOD+AI compliant leakage control, and interactive clinical decision support interface development. The candidate successfully completed all assigned technical milestones, including:
- Engineering a non-leaking preprocessing stack handling Missing Not At Random (MNAR) categorical clinical test results via non-colliding sentinel imputation (`-1`) and explicit missingness flags prior to Multivariate Imputation by Chained Equations (MICE).
- Implementing a two-tier cascade architecture dividing low-cost emergency intake vitals from high-cost laboratory and imaging features.
- Conducting thorough publication-grade safety audits, including threshold-sweep bypass accounting, Decision Curve Analysis (DCA), expected calibration error (ECE-10) evaluation, Leave-One-Site-Out (LOSO) stress testing across four international hospital cohorts, and demographic subgroup auditing.

To the best of our knowledge, the results reported herein are original, scientifically rigorous, and meet the high institutional standards expected by CSIR-CSIO and CCET Chandigarh.

**Dr. Sanjeev Verma**  
Senior Scientist & Research Mentor  
CSIR-Central Scientific Instruments Organisation (CSIO)  
Sector 30-C, Chandigarh - 160030  
Signature: ___________________________  
Date: July 12, 2026

**Dr. Durgesh Mishra**  
Senior Scientist & Research Mentor  
CSIR-Central Scientific Instruments Organisation (CSIO)  
Sector 30-C, Chandigarh - 160030  
Signature: ___________________________  
Date: July 12, 2026

**Head of Department**  
Department of Computer Science & Engineering  
Chandigarh College of Engineering and Technology (CCET), Chandigarh  
Signature: ___________________________  
Date: July 12, 2026

---

### 4. ACKNOWLEDGEMENT

I wish to express my deepest gratitude to the **Council of Scientific and Industrial Research - Central Scientific Instruments Organisation (CSIR-CSIO)**, Chandigarh, for providing an exceptional national research environment, world-class computational infrastructure, and a rigorous scientific culture during this six-week summer internship program.

I am profoundly indebted to my mentors, **Dr. Sanjeev Verma** and **Dr. Durgesh Mishra**, Senior Scientists at CSIR-CSIO, for their continuous technical guidance, insightful critiques, and unwavering support. Their rigorous feedback on statistical methodology, TRIPOD+AI compliance, probability calibration, and clinical risk analysis shaped the scientific direction of this work and elevated the project from a standard predictive prototype into a formal workflow safety audit.

I also extend my sincere thanks to the Director of CSIR-CSIO, the Head of the Department of Computer Science and Engineering at **Chandigarh College of Engineering and Technology (CCET)**, Chandigarh, and all faculty members and technical staff who provided valuable suggestions and organizational support throughout the tenure of this project.

---

### 5. ABSTRACT

Machine-learning research in cardiovascular disease prediction routinely evaluates diagnostic classifiers under the unrealistic operational assumption that complete patient profiles—including invasive fluoroscopy, nuclear stress tests, and specialized biochemical assays—are available at initial intake. This all-at-once assumption ignores the clinical reality of emergency department triage, where basic vitals are acquired immediately at zero variable cost, whereas diagnostic procedures incur substantial financial expense, infrastructure demand, and patient waiting time. This report details **Smart Clinic Assistant**, a retrospective staged-acquisition clinical decision support workflow simulation engineered and audited during a six-week summer research internship at **CSIR-CSIO**.

The system evaluates a 920-record multi-site clinical dataset (Cleveland, Hungary, VA Long Beach, Switzerland) partitioned into an eight-feature **Tier 1 intake space** (age, sex, chest pain description, resting blood pressure) and a 37-feature encoded **Tier 2 diagnostic space**. To preserve strict data integrity without leakage, missing categorical values are imputed with a non-colliding sentinel value of **`-1`**, binary missingness indicators are explicitly generated, and continuous variables undergo Multivariate Imputation by Chained Equations (MICE) fitted strictly on training partitions. A shallow Random Forest gatekeeper evaluates intake vitals against pragmatic lower and upper routing thresholds ($\theta_L = 0.30$, $\theta_H = 0.70$). Under stratified 5-fold cross-validation, the implemented conservative cascade achieved **84.5% ± 2.2% accuracy** and **0.898 ± 0.034 AUROC** while avoiding **25.1% ± 3.3% of Tier 2 diagnostic profiles** ($p=1.000$ vs. full Tier 2 model on held-out audit).

Crucially, the report conducts exhaustive post-hoc safety and distribution audits. Leave-one-site-out (LOSO) testing revealed severe site shift, dropping pooled accuracy to **78.0%** and falling below trivial majority-class baselines at high-prevalence sites (Switzerland, VA Long Beach). Furthermore, held-out bypass-safety audits demonstrated that the pragmatic **0.30 threshold** bypassed 4 true positive cases among 45 low-risk patients (91.1% NPV), whereas a stricter **0.15 threshold** achieved zero bypassed positives with 9.8% profile avoidance. For explainability, local SHAP permutation attributions are integrated with an experimental Large Language Model (Google Gemini via OpenRouter API) to generate plain-language draft summaries. The project establishes a reproducible workflow and safety-auditing framework for resource-aware clinical triage while explicitly underscoring that prospective threshold calibration and clinical trial validation remain mandatory prior to hospital deployment.

---

### 6. TABLE OF CONTENTS

- **FRONT MATTER**
  - 1. Official Cover Page
  - 2. Candidate's Declaration
  - 3. Certificate of Supervision
  - 4. Acknowledgement
  - 5. Abstract
  - 6. Table of Contents
  - 7. List of Figures
  - 8. List of Tables
- **CHAPTER 1: INTRODUCTION**
  - 1.1 Overview of Cardiovascular Disease Triage
  - 1.2 Problem Statement: The "All-Features-Available" Fallacy in Machine Learning
  - 1.3 Objectives of the Internship Project
  - 1.4 Scope of Work at CSIR-CSIO
  - 1.5 Report Organization
- **CHAPTER 2: LITERATURE REVIEW**
  - 2.1 Evolution of Cardiovascular Risk Prediction Models
  - 2.2 Critical Appraisal of Static Complete-Case ML Benchmarks
  - 2.3 Staged Feature Acquisition, Cascade Classifiers, and Cost-Sensitive Learning
  - 2.4 Selective Classification, Deferral Policies, and Uncertainty Quantification
  - 2.5 Explainable AI (SHAP) and Large Language Model Integration in Medicine
  - 2.6 Methodological Reporting Guidelines (TRIPOD+AI) and Dataset Shift
- **CHAPTER 3: TECHNOLOGIES AND TOOLS USED**
  - 3.1 Core Scientific Python Ecosystem (Python, Scikit-Learn, Pandas, NumPy)
  - 3.2 Model Persistence and Engineering Tools (Joblib, Git, GitHub)
  - 3.3 Interactive Clinical Interface Framework (Streamlit, Plotly, WebGL Shaders)
  - 3.4 Explainability and Generative AI Stack (SHAP Permutation, Google Gemini via OpenRouter)
- **CHAPTER 4: METHODOLOGY AND SYSTEM ARCHITECTURE**
  - 4.1 Multi-Site Clinical Dataset Composition and Target Definition
  - 4.2 Strict TRIPOD+AI Leakage Control and Preprocessing Pipeline
  - 4.3 Sentinel Categorical Imputation (`-1`) and Missingness Flag Generation
  - 4.4 Feature Space Partitioning: Tier 1 Intake Space vs. Tier 2 Diagnostic Space
  - 4.5 Tier 1 Shallow Random Forest Gatekeeper Architecture
  - 4.6 Tier 2 Heterogeneous Soft-Voting Diagnostic Ensemble Architecture
  - 4.7 Triage Routing Policies and Mathematical Gate Formulation
  - 4.8 Evaluation Framework: Cross-Validation, DCA, Calibration, and Safety Accounting
- **CHAPTER 5: IMPLEMENTATION AND INTERFACE**
  - 5.1 Object-Oriented Backend Engineering (`ModelHandler` and Pipeline Encapsulation)
  - 5.2 Offline Evaluation Engine Engineering (`publication_evaluation.py`)
  - 5.3 Local SHAP Attribution Engine and LLM Translation Layer
  - 5.4 Interactive Streamlit Clinical Decision Support Dashboard Architecture
  - 5.5 Reactive UI State Machine and Dynamic Diagnostic Workup Unlocking
- **CHAPTER 6: RESULTS AND ANALYSIS**
  - 6.1 Primary Generalization Benchmark: Stratified 5-Fold Cross-Validation
  - 6.2 Development Split Performance Benchmark ($N=184$)
  - 6.3 Resource Stewardship and Tier 2 Profile Avoidance Analysis
  - 6.4 Clinical Bypass Safety Audit and Low-Threshold Sensitivity Sweep
  - 6.5 Probability Calibration Audit and Reliability Diagram Analysis
  - 6.6 Decision Curve Analysis (Net Benefit Evaluation)
  - 6.7 Institutional Generalizability Audit: Leave-One-Site-Out (LOSO) Stress Testing
  - 6.8 Exploratory Subgroup Fairness Audit across Biological Sex and Age Bands
  - 6.9 End-to-End Deep Explainability and Draft Generative Output Walkthrough
- **CHAPTER 7: CONCLUSION AND FUTURE WORK**
  - 7.1 Summary of Engineering and Methodological Contributions
  - 7.2 Rigorous Accounting of Methodological and Clinical Limitations
  - 7.3 Roadmap for Clinical Integration and Future Scope at CSIR-CSIO
- **REFERENCES**

---

### 7. LIST OF FIGURES

- **Figure 4.1:** End-to-End Architectural Flowchart of the Two-Tier Staged Clinical Triage Pipeline
- **Figure 5.1:** Streamlit Dashboard Architecture: Stage 1 Intake Form and Tier 1 Gatekeeper Card
- **Figure 5.2:** Streamlit Dashboard Architecture: Dynamic Tier 2 Laboratory Panel and Explanation Workspace
- **Figure 6.1:** Decision Curve Analysis (DCA): Net Benefit vs. Clinical Threshold Probability ($p_t \in [0.05, 0.45]$)
- **Figure 6.2:** Probability Calibration Reliability Histogram: Raw vs. Isotonic Calibrated Tier 1 Probabilities

---

### 8. LIST OF TABLES

- **Table 4.1:** Multi-Site Institutional Composition and Prevalence of the 920-Record Clinical Dataset
- **Table 4.2:** Complete Feature Partitioning between Tier 1 Intake Subspace ($X_1$) and Tier 2 Diagnostic Subspace ($X_2$)
- **Table 6.1:** Primary Stratified 5-Fold Cross-Validation Generalization Performance ($N=920$)
- **Table 6.2:** Held-Out Development Split Benchmark across Standalone Models and Cascade Policies ($N=184$)
- **Table 6.3:** Held-Out Low-Threshold Sweep ($\theta_L \in [0.05, 0.50]$) and Bypass Safety Accounting
- **Table 6.4:** Leave-One-Site-Out (LOSO) Stress Testing and Comparison against Institutional Majority-Class Baselines
- **Table 6.5:** Exploratory Held-Out Subgroup Audit Disaggregated by Biological Sex and Age Bands

---

## CHAPTER 1: INTRODUCTION

### 1.1 Overview of Cardiovascular Disease Triage

Cardiovascular disease (CVD) represents the single largest cause of mortality globally, accounting for nearly 18 million deaths annually according to the World Health Organization. In acute hospital settings—particularly overcrowded emergency departments, community triage clinics, and rural health centers—the rapid and accurate stratification of patients presenting with chest pain or suspected coronary artery disease (CAD) is a paramount operational objective. Emergency department physicians operate under severe temporal, infrastructure, and staffing constraints. They must continuously navigate a challenging clinical trade-off between two asymmetrical risks:
1. **False-Negative Risk (Under-Triage):** Discharging an acute coronary patient without specialized intervention or hospitalization carries catastrophic mortality and morbidity risks.
2. **False-Positive Risk (Over-Triage):** Ordering invasive, expensive, or time-consuming diagnostic procedures—such as complete biochemical panels, nuclear stress imaging, or cardiac catheterization—for every presenting patient congests diagnostic infrastructure, depletes clinical resources, inflates healthcare costs, and subjects low-risk patients to iatrogenic complications.

Over the past two decades, clinical decision support systems (CDSS) have evolved to incorporate machine-learning classifiers trained on patient electronic health records (EHR). However, for a predictive algorithmic tool to be genuinely effective in real-world clinical operations, it must mirror the operational physical realities of medical triage workflows.

### 1.2 Problem Statement: The "All-Features-Available" Fallacy in Machine Learning

A pervasive methodological flaw dominates contemporary academic literature on cardiovascular machine learning: researchers almost exclusively evaluate predictive classifiers under the **"all-at-once" feature availability assumption**. In standard academic benchmarking pipelines, classifiers are trained and evaluated by simultaneously injecting the complete patient feature vector $X = \{x_1, x_2, \dots, x_M\}$ into the model.

This mathematical paradigm implicitly treats every physiological, laboratory, and imaging feature as if it were instantly and costlessly available at initial patient presentation. In physical hospital practice, however, clinical features exhibit extreme operational asymmetries across three real-world axes:
1. **Acquisition Expense and Invasiveness:** Primary triage vitals—such as patient age, biological sex, subjective description of chest pain, and resting blood pressure—are non-invasive, essentially zero-cost, and recorded by nursing staff within minutes of emergency intake. Conversely, diagnostic assays—such as fasting blood sugar assays, serum cholesterol panels, 12-lead resting electrocardiograms, thallium stress tests, and fluoroscopic coronary vessel counts—require specialized laboratory processing, specialized diagnostic hardware, phlebotomy, and physician interpretation.
2. **Temporal Latency:** Intake vitals are available instantly at time $t=0$, whereas blood assays and nuclear imaging introduce diagnostic latencies ranging from hours to overnight observation.
3. **Workflow Mismatch:** An algorithm that mandates complete diagnostic feature vectors before computing a risk probability cannot function as a front-line triage gatekeeper. Requiring invasive fluoroscopic vessel counts simply to run a prediction model defeats the purpose of triage.

### 1.3 Objectives of the Internship Project

During the six-week research internship tenure at **CSIR-CSIO**, this project aimed to replace the static "all-at-once" classification paradigm with an auditable, staged-acquisition clinical decision support architecture. The specific technical objectives were:
1. **Architecting a Staged Feature Pipeline:** To formalize and engineer a strict two-tier data separation dividing low-cost, immediately available intake vitals (**Tier 1**) from downstream, high-cost laboratory and imaging measurements (**Tier 2**).
2. **Developing a Leakage-Controlled Engineering Stack:** To implement an end-to-end Python preprocessing pipeline ensuring strict TRIPOD+AI compliance by performing train-test partitioning prior to imputation, utilizing non-colliding sentinel values (`-1`) for missing categorical data, and generating explicit missingness indicators.
3. **Designing and Auditing Cascade Routing Policies:** To implement a gatekeeper model coupled with pragmatic routing thresholds ($\theta_L=0.30$, $\theta_H=0.70$) that bypasses complete Tier 2 diagnostic workups for low-risk patients while rigorously auditing bypass safety across multi-site and subgroup stress tests.
4. **Implementing Human-Centered Explainability:** To integrate local Shapley Additive exPlanations (SHAP) with an experimental Large Language Model (Google Gemini 2.5 Flash via OpenRouter API) to translate complex ensemble attributions into draft clinical communication summaries requiring physician oversight.

### 1.4 Scope of Work at CSIR-CSIO

The scope of work executed at **CSIR-CSIO** encompassed complete system research, software engineering, empirical auditing, and scientific documentation:
- **Data Engineering:** Ingestion, cleaning, and reconciliation of the 920-record multi-site UCI Heart Disease data file across four international medical centers.
- **Model Design & Persistence:** Object-oriented implementation of a Tier 1 shallow Random Forest gatekeeper (`max_depth=4`) and a Tier 2 heterogeneous soft-voting ensemble comprising Random Forests, Logistic Regression, and $k$-Nearest Neighbors.
- **Auditing Framework:** Construction of `publication_evaluation.py` to generate Decision Curve Analysis (DCA) coordinates, expected calibration error (ECE-10) metrics, reliability bins, threshold-sweep sensitivity tables, leave-one-site-out (LOSO) generalizability tables, and demographic subgroup audits.
- **Interface Construction:** Development of a production-grade interactive Streamlit web dashboard (`app.py`, 1,844 lines) featuring real-time staged inference, Plotly visual charts, custom WebGL shader backgrounds, and LLM explanation rendering.

### 1.5 Report Organization

This technical report is organized into seven core chapters:
- **Chapter 1 (Introduction):** Formulates the clinical problem, exposes the "all-at-once" fallacy, and outlines project objectives.
- **Chapter 2 (Literature Review):** Critically synthesizes prior research in cardiovascular ML, staged acquisition, selective classification, explainable AI, and TRIPOD+AI reporting standards.
- **Chapter 3 (Technologies and Tools Used):** Details the Python scientific stack, Streamlit UI framework, and generative AI interfaces.
- **Chapter 4 (Methodology and System Architecture):** Provides an exhaustive mathematical and structural deep dive into dataset composition, leakage control, feature tiering, ensemble design, routing policies, and evaluation protocols.
- **Chapter 5 (Implementation and Interface):** Explains the backend object-oriented architecture, evaluation scripting, and Streamlit user interface implementation.
- **Chapter 6 (Results and Analysis):** Presents comprehensive empirical findings across cross-validation, held-out auditing, threshold sensitivity sweeps, calibration, Decision Curve Analysis, Leave-One-Site-Out stress testing, and subgroup analysis.
- **Chapter 7 (Conclusion and Future Work):** Summarizes engineering feasibility, acknowledges strict clinical limitations, and outlines a roadmap for hospital validation.

---

## CHAPTER 2: LITERATURE REVIEW

### 2.1 Evolution of Cardiovascular Risk Prediction Models

Cardiovascular risk assessment has evolved through distinct epidemiological and algorithmic eras. Classical risk scores—such as the Framingham Risk Score (D'Agostino *et al.*, 2008), the Systematic Coronary Risk Evaluation (SCORE; Conroy *et al.*, 2003), and the pooled atherosclerotic cardiovascular disease (ASCVD) risk equations—relied on parametric linear survival models (Cox proportional hazards) or multivariable logistic regressions applied to narrow demographic and serum biomarkers. While highly interpretative and clinically ubiquitous, these parametric equations impose rigid linear additivity assumptions and struggle to capture complex non-linear feature interactions.

Early computational benchmarking on the UCI Heart Disease dataset (Detrano *et al.*, 1989) demonstrated that statistical algorithms could predict angiographically confirmed coronary artery disease from clinical profiles. Over the past decade, supervised machine-learning models—including Random Forests (Breiman, 2001), Support Vector Machines (Cortes & Vapnik, 1995), Gradient Boosted Decision Trees (XGBoost by Chen & Guestrin, 2016; LightGBM by Ke *et al.*, 2017), and Deep Neural Networks—have repeatedly demonstrated superior statistical discrimination over classical linear scores on retrospective cohorts.

### 2.2 Critical Appraisal of Static Complete-Case ML Benchmarks

A prominent contemporary example of this complete-case modeling paradigm is the recent study by Raman *et al.* (*Epidemiologia*, 2026), which evaluated non-linear classifiers across the Framingham and Cleveland UCI cohorts and incorporated SHAP feature attributions. While Raman *et al.* achieved strong retrospective AUROC metrics, their methodology exhibits four classic limitations shared by the broader medical ML literature:
1. **Complete-Case Filtering:** Raman *et al.* restricted their Cleveland evaluation to 297 complete records after discarding instances with missing values, ignoring informative missingness patterns.
2. **Single-Site Selection Bias:** By focusing exclusively on the Cleveland Clinic cohort, the study ignored the severe institutional shifts present across the remaining European and VA sites in the complete 920-record repository.
3. **Absence of Cost-Aware Workflow Modeling:** All 13 diagnostic attributes—including invasive fluoroscopy (`ca`) and thallium stress imaging (`thal`)—were fed to the models simultaneously, providing zero mechanism for early patient discharge.
4. **Lack of Threshold Safety Auditing:** The study reported standard default classification accuracy ($p \ge 0.50$) without evaluating the clinical net benefit or false-negative bypass rates at low emergency triage thresholds.

### 2.3 Staged Feature Acquisition, Cascade Classifiers, and Cost-Sensitive Learning

To overcome the operational unfeasibility of full-profile inference, cost-sensitive learning and budgeted feature acquisition formalize learning settings where feature acquisition incurs explicit costs. Viola and Jones (2001) revolutionized real-time computer vision by introducing **cascade classification**, demonstrating that evaluating instances through a sequential pipeline of increasingly complex classifiers allows simple, computationally cheap stages to reject easy negative cases immediately, reserving expensive downstream evaluation only for ambiguous instances.

In medical informatics, cost-sensitive feature selection (CSFS; Turney, 1995; Kachuee *et al.*, 2019) integrates acquisition costs into the optimization objective:
$$\min_{f, S} \; \mathcal{L}(y, f(X_S)) + \lambda \sum_{j \in S} c_j,$$
where $S \subseteq \{1, \dots, M\}$ is the subset of acquired features, $c_j$ represents the financial, temporal, or physiological cost of feature $j$, and $\mathcal{L}$ is the predictive loss. While theoretical frameworks for active feature acquisition (AFA) using reinforcement learning exist, practical clinical deployment requires transparent, deterministic stage boundaries that match existing hospital triage protocols.

### 2.4 Selective Classification, Deferral Policies, and Uncertainty Quantification

Selective classification and reject-option learning (Chow, 1970; El-Yaniv & Wiener, 2010; Geifman & El-Yaniv, 2017) allow a classifier $f(x)$ to abstain from making an autonomous decision when predictive confidence falls within an uncertainty region:
$$f(x) = \begin{cases} \hat{y}, & \text{if } \text{Confidence}(x) \ge \tau, \\ \text{Abstain / Defer}, & \text{if } \text{Confidence}(x) < \tau. \end{cases}$$

In medical decision support, recent "learning to defer" frameworks (Madras *et al.*, 2018; Mozannar & Sontag, 2020) and uncertainty-aware clinical triage (Kompa *et al.*, 2021) argue that autonomous clinical AI is fundamentally unsafe without explicit uncertainty gates. When an intake gatekeeper exhibits posterior uncertainty ($0.30 < p_1 < 0.70$), the system must defer autonomous discharge and request secondary expert workup.

### 2.5 Explainable AI (SHAP) and Large Language Model Integration in Medicine

The adoption of high-dimensional tree ensembles and neural networks in clinical environments is impeded by their black-box opacity. Local feature attribution methods—most notably **Shapley Additive exPlanations (SHAP)** (Lundberg & Lee, 2017)—provide a mathematically rigorous, axiomatic framework based on cooperative game theory. SHAP assigns each feature an additive local attribution $\phi_i(x)$ satisfying local accuracy, missingness, and consistency.

However, clinical human-computer interaction studies (Tonekaboni *et al.*, 2019) demonstrate that raw numerical attribution vectors and complex force plots create cognitive overload for busy emergency physicians. To bridge this communication gap, recent studies (Singhal *et al.*, 2023) have explored coupling quantitative ML attributions with Large Language Models (LLMs) to generate natural-language clinical narratives. While LLMs excel at syntactic fluency, they exhibit severe clinical hazards, including ungrounded factual hallucinations, uncalibrated confidence, and sensitivity to prompt perturbations (Ji *et al.*, 2023). Consequently, regulatory bodies such as the U.S. FDA (2022 guidance on Clinical Decision Support Software) dictate that generative text in CDSS must be presented strictly as unvalidated draft assistance requiring mandatory human clinician review.

### 2.6 Methodological Reporting Guidelines (TRIPOD+AI) and Dataset Shift

To combat widespread reproducibility failures in clinical AI, the international **TRIPOD+AI statement** (Transparent Reporting of a multivariable prediction model for Individual Prognosis Or Diagnosis paired with AI; Collins *et al.*, 2024) mandates strict reporting standards. TRIPOD+AI explicitly requires:
- Clear separation of data partitioning prior to any preprocessing or imputation to prevent data leakage.
- Explicit reporting of missing data mechanisms and imputation models.
- Reporting of probability calibration alongside discrimination metrics.
- External validation across distinct geographical or institutional cohorts.

Crucially, medical AI models are highly vulnerable to **dataset shift** (Finlayson *et al.*, 2021), where changes in institutional protocols, patient demographics, or disease prevalence degrade classifier performance outside the training environment. Evaluating models under Leave-One-Site-Out (LOSO) cross-validation is essential to uncover hidden vulnerability to institutional shift.

---

## CHAPTER 3: TECHNOLOGIES AND TOOLS USED

### 3.1 Core Scientific Python Ecosystem (Python, Scikit-Learn, Pandas, NumPy)
- **Python (v3.10+):** Selected as the foundational engineering language due to its universal standard status in scientific computing and clinical machine learning.
- **Scikit-Learn (v1.4+):** Utilized as the core machine-learning engine. Scikit-learn provides robust, well-tested implementations of foundational supervised algorithms (`RandomForestClassifier`, `LogisticRegression`, `KNeighborsClassifier`, `VotingClassifier`), data transformation pipelines (`OneHotEncoder`, `MinMaxScaler`, `SimpleImputer`, `IterativeImputer`), and standard evaluation metrics (`accuracy_score`, `roc_auc_score`, `brier_score_loss`, `matthews_corrcoef`).
- **Pandas & NumPy:** Employed for structured tabular dataframe manipulation, efficient numerical slicing, Boolean masking, and vectorized matrix algebra.

### 3.2 Model Persistence and Engineering Tools (Joblib, Git, GitHub)
- **Joblib:** Utilized for lightweight, zero-copy object serialization (`joblib.dump`, `joblib.load`), enabling fast persistence and lazy loading of compiled ensemble weights (`Tier_1_model.pkl`, `Tier_2_model.pkl`) without retraining during UI inference.
- **Git & GitHub:** Employed for distributed version control, tracking code modifications, and documenting experimental iterations across the six-week research tenure.

### 3.3 Interactive Clinical Interface Framework (Streamlit, Plotly, WebGL Shaders)
- **Streamlit (v1.35+):** Selected to engineer the full-stack web dashboard (`app.py`). Streamlit's reactive execution model enables dynamic UI state machine transitions: when a patient's Tier 1 probability falls within the uncertainty gate ($p_1 > 0.30$), Streamlit automatically re-renders the DOM to unlock and display the Tier 2 diagnostic input panel.
- **Plotly Enterprise Charting:** Integrated to construct interactive, responsive, client-side visual artifacts, including multi-zone risk gauges, horizontal SHAP attribution bar charts, radar feature comparison profiles, and factor balance donuts.
- **Custom CSS & WebGL Shaders:** Custom CSS enforced a high-contrast warm glassmorphic clinical palette (`--paper: #fffaf2`, `--ink: #28313a`), augmented by a custom React Bits WebGL SideRays animated background shader.

### 3.4 Explainability and Generative AI Stack (SHAP Permutation, Google Gemini via OpenRouter)
- **SHAP (Shapley Additive exPlanations):** Utilized `shap.Explainer` configured in model-agnostic permutation mode (`algorithm="permutation"`) to compute exact local additive feature attributions around non-linear voting boundaries.
- **Google Gemini 2.5 Flash via OpenRouter API:** Integrated via OpenAI-compatible RESTful JSON endpoints (`https://openrouter.ai/api/v1`) to synthesize clinical draft narratives from structured SHAP rankings. Strict system prompts enforce professional doctor-to-patient consultation tone, forbid technical ML terminology, and implement a deterministic offline rule-based fallback generator if API access is unavailable.

---

## CHAPTER 4: METHODOLOGY AND SYSTEM ARCHITECTURE (CRITICAL SECTION)

### 4.1 Multi-Site Clinical Dataset Composition and Target Definition

The project evaluated a composite **920-record clinical dataset** compiled from four international medical centers available through the UCI Machine Learning Repository (Detrano *et al.*, 1988). Rather than filtering the cohort to complete-case Cleveland records, the entire 920-record cohort was analyzed to audit institutional robustness:

```
========================================================================================
TABLE 4.1: MULTI-SITE INSTITUTIONAL COMPOSITION AND PREVALENCE OF THE 920-RECORD DATASET
========================================================================================
Source Site / Medical Center          Records (N)   Positive CAD Cases (y=1)  Prevalence
----------------------------------------------------------------------------------------
Cleveland Clinic Foundation (USA)         304                 139              0.457
Hungarian Institute of Cardiology         293                 106              0.362
VA Medical Center Long Beach (USA)        200                 149              0.745
University Hospital Zurich (Switzerland)  123                 115              0.935
----------------------------------------------------------------------------------------
Complete Pooled Multi-Site Dataset        920                 509              0.553
Held-Out Test Partition (20% Stratified)  184                 102              0.554
========================================================================================
```

The binary diagnostic target $y \in \{0, 1\}$ was formulated from the raw ordinal angiographic narrowing score (`target` $\in \{0, 1, 2, 3, 4\}$, representing the number of coronary vessels showing $>50\%$ diameter stenosis):
$$y = \begin{cases} 1, & \text{if } \text{target} > 0 \text{ (clinically significant CAD present)}, \\ 0, & \text{if } \text{target} = 0 \text{ (normal angiogram / no CAD)}. \end{cases}$$

### 4.2 Strict TRIPOD+AI Leakage Control and Preprocessing Pipeline

To enforce absolute compliance with **TRIPOD+AI reporting guidelines** and prevent subtle data leakage, dataset splitting was performed strictly prior to any feature transformation, scaling, or imputation. The raw 920-record dataframe was partitioned into a stratified 80% training set ($N=736$) and a 20% held-out test set ($N=184$) using fixed seed 43 (`train_test_split(..., test_size=0.2, stratify=Y, random_state=43)`).

```
+-----------------------------------------------------------------------------------+
|                     RAW MULTI-SITE UCI DATASET (N = 920 RECORDS)                  |
+-----------------------------------------------------------------------------------+
                                          |
                                          v  [ STRATIFIED SPLIT BEFORE IMPUTATION ]
                        +-----------------+-----------------+
                        | (80% Train, N=736)                | (20% Test, N=184)
                        v                                   v
+-----------------------------------------------------------------------------------+
|                     LEAKAGE-CONTROLLED PREPROCESSING PIPELINE                     |
|  1. Categorical Sentinel Imputation: Missing Categoricals Imputed with -1.0       |
|  2. Continuous Explicit Flags: ca_missing, chol_missing_or_zero, oldpeak_missing  |
|  3. MICE Continuous Imputation: IterativeImputer(max_iter=15) FITTED ON TRAIN     |
|  4. One-Hot Encoding & Scaling: OneHotEncoder & MinMaxScaler FITTED ON TRAIN      |
+-----------------------------------------------------------------------------------+
                                          |
                        +-----------------+-----------------+
                        |                                   |
                        v                                   v
+---------------------------------------+   +---------------------------------------+
|  TIER 1 INTAKE FEATURE SPACE (X1)     |   |  TIER 2 DIAGNOSTIC SPACE (X2)         |
|  8 Encoded Features (Vitals Only)     |   |  37 Encoded Features (Full Workup)    |
+---------------------------------------+   +---------------------------------------+

Figure 4.1: End-to-End Architectural Flowchart of the Two-Tier Staged Clinical Triage Pipeline.
```

### 4.3 Sentinel Categorical Imputation (`-1`) and Missingness Flag Generation

A critical engineering correction was implemented to resolve a fatal bug present in early exploratory drafts. Initially, missing categorical variables (`fbs`, `restecg`, `exang`, `slope`, `thal`, `ca`) were imputed using integer `1.0`. This caused severe feature collisions: imputing `1.0` was indistinguishable from true positive binary findings (e.g., Fasting Blood Sugar $>120\text{ mg/dL} = 1.0$).

In `Data_Processing_final.py`, this collision was resolved by implementing a dedicated non-colliding sentinel imputation step using **`-1`**:
```python
cat_imputer = SimpleImputer(strategy='constant', fill_value=-1)
```
When subsequently passed to `OneHotEncoder(sparse_output=False, handle_unknown='ignore')`, missing categorical values generate dedicated explicit indicator columns (e.g., `ca_-1.0`, `thal_-1.0`).

For continuous features (`age`, `trestbps`, `chol`, `thalch`, `oldpeak`), physiological zero entries in serum cholesterol (`chol == 0`, present in 172 records) were converted to `np.nan`. Explicit binary missingness indicators were generated:
```python
ca_missing = (ca == -1)
chol_missing_or_zero = np.isnan(chol)
oldpeak_missing = np.isnan(oldpeak)
```
Continuous imputation was executed using scikit-learn's Multivariate Imputation by Chained Equations (`IterativeImputer(random_state=43, max_iter=15)`) fitted strictly on the training partition. All continuous features were scaled into $[0, 1]$ using `MinMaxScaler`.

### 4.4 Feature Space Partitioning: Tier 1 Intake Space vs. Tier 2 Diagnostic Space

The resulting 37-dimensional design matrix $\Phi(X_{\mathrm{raw}}) \in \mathbb{R}^{37}$ was divided into two clinical workflow tiers:

```
========================================================================================
TABLE 4.2: COMPLETE FEATURE PARTITIONING BETWEEN TIER 1 INTAKE AND TIER 2 DIAGNOSTIC SPACE
========================================================================================
Tier Category      Dimensionality    Included Physiological & Clinical Variables
----------------------------------------------------------------------------------------
Tier 1: Intake     8 Features        - Scaled Age (continuous)
Subspace (X1)                        - Gender Indicators: gender_Female, gender_Male
                                     - Chest Pain One-Hot: cp_asymptomatic, cp_atypical,
                                       cp_non-anginal, cp_typical
                                     - Scaled Resting Blood Pressure (trestbps)
----------------------------------------------------------------------------------------
Tier 2: Diagnostic 37 Features       - Complete Tier 1 Intake Vector (8 features)
Subspace (X2)      (Full Profile)    - Laboratory Assays: Scaled Cholesterol, Fasting
                                       Blood Sugar indicators (fbs_-1, fbs_0, fbs_1)
                                     - Electrophysiology: restecg (-1, 0, 1, 2),
                                       Scaled Max Heart Rate (thalch), Exercise Angina
                                     - Stress & Angiography: ST Depression (oldpeak),
                                       Slope indicators, Fluoroscopic Vessels (ca_0 to 3),
                                       Thalassemia categories (thal_-1 to 2)
                                     - Explicit Missingness Flags: ca_missing,
                                       chol_missing_or_zero, oldpeak_missing
========================================================================================
```

### 4.5 Tier 1 Shallow Random Forest Gatekeeper Architecture

The Tier 1 Gatekeeper ($f_1: \mathbb{R}^8 \to [0, 1]$) was engineered as a regularized Random Forest classifier:
```python
tier1_model = RandomForestClassifier(
    n_estimators=600,
    max_depth=4,
    min_samples_split=2,
    random_state=43
)
```
Tree depth was explicitly bounded at `max_depth=4` to prevent high-variance decision trees from overfitting on low-dimensional basic vitals, ensuring smooth posterior risk surfaces across primary triage inputs.

### 4.6 Tier 2 Heterogeneous Soft-Voting Diagnostic Ensemble Architecture

The Tier 2 Diagnostic Model ($f_2: \mathbb{R}^{37} \to [0, 1]$) was engineered as a heterogeneous soft-voting ensemble (`VotingClassifier(voting='soft')`) combining four functionally diverse estimators:
1. **`rf_best`:** `RandomForestClassifier(n_estimators=200, max_depth=12, min_samples_split=10, random_state=7)` — Captures deep non-linear feature interactions.
2. **`rf_current`:** `RandomForestClassifier(n_estimators=1000, max_depth=5, min_samples_split=10, random_state=43)` — Provides regularized broad ensemble consensus.
3. **`logreg`:** `LogisticRegression(max_iter=5000)` — Enforces stable linear hyperplanes across correlated blood assays.
4. **`knn7`:** `KNeighborsClassifier(n_neighbors=7)` — Incorporates local non-parametric instance similarity across patient clusters.

By averaging soft predicted probability distributions across linear parametric, tree-ensemble, and instance-based architectures, the voting classifier mitigates single-model variance across multi-site datasets.

### 4.7 Triage Routing Policies and Mathematical Gate Formulation

Let $p_1 = f_1(X_1) \in [0, 1]$ represent the gatekeeper posterior probability. The system establishes lower and upper triage thresholds:
$$\theta_L = 0.30, \qquad \theta_H = 0.70.$$

Two routing policies were mathematically formalized:

#### A. Conservative Application Policy ($r_{\mathrm{app}}$ — Deployed in `app.py`)
Designed for emergency safety, where all patients with intermediate or high risk undergo complete laboratory testing before diagnostic sign-off:
$$r_{\mathrm{app}}(p_1) = \begin{cases} \text{Stop at Tier 1 (Safe Discharge Review / Avoid Tier 2)}, & \text{if } p_1 \le \theta_L \; (0.30), \\ \text{Request Complete Tier 2 Diagnostic Profile}, & \text{if } p_1 > \theta_L \; (0.30). \end{cases}$$

#### B. Uncertainty-Band Policy ($r_{\mathrm{band}}$ — Cost-First Ablation)
Designed for theoretical resource maximization, where both confidently low-risk and confidently high-risk patients bypass Tier 2 testing:
$$r_{\mathrm{band}}(p_1) = \begin{cases} \text{Low-Risk Bypass (Avoid Tier 2)}, & \text{if } p_1 \le \theta_L \; (0.30), \\ \text{Request Complete Tier 2 Diagnostic Profile}, & \text{if } \theta_L < p_1 < \theta_H, \\ \text{High-Risk Escalation Bypass (Avoid Tier 2)}, & \text{if } p_1 \ge \theta_H \; (0.70). \end{cases}$$

To quantify resource stewardship without assigning arbitrary monetary costs, the primary efficiency metric is **Tier 2 Profile Avoidance ($A_{T2}$)**:
$$A_{T2} = \frac{N - N_{T2}}{N} \times 100\%,$$
where $N_{T2}$ is the count of patients routed to undergo Tier 2 evaluation.

### 4.8 Evaluation Framework: Cross-Validation, DCA, Calibration, and Safety Accounting

To audit performance across both held-out and cross-validated settings, `publication_evaluation.py` implements:
- **Exhaustive Discrimination Accounting:** Computes Accuracy, Balanced Accuracy, Sensitivity ($\frac{\mathrm{TP}}{\mathrm{TP}+\mathrm{FN}}$), Specificity ($\frac{\mathrm{TN}}{\mathrm{TN}+\mathrm{FP}}$), Matthews Correlation Coefficient ($\mathrm{MCC}$), F1-score, and AUROC.
- **Bypass Safety Accounting:** For patients bypassing Tier 2 ($p_1 \le \theta_L$), computes Bypass Negative Predictive Value ($\mathrm{NPV} = \frac{\mathrm{TN}_{\mathrm{bypass}}}{\mathrm{TN}_{\mathrm{bypass}} + \mathrm{FN}_{\mathrm{bypass}}}$) and total Missed Positive cases ($\mathrm{FN}_{\mathrm{bypass}}$).
- **Decision Curve Analysis (DCA):** Computes Vickers Net Benefit across threshold probabilities $p_t \in [0.05, 0.50]$:
  $$\mathrm{NB}(p_t) = \frac{\mathrm{TP}}{N} - \frac{\mathrm{FP}}{N}\left(\frac{p_t}{1-p_t}\right).$$
- **Probability Calibration Audit:** Evaluates Expected Calibration Error across $K=10$ equal-width probability bins:
  $$\mathrm{ECE} = \sum_{k=1}^{K} \frac{|B_k|}{N} \left| \bar{p}_k - \bar{y}_k \right|,$$
  where $\bar{p}_k$ is mean predicted probability and $\bar{y}_k$ is empirical prevalence in bin $B_k$.

---

## CHAPTER 5: IMPLEMENTATION AND INTERFACE

### 5.1 Object-Oriented Backend Engineering (`ModelHandler` and Pipeline Encapsulation)
The backend architecture is encapsulated inside `Model_handler_final.py` through the object-oriented `ModelHandler` class. Upon initialization, `ModelHandler` verifies the local filesystem for pre-compiled binary model artifacts (`Tier_1_model.pkl`, `Tier_2_model.pkl`). If artifacts are missing, it automatically invokes `Data_Processing_final.py`, executes training across the 736-record training split, logs model convergence, and persists serialized weights.

### 5.2 Offline Evaluation Engine Engineering (`publication_evaluation.py`)
To ensure publication reproducibility without polluting deployed dashboard code, `publication_evaluation.py` was constructed as an offline auditing suite. It loads `x_train_tier1`, `x_test_tier1`, `x_train_tier2`, and `x_test_tier2` and exports comprehensive publication CSV tables (`heldout_performance.csv`, `bypass_safety.csv`, `threshold_sweep.csv`, `decision_curve.csv`, `heldout_subgroup_and_site_metrics.csv`) accompanied by a cryptographic JSON manifest (`manifest.json`).

### 5.3 Local SHAP Attribution Engine and LLM Translation Layer
When a patient undergoes Tier 2 evaluation, `Ai_explainer_shap.py` and `app.py` instantiate a permutation-based Shapley explainer around the soft-voting ensemble:
```python
explainer = shap.Explainer(predict_positive, background_data, algorithm="permutation")
```
Local additive attributions $\phi_i(x)$ are extracted such that $\sum_{i=1}^{37} \phi_i(x) = f_2(x) - \phi_0$. The top risk-raising ($\phi_i > 0$) and protective ($\phi_i < 0$) features are structured into a constrained prompt submitted to `google/gemini-2.5-flash` via OpenRouter. To guarantee robustness against API downtime, a deterministic local rule-based fallback generator produces standardized clinical notes whenever network latency exceeds timeout thresholds.

### 5.4 Interactive Streamlit Clinical Decision Support Dashboard Architecture
The clinician dashboard (`app.py`, 1,844 lines) implements the four-stage triage workflow:

```
+-----------------------------------------------------------------------------------+
|                        SMART CLINIC ASSISTANT DASHBOARD                           |
|       [ STAGED TRIAGE INTAKE ] -> [ GATEKEEPER DECISION ] -> [ WORKUP ]           |
+-----------------------------------------------------------------------------------+
|  TIER 1 INTAKE VITALS:                        TRIAGE ROUTING GATE OUTPUT:         |
|  - Patient Age: [ 58 ]                        +---------------------------------+ |
|  - Biological Sex: [ Male ]                   | TIER 1 POSTERIOR RISK: 24.1%    | |
|  - Chest Pain: [ Asymptomatic ]               | STATUS: LOW-RISK CANDIDATE      | |
|  - Resting BP: [ 135 mmHg ]                   | RECOMMENDATION: DISCHARGE REVIEW| |
|  [ RUN INITIAL TRIAGE GATE ]                  +---------------------------------+ |
+-----------------------------------------------------------------------------------+

Figure 5.1: Streamlit Dashboard Architecture: Stage 1 Intake Form and Tier 1 Gatekeeper Card.
```

### 5.5 Reactive UI State Machine and Dynamic Diagnostic Workup Unlocking
When $p_1 > 0.30$, Streamlit automatically unlocks Stage 3 (Diagnostic Workup Input) and Stage 4 (Deep Explainability Dashboard).

```
+-----------------------------------------------------------------------------------+
|                   TIER 2 DIAGNOSTIC & EXPLAINABILITY WORKSPACE                    |
+-----------------------------------------------------------------------------------+
|  SHAP ATTRIBUTION IMPACT BARS:                CLINICIAN DRAFT NOTE (LLM):         |
|  Fluoroscopic Vessels (ca=2) [ +0.185 ] (RED) +---------------------------------+ |
|  Thalassemia: Reversable     [ +0.142 ] (RED) | [! CLINICIAN REVIEW MANDATORY ] | |
|  Max Heart Rate (145 bpm)    [ -0.064 ] (GRN) | The patient shows elevated risk | |
|  Resting ECG: Normal         [ -0.032 ] (GRN) | driven by multi-vessel narrowing| |
|                                               | and reversable defect...        | |
|  [ DOWNLOAD PDF REPORT ]                      +---------------------------------+ |
+-----------------------------------------------------------------------------------+

Figure 5.2: Streamlit Dashboard Architecture: Dynamic Tier 2 Laboratory Panel and Explanation Workspace.
```

---

## CHAPTER 6: RESULTS AND ANALYSIS

### 6.1 Primary Generalization Benchmark: Stratified 5-Fold Cross-Validation

To ensure methodological rigor and avoid single-split optimism, the primary generalization benchmark was evaluated across the complete 920-record dataset via stratified 5-fold cross-validation:

```
========================================================================================
TABLE 6.1: PRIMARY STRATIFIED 5-FOLD CROSS-VALIDATION GENERALIZATION PERFORMANCE (N=920)
========================================================================================
Model / Routing Policy             Mean Accuracy     Mean AUROC      Tier 2 Avoided
----------------------------------------------------------------------------------------
Tier 1 Gatekeeper Only            77.7% ± 2.7%      0.843 ± 0.036         --
Tier 2 Full Diagnostic Profile    84.3% ± 1.8%      0.904 ± 0.035        0.0%
Conservative Cascade (App Policy) 84.5% ± 2.2%      0.898 ± 0.034       25.1% ± 3.3%
Uncertainty-Band Policy           82.6% ± 2.6%      0.854 ± 0.042       72.4% ± 2.5%
========================================================================================
```

Under 5-fold cross-validation, the deployed **Conservative Cascade** achieved an accuracy of **84.5% ± 2.2%** and an AUROC of **0.898 ± 0.034**, statistically matching the full Tier 2 model ($84.3\% \pm 1.8\%$) while avoiding **25.1% ± 3.3%** of complete Tier 2 diagnostic workups.

### 6.2 Development Split Performance Benchmark ($N=184$)

```
========================================================================================
TABLE 6.2: HELD-OUT DEVELOPMENT SPLIT BENCHMARK ACROSS STANDALONE AND CASCADES (N=184)
========================================================================================
Model or Policy             Acc.    Bal.Acc. Sens.   Spec.   MCC    AUROC   Brier  FER
----------------------------------------------------------------------------------------
Tier 1 Only                 0.788   0.780    0.853   0.707  0.569   0.844   0.153  --
Tier 2 Full Profile         0.902   0.895    0.961   0.829  0.805   0.920   0.107  0.0%
Conservative Cascade        0.897   0.890    0.951   0.829  0.793   0.922   --    24.5%
Uncertainty-Band Policy     0.864   0.854    0.951   0.756  0.730   0.874   --    66.8%
========================================================================================
```

On the fixed 184-record held-out development split, exact McNemar paired testing between the full Tier 2 model and Conservative Cascade yielded $p=1.000$ (1 discordant case). In formal reporting, **84.5%** serves as the definitive CV generalization estimate, while **89.7%** represents the held-out development check.

### 6.3 Resource Stewardship and Tier 2 Profile Avoidance Analysis

The deployed Conservative Cascade ($\theta_L=0.30$) consistently avoided **25.1%** of diagnostic profiles across cross-validation folds. In hospital operational terms, routing one quarter of presenting emergency patients to immediate safe discharge review without ordering downstream biochemical assays or nuclear imaging represents substantial stewardship of diagnostic capacity.

### 6.4 Clinical Bypass Safety Audit and Low-Threshold Sensitivity Sweep

To audit the safety of the pragmatic $\theta_L=0.30$ discharge threshold, an exhaustive threshold-sweep accounting audit was executed on the held-out split:

```
========================================================================================
TABLE 6.3: HELD-OUT LOW-THRESHOLD SWEEP AND BYPASS SAFETY ACCOUNTING (N=184)
========================================================================================
Low Threshold (θL)  Tier 2 Avoided   Bypass Count (N)  Bypassed Positives  Missed Pos.
----------------------------------------------------------------------------------------
0.05                    1.1%                2                  0                0
0.10                    4.9%                9                  0                0
0.15                    9.8%               18                  0                0
0.20                   15.8%               29                  1                1
0.25                   22.3%               41                  3                3
0.30 (Deployed App)    24.5%               45                  4                4
0.40                   30.4%               56                  8                8
0.50                   39.7%               73                 15               15
========================================================================================
```

At the deployed **0.30 threshold**, 45 patients bypassed Tier 2 testing. Crucially, **4 of these 45 patients were true positive cases**, resulting in a bypass Negative Predictive Value (NPV) of **91.1%**. In emergency cardiac triage, a 91.1% NPV is clinically unsafe as an autonomous discharge rule. Achieving **zero missed positive cases (100% NPV)** on this held-out cohort requires lowering the threshold to **`0.15`**, which reduces profile avoidance from 24.5% to **9.8%**. This empirical trade-off proves why heuristic probability thresholds cannot be deployed autonomously without clinical utility calibration.

### 6.5 Probability Calibration Audit and Reliability Diagram Analysis

```
       Observed
          1.0 +--------------------------------------------------*--+
              |                                               *-/   |
          0.8 |                                            *-/      |
              |                                         *-/         |
          0.6 |                                      *-/            |
              |                           *-------*--               |
          0.4 |                        *-/                          |
              |              *------*-/   Raw Tier 1 Curve          |
          0.2 |        *----/                                       |
              |  *----/                                             |
          0.0 +--*--------------------------------------------------+
              0.0                   0.5                             1.0
                                 Predicted Probability

Figure 6.2: Probability Calibration Reliability Histogram: Raw vs. Isotonic Calibrated Tier 1 Probabilities.
```

Probability calibration auditing revealed that raw Tier 1 probabilities exhibited an expected calibration error (ECE-10) of `0.052` (Brier score `0.153`). Applying 5-fold Isotonic Regression slightly improved ECE-10 to `0.048` but degraded Brier score (`0.157`) and AUROC (`0.840`) on small held-out splits. Consequently, raw probabilities were retained for application routing while underscoring that prospective calibration against clinical utility cost matrices is mandatory.

### 6.6 Decision Curve Analysis (Net Benefit Evaluation)

```
       Net Benefit
         0.55 +-------------Tier 2 / Cascade-----------------------+
              |           /---**                                   |
         0.45 |         /----   ***---                             |
              |       /--             ***--                        |
         0.35 |     /--                    ***---   Treat All      |
              |   /--                            ***--             |
         0.25 | /--                                   ***--        |
              |/                                           ***     |
         0.00 +-----------------------------------------------****-+ Treat None
             0.05           0.15           0.25           0.35           0.45
                                 Threshold Probability (pt)

Figure 6.1: Decision Curve Analysis (DCA): Net Benefit vs. Clinical Threshold Probability.
```

Decision Curve Analysis (DCA) confirmed that both the full Tier 2 model and the Conservative Cascade achieved superior net benefit across clinical threshold probabilities $p_t \in [0.05, 0.45]$ compared to default treat-all and treat-none strategies.

### 6.7 Institutional Generalizability Audit: Leave-One-Site-Out (LOSO) Stress Testing

To test whether high random-split accuracy masked vulnerability to institutional shift across hospitals, **Leave-One-Site-Out (LOSO)** cross-validation was executed across the four source medical centers:

```
========================================================================================
TABLE 6.4: LEAVE-ONE-SITE-OUT (LOSO) STRESS TESTING AND MAJORITY-CLASS BASELINE COMPARISON
========================================================================================
Held-Out Hospital Cohort     Records (N) Prevalence Maj.Base Acc.  Sens.  Spec.  MCC
----------------------------------------------------------------------------------------
Cleveland Clinic (USA)          304        0.457      0.543  0.783  0.820  0.752  0.570
Hungarian Institute             293        0.362      0.638  0.799  0.896  0.743  0.615
Switzerland Hospital            123        0.935      0.935  0.813  0.826  0.625  0.276
VA Long Beach Medical Center    200        0.745      0.745  0.730  0.779  0.588  0.344
----------------------------------------------------------------------------------------
Pooled LOSO Total               920        0.553      0.553  0.780  0.825  0.725  0.554
========================================================================================
```

The LOSO stress test revealed a critical scientific finding: pooled generalizability accuracy fell from $84.5\%$ (random CV) to **78.0%**. Most alarmingly, at the two high-prevalence sites—**Switzerland (93.5% prevalence)** and **VA Long Beach (74.5% prevalence)**—the conservative cascade achieved accuracies of **81.3%** and **73.0%**, respectively, falling **below trivial institutional majority-class rules** ($93.5\%$ and $74.5\%$). This audit proves that institutional shift and base-rate imbalance profoundly degrade uncalibrated clinical classifiers.

### 6.8 Exploratory Subgroup Fairness Audit across Biological Sex and Age Bands

```
========================================================================================
TABLE 6.5: EXPLORATORY HELD-OUT SUBGROUP AUDIT DISAGGREGATED BY BIOLOGICAL SEX AND AGE
========================================================================================
Subgroup Cohort        Records (N) Prevalence  Accuracy  Sensitivity  Specificity  MCC
----------------------------------------------------------------------------------------
Female Subgroup            38        0.263      0.974       0.900        1.000    0.932
Male Subgroup             146        0.630      0.877       0.957        0.741    0.734
Age < 50 Years             49        0.388      0.959       0.895        1.000    0.916
Age 50 - 65 Years         115        0.609      0.887       0.971        0.756    0.765
Age > 65 Years             20        0.650      0.800       0.923        0.571    0.545
========================================================================================
```

Exploratory subgroup auditing showed lower specificity (`0.571`) and MCC (`0.545`) in elderly patients ($>65$ years). Because subgroup sample sizes ($N=38$ females, $N=20$ elderly) are severely underpowered, these results are reported strictly for scientific transparency rather than claiming demographic fairness.

### 6.9 End-to-End Deep Explainability and Draft Generative Output Walkthrough

When a high-risk patient is evaluated, `app.py` renders local SHAP attributions alongside the following structured draft summary:

> **[! CLINICIAN REVIEW MANDATORY: DRAFT AI COMMUNICATION SUMMARY ONLY ]**  
> **Clinical Risk Profile Summary:** The patient presents with an elevated probability of cardiovascular disease ($p_2 = 0.842$).  
> **Dominant Attributions:** Risk is driven predominantly by multi-vessel fluoroscopic narrowing (`ca=2`, $\phi=+0.185$) and reversable thalassemia defect (`thal=2`, $\phi=+0.142$), partially mitigated by normal resting ECG (`restecg=0`, $\phi=-0.032$).  
> **Recommendation:** Advise cardiology consult for formal angiographic correlation. This note is unvalidated draft assistance and must be verified by the attending clinician.

---

## CHAPTER 7: CONCLUSION AND FUTURE WORK

### 7.1 Summary of Engineering and Methodological Contributions

During this six-week research internship at **CSIR-CSIO**, **Smart Clinic Assistant** successfully demonstrated the feasibility of resource-aware staged feature acquisition for cardiovascular triage. The project delivered:
1. A leakage-controlled Python preprocessing pipeline enforcing TRIPOD+AI partitioning, non-colliding categorical sentinel imputation (`-1`), and MICE continuous imputation.
2. An auditable two-tier ensemble architecture achieving **84.5% ± 2.2% cross-validated accuracy** and **0.898 ± 0.034 AUROC** while avoiding **25.1% ± 3.3% of complete diagnostic profiles**.
3. A rigorous post-hoc auditing framework exposing the clinical limitations of uncalibrated thresholds, revealing that the pragmatic **0.30 threshold** bypasses positive cases (91.1% NPV) and that multi-site LOSO accuracy drops to **78.0%**.
4. An interactive Streamlit decision support interface coupling permutation SHAP attributions with draft LLM communication notes.

### 7.2 Rigorous Accounting of Methodological and Clinical Limitations

To maintain absolute institutional honesty, the following limitations preclude deployment readiness:
- **Retrospective Public Data:** The analysis relies on a retrospective 920-record dataset originating from 1988 with significant inter-hospital prevalence shift.
- **Unsafe Pragmatic Thresholds:** The 0.30 bypass threshold missed 4 positive cases among 45 held-out patients; achieving 100% NPV required lowering the threshold to 0.15, reducing profile avoidance to 9.8%.
- **Unvalidated Generative AI:** Generated clinical summaries were not formally evaluated by clinician panels for faithfulness, readability, or hallucination rates.
- **Underpowered Subgroups:** Small subgroup sample sizes ($N=20$ for age $>65$, $N=38$ for females) prevent conclusive demographic fairness claims.

### 7.3 Roadmap for Clinical Integration and Future Scope at CSIR-CSIO

Prior to prospective clinical integration, future work at **CSIR-CSIO** must address four technical milestones:
1. **Utility-Optimized Threshold Calibration:** Deriving routing gates from formal clinical cost-loss matrices rather than heuristic probability bands.
2. **Prospective Multi-Center External Validation:** Conducting external validation on modern EMR cohorts across geographically diverse Indian medical institutes.
3. **Formal Blinded LLM Clinical Scoring:** Executing structured Likert-scale audits evaluating generated clinical text against clinician gold-standard notes.
4. **On-Premise Privacy & EMR Governance:** Replacing third-party public API endpoints with secure, locally deployed open-weight medical language models adhering to hospital HIPAA/DISHA data governance.

---

## REFERENCES

1. A. Janosi, W. Steinbrunn, M. Pfisterer, and R. Detrano, "Heart Disease," *UCI Machine Learning Repository*, 1988, doi: 10.24432/C52P4X.
2. R. Detrano *et al.*, "International application of a new probability algorithm for the diagnosis of coronary artery disease," *American Journal of Cardiology*, vol. 64, no. 5, pp. 304–310, 1989.
3. S. Raman *et al.*, "Machine learning for coronary heart disease prediction: Comparative analysis of Framingham and Cleveland subset of the UCI dataset with SHAP-based interpretability," *Epidemiologia*, vol. 7, no. 3, Art. no. 75, 2026.
4. G. S. Collins *et al.*, "TRIPOD+AI statement: Updated guidance for reporting clinical prediction models that use regression or machine learning methods," *BMJ*, vol. 385, Art. no. e078378, 2024.
5. S. G. Finlayson *et al.*, "The clinician and dataset shift in artificial intelligence," *New England Journal of Medicine*, vol. 385, no. 3, pp. 283–286, 2021.
6. A. J. Vickers and E. B. Elkin, "Decision curve analysis: A novel method for evaluating prediction models," *Medical Decision Making*, vol. 26, no. 6, pp. 565–574, 2006.
7. S. M. Lundberg and S.-I. Lee, "A unified approach to interpreting model predictions," in *Proc. Advances in Neural Information Processing Systems (NeurIPS)*, 2017, pp. 4765–4774.
8. S. Tonekaboni *et al.*, "What clinicians want: Contextualizing explainable machine learning for clinical end use," in *Proc. Machine Learning for Healthcare (ML4H)*, PMLR, 2019, pp. 359–380.
9. B. Kompa, J. Snoek, and A. L. Beam, "Second opinion needed: communicating uncertainty in medical machine learning," *npj Digital Medicine*, vol. 4, Art. no. 4, 2021.
10. K. Singhal *et al.*, "Large language models encode clinical knowledge," *Nature*, vol. 620, pp. 172–180, 2023.
11. Z. Ji *et al.*, "Survey of hallucination in natural language generation," *ACM Computing Surveys*, vol. 55, no. 12, Art. no. 248, 2023.
12. U.S. Food and Drug Administration, "Clinical Decision Support Software: Guidance for Industry and Food and Drug Administration Staff," Final Guidance, Sep. 2022.
13. P. Viola and M. Jones, "Rapid object detection using a boosted cascade of simple features," in *Proc. IEEE Computer Society Conference on Computer Vision and Pattern Recognition (CVPR)*, 2001.
14. R. Mozannar and D. Sontag, "Consistent estimators for learning to defer to an expert," in *Proc. International Conference on Machine Learning (ICML)*, PMLR, 2020, pp. 7076–7087.
15. C. Guo, G. Pleiss, Y. Sun, and K. Q. Weinberger, "On calibration of modern neural networks," in *Proc. International Conference on Machine Learning (ICML)*, PMLR, 2017, pp. 1321–1330.

</div>
