\documentclass[conference]{IEEEtran}

\usepackage{cite}
\usepackage{amsmath,amssymb}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{array}
\usepackage{url}
\usepackage{xcolor}
\usepackage{tikz}
\usetikzlibrary{arrows.meta,positioning,shapes.geometric}

\hyphenation{Open-Router cardio-vascular}

\begin{document}

\title{Safety-Audited Staged Feature Acquisition for Retrospective Heart Disease Triage: A Two-Tier Workflow Simulation With Threshold Sensitivity and Site-Held-Out Stress Testing}

\author{
\IEEEauthorblockN{Sparsh Chopra}
\IEEEauthorblockA{
Chandigarh College of Engineering and Technology\\
Chandigarh, India\\
Email: co24365@ccet.ac.in}
}

\maketitle

\begin{abstract}
Machine-learning studies on heart disease prediction often evaluate a complete diagnostic feature vector as if all variables were available at first contact. This assumption is convenient for benchmarking but weak for triage, where intake variables are available immediately and downstream diagnostic variables may require additional time, cost, or infrastructure. This paper evaluates Smart Clinic Assistant as a retrospective staged-acquisition simulation on a 920-record multi-site UCI Heart Disease data file. Tier 1 used age, sex, chest pain type, and resting blood pressure. Tier 2 used a 37-feature encoded diagnostic profile. The primary evidence is stratified 5-fold cross-validation and leave-one-site-out testing; a fixed 184-record held-out split is used only for development audits. In cross-validation, the implemented conservative cascade achieved $84.5\%\pm2.2\%$ accuracy and $0.898\pm0.034$ AUROC while avoiding $25.1\%\pm3.3\%$ of Tier 2 profiles. The full Tier 2 model achieved $84.3\%\pm1.8\%$ accuracy and $0.904\pm0.035$ AUROC. Leave-one-site-out evaluation reduced pooled accuracy to 78.0\%, and the model fell below a trivial majority-class accuracy baseline at two high-prevalence sites. On the held-out split, the implemented 0.30 low-risk threshold avoided 24.5\% of Tier 2 profiles but bypassed 4 positive cases among 45 bypassed patients. A stricter 0.15 threshold avoided 9.8\% of profiles with no bypassed positives on this held-out split. These calibration, threshold-sweep, bypass-safety, and decision-curve audits support staged feature acquisition as a workflow-auditing case study, not as a validated discharge rule, diagnostic device, or deployment-ready clinical risk score.
\end{abstract}

\begin{IEEEkeywords}
clinical decision support, heart disease prediction, staged feature acquisition, selective classification, calibration, decision curve analysis
\end{IEEEkeywords}

\section{Introduction}
Cardiovascular prediction has long been used as a benchmark problem for statistical and machine-learning decision support. The UCI Heart Disease repository remains one of the most reused public cardiac datasets, with roots in the angiographic diagnostic work of Detrano \emph{et al.} \cite{uci_heart,detrano1989}. This benchmark tradition sits inside a broader clinical-AI literature arguing that predictive systems must be evaluated in relation to their intended workflow, not only by aggregate discrimination on convenient retrospective splits \cite{rajkomar2019,topol2019,kelly2019}. Static benchmarking is useful, but it can also hide an important operational assumption: many studies model heart disease prediction after all variables are already known.

That assumption is not neutral. Intake variables such as age, sex, chest pain description, and resting blood pressure are usually available early in triage. Variables such as thallium stress-test category, number of major vessels, electrocardiographic findings, and laboratory results may require downstream testing, specialized staff, equipment, or additional waiting time. A full-vector model can therefore have attractive statistical performance while still being inefficient as an initial triage workflow.

Prior cardiovascular machine-learning studies show that conventional models can achieve strong discrimination when complete variables are supplied \cite{weng2017,motwani2017,raman2026}. Raman \emph{et al.}, for example, evaluated Framingham data and the Cleveland subset of the UCI dataset with SHAP-based interpretation, reporting strong full-feature performance on the Cleveland subset \cite{raman2026}. Gradient-boosted tree methods such as XGBoost have become the de facto standard for structured clinical tabular data \cite{chen2016}, and their performance on heart disease benchmarks provides an important reference point for Tier 2 model design. The present paper asks a narrower and different question: whether a fixed, clinically interpretable staged feature split can preserve much of full-profile performance while reducing the number of patients for whom a complete diagnostic profile is requested. We position this explicitly as a workflow-auditing feasibility study rather than an algorithmic contribution, because the staged routing architecture deliberately employs a fixed threshold policy instead of a learned acquisition, deferral, or reject-option classifier \cite{cortes2016,geifman2019,mozannar2020}.

The second issue is explanation delivery. Explainable-AI methods such as SHAP expose local feature contributions \cite{lundberg2017}, but local attribution values alone do not establish clinical usefulness. Human-centered clinical AI research emphasizes workflow fit, uncertainty, and clinician oversight \cite{tonekaboni2019}. In this study, local attributions are used inside a prototype interface, and the optional language-generation component is treated strictly as draft communication requiring clinician review. It is not evaluated as a clinical text-generation method and is excluded from the quantitative claims.

This paper makes five limited contributions:
\begin{itemize}
    \item it evaluates a fixed two-tier cardiac triage simulation separating low-cost intake variables from downstream diagnostic variables;
    \item it compares conservative and cost-first routing policies against Tier 1-only and full-profile baselines;
    \item it reports cross-validation, held-out, and leave-one-site-out results to distinguish favorable random-split performance from site-shift behavior;
    \item it audits the routing gate using bypass safety, calibration, threshold sweep, and decision-curve analyses;
    \item it reports exploratory subgroup results to show where the evidence is underpowered rather than to claim fairness.
\end{itemize}
The contribution is therefore a reproducible workflow simulation and safety audit, not a new machine-learning algorithm, calibrated clinical risk score, validated explanation generator, or deployable medical device. Fig.~\ref{fig:architecture} summarizes the evaluated workflow.

\begin{figure}[!t]
\centering
\footnotesize
\begin{tikzpicture}[
    node distance=0.46cm,
    block/.style={rectangle, rounded corners=2pt, draw=black!65, fill=blue!7, minimum height=0.58cm, text width=3.15cm, align=center},
    gate/.style={diamond, aspect=2.2, draw=black!65, fill=yellow!18, inner sep=1pt, text width=2.45cm, align=center},
    out/.style={rectangle, rounded corners=2pt, draw=black!65, fill=green!9, minimum height=0.55cm, text width=3.1cm, align=center},
    warn/.style={rectangle, rounded corners=2pt, draw=black!65, fill=red!8, minimum height=0.55cm, text width=3.1cm, align=center},
    arrow/.style={-{Latex[length=2mm]}, thick}
]
\node[block] (intake) {Tier 1 intake\\age, sex, chest pain, BP};
\node[block, below=of intake] (rf1) {Tier 1 Random Forest\\$p_1=f_1(X_1)$};
\node[gate, below=0.55cm of rf1] (gate) {Routing gate\\0.30 and 0.70};
\node[out, below left=0.65cm and -0.1cm of gate] (low) {Low-risk review\\Tier 2 avoided};
\node[block, below right=0.65cm and -0.1cm of gate] (tier2) {Tier 2 profile\\37 encoded features};
\node[block, below=of tier2] (vote) {Soft-voting ensemble\\$p_2=f_2(X_2)$};
\node[block, below=of vote] (shap) {Local attribution\\rank patient factors};
\node[warn, below=of shap] (note) {Draft explanation\\clinician review};
\draw[arrow] (intake) -- (rf1);
\draw[arrow] (rf1) -- (gate);
\draw[arrow] (gate) -- node[left]{low} (low);
\draw[arrow] (gate) -- node[right]{gray or high in app} (tier2);
\draw[arrow] (tier2) -- (vote);
\draw[arrow] (vote) -- (shap);
\draw[arrow] (shap) -- (note);
\end{tikzpicture}
\caption{Staged triage workflow evaluated in this study. The implemented application bypasses Tier 2 only for low-risk Tier 1 cases. The uncertainty-band policy, which also bypasses high-risk Tier 1 cases, is evaluated only as a cost-first ablation.}
\label{fig:architecture}
\end{figure}

\section{Related Work}
\subsection{Cardiovascular Prediction Models}
Clinical prediction models are useful only when discrimination, calibration, validation setting, and intended use are reported together \cite{steyerberg2010,collins2015,collins2024}. Machine learning has been applied to cardiovascular risk prediction using routine clinical records, imaging, and public benchmark datasets \cite{weng2017,motwani2017,raman2026}. These studies show that nonlinear and ensemble models can improve apparent performance in some settings, but external validity is often limited by cohort composition, missingness, and site-specific data-generating processes \cite{steyerberg2016,finlayson2021,kelly2019}. The present work therefore reports both random-split and leave-one-site-out results.

\subsection{Staged Acquisition, Selective Prediction, and Deferral}
Cost-sensitive learning and budgeted feature acquisition study settings in which variables have different acquisition costs \cite{nan2015,kachuee2019}. Cascade classifiers have a long history in machine learning as a way to reserve expensive computation or features for harder cases \cite{viola2001}. Selective classification and reject-option learning formalize related problems in which a model abstains rather than forcing an immediate prediction \cite{chow1970,elyaniv2010,geifman2017}. Learning-to-defer extends this idea by routing cases to another decision-maker or expert system \cite{madras2018,mozannar2020}. This paper uses a fixed two-tier split rather than a learned acquisition or deferral policy: Tier 1 is limited to immediately available intake variables, and Tier 2 contains downstream diagnostic variables. The novelty is not algorithmic; it is the explicit safety and workflow audit on a heterogeneous cardiac dataset.

\subsection{Evaluation, Calibration, and Clinical Utility}
Accuracy alone is a weak primary endpoint for clinical prediction, especially with imbalanced or site-shifted data. AUROC, sensitivity, specificity, balanced accuracy, MCC, calibration, and decision-analytic utility answer different questions \cite{steyerberg2010,brier1950}. Probability calibration is particularly important when a model output is compared with fixed thresholds \cite{niculescu2005,vancalster2019}. Decision-curve analysis evaluates clinical net benefit across threshold probabilities and is more relevant than discrimination alone when the intended use is decision support \cite{vickers2006}. This study reports these diagnostics as post-hoc audits of a fixed routing policy rather than as evidence of clinical deployment readiness.

\subsection{Missing Data, Fairness, and Explanation}
Healthcare datasets often contain informative missingness rather than random absence. Multiple imputation and missingness indicators can reduce bias, but they also require transparent reporting because missingness may encode site or workflow artifacts \cite{sterne2009}. Fairness and subgroup analysis are similarly important because aggregate performance can conceal unequal error patterns across demographic groups or care sites \cite{obermeyer2019}. For explanations, SHAP offers a principled additive attribution framework \cite{lundberg2017}, but explanation usefulness must be evaluated in clinical context \cite{tonekaboni2019}. Large language models can generate fluent medical text \cite{singhal2023}, but hallucination and unsafe omission remain concerns \cite{ji2023}. The language layer in this work is therefore scoped as an unvalidated communication prototype.

\section{Methods}
\subsection{Study Design}
This was a retrospective machine-learning simulation using public, de-identified heart disease data. No patient interaction occurred, no identifiable personal information was collected, and no intervention was performed. The study follows reporting principles from TRIPOD and TRIPOD+AI for prediction-model studies \cite{collins2015,collins2024}; risk-of-bias issues are discussed using the logic of PROBAST and clinical-AI validation guidance \cite{wolff2019,kelly2019}. Later-stage clinical AI evaluation guidelines such as CONSORT-AI, SPIRIT-AI, and DECIDE-AI are relevant to future prospective work, but this study did not conduct a clinical trial or clinical deployment evaluation \cite{liu2020,rivera2020,vasey2022}.

Four modes were evaluated:
\begin{enumerate}
    \item Tier 1 only: an intake model using basic triage variables;
    \item Tier 2 full profile: a complete diagnostic model using all processed features;
    \item conservative cascade: the implemented application policy, where low-risk Tier 1 cases bypass Tier 2 and all other cases proceed to Tier 2;
    \item uncertainty-band policy: a cost-first ablation where low-risk and high-risk Tier 1 cases bypass Tier 2 and only intermediate cases receive Tier 2.
\end{enumerate}
The primary generalization evidence was stratified 5-fold cross-validation and leave-one-site-out evaluation. The 80/20 held-out split was used as a development audit for concrete confusion matrices, calibration diagnostics, threshold sensitivity, bypass-safety accounting, and decision-curve analysis.

\subsection{Dataset}
The local data file contained 920 records and 16 columns from Cleveland, Hungary, VA Long Beach, and Switzerland. The file differs from the filtered Cleveland-only subset used in several benchmark papers \cite{raman2026}. The binary target was
\begin{equation}
y =
\begin{cases}
1, & \text{target} > 0,\\
0, & \text{target} = 0.
\end{cases}
\end{equation}
This produced 509 positive and 411 negative cases. A stratified 80/20 split produced 736 training records and 184 held-out records. Table~\ref{tab:dataset} summarizes the data. The source sites had sharply different prevalence, which can affect both apparent accuracy and calibration.

\begin{table}[!t]
\caption{Dataset Summary}
\label{tab:dataset}
\centering
\footnotesize
\renewcommand{\arraystretch}{1.12}
\begin{tabular}{lrrr}
\toprule
\textbf{Source site or split} & \textbf{Records} & \textbf{Positive} & \textbf{Prevalence}\\
\midrule
Cleveland & 304 & 139 & 0.457\\
Hungary & 293 & 106 & 0.362\\
VA Long Beach & 200 & 149 & 0.745\\
Switzerland & 123 & 115 & 0.935\\
\midrule
Full dataset & 920 & 509 & 0.553\\
Held-out development split & 184 & 102 & 0.554\\
\bottomrule
\end{tabular}
\end{table}

\subsection{Preprocessing}
Every split was created before imputation, encoding, and scaling. Continuous variables included age, resting blood pressure, cholesterol, maximum heart rate, and ST depression. Categorical or binary variables included sex, chest pain type, fasting blood sugar, resting electrocardiogram, exercise-induced angina, slope, number of major vessels, and thalassemia or stress-test category.

Cholesterol values of zero were treated as invalid and converted to missing values before continuous imputation; 172 of 920 records had cholesterol equal to zero in the local file. Missingness indicators were added for clinically meaningful absence patterns: \texttt{ca\_missing}, \texttt{chol\_missing\_or\_zero}, and \texttt{oldpeak\_missing}. The local file contained 611 missing \texttt{ca} values and 62 missing \texttt{oldpeak} values. Missing categorical values were imputed with a distinct sentinel value of $-1$ before one-hot encoding. This avoids collision with genuine positive binary values such as fasting blood sugar $=1$ or exercise-induced angina $=1$. Continuous variables were imputed with iterative chained imputation fitted on the training split and then scaled using min--max normalization \cite{sterne2009}. The final Tier 2 matrix contained 37 encoded features.

\subsection{Feature Tiers}
Tier 1 was restricted to low-cost intake variables:
\begin{equation}
X_1 = \{\text{age}, \text{sex}, \text{chest pain type}, \text{resting blood pressure}\}.
\end{equation}
After one-hot encoding, Tier 1 contained eight columns: age, two sex indicators, four chest-pain indicators, and resting blood pressure.

Tier 2 used the complete encoded diagnostic profile:
\begin{equation}
X_2 = X_1 \cup X_{\mathrm{diagnostic}},
\end{equation}
where $X_{\mathrm{diagnostic}}$ includes cholesterol, fasting blood sugar, resting electrocardiogram, maximum heart rate, exercise-induced angina, ST depression, slope, major vessels, thalassemia or stress-test category, and missingness indicators. Several Tier 2 features, especially major-vessel and thalassemia/stress-test variables, are downstream diagnostic measurements. Their downstream nature is the reason they are staged rather than treated as free intake information.

\subsection{Model Architecture}
The Tier 1 gatekeeper was a Random Forest classifier with 600 estimators, maximum depth 4, minimum samples split 2, and random state 43 \cite{breiman2001}. The Tier 2 model was a soft-voting ensemble containing two Random Forest classifiers, Logistic Regression, and KNN. The first Tier 2 Random Forest used 200 estimators, maximum depth 12, minimum samples split 10, and random state 7. The second used 1000 estimators, maximum depth 5, minimum samples split 10, and random state 43. Logistic Regression used 5000 maximum iterations, and KNN used seven neighbors.

The hyperparameters were fixed during prototype development before the final publication audit and were not tuned on the held-out split or on the publication-output tables. No nested hyperparameter optimization is claimed. This limits conclusions about the absolute best achievable accuracy, but it also prevents the held-out development split from being presented as an optimization target. Each model returned a positive-class probability:
\begin{equation}
p_1 = f_1(X_1), \qquad p_2 = f_2(X_2).
\end{equation}

\subsection{Routing Policies}
The implemented application gate used lower and upper thresholds
\begin{equation}
\theta_L = 0.30, \qquad \theta_H = 0.70.
\end{equation}
The conservative application policy was
\begin{equation}
r_{\mathrm{app}}(p_1)=
\begin{cases}
\text{stop after Tier 1}, & p_1 \leq \theta_L,\\
\text{request Tier 2}, & p_1 > \theta_L.
\end{cases}
\label{eq:appgate}
\end{equation}
The cost-first uncertainty-band ablation was
\begin{equation}
r_{\mathrm{band}}(p_1)=
\begin{cases}
\text{low-risk bypass}, & p_1 \leq \theta_L,\\
\text{request Tier 2}, & \theta_L < p_1 < \theta_H,\\
\text{high-risk bypass}, & p_1 \geq \theta_H.
\end{cases}
\label{eq:bandgate}
\end{equation}
The thresholds were pragmatic prototype bands, not calibrated clinical cutoffs. The post-hoc audit therefore evaluates how sensitive the results are to threshold movement. The deployed application behavior was not changed by the publication audit.

\subsection{Attribution and Draft Explanation Layer}
For Tier 2 cases, the application computes local attributions for the positive-class probability using a model-agnostic SHAP permutation explainer. Factors are sorted by absolute local impact:
\begin{equation}
\operatorname{Impact}_i(x)=|\phi_i(x)|.
\label{eq:impact}
\end{equation}
The interface can display the top risk-raising and risk-lowering factors. It can also pass the final risk estimate and top-ranked factors to a language model to produce a plain-language draft note. That note is not evaluated in this paper and must not be interpreted as a validated clinical explanation. The interface marks the note as machine generated and requiring clinician review, consistent with clinical decision-support oversight principles \cite{fda_cds}. Because no note-level human evaluation was performed, language generation is excluded from the quantitative claims.

\subsection{Metrics and Statistical Analysis}
Predictive performance was evaluated with accuracy, balanced accuracy, precision, recall, specificity, F1-score, MCC, AUROC, and confusion matrices. Brier score is reported only for stand-alone Tier 1 and Tier 2 probability models. It is not used to rank cascaded policies because those policies mix probabilities from different model stages. Tier 2 profile avoidance was measured as
\begin{equation}
A_{T2} = \frac{N - N_{T2}}{N},
\end{equation}
where $N$ is the number of evaluated patients and $N_{T2}$ is the number routed to Tier 2. This is a profile-avoidance metric, not a dollar-cost estimate.

For cascaded policies, each patient was assigned the probability from the stage that determined the reported decision. In the conservative cascade, low-risk bypassed patients retained $p_1$, and Tier 2-routed patients used $p_2$. In the uncertainty-band policy, low-risk and high-risk bypassed patients retained $p_1$, while intermediate patients used $p_2$. These mixed-stage scores are used for AUROC only. Brier scores for mixed-stage cascades are intentionally omitted because proper probability scoring across two differently calibrated score sources is not directly interpretable.

Decision-curve net benefit was computed as
\begin{equation}
\mathrm{NB}(p_t)=\frac{\mathrm{TP}}{N}-
\frac{\mathrm{FP}}{N}\cdot\frac{p_t}{1-p_t},
\end{equation}
where $p_t$ is the decision threshold \cite{vickers2006}. Calibration was summarized using Brier score, 10-bin expected calibration error (ECE), and reliability bins. Bypass safety was reported as the number of true positive cases in the low-risk bypass branch. Because this is a triage setting, the presence of bypassed positives is interpreted as a safety failure for any discharge-style use.

\section{Results}
\subsection{Held-Out Development Split}
Table~\ref{tab:heldout} summarizes the 184-record held-out development split. The full Tier 2 model achieved 90.2\% accuracy, 0.895 balanced accuracy, 0.805 MCC, and 0.920 AUROC. The conservative cascade achieved 89.7\% accuracy, 0.890 balanced accuracy, 0.793 MCC, and 0.922 AUROC while avoiding 24.5\% of Tier 2 profiles. The uncertainty-band policy achieved 86.4\% accuracy and avoided 66.8\% of Tier 2 profiles. These held-out point estimates are reported as development evidence, not as the primary generalization estimate.

\begin{table*}[!t]
\caption{Held-Out Development Performance}
\label{tab:heldout}
\centering
\footnotesize
\renewcommand{\arraystretch}{1.12}
\begin{tabular}{lrrrrrrrrr}
\toprule
\textbf{Model or policy} & \textbf{Acc.} & \textbf{Bal.} & \textbf{Prec.} & \textbf{Sens.} & \textbf{Spec.} & \textbf{F1} & \textbf{MCC} & \textbf{AUROC} & \textbf{Brier}\\
\midrule
Tier 1 only & 0.788 & 0.780 & 0.784 & 0.853 & 0.707 & 0.817 & 0.569 & 0.844 & 0.153\\
Tier 2 full profile & 0.902 & 0.895 & 0.875 & 0.961 & 0.829 & 0.916 & 0.805 & 0.920 & 0.107\\
Conservative cascade & 0.897 & 0.890 & 0.874 & 0.951 & 0.829 & 0.911 & 0.793 & 0.922 & --\\
Uncertainty-band policy & 0.864 & 0.854 & 0.829 & 0.951 & 0.756 & 0.886 & 0.730 & 0.874 & --\\
\bottomrule
\end{tabular}
\end{table*}

The held-out confusion matrices were: Tier 1, TN=58, FP=24, FN=15, TP=87; Tier 2, TN=68, FP=14, FN=4, TP=98; conservative cascade, TN=68, FP=14, FN=5, TP=97; and uncertainty-band policy, TN=62, FP=20, FN=5, TP=97. Exact McNemar testing between Tier 2 and the conservative cascade produced $p=1.000$, but this comparison had only one discordant pair and should not be interpreted as evidence of equivalence.

\subsection{Bypass Safety and Threshold Sensitivity}
The implemented 0.30 low-risk threshold routed 45 held-out patients to the bypass branch. Four of those 45 patients were positive cases, yielding a bypass NPV of 91.1\%. This is not adequate evidence for discharge or rule-out use. The threshold sweep in Table~\ref{tab:threshold} shows why profile avoidance and bypass safety must be reported together. On this held-out split, a 0.15 threshold avoided 18 of 184 profiles (9.8\%) and had no bypassed positives. At 0.20, the first bypassed positive appeared.

\begin{table}[!t]
\caption{Held-Out Low-Threshold Sweep for Conservative Routing}
\label{tab:threshold}
\centering
\scriptsize
\renewcommand{\arraystretch}{1.12}
\begin{tabular}{rrrrr}
\toprule
\textbf{$\theta_L$} & \textbf{T2 avoided} & \textbf{Bypass N} & \textbf{Bypass pos.} & \textbf{Missed pos.}\\
\midrule
0.05 & 1.1\% & 2 & 0 & 0\\
0.10 & 4.9\% & 9 & 0 & 0\\
0.15 & 9.8\% & 18 & 0 & 0\\
0.20 & 15.8\% & 29 & 1 & 1\\
0.25 & 22.3\% & 41 & 3 & 3\\
0.30 & 24.5\% & 45 & 4 & 4\\
0.40 & 30.4\% & 56 & 8 & 8\\
0.50 & 39.7\% & 73 & 15 & 15\\
0.60 & 47.8\% & 88 & 24 & 15\\
\bottomrule
\end{tabular}
\end{table}

Thus, the 25\% profile-avoidance result is best interpreted as an efficiency-oriented prototype operating point, not as a clinically safe rule-out threshold. If zero bypassed positives is required on this held-out audit, the empirically observed threshold is 0.15, with substantially lower profile avoidance. This trade-off is the central safety finding of the paper.

\subsection{Decision-Curve and Calibration Audits}
Fig.~\ref{fig:dca} shows held-out decision-curve results for the full Tier 2 model and conservative cascade. Both models had higher net benefit than treat-all and treat-none strategies across the plotted threshold range. The cascade was close to the full Tier 2 model at several thresholds, but no confidence bands were estimated; the figure is therefore an audit of apparent utility on the held-out split rather than a definitive clinical utility claim.

\begin{figure}[!t]
\centering
\footnotesize
\begin{tikzpicture}[x=7.0cm,y=6.0cm]
\draw[->] (0.04,0) -- (0.53,0) node[right]{Threshold};
\draw[->] (0.05,0) -- (0.05,0.58) node[above]{Net benefit};
\foreach \x/\lab in {0.05/.05,0.15/.15,0.25/.25,0.35/.35,0.45/.45}
  \draw (\x,0.006) -- (\x,-0.006) node[below]{\lab};
\foreach \y/\lab in {0/0,0.2/.20,0.4/.40}
  \draw (0.048,\y) -- (0.052,\y) node[left]{\lab};
\draw[gray!60, thick] (0.05,0) -- (0.50,0) node[right]{None};
\draw[black!60, dashed]
  (0.05,0.531) -- (0.10,0.505) -- (0.15,0.476) -- (0.20,0.443) -- (0.25,0.406) -- (0.30,0.363) -- (0.35,0.314) -- (0.40,0.257) -- (0.45,0.190) -- (0.50,0.109) node[right]{All};
\draw[blue!70!black, thick]
  (0.05,0.533) -- (0.10,0.516) -- (0.15,0.495) -- (0.20,0.484) -- (0.25,0.480) -- (0.30,0.473) -- (0.35,0.459) -- (0.40,0.455) -- (0.45,0.449) -- (0.50,0.457) node[right]{Tier 2};
\draw[green!50!black, thick]
  (0.05,0.531) -- (0.10,0.512) -- (0.15,0.498) -- (0.20,0.492) -- (0.25,0.486) -- (0.30,0.477) -- (0.35,0.462) -- (0.40,0.457) -- (0.45,0.448) -- (0.50,0.451) node[right]{Cascade};
\end{tikzpicture}
\caption{Held-out decision-curve audit for the full Tier 2 model and conservative cascade. Net benefit is shown without confidence bands and should be interpreted as development-split evidence only.}
\label{fig:dca}
\end{figure}

Table~\ref{tab:calibration} summarizes the held-out probability audit. Raw Tier 1 probabilities produced Brier score 0.153, AUROC 0.844, and ECE-10 of 0.052. Five-fold isotonic calibration fitted on the training split slightly reduced ECE to 0.048 but worsened Brier score and AUROC. The calibrated model was therefore not substituted into the deployed pipeline or reported cascade metrics. These results support the paper's conservative interpretation: the 0.30 and 0.70 gates are prototype routing thresholds, not calibrated clinical risk cutoffs.

\begin{table}[!t]
\caption{Held-Out Probability Calibration Audit}
\label{tab:calibration}
\centering
\scriptsize
\renewcommand{\arraystretch}{1.12}
\begin{tabular}{lrrr}
\toprule
\textbf{Model} & \textbf{Brier} & \textbf{ECE-10} & \textbf{AUROC}\\
\midrule
Tier 1 raw & 0.153 & 0.052 & 0.844\\
Tier 1 isotonic CV5 & 0.157 & 0.048 & 0.840\\
Tier 2 raw & 0.107 & 0.097 & 0.920\\
\bottomrule
\end{tabular}
\end{table}

\begin{figure}[!t]
\centering
\footnotesize
\begin{tikzpicture}[x=4.2cm,y=4.2cm]
\draw[->] (0,0) -- (1.05,0) node[right]{Predicted};
\draw[->] (0,0) -- (0,1.05) node[above]{Observed};
\draw[dashed, gray!70] (0,0) -- (1,1);
\foreach \x/\lab in {0/0,0.25/.25,0.50/.50,0.75/.75,1.0/1.0}
  \draw (\x,0.012) -- (\x,-0.012) node[below]{\lab};
\foreach \y/\lab in {0/0,0.25/.25,0.50/.50,0.75/.75,1.0/1.0}
  \draw (0.012,\y) -- (-0.012,\y) node[left]{\lab};
\draw[blue!70!black, thick]
  (0.064,0.000) -- (0.151,0.050) -- (0.241,0.188) -- (0.352,0.364) -- (0.439,0.412) -- (0.553,0.600) -- (0.645,0.667) -- (0.764,0.857) -- (0.847,0.833);
\foreach \x/\y in {0.064/0.000,0.151/0.050,0.241/0.188,0.352/0.364,0.439/0.412,0.553/0.600,0.645/0.667,0.764/0.857,0.847/0.833}
  \filldraw[blue!70!black] (\x,\y) circle (1.3pt);
\draw[red!65!black, dashed] (0.30,0) -- (0.30,1.0) node[above]{0.30};
\end{tikzpicture}
\caption{Held-out reliability bins for raw Tier 1 probabilities. The 0.20--0.30 bin had an observed positive rate of 18.8\%, explaining why the 0.30 low-risk bypass branch cannot be considered clinically safe.}
\label{fig:reliability}
\end{figure}

\subsection{Cross-Validation and Site-Held-Out Stress Tests}
Stratified 5-fold cross-validation produced lower and more stable estimates than the held-out development split. Table~\ref{tab:cv} shows that the conservative cascade reached $0.845\pm0.022$ mean accuracy and $0.898\pm0.034$ AUROC while avoiding $25.1\%\pm3.3\%$ of Tier 2 profiles. The difference between the conservative cascade and full Tier 2 model is smaller than fold-level variability and is not evidence that the cascade is intrinsically more accurate.

\begin{table}[!t]
\caption{Five-Fold Cross-Validation Summary}
\label{tab:cv}
\centering
\scriptsize
\renewcommand{\arraystretch}{1.12}
\begin{tabular}{lrrr}
\toprule
\textbf{Model or policy} & \textbf{Acc.} & \textbf{AUROC} & \textbf{T2 avoided}\\
\midrule
Tier 1 only & $0.777\pm0.027$ & $0.843\pm0.036$ & --\\
Tier 2 full profile & $0.843\pm0.018$ & $0.904\pm0.035$ & 0.0\%\\
Conservative cascade & $0.845\pm0.022$ & $0.898\pm0.034$ & $25.1\pm3.3\%$\\
Uncertainty-band policy & $0.826\pm0.026$ & $0.854\pm0.042$ & $72.4\pm2.5\%$\\
\bottomrule
\end{tabular}
\end{table}

Leave-one-site-out validation was a stronger stress test. Table~\ref{tab:loso} shows a pooled conservative-cascade accuracy of 0.780, balanced accuracy of 0.775, MCC of 0.554, sensitivity of 0.825, and specificity of 0.725. Switzerland and VA Long Beach had high disease prevalence, and the cascade fell below the corresponding majority-class accuracy baseline at both sites. The performance drop from random split to site-held-out testing is one of the main findings of the paper.

\begin{table}[!t]
\caption{Leave-One-Site-Out Conservative Cascade}
\label{tab:loso}
\centering
\tiny
\renewcommand{\arraystretch}{1.12}
\resizebox{\columnwidth}{!}{%
\begin{tabular}{lrrrrrrrr}
\toprule
\textbf{Held-out site} & \textbf{N} & \textbf{Prev.} & \textbf{Maj.} & \textbf{Acc.} & \textbf{Bal.} & \textbf{MCC} & \textbf{Sens.} & \textbf{Spec.}\\
\midrule
Cleveland & 304 & 0.457 & 0.543 & 0.783 & 0.786 & 0.570 & 0.820 & 0.752\\
Hungary & 293 & 0.362 & 0.638 & 0.799 & 0.820 & 0.615 & 0.896 & 0.743\\
Switzerland & 123 & 0.935 & 0.935 & 0.813 & 0.726 & 0.276 & 0.826 & 0.625\\
VA Long Beach & 200 & 0.745 & 0.745 & 0.730 & 0.683 & 0.344 & 0.779 & 0.588\\
\midrule
Pooled & 920 & 0.553 & 0.553 & 0.780 & 0.775 & 0.554 & 0.825 & 0.725\\
\bottomrule
\end{tabular}
}
\end{table}

\subsection{Exploratory Subgroup Audit}
Table~\ref{tab:subgroup} reports held-out conservative-cascade performance by sex and age band. The analysis is intentionally described as exploratory because subgroup sample sizes are small. The female subgroup had 38 records, and the oldest age band had 20 records. These values are inadequate for a fairness claim.

\begin{table}[!t]
\caption{Held-Out Conservative Cascade by Subgroup}
\label{tab:subgroup}
\centering
\scriptsize
\renewcommand{\arraystretch}{1.12}
\begin{tabular}{lrrrrrr}
\toprule
\textbf{Subgroup} & \textbf{N} & \textbf{Prev.} & \textbf{Acc.} & \textbf{Sens.} & \textbf{Spec.} & \textbf{MCC}\\
\midrule
Female & 38 & 0.263 & 0.974 & 0.900 & 1.000 & 0.932\\
Male & 146 & 0.630 & 0.877 & 0.957 & 0.741 & 0.734\\
$<50$ years & 49 & 0.388 & 0.959 & 0.895 & 1.000 & 0.916\\
50--65 years & 115 & 0.609 & 0.887 & 0.971 & 0.756 & 0.765\\
$>65$ years & 20 & 0.650 & 0.800 & 0.923 & 0.571 & 0.545\\
\bottomrule
\end{tabular}
\end{table}

The lower specificity and MCC in older patients suggest that future validation should pre-specify age-stratified safety endpoints. No equality-of-odds, demographic parity, or subgroup safety claim is made.

\subsection{Attribution Output}
The application computes local Tier 2 attributions and displays the largest risk-raising and risk-lowering factors. In a tested profile with 31.7\% final Tier 2 risk, the highest-impact factors included major-vessel encoding, asymptomatic chest pain, thalassemia or stress-test category, ST-segment slope, and exercise-induced angina indicators. This example is included only to document the interface behavior. It is not evidence that the generated plain-language note is faithful, safe, or clinically useful.

\section{Discussion}
This study reframes heart disease prediction as staged feature acquisition rather than static full-vector classification. The main empirical result is modest but defensible: under cross-validation, a conservative cascade avoided about one quarter of Tier 2 profiles while matching the full-profile model within fold-level noise. The held-out development split showed the same qualitative trade-off, but also revealed the safety problem directly: 4 positive patients appeared in the low-risk bypass branch at the implemented 0.30 threshold.

The threshold sweep changes the interpretation of the resource claim. A 0.30 threshold is attractive operationally because it avoids 24.5\% of held-out Tier 2 profiles, but it is not acceptable as a rule-out threshold in this audit. The 0.15 threshold avoided only 9.8\% of profiles but had no bypassed positives on the same held-out split. Therefore, the evidence supports a staged-acquisition audit and a safety-efficiency frontier, not the claim that the current application threshold is clinically safe.

The site-held-out results are more important than the favorable held-out point estimate. Pooled leave-one-site-out accuracy fell to 78.0\%, and site-specific specificity was weak for VA Long Beach and Switzerland. Critically, the cascade did not beat a trivial majority-class accuracy baseline at those two high-prevalence sites. These results are consistent with known concerns about dataset shift in clinical AI \cite{finlayson2021,kelly2019}. The local multi-site UCI-style file combines sites with different prevalence and missingness patterns; Switzerland in particular has very high disease prevalence in the local file. A staged workflow can therefore be plausible on random splits while still failing to generalize robustly across source sites.

The calibration audit also limits the interpretation of the thresholds. Raw Tier 1 ECE was not extreme, and isotonic calibration slightly improved ECE, but it did not improve Brier score or discrimination. More importantly, the low-probability bins still contained positive cases. A future clinical version would need to derive thresholds from calibrated probabilities and an explicit utility function, not from pragmatic bands.

The explanation layer is secondary. SHAP attributions provide a traceable way to rank local drivers, but the optional language-model output has not been validated. The correct interpretation is that the prototype explores how attribution summaries might be translated into clinician-reviewed plain language. It does not establish that an LLM can safely produce patient-facing cardiac explanations. Any deployment involving third-party language-model services would require privacy review, logging controls, prompt/version governance, and clinician-facing disclosure.

\subsection{Limitations}
Several limitations prevent clinical deployment claims. First, the dataset is public, retrospective, old, and heterogeneous across source sites. Second, no prospective external cohort was evaluated. Third, the routing thresholds were not derived from calibrated clinical risk, decision-curve optimization, or a pre-specified utility function. Fourth, the held-out bypass audit found 4 positive cases among 45 low-risk bypassed patients at the implemented threshold, so the current gate is not suitable for discharge decisions. Fifth, profile avoidance is only an operational proxy; no hospital cost model or time-motion study was performed. Sixth, the subgroup audit is underpowered and cannot establish fairness. Seventh, hyperparameters were fixed for the prototype and not optimized with nested cross-validation. Eighth, some Tier 2 features are downstream diagnostic variables close to the target definition; this is acceptable for staged simulation but not proof of causal diagnosis. Ninth, generated explanations were not evaluated for faithfulness, hallucination, readability, clinician acceptability, or safety. Finally, real clinical use would require prospective validation, governance, privacy controls, and regulatory assessment.

\subsection{Reproducibility}
The analysis used Python with scikit-learn, pandas, NumPy, SHAP, joblib, Streamlit, and the OpenAI-compatible SDK used for the optional language-model call. The public source data are available from the UCI Heart Disease repository \cite{uci_heart}. A standalone publication-evaluation script regenerates held-out metrics, bypass safety, calibration summaries, reliability bins, threshold sweeps, decision-curve data, subgroup metrics, and site prevalence outputs. The language-model API is configured through environment variables and is not required to regenerate the quantitative results.

\subsection{Future Work}
Future work should evaluate the routing gate using cross-validated calibration, utility-optimized thresholds, and prospective external cohorts. Bypass safety should be reported across cross-validation folds and leave-one-site-out settings, not only on a single held-out split. A realistic cost analysis should weight diagnostic variables by site-specific monetary cost, waiting time, invasiveness, and staffing requirements. The explanation layer should undergo a formal faithfulness and safety audit, including whether generated text mentions the correct top-ranked factors, preserves their direction, avoids unsupported claims, and remains acceptable to clinicians.

\section{Conclusion}
Smart Clinic Assistant is best understood as a retrospective staged-acquisition simulation. On a 920-record multi-site UCI-style heart disease file, the implemented conservative cascade achieved $84.5\%\pm2.2\%$ cross-validation accuracy while avoiding $25.1\%\pm3.3\%$ of Tier 2 profiles. Held-out development results were stronger, but leave-one-site-out accuracy dropped to 78.0\%, and the low-risk bypass branch contained missed positive cases at the implemented threshold. A stricter 0.15 threshold had no bypassed positives on the held-out split but avoided only 9.8\% of profiles. The evidence supports the feasibility of resource-aware workflow evaluation, not clinical deployment. Before such a system could guide care, it would need calibrated threshold selection, prospective validation, stronger site and subgroup robustness, real cost modeling, and formal evaluation of any generated clinical text.

\section*{Acknowledgment}
The author acknowledges the public UCI Heart Disease dataset and the open-source scientific Python ecosystem used to build and evaluate the prototype.

\begin{thebibliography}{99}

\bibitem{uci_heart}
A. Janosi, W. Steinbrunn, M. Pfisterer, and R. Detrano, ``Heart Disease,'' UCI Machine Learning Repository, 1988, doi: 10.24432/C52P4X. [Online]. Available: \url{https://archive.ics.uci.edu/dataset/45/heart+disease}

\bibitem{detrano1989}
R. Detrano, A. Janosi, W. Steinbrunn, M. Pfisterer, J.-J. Schmid, S. Sandhu, K. H. Guppy, S. Lee, and V. Froelicher, ``International application of a new probability algorithm for the diagnosis of coronary artery disease,'' \emph{American Journal of Cardiology}, vol. 64, no. 5, pp. 304--310, 1989.

\bibitem{rajkomar2019}
A. Rajkomar, J. Dean, and I. Kohane, ``Machine learning in medicine,'' \emph{New England Journal of Medicine}, vol. 380, no. 14, pp. 1347--1358, 2019.

\bibitem{topol2019}
E. J. Topol, ``High-performance medicine: The convergence of human and artificial intelligence,'' \emph{Nature Medicine}, vol. 25, pp. 44--56, 2019.

\bibitem{kelly2019}
C. J. Kelly, A. Karthikesalingam, M. Suleyman, G. Corrado, and D. King, ``Key challenges for delivering clinical impact with artificial intelligence,'' \emph{BMC Medicine}, vol. 17, Art. no. 195, 2019.

\bibitem{weng2017}
S. F. Weng, J. Reps, J. Kai, J. M. Garibaldi, and N. Qureshi, ``Can machine-learning improve cardiovascular risk prediction using routine clinical data?,'' \emph{PLOS ONE}, vol. 12, no. 4, Art. no. e0174944, 2017.

\bibitem{motwani2017}
M. Motwani, D. Dey, D. S. Berman, G. Germano, S. Achenbach, F. Al-Mallah, T. Andreini, M. Budoff, F. Cademartiri, T. Callister, H. Chang, V. Cheng, B. Chinnaiyan, B. Chow, A. Cury, L. Delago, G. Feuchtner, M. Hadamitzky, J. Hausleiter, P. Kaufmann, Y. Kim, J. Leipsic, E. Maffei, G. Raff, L. Shaw, T. Villines, and P. Slomka, ``Machine learning for prediction of all-cause mortality in patients with suspected coronary artery disease: A 5-year multicentre prospective registry analysis,'' \emph{European Heart Journal}, vol. 38, no. 7, pp. 500--507, 2017.

\bibitem{raman2026}
S. Raman, D. Thakkar, J. Calixte, R. Kumar, K. Sporn, K. Marla, D. Goel, R. Gopali, N. Chetla, S. Pasha, N. Ravisankar, R. Lee, and C. Ionita, ``Machine learning for coronary heart disease prediction: Comparative analysis of Framingham and Cleveland subset of the UCI dataset with SHAP-based interpretability,'' \emph{Epidemiologia}, vol. 7, no. 3, Art. no. 75, 2026, doi: 10.3390/epidemiologia7030075.

\bibitem{steyerberg2010}
E. W. Steyerberg, A. J. Vickers, N. R. Cook, T. Gerds, M. Gonen, N. Obuchowski, M. J. Pencina, and M. W. Kattan, ``Assessing the performance of prediction models: A framework for traditional and novel measures,'' \emph{Epidemiology}, vol. 21, no. 1, pp. 128--138, 2010.

\bibitem{collins2015}
G. S. Collins, J. B. Reitsma, D. G. Altman, and K. G. M. Moons, ``Transparent reporting of a multivariable prediction model for individual prognosis or diagnosis (TRIPOD): The TRIPOD statement,'' \emph{Annals of Internal Medicine}, vol. 162, no. 1, pp. 55--63, 2015.

\bibitem{collins2024}
G. S. Collins, K. G. M. Moons, P. Dhiman, \emph{et al.}, ``TRIPOD+AI statement: Updated guidance for reporting clinical prediction models that use regression or machine learning methods,'' \emph{BMJ}, vol. 385, Art. no. e078378, 2024, doi: 10.1136/bmj-2023-078378.

\bibitem{steyerberg2016}
E. W. Steyerberg and F. E. Harrell, Jr., ``Prediction models need appropriate internal, internal-external, and external validation,'' \emph{Journal of Clinical Epidemiology}, vol. 69, pp. 245--247, 2016.

\bibitem{finlayson2021}
S. G. Finlayson, A. Subbaswamy, K. Singh, J. Bowers, A. Kupke, J. Zittrain, I. S. Kohane, and S. Saria, ``The clinician and dataset shift in artificial intelligence,'' \emph{New England Journal of Medicine}, vol. 385, no. 3, pp. 283--286, 2021.

\bibitem{brier1950}
G. W. Brier, ``Verification of forecasts expressed in terms of probability,'' \emph{Monthly Weather Review}, vol. 78, no. 1, pp. 1--3, 1950.

\bibitem{niculescu2005}
A. Niculescu-Mizil and R. Caruana, ``Predicting good probabilities with supervised learning,'' in \emph{Proc. Int. Conf. Machine Learning}, 2005, pp. 625--632.

\bibitem{vancalster2019}
B. Van Calster, D. J. McLernon, M. van Smeden, L. Wynants, and E. W. Steyerberg, ``Calibration: The Achilles heel of predictive analytics,'' \emph{BMC Medicine}, vol. 17, Art. no. 230, 2019.

\bibitem{vickers2006}
A. J. Vickers and E. B. Elkin, ``Decision curve analysis: A novel method for evaluating prediction models,'' \emph{Medical Decision Making}, vol. 26, no. 6, pp. 565--574, 2006.

\bibitem{wolff2019}
R. F. Wolff, K. G. M. Moons, R. D. Riley, P. F. Whiting, M. Westwood, G. S. Collins, J. B. Reitsma, J. Kleijnen, and S. Mallett, ``PROBAST: A tool to assess the risk of bias and applicability of prediction model studies,'' \emph{Annals of Internal Medicine}, vol. 170, no. 1, pp. 51--58, 2019.

\bibitem{liu2020}
X. Liu, S. Cruz Rivera, D. Moher, M. J. Calvert, A. K. Denniston, and the SPIRIT-AI and CONSORT-AI Working Group, ``Reporting guidelines for clinical trial reports for interventions involving artificial intelligence: The CONSORT-AI extension,'' \emph{Nature Medicine}, vol. 26, pp. 1364--1374, 2020.

\bibitem{rivera2020}
S. C. Rivera, X. Liu, A.-W. Chan, A. K. Denniston, M. J. Calvert, and the SPIRIT-AI and CONSORT-AI Working Group, ``Guidelines for clinical trial protocols for interventions involving artificial intelligence: The SPIRIT-AI extension,'' \emph{Nature Medicine}, vol. 26, pp. 1351--1363, 2020.

\bibitem{vasey2022}
B. Vasey, S. Nagendran, B. Campbell, D. A. Clifton, G. S. Collins, A. K. Denniston, G. Faes, B. Geerts, M. Ibrahim, X. Liu, \emph{et al.}, ``Reporting guideline for the early-stage clinical evaluation of decision support systems driven by artificial intelligence: DECIDE-AI,'' \emph{Nature Medicine}, vol. 28, pp. 924--933, 2022.

\bibitem{nan2015}
F. Nan, J. Wang, and V. Saligrama, ``Feature-budgeted random forest,'' in \emph{Proc. Int. Conf. Machine Learning}, 2015, pp. 1983--1991.

\bibitem{kachuee2019}
M. Kachuee, K. Karkkainen, O. Goldstein, \emph{et al.}, ``Cost-sensitive diagnosis and learning leveraging public health data,'' arXiv:1902.07102, 2019.

\bibitem{viola2001}
P. Viola and M. Jones, ``Rapid object detection using a boosted cascade of simple features,'' in \emph{Proc. IEEE Conf. Computer Vision and Pattern Recognition}, 2001, pp. I-511--I-518.

\bibitem{chow1970}
C. K. Chow, ``On optimum recognition error and reject tradeoff,'' \emph{IEEE Transactions on Information Theory}, vol. 16, no. 1, pp. 41--46, 1970.

\bibitem{elyaniv2010}
R. El-Yaniv and Y. Wiener, ``On the foundations of noise-free selective classification,'' \emph{Journal of Machine Learning Research}, vol. 11, pp. 1605--1641, 2010.

\bibitem{geifman2017}
Y. Geifman and R. El-Yaniv, ``Selective classification for deep neural networks,'' arXiv:1705.08500, 2017.

\bibitem{madras2018}
D. Madras, T. Pitassi, and R. Zemel, ``Predict responsibly: Improving fairness and accuracy by learning to defer,'' in \emph{Proc. Advances in Neural Information Processing Systems}, 2018, pp. 6150--6160.

\bibitem{mozannar2020}
H. Mozannar and D. Sontag, ``Consistent estimators for learning to defer to an expert,'' in \emph{Proc. Int. Conf. Artificial Intelligence and Statistics}, PMLR, vol. 108, 2020, pp. 707--717.

\bibitem{sterne2009}
J. A. C. Sterne, I. R. White, J. B. Carlin, M. Spratt, P. Royston, M. G. Kenward, A. M. Wood, and J. R. Carpenter, ``Multiple imputation for missing data in epidemiological and clinical research: Potential and pitfalls,'' \emph{BMJ}, vol. 338, Art. no. b2393, 2009.

\bibitem{breiman2001}
L. Breiman, ``Random forests,'' \emph{Machine Learning}, vol. 45, pp. 5--32, 2001.

\bibitem{lundberg2017}
S. M. Lundberg and S.-I. Lee, ``A unified approach to interpreting model predictions,'' in \emph{Proc. Advances in Neural Information Processing Systems}, 2017, pp. 4765--4774.

\bibitem{tonekaboni2019}
S. Tonekaboni, S. Joshi, M. D. McCradden, and A. Goldenberg, ``What clinicians want: Contextualizing explainable machine learning for clinical end use,'' in \emph{Proc. Machine Learning for Healthcare}, PMLR, vol. 106, 2019, pp. 359--380.

\bibitem{obermeyer2019}
Z. Obermeyer, B. Powers, C. Vogeli, and S. Mullainathan, ``Dissecting racial bias in an algorithm used to manage the health of populations,'' \emph{Science}, vol. 366, no. 6464, pp. 447--453, 2019.

\bibitem{singhal2023}
K. Singhal, S. Azizi, T. Tu, \emph{et al.}, ``Large language models encode clinical knowledge,'' \emph{Nature}, vol. 620, pp. 172--180, 2023.

\bibitem{ji2023}
Z. Ji, N. Lee, R. Frieske, \emph{et al.}, ``Survey of hallucination in natural language generation,'' \emph{ACM Computing Surveys}, vol. 55, no. 12, Art. no. 248, 2023.

\bibitem{fda_cds}
U.S. Food and Drug Administration, ``Clinical Decision Support Software: Guidance for Industry and Food and Drug Administration Staff,'' final guidance, Sep. 2022. [Online]. Available: \url{https://www.fda.gov/regulatory-information/search-fda-guidance-documents/clinical-decision-support-software}

\end{thebibliography}

\end{document}
