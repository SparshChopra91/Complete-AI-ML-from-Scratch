
# =============================================================================
# app.py  —  Streamlit Frontend for the Two-Tier Heart Disease Prediction System
# =============================================================================
# ⚠️ ANTIGRAVITY CREATED THIS FILE — your original code files are NOT modified.
#    This file ONLY imports from your modules and wraps them in a UI.
#    All logic stays 100% inside data_processor.py, model_handler.py, ai_explainer.py
# =============================================================================

import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os
import shap
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
from openai import OpenAI
from dotenv import load_dotenv
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG  (must be FIRST streamlit call)
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CardioScan AI — Heart Disease Risk Analyzer",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# CUSTOM CSS  — premium dark medical theme
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Google Font ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

/* ── Root palette ── */
:root {
    --bg-primary:   #0d1117;
    --bg-card:      #161b22;
    --bg-input:     #1c2230;
    --accent-red:   #e5484d;
    --accent-blue:  #3b82f6;
    --accent-green: #22c55e;
    --accent-amber: #f59e0b;
    --text-primary: #f0f6fc;
    --text-muted:   #8b949e;
    --border:       #30363d;
    --glow-red:     0 0 20px rgba(229,72,77,0.3);
    --glow-blue:    0 0 20px rgba(59,130,246,0.3);
    --glow-green:   0 0 20px rgba(34,197,94,0.3);
}

/* ── Base ── */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif !important;
    background-color: var(--bg-primary) !important;
    color: var(--text-primary) !important;
}

/* hide default streamlit header/footer */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 1.5rem 2rem 3rem 2rem !important; max-width: 1200px; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: var(--bg-card) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] * { color: var(--text-primary) !important; }

/* ── Cards / containers ── */
.card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.5rem;
    margin-bottom: 1rem;
    transition: box-shadow .3s ease;
}
.card:hover { box-shadow: 0 4px 24px rgba(0,0,0,0.4); }

/* ── Hero banner ── */
.hero {
    background: linear-gradient(135deg, #1a1f35 0%, #0d1117 50%, #1a1125 100%);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 2.5rem 2rem;
    text-align: center;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute; inset: 0;
    background: radial-gradient(ellipse at 50% 0%, rgba(229,72,77,0.12) 0%, transparent 70%);
}
.hero h1 { font-size: 2.8rem; font-weight: 800; margin: 0;
    background: linear-gradient(135deg, #f0f6fc 0%, #e5484d 50%, #3b82f6 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;
}
.hero p { font-size: 1.05rem; color: var(--text-muted); margin: .5rem 0 0; }

/* ── Section heading ── */
.section-title {
    font-size: 1.1rem; font-weight: 700; color: var(--text-primary);
    border-left: 3px solid var(--accent-red);
    padding-left: .6rem; margin-bottom: 1rem;
}

/* ── Risk badge ── */
.risk-badge {
    border-radius: 100px; display: inline-block;
    padding: .3rem .9rem; font-weight: 700; font-size: .85rem;
}
.risk-low    { background: rgba(34,197,94,.15);  color: #22c55e; border: 1px solid rgba(34,197,94,.3); }
.risk-medium { background: rgba(245,158,11,.15); color: #f59e0b; border: 1px solid rgba(245,158,11,.3); }
.risk-high   { background: rgba(229,72,77,.15);  color: #e5484d; border: 1px solid rgba(229,72,77,.3); }

/* ── Big risk number ── */
.risk-number {
    font-size: 4rem; font-weight: 800; line-height: 1;
    text-align: center; padding: .5rem;
}
.risk-low-num    { color: #22c55e; text-shadow: 0 0 30px rgba(34,197,94,.5); }
.risk-medium-num { color: #f59e0b; text-shadow: 0 0 30px rgba(245,158,11,.5); }
.risk-high-num   { color: #e5484d; text-shadow: 0 0 30px rgba(229,72,77,.5); }

/* ── Step indicators ── */
.step-row { display: flex; gap: .5rem; margin-bottom: 1.5rem; justify-content: center; }
.step-dot {
    width: 10px; height: 10px; border-radius: 50%;
    background: var(--border); transition: all .3s;
}
.step-dot.active { background: var(--accent-red); box-shadow: 0 0 8px rgba(229,72,77,.6); }
.step-dot.done   { background: var(--accent-green); }

/* ── Info box ── */
.info-box {
    background: rgba(59,130,246,.08);
    border: 1px solid rgba(59,130,246,.25);
    border-radius: 10px; padding: 1rem 1.2rem;
    font-size: .9rem; color: #93c5fd;
    margin: .75rem 0;
}
.warn-box {
    background: rgba(245,158,11,.08);
    border: 1px solid rgba(245,158,11,.25);
    border-radius: 10px; padding: 1rem 1.2rem;
    font-size: .9rem; color: #fcd34d;
    margin: .75rem 0;
}
.success-box {
    background: rgba(34,197,94,.08);
    border: 1px solid rgba(34,197,94,.25);
    border-radius: 10px; padding: 1rem 1.2rem;
    font-size: .9rem; color: #86efac;
    margin: .75rem 0;
}

/* ── Streamlit widget overrides ── */
.stSlider > div > div > div { background: var(--accent-red) !important; }
.stSelectbox div[data-baseweb="select"] > div,
.stNumberInput input {
    background: var(--bg-input) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    color: var(--text-primary) !important;
}
.stButton > button {
    background: linear-gradient(135deg, #e5484d, #c53030) !important;
    color: white !important; font-weight: 700 !important;
    border: none !important; border-radius: 10px !important;
    padding: .65rem 2rem !important; font-size: 1rem !important;
    transition: all .3s !important; width: 100%;
    box-shadow: 0 4px 15px rgba(229,72,77,.3) !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 25px rgba(229,72,77,.5) !important;
}
div[data-testid="metric-container"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important; padding: .8rem !important;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    border-bottom: 2px solid transparent !important;
    color: var(--text-muted) !important;
    font-weight: 500 !important;
}
.stTabs [aria-selected="true"] {
    border-bottom: 2px solid var(--accent-red) !important;
    color: var(--text-primary) !important;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS — these functions replicate ONLY what's needed for the UI.
#            Your original module-level code runs at import time which causes
#            side-effects (exit(), hardcoded patient, etc.) so we call the
#            trained .pkl files directly here instead of importing the modules.
#
# ⚠️  ANTIGRAVITY NOTE: We do NOT import model_handler / ai_explainer directly
#     because those files run the whole pipeline at module-level (including
#     exit()) which breaks Streamlit. Instead we load the same .pkl files
#     that YOUR code already saved and replicate the inference logic inline.
#     YOUR original files are completely untouched.
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading Tier-1 model…")
def load_tier1():
    return joblib.load("Tier1_model_trained.pkl")

@st.cache_resource(show_spinner="Loading Tier-2 model…")
def load_tier2():
    return joblib.load("Tier2_model_trained.pkl")

def run_tier1(model, age, gender, cp, trestbps):
    """Replicates model_handler.py Tier-1 inference."""
    patient = [[age, gender, cp, trestbps]]
    proba = model.predict_proba(patient)
    return round(proba[0][1] * 100, 2)

def run_tier2(model, age, gender, cp, trestbps, chol, fbs, restecg, thalch):
    """Replicates model_handler.py Tier-2 inference."""
    patient_final = [[age, gender, cp, trestbps, chol, fbs, restecg, thalch]]
    proba = model.predict_proba(patient_final)
    return round(proba[0][1] * 100, 2), patient_final

def get_shap_values(model, patient_final):
    """Replicates ai_explainer.py SHAP computation."""
    patient_np = np.array(patient_final)
    explainer = shap.Explainer(model)
    sv = explainer(patient_np)
    contributions = np.round(sv.values[0][:, 1], 4)
    return contributions, sv

def get_ai_explanation(heart_disease_pct, shap_vals):
    """Replicates ai_explainer.py LLM explanation."""
    load_dotenv()
    api_key = os.getenv("open_router_api_key")
    if not api_key:
        return "⚠️ API key not found in .env file."
    client = OpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1")
    names = ["Age","Gender","Chest Pain Type","Resting Blood Pressure",
             "Serum Cholesterol Level","Fasting Blood Sugar",
             "Resting Electrocardiogram Result","Maximum Heart Rate Achieved"]
    shap_lines = "\n".join([f"- {names[i]}: {shap_vals[i]}" for i in range(len(names))])
    prompt = f"""
    You are a senior consultant cardiologist with over 20 years of experience explaining a patient's heart health assessment during a clinical consultation.

Patient Assessment:

* Estimated Heart Disease Risk: {heart_disease_pct:.1f}%

Factors and Contributions:
{shap_lines}

Important Interpretation:

* Positive values indicate factors that increased the patient's heart disease risk.
* Negative values indicate factors that reduced the patient's heart disease risk.
* Focus primarily on the factors with the largest impact.
* Mention both risk-increasing and risk-reducing factors when relevant.

Your Task:
Write a personalized explanation for the patient as if you are speaking directly to them in a hospital consultation room.

Requirements:

1. Use clear, simple, patient-friendly language.
2. Explain WHY the most important factors may have influenced the result.
3. Mention the estimated risk percentage naturally.
4. Discuss both concerning findings and reassuring findings.
5. Provide balanced medical context without causing unnecessary fear.
6. Include one practical recommendation for improving heart health.
7. Do not mention SHAP values, machine learning, AI, algorithms, models, feature importance, data science, probabilities, or technical analysis.
8. Do not list factors as bullet points.
9. Write naturally as a real doctor would speak.
10. Write approximately 120-180 words.
11. don't greet the patient with the name or something else just start the explanation
12. add the bullet points in the explanation with the emojis so the explanation looks good 
13. explain to the patient in a best analystical and best way so that they can understand it 
14.don't just make the one para explain in points type within the para 
15. in the end draw the conclusion properly 

Return only the patient explanation.

    """
    resp = client.chat.completions.create(
        model="google/gemini-2.5-flash",
        messages=[{"role":"user","content":prompt}],
        max_tokens=400, temperature=0.3
    )
    return resp.choices[0].message.content


def risk_class(pct):
    if pct < 30:   return "low",    "risk-low",    "risk-low-num",    "✅ Low Risk"
    if pct < 60:   return "medium", "risk-medium", "risk-medium-num", "⚠️ Moderate Risk"
    return             "high",   "risk-high",   "risk-high-num",   "🚨 High Risk"


# ─────────────────────────────────────────────────────────────────────────────
# LOAD MODELS
# ─────────────────────────────────────────────────────────────────────────────
try:
    tier1_model = load_tier1()
    tier2_model = load_tier2()
    models_ok = True
except Exception as e:
    models_ok = False
    model_error = str(e)


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 1rem 0;'>
        <div style='font-size:3rem;'>🫀</div>
        <div style='font-size:1.2rem; font-weight:800; color:#f0f6fc;'>CardioScan AI</div>
        <div style='font-size:.8rem; color:#8b949e; margin-top:.2rem;'>Two-Tier Risk Analysis</div>
    </div>
    <hr style='border-color:#30363d; margin:1rem 0;'>
    """, unsafe_allow_html=True)

    st.markdown("**🔬 How It Works**")
    st.markdown("""
    <div class='info-box'>
    <b>Tier 1 — Quick Screen</b><br>
    Uses Age, Gender, Chest Pain, Blood Pressure to give a fast initial risk score.
    </div>
    <div class='info-box'>
    <b>Tier 2 — Deep Analysis</b><br>
    If Tier-1 risk ≥ 30%, adds Cholesterol, Blood Sugar, ECG & Heart Rate for a precise final score.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<hr style='border-color:#30363d;'>", unsafe_allow_html=True)
    st.markdown("**📊 Model Info**")
    st.markdown(f"""
    <div style='font-size:.85rem; color:#8b949e;'>
    • Algorithm: Random Forest<br>
    • Tier 1 features: 4<br>
    • Tier 2 features: 8<br>
    • Dataset: UCI Heart Disease<br>
    • AI Explainer: Gemini 2.5 Flash (via OpenRouter)<br>
    • SHAP: Tree Explainer
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<hr style='border-color:#30363d;'>", unsafe_allow_html=True)
    if models_ok:
        st.markdown('<div class="success-box">✅ Both models loaded</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div style="background:rgba(229,72,77,.1);border:1px solid rgba(229,72,77,.3);border-radius:8px;padding:.75rem;color:#fca5a5;font-size:.85rem;">❌ Model error: {model_error}</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# HERO BANNER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class='hero'>
    <h1>🫀 CardioScan AI</h1>
    <p>AI-powered Two-Tier Heart Disease Risk Assessment System</p>
    <p style='font-size:.82rem; margin-top:.3rem; color:#484f58;'>
        Built with Random Forest · SHAP Explainability · LLM Patient Summary
    </p>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN TABS
# ─────────────────────────────────────────────────────────────────────────────
tab_predict, tab_about, tab_data = st.tabs([
    "🩺  Run Analysis", "ℹ️  About This Project", "📋  Dataset Info"
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — PREDICTION
# ══════════════════════════════════════════════════════════════════════════════
with tab_predict:

    if not models_ok:
        st.error(f"❌ Cannot run analysis — models failed to load: {model_error}")
        st.stop()

    # ── Patient Info Input ────────────────────────────────────────────────
    st.markdown("<div class='section-title'>👤 Tier-1: Basic Patient Information</div>", unsafe_allow_html=True)
    st.markdown('<div class="info-box">🔹 These 4 features are used for the <b>Tier-1 quick screen</b>. Fill them in and click <b>Run Tier-1 Analysis</b>.</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("🎂 Age (years)", min_value=18, max_value=100, value=55, step=1,
                              help="Patient's age in years")
        gender_str = st.selectbox("⚧ Gender", ["Male", "Female"],
                                  help="Patient's biological sex")
        gender = 1 if gender_str == "Male" else 0

    with col2:
        cp_options = {
            "Typical Angina": 0,
            "Atypical Angina": 1,
            "Non-Anginal Pain": 2,
            "Asymptomatic": 3
        }
        cp_str = st.selectbox("💔 Chest Pain Type", list(cp_options.keys()),
                              help="Type of chest pain reported by patient")
        cp = cp_options[cp_str]
        trestbps = st.number_input("🩸 Resting Blood Pressure (mm Hg)", min_value=80, max_value=250,
                                   value=130, step=1, help="Resting blood pressure in mm Hg")

    col_btn1, _ = st.columns([1, 2])
    with col_btn1:
        run_tier1_btn = st.button("🔍  Run Tier-1 Screening", use_container_width=True)

    # ── TIER-1 RESULT ─────────────────────────────────────────────────────
    if run_tier1_btn or st.session_state.get("tier1_done"):

        if run_tier1_btn:
            with st.spinner("🔄 Running Tier-1 model…"):
                t1_pct = run_tier1(tier1_model, age, gender, cp, trestbps)
            st.session_state["tier1_pct"]    = t1_pct
            st.session_state["tier1_inputs"] = (age, gender, cp, trestbps)
            st.session_state["tier1_done"]   = True
            st.session_state["tier2_done"]   = False  # reset tier2

        t1_pct = st.session_state["tier1_pct"]
        _, badge_cls, num_cls, label = risk_class(t1_pct)

        st.markdown("<hr style='border-color:#30363d; margin:1.5rem 0;'>", unsafe_allow_html=True)
        st.markdown("<div class='section-title'>📊 Tier-1 Screening Result</div>", unsafe_allow_html=True)

        col_r1, col_r2, col_r3 = st.columns([1, 1, 1])
        with col_r1:
            st.markdown(f"""
            <div class='card' style='text-align:center;'>
                <div style='color:#8b949e; font-size:.85rem; margin-bottom:.4rem;'>Heart Disease Risk Score</div>
                <div class='risk-number {num_cls}'>{t1_pct}%</div>
                <div style='margin-top:.6rem;'>
                    <span class='risk-badge {badge_cls}'>{label}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
        with col_r2:
            healthy_pct = round(100 - t1_pct, 2)
            st.metric("Healthy Probability", f"{healthy_pct}%", delta=None)
            st.metric("Risk Probability",    f"{t1_pct}%",      delta=None)
        with col_r3:
            # Simple donut-like bar
            fig, ax = plt.subplots(figsize=(3, 3), facecolor='none')
            sizes  = [t1_pct, 100 - t1_pct]
            colors = ['#e5484d', '#22c55e']
            wedges, _ = ax.pie(sizes, colors=colors, startangle=90,
                               wedgeprops=dict(width=0.45, edgecolor='#161b22', linewidth=3))
            ax.text(0, 0, f'{t1_pct}%', ha='center', va='center',
                    fontsize=18, fontweight='bold', color='white',
                    fontfamily='monospace')
            ax.set_facecolor('none')
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close()

        # Decision gate
        if t1_pct < 30:
            st.markdown("""
            <div class='success-box'>
            ✅ <b>Tier-1 Decision: CLEAR</b><br>
            The initial screening shows a low risk of heart disease. No further testing is needed based on this analysis.
            This mirrors the <code>exit(0)</code> decision in <code>model_handler.py</code>.
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class='warn-box'>
            ⚠️ <b>Tier-1 Decision: ESCALATE TO TIER-2</b><br>
            Risk is in the gray zone (≥ 30%). Additional clinical parameters are needed for a precise assessment.
            This triggers the Tier-2 model in <code>model_handler.py</code>.
            </div>
            """, unsafe_allow_html=True)

            # ── TIER-2 INPUT ──────────────────────────────────────────────
            st.markdown("<hr style='border-color:#30363d; margin:1.5rem 0;'>", unsafe_allow_html=True)
            st.markdown("<div class='section-title'>🔬 Tier-2: Extended Clinical Parameters</div>", unsafe_allow_html=True)
            st.markdown('<div class="info-box">🔹 Please fill in the additional clinical values below for the <b>Tier-2 deep analysis</b>.</div>', unsafe_allow_html=True)

            col3, col4 = st.columns(2)
            with col3:
                chol = st.number_input("🧪 Serum Cholesterol (mg/dl)", min_value=100, max_value=600,
                                       value=233, step=1, help="Cholesterol level in mg/dl")
                fbs_str = st.selectbox("🍬 Fasting Blood Sugar > 120 mg/dl",
                                       ["No (False)", "Yes (True)"],
                                       help="Is fasting blood sugar > 120 mg/dl?")
                fbs = 1 if "Yes" in fbs_str else 0
            with col4:
                restecg_opts = {
                    "Normal": 0,
                    "ST-T Wave Abnormality": 1,
                    "Left Ventricular Hypertrophy": 2
                }
                restecg_str = st.selectbox("📈 Resting ECG Result", list(restecg_opts.keys()),
                                           help="Resting electrocardiogram results")
                restecg = restecg_opts[restecg_str]
                thalch = st.number_input("❤️ Max Heart Rate Achieved (bpm)", min_value=60, max_value=250,
                                         value=150, step=1, help="Maximum heart rate achieved during exercise")

            col_btn2, _ = st.columns([1, 2])
            with col_btn2:
                run_tier2_btn = st.button("🏥  Run Full Tier-2 Analysis", use_container_width=True)

            # ── TIER-2 RESULT ─────────────────────────────────────────────
            if run_tier2_btn or st.session_state.get("tier2_done"):

                if run_tier2_btn:
                    with st.spinner("🔄 Running Tier-2 deep analysis…"):
                        t2_pct, patient_final = run_tier2(
                            tier2_model, age, gender, cp, trestbps,
                            chol, fbs, restecg, thalch
                        )
                    st.session_state["tier2_pct"]    = t2_pct
                    st.session_state["patient_final"] = patient_final
                    st.session_state["tier2_done"]   = True

                t2_pct       = st.session_state["tier2_pct"]
                patient_final = st.session_state["patient_final"]
                _, badge2_cls, num2_cls, label2 = risk_class(t2_pct)

                st.markdown("<hr style='border-color:#30363d; margin:1.5rem 0;'>", unsafe_allow_html=True)
                st.markdown("<div class='section-title'>🎯 Final Tier-2 Prediction</div>", unsafe_allow_html=True)

                col_f1, col_f2 = st.columns([1, 1])
                with col_f1:
                    st.markdown(f"""
                    <div class='card' style='text-align:center; padding:2rem;'>
                        <div style='color:#8b949e; font-size:.9rem; margin-bottom:.5rem;'>Final Heart Disease Risk</div>
                        <div class='risk-number {num2_cls}'>{t2_pct}%</div>
                        <div style='margin-top:.8rem;'>
                            <span class='risk-badge {badge2_cls}'>{label2}</span>
                        </div>
                        <div style='margin-top:1rem; font-size:.82rem; color:#8b949e;'>
                            Based on 8 clinical features · Tier-2 Random Forest
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                with col_f2:
                    # Feature values summary
                    feat_names = ["Age","Gender","Chest Pain","Resting BP",
                                  "Cholesterol","Fasting BS","ECG","Max HR"]
                    feat_vals  = [age, gender_str, cp_str.split()[0],
                                  trestbps, chol,
                                  "Yes" if fbs else "No",
                                  restecg_str.split()[0], thalch]
                    df_summary = pd.DataFrame({
                        "Feature": feat_names,
                        "Value":   feat_vals
                    })
                    st.markdown("**Patient Summary**")
                    st.dataframe(df_summary, use_container_width=True, hide_index=True,
                                 column_config={
                                     "Feature": st.column_config.TextColumn("Feature"),
                                     "Value":   st.column_config.TextColumn("Value"),
                                 })

                # ── SHAP EXPLAINABILITY ───────────────────────────────────
                st.markdown("<hr style='border-color:#30363d; margin:1.5rem 0;'>", unsafe_allow_html=True)
                st.markdown("<div class='section-title'>🧠 SHAP Feature Contributions</div>", unsafe_allow_html=True)
                st.markdown('<div class="info-box">🔹 SHAP values show <b>how much each feature pushed the risk score up or down</b>. Positive = increased risk. Negative = reduced risk. This is exactly what your <code>ai_explainer.py</code> computes.</div>', unsafe_allow_html=True)

                with st.spinner("⚙️ Computing SHAP values…"):
                    contributions, sv = get_shap_values(tier2_model, patient_final)

                feature_names = [
                    "Age", "Gender", "Chest Pain Type",
                    "Resting Blood Pressure", "Serum Cholesterol",
                    "Fasting Blood Sugar", "ECG Result", "Max Heart Rate"
                ]

                shap_df = pd.DataFrame({
                    "Feature":      feature_names,
                    "SHAP Value":   contributions,
                    "Direction":    ["⬆️ Increases Risk" if v > 0 else "⬇️ Reduces Risk" for v in contributions]
                }).sort_values("SHAP Value", ascending=False)

                col_s1, col_s2 = st.columns([3, 2])
                with col_s1:
                    fig2, ax2 = plt.subplots(figsize=(7, 4), facecolor='#161b22')
                    colors = ['#e5484d' if v > 0 else '#22c55e' for v in shap_df["SHAP Value"]]
                    bars = ax2.barh(shap_df["Feature"], shap_df["SHAP Value"],
                                   color=colors, edgecolor='none', height=0.6)
                    ax2.axvline(0, color='#8b949e', linewidth=1, linestyle='--')
                    ax2.set_facecolor('#161b22')
                    fig2.patch.set_facecolor('#161b22')
                    ax2.tick_params(colors='#8b949e', labelsize=9)
                    ax2.spines[:].set_color('#30363d')
                    ax2.set_xlabel("SHAP Value (contribution to heart disease risk)", color='#8b949e', fontsize=9)
                    ax2.set_title("Feature Contributions to Final Prediction", color='#f0f6fc', fontsize=11, fontweight='bold')
                    plt.tight_layout()
                    st.pyplot(fig2, use_container_width=True)
                    plt.close()

                with col_s2:
                    st.markdown("**Contribution Table**")
                    st.dataframe(
                        shap_df[["Feature","SHAP Value","Direction"]].reset_index(drop=True),
                        use_container_width=True, hide_index=True
                    )

                # ── AI EXPLANATION ────────────────────────────────────────
                st.markdown("<hr style='border-color:#30363d; margin:1.5rem 0;'>", unsafe_allow_html=True)
                st.markdown("<div class='section-title'>🤖 AI Patient Explanation (Gemini 2.5 Flash)</div>", unsafe_allow_html=True)
                st.markdown('<div class="info-box">🔹 This uses your <code>ai_explainer.py</code> logic — SHAP values are sent to <b>Gemini 2.5 Flash via OpenRouter</b> to generate a plain-English explanation for the patient.</div>', unsafe_allow_html=True)

                if st.button("✨  Generate AI Explanation", use_container_width=False):
                    with st.spinner("🤖 Asking Gemini 2.5 Flash…"):
                        try:
                            explanation = get_ai_explanation(t2_pct, contributions)
                            st.session_state["ai_explanation"] = explanation
                        except Exception as e:
                            st.session_state["ai_explanation"] = f"Error: {e}"

                if "ai_explanation" in st.session_state:
                    st.markdown(f"""
                    <div class='card' style='border-left: 3px solid #e5484d;'>
                        <div style='color:#8b949e; font-size:.8rem; margin-bottom:.6rem;'>
                            🤖 AI Cardiologist Explanation
                        </div>
                        <div style='font-size:1rem; line-height:1.75; color:#f0f6fc;'>
                            {st.session_state["ai_explanation"]}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — ABOUT
# ══════════════════════════════════════════════════════════════════════════════
with tab_about:
    st.markdown("""
    <div class='section-title'>🏗️ Architecture Overview</div>
    """, unsafe_allow_html=True)

    col_a1, col_a2 = st.columns(2)
    with col_a1:
        st.markdown("""
        <div class='card'>
        <b style='color:#e5484d;'>📄 data_processor.py</b>
        <div style='color:#8b949e; font-size:.88rem; margin-top:.5rem;'>
        • Loads UCI Heart Disease CSV<br>
        • Cleans & maps categorical columns (cp, gender, restecg, fbs)<br>
        • Prepares Tier-1 dataset (4 features + target)<br>
        • Prepares Tier-2 dataset (8 features, merged on patient ID)<br>
        • Handles missing values via NaN replacement
        </div>
        </div>

        <div class='card'>
        <b style='color:#3b82f6;'>🤖 model_handler.py</b>
        <div style='color:#8b949e; font-size:.88rem; margin-top:.5rem;'>
        • Trains/loads <b>Tier-1 Random Forest</b> (age, gender, cp, trestbps)<br>
        • Decision gate: if risk < 30% → exit, else escalate<br>
        • Trains/loads <b>Tier-2 Random Forest</b> (all 8 features)<br>
        • Median imputation for missing values<br>
        • Saves models as <code>.pkl</code> files using joblib
        </div>
        </div>
        """, unsafe_allow_html=True)

    with col_a2:
        st.markdown("""
        <div class='card'>
        <b style='color:#22c55e;'>🧠 ai_explainer.py</b>
        <div style='color:#8b949e; font-size:.88rem; margin-top:.5rem;'>
        • Uses <b>SHAP TreeExplainer</b> on the Tier-2 model<br>
        • Extracts per-feature contributions (disease class column)<br>
        • Builds a structured prompt with SHAP values<br>
        • Sends to <b>Gemini 2.5 Flash via OpenRouter API</b><br>
        • Returns plain-English patient explanation
        </div>
        </div>

        <div class='card'>
        <b style='color:#f59e0b;'>📋 pdf_generator.py</b>
        <div style='color:#8b949e; font-size:.88rem; margin-top:.5rem;'>
        • Planned feature (currently empty)<br>
        • Will generate a downloadable PDF report<br>
        • Will include risk score, SHAP chart, and AI explanation
        </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class='section-title' style='margin-top:1rem;'>🔄 Pipeline Flow</div>
    <div class='card'>
    <pre style='color:#8b949e; font-size:.85rem; line-height:1.8; background:transparent; border:none;'>
    Patient Input (4 features)
          │
          ▼
    ┌─────────────────────────────────┐
    │  TIER-1: Random Forest Screen   │  ←  model_handler.py
    │  Features: age, gender, cp,     │
    │            resting BP           │
    └────────────┬────────────────────┘
                 │
         ┌───────┴─────────┐
         │  risk < 30% ?   │
         └───────┬─────────┘
                 │
        YES ─────┘                 NO
        │                          │
    CLEAR — No Heart            ESCALATE
    Disease Detected       (enter 4 more values)
                                   │
                                   ▼
                    ┌──────────────────────────────────┐
                    │  TIER-2: Random Forest Deep       │  ←  model_handler.py
                    │  Features: + chol, fbs,          │
                    │            restecg, max HR        │
                    └──────────────┬───────────────────┘
                                   │
                                   ▼
                    ┌──────────────────────────────────┐
                    │  SHAP Explainability              │  ←  ai_explainer.py
                    │  (feature contribution values)   │
                    └──────────────┬───────────────────┘
                                   │
                                   ▼
                    ┌──────────────────────────────────┐
                    │  LLM Explanation (Gemini 2.5)    │  ←  ai_explainer.py
                    │  Plain-English patient summary   │
                    └──────────────────────────────────┘
    </pre>
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — DATASET INFO
# ══════════════════════════════════════════════════════════════════════════════
with tab_data:
    st.markdown("<div class='section-title'>📋 UCI Heart Disease Dataset</div>", unsafe_allow_html=True)

    try:
        df_raw = pd.read_csv("heart_disease_uci.csv")
        col_d1, col_d2, col_d3, col_d4 = st.columns(4)
        with col_d1: st.metric("Total Records", len(df_raw))
        with col_d2: st.metric("Total Features", len(df_raw.columns))
        with col_d3: st.metric("Positive Cases", int((df_raw.get("target", pd.Series([0])) > 0).sum()))
        with col_d4: st.metric("Missing Values", int(df_raw.isnull().sum().sum()))

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("**Raw Dataset Preview (first 10 rows)**")
        st.dataframe(df_raw.head(10), use_container_width=True, hide_index=True)

        st.markdown("**Feature Data Types & Null Count**")
        dtype_df = pd.DataFrame({
            "Column":    df_raw.columns,
            "Dtype":     df_raw.dtypes.values.astype(str),
            "Nulls":     df_raw.isnull().sum().values,
            "Non-Null":  df_raw.notna().sum().values,
        })
        st.dataframe(dtype_df, use_container_width=True, hide_index=True)
    except FileNotFoundError:
        st.error("heart_disease_uci.csv not found in the project directory.")


# ─────────────────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div style='text-align:center; padding:2rem 0 1rem; color:#484f58; font-size:.82rem;'>
    CardioScan AI · Built with ❤️ · Two-Tier Heart Disease Risk Analysis System<br>
    <span style='color:#30363d;'>data_processor.py · model_handler.py · ai_explainer.py</span>
</div>
""", unsafe_allow_html=True)
