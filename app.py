"""
Smart Clinic Triage Engine — Premium Streamlit Front-End
=========================================================
Two-Tier ML Cascade Router + SHAP Explainability + GenAI Clinical Notes.

Run:
    streamlit run app.py
"""

from __future__ import annotations

import os
import json
import time
from dataclasses import dataclass, asdict
from html import escape
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st
import joblib

try:
    import shap
except ImportError:
    shap = None

try:
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


# ╔═══════════════════════════════════════════════════════════════╗
# ║                     PROJECT CONSTANTS                        ║
# ╚═══════════════════════════════════════════════════════════════╝

BASE_DIR         = Path(__file__).resolve().parent
DATA_PATH        = BASE_DIR / "heart_disease_uci.csv"
TIER1_MODEL_PATH = BASE_DIR / "Tier_1_model.pkl"
TIER2_MODEL_PATH = BASE_DIR / "Tier_2_model.pkl"

LOW_THRESHOLD  = 0.30
HIGH_THRESHOLD = 0.70

TIER1_COLS = [
    "age", "gender_Female", "gender_Male",
    "cp_asymptomatic", "cp_atypical angina", "cp_non-anginal", "cp_typical angina",
    "trestbps",
]

CP_OPTS      = ["asymptomatic", "atypical angina", "non-anginal", "typical angina"]
SEX_OPTS     = ["Female", "Male"]
RESTECG_OPTS = ["normal", "st-t abnormality", "lv hypertrophy"]
RESTECG_MAP  = {"normal": 0, "st-t abnormality": 1, "lv hypertrophy": 2}
SLOPE_OPTS   = ["downsloping", "flat", "upsloping"]
SLOPE_MAP    = {"downsloping": 0, "flat": 1, "upsloping": 2}
THAL_OPTS    = ["fixed defect", "normal", "reversable defect"]
THAL_MAP     = {"fixed defect": 0, "normal": 1, "reversable defect": 2}

REF_RANGES = {
    "age":      {"min": 28.0,  "max": 77.0},
    "trestbps": {"min": 92.0,  "max": 200.0},
    "chol":     {"min": 85.0,  "max": 603.0},
    "thalch":   {"min": 60.0,  "max": 202.0},
    "oldpeak":  {"min": 0.0,   "max": 6.2},
}

SHAP_LABELS = [
    "Age", "Gender", "Chest Pain Type", "Resting BP",
    "Cholesterol", "Fasting Blood Sugar",
    "Resting ECG", "Max Heart Rate",
]


# ╔═══════════════════════════════════════════════════════════════╗
# ║            CSS INJECTION — WARM THEME + SIDE RAYS            ║
# ╚═══════════════════════════════════════════════════════════════╝

def inject_css():
    """Inject the full premium warm-gold theme CSS + SideRays background."""
    st.markdown("""
<!-- Google Fonts — Inter -->
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap" rel="stylesheet">

<style>
/* ═══════════════════════════════════════════════════
   1.  DESIGN TOKENS — Warm Gold + Soft Blue palette
       Colours inspired by SideRays (EAB308 / 96c8ff)
   ═══════════════════════════════════════════════════ */
:root {
    --bg-page:         #FFF9F0;
    --bg-surface:      #FFFFFF;
    --bg-surface-alt:  #FFFAF3;
    --bg-inner:        #FFF6EC;
    --bg-glass:        rgba(255, 255, 255, 0.88);

    --gold-900: #78350F;
    --gold-700: #B45309;
    --gold-600: #CA8A04;
    --gold-500: #EAB308;
    --gold-400: #FACC15;
    --gold-300: #FDE68A;
    --gold-200: #FEF3C7;
    --gold-100: #FEFCE8;
    --gold-glow: rgba(234, 179, 8, 0.14);

    --blue-soft: #96C8FF;
    --blue-500:  #3B82F6;
    --blue-600:  #2563EB;
    --blue-100:  #DBEAFE;
    --blue-glow: rgba(150, 200, 255, 0.12);

    --green-600: #16A34A;
    --green-100: #DCFCE7;
    --amber-600: #D97706;
    --amber-100: #FEF3C7;
    --red-600:   #DC2626;
    --red-100:   #FEE2E2;
    --purple-600:#7C3AED;
    --purple-100:#EDE9FE;
    --teal-600:  #0D9488;

    --text-primary:   #2D2A26;
    --text-secondary: #6B5E50;
    --text-muted:     #A89F93;
    --border:         #E8DFD3;
    --border-soft:    #F0E8DE;

    --shadow-xs:  0 1px 2px rgba(45, 42, 38, 0.04);
    --shadow-sm:  0 2px 8px rgba(45, 42, 38, 0.06);
    --shadow-md:  0 4px 20px rgba(45, 42, 38, 0.08);
    --shadow-lg:  0 8px 40px rgba(45, 42, 38, 0.10);
    --shadow-gold:0 4px 24px rgba(234, 179, 8, 0.12);

    --radius-sm: 10px;
    --radius-md: 14px;
    --radius-lg: 18px;
    --radius-xl: 22px;

    --font: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
}

/* ═══════════════════════════════════════════════════
   2.  GLOBAL RESETS + APP BACKGROUND
   ═══════════════════════════════════════════════════ */
html, body, .stApp, [data-testid="stAppViewContainer"],
p, span, div, label, input, textarea, select, button, a, li, td, th {
    font-family: var(--font) !important;
    -webkit-font-smoothing: antialiased;
}
.stApp {
    background: var(--bg-page) !important;
    color: var(--text-primary) !important;
}
.block-container {
    max-width: 1440px !important;
    padding: 1.2rem 2rem 3rem 2rem !important;
    position: relative;
    z-index: 2;
}

/* Header + Sidebar */
[data-testid="stHeader"] {
    background: rgba(255,249,240,0.6) !important;
    backdrop-filter: blur(18px) saturate(1.4);
    border-bottom: 1px solid var(--border-soft);
}
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #FFFDF8, #FFF6EC) !important;
    border-right: 1px solid var(--border);
}
/* Hide Streamlit branding */
#MainMenu, footer, header [data-testid="stStatusWidget"] { visibility: hidden; }

/* ═══════════════════════════════════════════════════
   3.  SIDE RAYS — CSS-based Volumetric Light Effect
       Replicates the ReactBits SideRays component
       using pure CSS (gold #EAB308 + blue #96c8ff)
       emanating from the top-right corner.
   ═══════════════════════════════════════════════════ */
.side-rays {
    position: fixed;
    inset: 0;
    z-index: 0;
    pointer-events: none;
    overflow: hidden;
}

/* Golden light source at top-right */
.side-rays .glow {
    position: absolute;
    top: -12%;
    right: -8%;
    width: 520px;
    height: 520px;
    background: radial-gradient(ellipse at center,
        rgba(234,179,8,0.18) 0%,
        rgba(234,179,8,0.08) 30%,
        rgba(150,200,255,0.04) 55%,
        transparent 75%);
    border-radius: 50%;
    filter: blur(50px);
    animation: glow-pulse 7s ease-in-out infinite alternate;
}
.side-rays .glow-2 {
    position: absolute;
    top: 5%;
    right: 2%;
    width: 300px;
    height: 300px;
    background: radial-gradient(ellipse at center,
        rgba(150,200,255,0.12) 0%,
        rgba(150,200,255,0.04) 40%,
        transparent 70%);
    border-radius: 50%;
    filter: blur(40px);
    animation: glow-pulse 9s ease-in-out 2s infinite alternate;
}

/* Ray beams — golden */
.ray { position: absolute; top: 0; right: 0; transform-origin: top right; }

.ray.g1 { width: 220%; height: 2px;
    background: linear-gradient(270deg, rgba(234,179,8,0.12), rgba(234,179,8,0.03) 60%, transparent);
    transform: rotate(-18deg); filter: blur(6px);
    animation: ray-breathe 7s ease-in-out infinite alternate; }
.ray.g2 { width: 200%; height: 3px;
    background: linear-gradient(270deg, rgba(234,179,8,0.10), rgba(234,179,8,0.02) 55%, transparent);
    transform: rotate(-28deg); filter: blur(10px);
    animation: ray-breathe 9s ease-in-out 1s infinite alternate; }
.ray.g3 { width: 180%; height: 2px;
    background: linear-gradient(270deg, rgba(234,179,8,0.08), transparent 50%);
    transform: rotate(-40deg); filter: blur(8px);
    animation: ray-breathe 11s ease-in-out 2s infinite alternate; }
.ray.g4 { width: 160%; height: 1.5px;
    background: linear-gradient(270deg, rgba(234,179,8,0.07), transparent 45%);
    transform: rotate(-52deg); filter: blur(7px);
    animation: ray-breathe 8s ease-in-out 3s infinite alternate; }

/* Ray beams — soft blue */
.ray.b1 { width: 190%; height: 2px;
    background: linear-gradient(270deg, rgba(150,200,255,0.10), rgba(150,200,255,0.02) 55%, transparent);
    transform: rotate(-23deg); filter: blur(8px);
    animation: ray-breathe 10s ease-in-out 0.5s infinite alternate; }
.ray.b2 { width: 170%; height: 2.5px;
    background: linear-gradient(270deg, rgba(150,200,255,0.08), transparent 50%);
    transform: rotate(-34deg); filter: blur(12px);
    animation: ray-breathe 12s ease-in-out 1.5s infinite alternate; }
.ray.b3 { width: 150%; height: 1.5px;
    background: linear-gradient(270deg, rgba(150,200,255,0.06), transparent 45%);
    transform: rotate(-46deg); filter: blur(9px);
    animation: ray-breathe 9s ease-in-out 2.5s infinite alternate; }

@keyframes glow-pulse {
    0%   { opacity: 0.6; transform: scale(1); }
    100% { opacity: 1;   transform: scale(1.08); }
}
@keyframes ray-breathe {
    0%   { opacity: 0.25; }
    50%  { opacity: 0.85; }
    100% { opacity: 0.4;  }
}
@media (prefers-reduced-motion: reduce) {
    .side-rays * { animation: none !important; }
}

/* ═══════════════════════════════════════════════════
   4.  TYPOGRAPHY
   ═══════════════════════════════════════════════════ */
h1, h2, h3, h4, h5, h6 {
    color: var(--text-primary) !important;
    font-weight: 700 !important;
    letter-spacing: -0.02em;
}

/* ═══════════════════════════════════════════════════
   5.  FORM CONTROLS — rounded, warm
   ═══════════════════════════════════════════════════ */
/* Number, text, textarea */
input[type="number"],
input[type="text"],
.stNumberInput > div > div > input,
.stTextInput > div > div > input,
.stTextArea textarea {
    border-radius: var(--radius-sm) !important;
    border: 1.5px solid var(--border) !important;
    background: var(--bg-surface) !important;
    color: var(--text-primary) !important;
    box-shadow: var(--shadow-xs) !important;
    transition: border 0.2s, box-shadow 0.2s !important;
}
input:focus, textarea:focus {
    border-color: var(--gold-500) !important;
    box-shadow: 0 0 0 3px var(--gold-glow) !important;
}

/* ── Selectbox — FIX for blank/black text ── */
div[data-baseweb="select"] {
    border-radius: var(--radius-sm) !important;
}
div[data-baseweb="select"] > div {
    background: var(--bg-surface) !important;
    color: var(--text-primary) !important;
    border-radius: var(--radius-sm) !important;
    border: 1.5px solid var(--border) !important;
    box-shadow: var(--shadow-xs) !important;
}
div[data-baseweb="select"] > div:hover {
    border-color: var(--gold-500) !important;
}
/* Selected value text */
div[data-baseweb="select"] span,
div[data-baseweb="select"] div[class*="ValueContainer"] span,
div[data-baseweb="select"] div[class*="singleValue"],
div[data-baseweb="select"] [data-testid="stMarkdownContainer"] {
    color: var(--text-primary) !important;
    opacity: 1 !important;
}
/* Placeholder text */
div[data-baseweb="select"] div[class*="placeholder"] {
    color: var(--text-muted) !important;
}
/* Dropdown menu */
div[data-baseweb="popover"],
div[data-baseweb="popover"] > div,
ul[role="listbox"] {
    background: var(--bg-surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-sm) !important;
    box-shadow: var(--shadow-md) !important;
}
/* Dropdown items */
li[role="option"] {
    color: var(--text-primary) !important;
    background: transparent !important;
}
li[role="option"]:hover,
li[role="option"][aria-selected="true"] {
    background: var(--gold-200) !important;
    color: var(--gold-900) !important;
}
/* Dropdown arrow */
div[data-baseweb="select"] svg {
    color: var(--text-secondary) !important;
    fill: var(--text-secondary) !important;
}

/* Slider */
.stSlider > div > div > div > div {
    background: linear-gradient(90deg, var(--gold-500), var(--blue-soft)) !important;
}
.stSlider [data-baseweb="slider"] div[role="slider"] {
    background: var(--gold-500) !important;
    border-color: var(--gold-600) !important;
}
/* Slider labels */
.stSlider label, .stSlider span {
    color: var(--text-primary) !important;
}

/* Label text across all inputs */
.stSelectbox label, .stNumberInput label, .stSlider label,
.stTextInput label, .stTextArea label, .stRadio label,
.stCheckbox label, .stMultiSelect label {
    color: var(--text-primary) !important;
    font-weight: 600 !important;
    font-size: 0.88rem !important;
}

/* Help text */
.stTooltipIcon svg {
    color: var(--text-muted) !important;
    fill: var(--text-muted) !important;
}

/* ═══════════════════════════════════════════════════
   6.  BUTTONS — warm gold gradient
   ═══════════════════════════════════════════════════ */
.stButton > button,
.stFormSubmitButton > button {
    border-radius: var(--radius-md) !important;
    border: none !important;
    background: linear-gradient(135deg, var(--gold-500) 0%, var(--gold-600) 100%) !important;
    color: var(--gold-900) !important;
    font-weight: 800 !important;
    font-size: 0.95rem !important;
    padding: 0.7rem 1.8rem !important;
    min-height: 52px !important;
    box-shadow: var(--shadow-gold) !important;
    transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1) !important;
    letter-spacing: 0.01em;
    width: 100%;
}
.stButton > button:hover,
.stFormSubmitButton > button:hover {
    background: linear-gradient(135deg, var(--gold-600) 0%, var(--gold-700) 100%) !important;
    box-shadow: 0 6px 28px rgba(234, 179, 8, 0.3) !important;
    transform: translateY(-1px);
    color: #FFFFFF !important;
}
.stButton > button:active,
.stFormSubmitButton > button:active {
    transform: translateY(0);
}
.stDownloadButton > button {
    border-radius: var(--radius-sm) !important;
    border: 1.5px solid var(--border) !important;
    background: var(--bg-surface) !important;
    color: var(--text-primary) !important;
    font-weight: 600 !important;
    box-shadow: var(--shadow-sm) !important;
}

/* ═══════════════════════════════════════════════════
   7.  BOX-IN-BOX CARD SYSTEM
   ═══════════════════════════════════════════════════ */
/* Outer container */
.outer-box {
    background: var(--bg-surface);
    border: 1px solid var(--border);
    border-radius: var(--radius-xl);
    padding: 1.6rem;
    box-shadow: var(--shadow-md);
    margin-bottom: 1rem;
    position: relative;
    overflow: hidden;
}
/* Inner card */
.inner-card {
    background: var(--bg-inner);
    border: 1px solid var(--border-soft);
    border-radius: var(--radius-md);
    padding: 1.2rem 1.4rem;
    position: relative;
}

/* ═══════════════════════════════════════════════════
   8.  METRIC CARDS
   ═══════════════════════════════════════════════════ */
.metric-card {
    background: var(--bg-surface);
    border: 1px solid var(--border);
    border-radius: var(--radius-lg);
    padding: 1.1rem 1.2rem;
    min-height: 120px;
    box-shadow: var(--shadow-sm);
    position: relative;
    overflow: hidden;
    transition: transform 0.25s ease, box-shadow 0.25s ease;
}
.metric-card:hover {
    transform: translateY(-2px);
    box-shadow: var(--shadow-md);
}
.metric-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; bottom: 0;
    width: 4px;
    border-radius: 4px 0 0 4px;
}
.metric-card.t-gold::before    { background: linear-gradient(180deg, var(--gold-500), var(--gold-400)); }
.metric-card.t-blue::before    { background: linear-gradient(180deg, var(--blue-500), var(--blue-soft)); }
.metric-card.t-green::before   { background: linear-gradient(180deg, var(--green-600), #22C55E); }
.metric-card.t-amber::before   { background: linear-gradient(180deg, var(--amber-600), #F59E0B); }
.metric-card.t-red::before     { background: linear-gradient(180deg, var(--red-600), #EF4444); }
.metric-card.t-purple::before  { background: linear-gradient(180deg, var(--purple-600), #8B5CF6); }

.mc-label { color: var(--text-muted); font-size: 0.76rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.06em; margin-bottom: 0.35rem; }
.mc-value { font-size: clamp(1.4rem, 2.8vw, 2rem); font-weight: 800; color: var(--text-primary); line-height: 1.1; }
.mc-detail { color: var(--text-muted); font-size: 0.8rem; margin-top: 0.45rem; font-weight: 500; }

/* ═══════════════════════════════════════════════════
   9.  HEADER BANNER
   ═══════════════════════════════════════════════════ */
.hero-banner {
    background: linear-gradient(135deg,
        rgba(234,179,8,0.06) 0%,
        rgba(255,255,255,0.5) 40%,
        rgba(150,200,255,0.04) 100%);
    border: 1px solid var(--border);
    border-radius: var(--radius-xl);
    padding: 2rem 2.4rem;
    margin-bottom: 1.5rem;
    box-shadow: var(--shadow-sm);
    position: relative;
    overflow: hidden;
}
.hero-banner::after {
    content: '';
    position: absolute;
    top: -40%; right: -6%;
    width: 260px; height: 260px;
    background: radial-gradient(circle, rgba(234,179,8,0.08), transparent 65%);
    border-radius: 50%;
    pointer-events: none;
}
.hero-eyebrow {
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    color: var(--gold-600);
    font-weight: 700;
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-bottom: 0.4rem;
}
.hero-eyebrow .pulse-dot {
    width: 8px; height: 8px;
    background: var(--gold-500);
    border-radius: 50%;
    animation: dot-pulse 2s ease-in-out infinite;
}
@keyframes dot-pulse {
    0%,100% { opacity: 1; transform: scale(1); }
    50%     { opacity: 0.4; transform: scale(1.4); }
}
.hero-title {
    font-size: clamp(1.6rem, 3.5vw, 2.4rem) !important;
    font-weight: 900 !important;
    color: var(--text-primary) !important;
    margin: 0.2rem 0 0.45rem 0 !important;
    line-height: 1.15 !important;
}
.hero-sub {
    color: var(--text-secondary);
    font-size: 0.98rem;
    line-height: 1.55;
    max-width: 680px;
    margin: 0;
}

/* ═══════════════════════════════════════════════════
   10.  RESULT / TRIAGE CARDS
   ═══════════════════════════════════════════════════ */
.result-box {
    background: var(--bg-surface);
    border: 1px solid var(--border);
    border-radius: var(--radius-xl);
    padding: 1.8rem 2rem;
    box-shadow: var(--shadow-lg);
    position: relative;
    overflow: hidden;
    margin-bottom: 1rem;
}
.result-box::before {
    content: '';
    position: absolute;
    top: 0; left: 0; bottom: 0;
    width: 5px;
}
.result-box.s-low::before    { background: linear-gradient(180deg, var(--green-600), #22C55E); }
.result-box.s-gray::before   { background: linear-gradient(180deg, var(--gold-500), var(--gold-400)); }
.result-box.s-high::before   { background: linear-gradient(180deg, var(--red-600), #EF4444); }

.result-tag {
    display: inline-block;
    padding: 0.22rem 0.7rem;
    border-radius: 20px;
    font-size: 0.7rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.07em;
}
.s-low .result-tag   { background: var(--green-100); color: var(--green-600); }
.s-gray .result-tag  { background: var(--gold-200);  color: var(--gold-700); }
.s-high .result-tag  { background: var(--red-100);   color: var(--red-600); }

.risk-pct {
    font-size: clamp(2.8rem, 6vw, 4.4rem);
    font-weight: 900;
    line-height: 1;
    margin: 0.4rem 0 0.3rem 0;
    color: var(--text-primary);
}
.s-low .risk-pct  { color: var(--green-600); }
.s-gray .risk-pct { color: var(--gold-600); }
.s-high .risk-pct { color: var(--red-600); }

.result-title { font-size: 1.2rem !important; font-weight: 700 !important; margin: 0.3rem 0 !important; color: var(--text-primary) !important; }
.result-desc  { color: var(--text-secondary); font-size: 0.92rem; line-height: 1.5; margin: 0; }
.result-next  { margin-top: 0.9rem; padding-top: 0.9rem; border-top: 1px solid var(--border-soft); color: var(--text-secondary); font-weight: 600; font-size: 0.85rem; }

/* Gate pills */
.gate-row {
    display: flex; gap: 0.5rem; flex-wrap: wrap; margin-top: 0.8rem;
}
.gate-pill {
    flex: 1; min-width: 120px; text-align: center;
    background: var(--bg-inner);
    border: 1px solid var(--border-soft);
    border-radius: var(--radius-sm);
    padding: 0.5rem 0.8rem;
}
.gate-pill .gp-label {
    display: block; font-size: 0.68rem; font-weight: 700;
    text-transform: uppercase; letter-spacing: 0.06em;
    color: var(--text-muted); margin-bottom: 0.15rem;
}
.gate-pill .gp-val {
    font-size: 0.95rem; font-weight: 800; color: var(--text-primary);
}

/* ═══════════════════════════════════════════════════
   11.  AI CLINICAL NOTE
   ═══════════════════════════════════════════════════ */
.ai-note {
    background: linear-gradient(135deg, var(--gold-100) 0%, var(--blue-100) 100%);
    border: 1px solid var(--border);
    border-radius: var(--radius-lg);
    padding: 1.6rem 1.8rem;
    box-shadow: var(--shadow-md), 0 0 30px rgba(234,179,8,0.06);
    position: relative;
    overflow: hidden;
    margin-top: 0.8rem;
}
.ai-note::before {
    content: '';
    position: absolute; top: 0; left: 0; right: 0; height: 3px;
    background: linear-gradient(90deg, var(--gold-500), var(--blue-soft), var(--purple-600));
}
.ai-note-hdr {
    display: flex; align-items: center; gap: 0.55rem; margin-bottom: 0.9rem;
}
.ai-note-icon {
    width: 34px; height: 34px;
    background: linear-gradient(135deg, var(--gold-500), var(--blue-soft));
    border-radius: 9px;
    display: flex; align-items: center; justify-content: center;
    font-size: 1rem;
    box-shadow: 0 4px 12px rgba(234,179,8,0.18);
}
.ai-note-hdr h4 {
    margin: 0 !important; font-size: 1rem !important; color: var(--text-primary) !important;
}
.ai-badge {
    background: var(--gold-200); color: var(--gold-700);
    padding: 0.12rem 0.5rem; border-radius: 10px;
    font-size: 0.65rem; font-weight: 700; letter-spacing: 0.04em;
}
.ai-note-body {
    color: var(--text-secondary); font-size: 0.95rem; line-height: 1.7; text-align: justify;
}

/* ═══════════════════════════════════════════════════
   12.  SECTION HEADERS
   ═══════════════════════════════════════════════════ */
.sec-hdr {
    display: flex; align-items: center; gap: 0.55rem;
    margin: 1rem 0 0.7rem 0;
    padding-bottom: 0.55rem;
    border-bottom: 2px solid var(--border-soft);
}
.sec-icon {
    width: 30px; height: 30px; border-radius: 8px;
    display: flex; align-items: center; justify-content: center;
    font-size: 0.95rem;
}
.sec-icon.gold   { background: var(--gold-200); }
.sec-icon.blue   { background: var(--blue-100); }
.sec-icon.green  { background: var(--green-100); }
.sec-icon.amber  { background: var(--amber-100); }
.sec-icon.purple { background: var(--purple-100); }
.sec-hdr h3 { margin: 0 !important; font-size: 1.05rem !important; }

/* ═══════════════════════════════════════════════════
   13.  PROGRESS BAR (risk gradient)
   ═══════════════════════════════════════════════════ */
.stProgress > div > div > div > div {
    background: linear-gradient(90deg,
        var(--green-600) 0%, var(--gold-500) 50%, var(--red-600) 100%) !important;
    border-radius: 8px !important;
}

/* ═══════════════════════════════════════════════════
   14.  EXPANDER
   ═══════════════════════════════════════════════════ */
div[data-testid="stExpander"] {
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-md) !important;
    background: var(--bg-surface) !important;
    box-shadow: var(--shadow-xs) !important;
}
.streamlit-expanderHeader {
    font-weight: 600 !important;
    color: var(--text-secondary) !important;
}

/* ═══════════════════════════════════════════════════
   15.  ANIMATIONS
   ═══════════════════════════════════════════════════ */
.fade-in {
    animation: fadeSlideIn 0.45s cubic-bezier(0.16,1,0.3,1) forwards;
}
@keyframes fadeSlideIn {
    from { opacity: 0; transform: translateY(10px); }
    to   { opacity: 1; transform: translateY(0); }
}

/* ═══════════════════════════════════════════════════
   16.  PLACEHOLDER / EMPTY STATE
   ═══════════════════════════════════════════════════ */
.empty-state {
    text-align: center;
    padding: 3rem 2rem;
}
.empty-state .icon { font-size: 3rem; margin-bottom: 0.6rem; }
.empty-state h3 { color: var(--text-secondary) !important; font-weight: 600 !important; }
.empty-state p  { color: var(--text-muted); max-width: 400px; margin: 0.4rem auto 0; }

/* ═══════════════════════════════════════════════════
   17.  FOOTER
   ═══════════════════════════════════════════════════ */
.app-ft {
    text-align: center; padding: 1.5rem 0; margin-top: 2rem;
    border-top: 1px solid var(--border-soft);
    color: var(--text-muted); font-size: 0.8rem;
}
</style>

<!-- ══════  SIDE RAYS BACKGROUND ELEMENTS  ══════ -->
<div class="side-rays" aria-hidden="true">
    <div class="glow"></div>
    <div class="glow-2"></div>
    <div class="ray g1"></div>
    <div class="ray g2"></div>
    <div class="ray g3"></div>
    <div class="ray g4"></div>
    <div class="ray b1"></div>
    <div class="ray b2"></div>
    <div class="ray b3"></div>
</div>
    """, unsafe_allow_html=True)


# ╔═══════════════════════════════════════════════════════════════╗
# ║                  MODEL & DATA HELPERS                        ║
# ╚═══════════════════════════════════════════════════════════════╝

@st.cache_resource(show_spinner=False)
def _load_model(path):
    p = Path(path)
    return joblib.load(p) if p.exists() else None

@st.cache_data(show_spinner=False)
def _load_csv():
    if not DATA_PATH.exists():
        return pd.DataFrame()
    df = pd.read_csv(DATA_PATH)
    df["target_binary"] = (pd.to_numeric(df["target"], errors="coerce").fillna(0) > 0).astype(int)
    return df

def _ref_ranges(df):
    """Compute scaling ranges from a stratified training split (mirrors Data_Processing_final.py)."""
    try:
        from sklearn.model_selection import train_test_split
        y = (pd.to_numeric(df["target"], errors="coerce").fillna(0) > 0).astype(int)
        tr, _ = train_test_split(df, test_size=0.2, stratify=y, random_state=43)
    except Exception:
        tr = df
    out = {}
    for c, fb in REF_RANGES.items():
        v = pd.to_numeric(tr[c], errors="coerce")
        if c == "chol":
            v = v.replace(0, np.nan)
        v = v.dropna()
        out[c] = {"min": float(v.min()), "max": float(v.max())} if not v.empty else fb
    return out

def _scale(val, col, refs):
    r = refs.get(col, REF_RANGES[col])
    lo, hi = r["min"], r["max"]
    if hi == lo:
        return 0.0
    return float(np.clip((float(val) - lo) / (hi - lo), 0.0, 1.0))


# ── Tier 1 vector ──
def _t1_vec(age, sex, cp, bp, refs):
    row = {f: 0.0 for f in TIER1_COLS}
    row["age"]     = _scale(age, "age", refs)
    row["trestbps"] = _scale(bp, "trestbps", refs)
    row[f"gender_{sex}"] = 1.0
    k = f"cp_{cp}"
    if k in row:
        row[k] = 1.0
    return pd.DataFrame([row], columns=TIER1_COLS)


# ── Tier 2 vector ──
def _t2_vec(age, sex, cp, bp, chol, fbs, ecg, thalch,
            exang, oldpeak, slope, ca, thal, refs, model):
    cols = list(getattr(model, "feature_names_in_", _default_t2_cols()))
    row = {f: 0.0 for f in cols}

    for c, v in [("age", age), ("trestbps", bp), ("chol", chol),
                 ("thalch", thalch), ("oldpeak", oldpeak)]:
        if c in row and v is not None:
            row[c] = _scale(v, c, refs)

    # Missing indicators
    for m in ["ca_missing", "chol_missing_or_zero", "oldpeak_missing"]:
        if m in row:
            row[m] = 0.0
    if "chol_missing_or_zero" in row and (chol is None or chol == 0):
        row["chol_missing_or_zero"] = 1.0

    # One-hot encodings
    for key in [f"gender_{sex}", f"cp_{cp}"]:
        if key in row:
            row[key] = 1.0

    ecg_code = RESTECG_MAP.get(ecg)
    if ecg_code is not None:
        k = f"restecg_{float(ecg_code)}"
        if k in row: row[k] = 1.0

    fbs_v = 1.0 if fbs else 0.0
    k = f"fbs_{fbs_v}"
    if k in row: row[k] = 1.0

    exang_v = 1.0 if exang else 0.0
    k = f"exang_{exang_v}"
    if k in row: row[k] = 1.0

    slope_code = SLOPE_MAP.get(slope)
    if slope_code is not None:
        k = f"slope_{float(slope_code)}"
        if k in row: row[k] = 1.0

    thal_code = THAL_MAP.get(thal)
    if thal_code is not None:
        k = f"thal_{float(thal_code)}"
        if k in row: row[k] = 1.0

    if ca is not None:
        k = f"ca_{float(ca)}"
        if k in row: row[k] = 1.0

    return pd.DataFrame([row], columns=cols)


def _default_t2_cols():
    return [
        "age", "trestbps", "chol", "thalch", "oldpeak",
        "ca_missing", "chol_missing_or_zero", "oldpeak_missing",
        "cp_asymptomatic", "cp_atypical angina", "cp_non-anginal", "cp_typical angina",
        "gender_Female", "gender_Male",
        "fbs_-1.0", "fbs_0.0", "fbs_1.0",
        "restecg_-1.0", "restecg_0.0", "restecg_1.0", "restecg_2.0",
        "exang_0.0", "exang_1.0",
        "slope_-1.0", "slope_0.0", "slope_1.0", "slope_2.0",
        "ca_-1.0", "ca_0.0", "ca_1.0", "ca_2.0", "ca_3.0",
        "thal_-1.0", "thal_0.0", "thal_1.0", "thal_2.0",
    ]


def _prob(model, X):
    if hasattr(model, "predict_proba"):
        p = model.predict_proba(X)
        cls = list(getattr(model, "classes_", range(p.shape[1])))
        idx = cls.index(1) if 1 in cls else p.shape[1] - 1
        return float(p[0, idx])
    return float(model.predict(X)[0])


def _route(prob, lo=LOW_THRESHOLD, hi=HIGH_THRESHOLD):
    if prob <= lo:
        return {"status": "low", "title": "Safe Discharge Candidate",
                "action": "Tier 1 vitals sit below the low-risk gate. Standard clinical review recommended before discharge.",
                "next": "No Tier 2 lab work triggered.", "icon": "✅"}
    if prob >= hi:
        return {"status": "high", "title": "Immediate Clinical Escalation",
                "action": "Tier 1 vitals exceed the high-risk gate. Prioritize clinician review and confirmatory cardiac workup.",
                "next": "Tier 2 panel auto-launched for full diagnostic assessment.", "icon": "🚨"}
    return {"status": "gray", "title": "Gray Zone — Trigger Tier 2 Labs",
            "action": "Tier 1 is uncertain. Collecting cholesterol, fasting blood sugar, ECG, max heart rate, and additional labs.",
            "next": "Complete the Tier 2 panel below for a definitive diagnosis.", "icon": "⚠️"}


# ╔═══════════════════════════════════════════════════════════════╗
# ║           SHAP EXPLAINABILITY + GenAI CLINICAL NOTE          ║
# ╚═══════════════════════════════════════════════════════════════╝

def _shap(model, X):
    if shap is None:
        return None
    try:
        exp = shap.Explainer(model)
        sv = exp(X.values)
        c = sv.values[0]
        return c[:, 1] if c.ndim == 2 else c
    except Exception:
        return None


def _ai_note(pct, contribs, names):
    if OpenAI is None:
        return None
    key = os.getenv("open_router_api_key") or os.getenv("OPENROUTER_API_KEY")
    if not key:
        return None

    factors = "\n".join(f"* {n}: {v:.4f}" for n, v in zip(names, contribs))
    prompt = f"""You are a senior cardiologist with 20+ years of clinical experience explaining a heart health assessment to a patient during a consultation.

Patient Information:
* Predicted Heart Disease Risk: {pct:.1f}%

Factors considered:
{factors}

Rules: Positive values = increased risk, negative = reduced risk. Focus on the 3-4 strongest factors. Explain naturally as a doctor would. Avoid ALL technical jargon, ML terms, SHAP values, probabilities, algorithms. Do not mention AI or computers.

Style: Warm, professional, reassuring. Use "you" and "your". Sound like a real cardiologist. Include one lifestyle recommendation.

Output: One paragraph, 120-180 words. Mention the risk percentage naturally. Return only the explanation."""

    try:
        client = OpenAI(api_key=key, base_url="https://openrouter.ai/api/v1")
        r = client.chat.completions.create(
            model="google/gemini-2.5-flash",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500, top_p=0.9, temperature=0.3)
        return r.choices[0].message.content
    except Exception as e:
        return f"Note generation error: {e}"


# ╔═══════════════════════════════════════════════════════════════╗
# ║                 VISUALISATION FUNCTIONS                      ║
# ╚═══════════════════════════════════════════════════════════════╝

def _plotly_gauge(prob, title="Cardiac Risk Score"):
    """Render a Plotly gauge / speedometer for probability."""
    if not HAS_PLOTLY:
        return
    color = "#16A34A" if prob < 0.3 else ("#D97706" if prob < 0.7 else "#DC2626")
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(prob * 100, 1),
        number={"suffix": "%", "font": {"size": 38, "family": "Inter", "color": "#2D2A26"}},
        title={"text": title, "font": {"size": 14, "family": "Inter", "color": "#6B5E50"}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#E8DFD3",
                     "tickfont": {"color": "#A89F93", "family": "Inter"}},
            "bar": {"color": color, "thickness": 0.3},
            "bgcolor": "#FFF6EC",
            "borderwidth": 1, "bordercolor": "#E8DFD3",
            "steps": [
                {"range": [0, 30],  "color": "#DCFCE7"},
                {"range": [30, 70], "color": "#FEF3C7"},
                {"range": [70, 100],"color": "#FEE2E2"},
            ],
            "threshold": {"line": {"color": "#2D2A26", "width": 3}, "thickness": 0.8, "value": prob * 100},
        }
    ))
    fig.update_layout(
        height=220, margin=dict(l=30, r=30, t=40, b=10),
        paper_bgcolor="rgba(0,0,0,0)", font={"family": "Inter"},
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def _plotly_shap_bars(contribs, names):
    """Horizontal SHAP contribution bar chart."""
    if not HAS_PLOTLY or contribs is None:
        return
    n = min(len(contribs), len(names))
    vals = contribs[:n]
    labs = names[:n]

    # Sort by absolute value
    order = np.argsort(np.abs(vals))
    vals = vals[order]
    labs = [labs[i] for i in order]

    colors = ["#DC2626" if v > 0 else "#16A34A" for v in vals]

    fig = go.Figure(go.Bar(
        x=vals, y=labs, orientation="h",
        marker_color=colors,
        marker_line_width=0,
        text=[f"{v:+.4f}" for v in vals],
        textposition="outside",
        textfont={"family": "Inter", "size": 11},
    ))
    fig.update_layout(
        height=max(220, n * 34),
        margin=dict(l=10, r=60, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(showgrid=True, gridcolor="#F0E8DE", zeroline=True,
                   zerolinecolor="#E8DFD3", zerolinewidth=2,
                   tickfont={"family": "Inter", "color": "#A89F93"}),
        yaxis=dict(tickfont={"family": "Inter", "color": "#2D2A26", "size": 12}),
        font={"family": "Inter"},
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def _plotly_pie(contribs, names):
    """Pie chart: risk-increasing vs risk-decreasing factors."""
    if not HAS_PLOTLY or contribs is None:
        return
    n = min(len(contribs), len(names))
    pos = sum(c for c in contribs[:n] if c > 0)
    neg = abs(sum(c for c in contribs[:n] if c < 0))
    if pos + neg == 0:
        return

    fig = go.Figure(go.Pie(
        labels=["Risk Factors", "Protective Factors"],
        values=[pos, neg],
        hole=0.55,
        marker=dict(colors=["#FEE2E2", "#DCFCE7"],
                    line=dict(color=["#DC2626", "#16A34A"], width=2)),
        textinfo="label+percent",
        textfont=dict(family="Inter", size=12, color="#2D2A26"),
        hoverinfo="label+value",
    ))
    fig.update_layout(
        height=260,
        margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        font={"family": "Inter"},
        showlegend=False,
        annotations=[dict(text="SHAP", x=0.5, y=0.5, font_size=14,
                          font_family="Inter", font_color="#A89F93", showarrow=False)],
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


# ╔═══════════════════════════════════════════════════════════════╗
# ║                 HTML CARD RENDERERS                          ║
# ╚═══════════════════════════════════════════════════════════════╝

def _mc(label, value, detail, tone="gold"):
    st.markdown(f"""<div class="metric-card t-{escape(tone)}">
        <div class="mc-label">{escape(label)}</div>
        <div class="mc-value">{escape(str(value))}</div>
        <div class="mc-detail">{escape(detail)}</div>
    </div>""", unsafe_allow_html=True)


def _result_card(prob, route):
    s = route["status"]
    tag = {"low": "Low Risk", "gray": "Gray Zone", "high": "High Risk"}.get(s, "")
    st.markdown(f"""<div class="result-box s-{escape(s)} fade-in">
        <span class="result-tag">{route["icon"]}  {tag}</span>
        <div class="risk-pct">{prob:.1%}</div>
        <div class="result-title">{escape(route["title"])}</div>
        <p class="result-desc">{escape(route["action"])}</p>
        <div class="result-next">&#8594; {escape(route["next"])}</div>
    </div>""", unsafe_allow_html=True)


def _gate_pills(prob, lo, hi):
    st.markdown(f"""<div class="gate-row">
        <div class="gate-pill"><span class="gp-label">Discharge</span><span class="gp-val">&le; {lo:.2f}</span></div>
        <div class="gate-pill"><span class="gp-label">P(Sick)</span><span class="gp-val" style="color:var(--gold-600)">{prob:.3f}</span></div>
        <div class="gate-pill"><span class="gp-label">Escalation</span><span class="gp-val">&ge; {hi:.2f}</span></div>
    </div>""", unsafe_allow_html=True)


def _sec(icon, title, color="gold"):
    st.markdown(f"""<div class="sec-hdr">
        <div class="sec-icon {escape(color)}">{icon}</div>
        <h3>{escape(title)}</h3>
    </div>""", unsafe_allow_html=True)


def _ai_box(text):
    st.markdown(f"""<div class="ai-note fade-in">
        <div class="ai-note-hdr">
            <div class="ai-note-icon">🤖</div>
            <h4>AI Clinical Note</h4>
            <span class="ai-badge">GEMINI 2.5</span>
        </div>
        <div class="ai-note-body">{escape(text)}</div>
    </div>""", unsafe_allow_html=True)


# ╔═══════════════════════════════════════════════════════════════╗
# ║                        MAIN APP                              ║
# ╚═══════════════════════════════════════════════════════════════╝

def main():
    st.set_page_config(
        page_title="Smart Clinic Triage Engine",
        page_icon="🏥",
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    inject_css()

    # ── Session state init ──
    for k, v in {
        "tier": None,           # None | "done_t1" | "need_t2" | "done_t2"
        "t1_prob": None,
        "t1_route": None,
        "t1_data": {},
        "t2_prob": None,
        "t2_route": None,
        "shap_vals": None,
        "ai_note_text": None,
    }.items():
        if k not in st.session_state:
            st.session_state[k] = v

    # ── Load resources ──
    t1_model = _load_model(TIER1_MODEL_PATH)
    t2_model = _load_model(TIER2_MODEL_PATH)
    dataset  = _load_csv()
    refs     = _ref_ranges(dataset) if not dataset.empty else REF_RANGES

    # ════════════════════════════════════════════════
    #                  HERO BANNER
    # ════════════════════════════════════════════════
    st.markdown("""
    <div class="hero-banner">
        <div class="hero-eyebrow"><span class="pulse-dot"></span>Smart Clinic Triage Engine</div>
        <h1 class="hero-title">🏥  AI-Driven, Cost-Aware Diagnostic Assistant</h1>
        <p class="hero-sub">
            Two-tier cascade routing engine — triages patients using free intake vitals first,
            triggering expensive lab work only when clinically necessary. Machine learning decisions
            are translated into plain-English clinical notes via Generative AI.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ── Status cards ──
    s1, s2, s3, s4 = st.columns(4)
    with s1:
        _mc("Patient Records", f"{len(dataset):,}" if not dataset.empty else "—",
            "UCI Cleveland Dataset", "blue")
    with s2:
        prev = dataset["target_binary"].mean() if not dataset.empty else 0
        _mc("Prevalence", f"{prev:.1%}", "Observed disease rate", "red")
    with s3:
        _mc("Tier 1 Model", "Loaded" if t1_model else "Missing",
            "RF Gatekeeper", "green" if t1_model else "amber")
    with s4:
        _mc("Tier 2 Model", "Ready" if t2_model else "Pending",
            "RF Tie-Breaker", "green" if t2_model else "amber")

    st.markdown("<div style='height:0.8rem'></div>", unsafe_allow_html=True)

    # ════════════════════════════════════════════════
    #          MAIN 2-COLUMN LAYOUT
    # ════════════════════════════════════════════════
    col_L, col_R = st.columns([1, 1.15], gap="large")

    # ────────────────────────────────────────────
    #  LEFT COLUMN — DATA ENTRY
    # ────────────────────────────────────────────
    with col_L:

        # ── Outer Box → Tier 1 Form ──
        _sec("🩺", "Tier 1 — Patient Vitals", "gold")

        st.markdown('<div class="outer-box">', unsafe_allow_html=True)
        st.markdown('<div class="inner-card">', unsafe_allow_html=True)

        with st.form("t1_form", clear_on_submit=False):
            c1, c2 = st.columns(2)
            with c1:
                age = st.number_input("Age", 18, 100, 54, help="Patient's age in years")
                cp  = st.selectbox("Chest Pain Type", CP_OPTS, help="Clinical chest pain classification")
            with c2:
                sex = st.selectbox("Sex", SEX_OPTS, index=1)
                bp  = st.slider("Resting Blood Pressure (mm Hg)", 80, 220, 130)

            t1_go = st.form_submit_button("🔬  Run Initial Triage")

        st.markdown('</div></div>', unsafe_allow_html=True)

        # ── Process Tier 1 ──
        if t1_go:
            if t1_model is None:
                st.error("Tier 1 model (Tier_1_model.pkl) not found.")
            else:
                vec = _t1_vec(age, sex, cp, bp, refs)
                prob = _prob(t1_model, vec)
                route = _route(prob)

                st.session_state.t1_prob  = prob
                st.session_state.t1_route = route
                st.session_state.t1_data  = {"age": age, "sex": sex, "cp": cp, "bp": bp}

                # ★ Auto-launch Tier 2 for gray AND high risk
                if route["status"] in ("gray", "high"):
                    st.session_state.tier = "need_t2"
                else:
                    st.session_state.tier = "done_t1"
                    # Clear previous Tier 2 results
                    st.session_state.t2_prob = None
                    st.session_state.t2_route = None
                    st.session_state.shap_vals = None
                    st.session_state.ai_note_text = None

        # ── Tier 2 Form — Auto-launches for gray/high ──
        if st.session_state.tier in ("need_t2", "done_t2"):

            st.markdown("<div style='height:0.6rem'></div>", unsafe_allow_html=True)
            _sec("🧪", "Tier 2 — Full Diagnostic Panel", "amber")

            status = st.session_state.t1_route["status"] if st.session_state.t1_route else "gray"
            if status == "gray":
                st.warning("⚡ **Uncertainty Gate activated** — Initial triage is inconclusive. "
                           "Provide the lab results below for a definitive diagnosis.")
            else:
                st.info("🔍 **Full workup recommended** — High-risk vitals detected. "
                        "Complete lab panel for comprehensive cardiac assessment.")

            st.markdown('<div class="outer-box fade-in">', unsafe_allow_html=True)
            st.markdown('<div class="inner-card">', unsafe_allow_html=True)

            with st.form("t2_form", clear_on_submit=False):
                r1, r2 = st.columns(2)
                with r1:
                    chol = st.number_input("Serum Cholesterol (mg/dL)", 50, 700, 220,
                                           help="Total cholesterol level")
                    fbs  = st.selectbox("Fasting Blood Sugar > 120 mg/dL", [False, True],
                                        format_func=lambda v: "Yes" if v else "No")
                    ecg  = st.selectbox("Resting ECG Result", RESTECG_OPTS,
                                        help="Electrocardiogram classification")
                    thalch = st.number_input("Max Heart Rate", 50, 250, 150,
                                             help="Maximum heart rate achieved during exercise")
                with r2:
                    exang = st.selectbox("Exercise-Induced Angina", [False, True],
                                         format_func=lambda v: "Yes" if v else "No")
                    oldpeak = st.number_input("ST Depression (Oldpeak)", 0.0, 7.0, 1.0, 0.1,
                                              help="ST depression induced by exercise vs rest")
                    slope = st.selectbox("Slope of Peak Exercise ST", SLOPE_OPTS)
                    ca = st.selectbox("Major Vessels (Fluoroscopy)", [0, 1, 2, 3],
                                      help="Number of major vessels colored (0-3)")

                thal = st.selectbox("Thalassemia", THAL_OPTS, index=1)

                t2_go = st.form_submit_button("🎯  Run Final Diagnosis")

            st.markdown('</div></div>', unsafe_allow_html=True)

            # ── Process Tier 2 ──
            if t2_go:
                if t2_model is None:
                    st.error("Tier 2 model (Tier_2_model.pkl) not found.")
                else:
                    p = st.session_state.t1_data
                    vec2 = _t2_vec(
                        p["age"], p["sex"], p["cp"], p["bp"],
                        chol, fbs, ecg, thalch,
                        exang, oldpeak, slope, ca, thal,
                        refs, t2_model)
                    prob2 = _prob(t2_model, vec2)

                    if prob2 >= 0.5:
                        route2 = {"status": "high",
                                  "title": "Positive Cardiac Risk — Admit & Evaluate",
                                  "action": "Full lab analysis confirms elevated cardiac risk. Cardiology consult recommended.",
                                  "next": "Proceed with confirmatory cardiac workup.", "icon": "🚨"}
                    else:
                        route2 = {"status": "low",
                                  "title": "Low Cardiac Risk — Safe for Discharge",
                                  "action": "Full lab profile confirms low cardiac risk. Standard follow-up recommended.",
                                  "next": "Schedule routine follow-up.", "icon": "✅"}

                    st.session_state.t2_prob  = prob2
                    st.session_state.t2_route = route2
                    st.session_state.tier     = "done_t2"

                    # SHAP
                    sv = _shap(t2_model, vec2)
                    st.session_state.shap_vals = sv

                    # AI Note
                    if sv is not None:
                        n = min(8, len(sv))
                        note = _ai_note(prob2 * 100, sv[:n], SHAP_LABELS[:n])
                        st.session_state.ai_note_text = note

    # ────────────────────────────────────────────
    #  RIGHT COLUMN — OUTPUT DASHBOARD
    # ────────────────────────────────────────────
    with col_R:

        _sec("📊", "Triage Decision Dashboard", "blue")

        # ── Empty state ──
        if st.session_state.t1_prob is None:
            st.markdown("""<div class="outer-box empty-state">
                <div class="icon">🏥</div>
                <h3>Awaiting Patient Data</h3>
                <p>Enter vitals on the left and click <b>"Run Initial Triage"</b> to begin.</p>
            </div>""", unsafe_allow_html=True)

        else:
            # ════  TIER 1 RESULTS  ════
            st.markdown("""<div class="outer-box fade-in">""", unsafe_allow_html=True)

            st.markdown("""<div class="sec-hdr" style="margin-top:0">
                <div class="sec-icon gold">🔬</div>
                <h3>Tier 1 — Gatekeeper Assessment</h3>
            </div>""", unsafe_allow_html=True)

            st.markdown('<div class="inner-card">', unsafe_allow_html=True)
            _result_card(st.session_state.t1_prob, st.session_state.t1_route)
            st.progress(float(np.clip(st.session_state.t1_prob, 0, 1)))
            _gate_pills(st.session_state.t1_prob, LOW_THRESHOLD, HIGH_THRESHOLD)
            st.markdown('</div>', unsafe_allow_html=True)

            # Tier 1 Gauge
            if HAS_PLOTLY:
                st.markdown('<div class="inner-card" style="margin-top:0.8rem">', unsafe_allow_html=True)
                _plotly_gauge(st.session_state.t1_prob, "Tier 1 Risk Score")
                st.markdown('</div>', unsafe_allow_html=True)

            st.markdown('</div>', unsafe_allow_html=True)  # close outer-box

            # ════  TIER 2 RESULTS  ════
            if st.session_state.t2_prob is not None:
                st.markdown("<div style='height:0.6rem'></div>", unsafe_allow_html=True)
                st.markdown("""<div class="outer-box fade-in">""", unsafe_allow_html=True)

                st.markdown("""<div class="sec-hdr" style="margin-top:0">
                    <div class="sec-icon amber">🎯</div>
                    <h3>Tier 2 — Definitive Diagnosis</h3>
                </div>""", unsafe_allow_html=True)

                st.markdown('<div class="inner-card">', unsafe_allow_html=True)
                _result_card(st.session_state.t2_prob, st.session_state.t2_route)
                st.progress(float(np.clip(st.session_state.t2_prob, 0, 1)))
                st.markdown('</div>', unsafe_allow_html=True)

                # Comparison cards
                m1, m2 = st.columns(2)
                with m1:
                    st.markdown('<div class="inner-card" style="margin-top:0.6rem">', unsafe_allow_html=True)
                    _mc("Tier 1 Risk", f"{st.session_state.t1_prob:.1%}", "Gatekeeper score", "amber")
                    st.markdown('</div>', unsafe_allow_html=True)
                with m2:
                    t2t = "red" if st.session_state.t2_prob >= 0.5 else "green"
                    st.markdown('<div class="inner-card" style="margin-top:0.6rem">', unsafe_allow_html=True)
                    _mc("Tier 2 Final", f"{st.session_state.t2_prob:.1%}", "Definitive score", t2t)
                    st.markdown('</div>', unsafe_allow_html=True)

                # Tier 2 Gauge
                if HAS_PLOTLY:
                    st.markdown('<div class="inner-card" style="margin-top:0.6rem">', unsafe_allow_html=True)
                    _plotly_gauge(st.session_state.t2_prob, "Tier 2 Risk Score")
                    st.markdown('</div>', unsafe_allow_html=True)

                st.markdown('</div>', unsafe_allow_html=True)  # close outer-box

                # ════  SHAP ANALYSIS  ════
                if st.session_state.shap_vals is not None:
                    st.markdown("<div style='height:0.6rem'></div>", unsafe_allow_html=True)
                    st.markdown("""<div class="outer-box fade-in">""", unsafe_allow_html=True)

                    st.markdown("""<div class="sec-hdr" style="margin-top:0">
                        <div class="sec-icon purple">📈</div>
                        <h3>SHAP Feature Attribution</h3>
                    </div>""", unsafe_allow_html=True)

                    # Bar chart
                    st.markdown('<div class="inner-card">', unsafe_allow_html=True)
                    st.caption("**Contribution of each clinical factor** — Red bars increase risk, green bars reduce it.")
                    _plotly_shap_bars(st.session_state.shap_vals, SHAP_LABELS)
                    st.markdown('</div>', unsafe_allow_html=True)

                    # Pie chart
                    st.markdown('<div class="inner-card" style="margin-top:0.6rem">', unsafe_allow_html=True)
                    st.caption("**Risk vs. Protective Factor Balance**")
                    _plotly_pie(st.session_state.shap_vals, SHAP_LABELS)
                    st.markdown('</div>', unsafe_allow_html=True)

                    # Top factors table
                    n = min(8, len(st.session_state.shap_vals))
                    sv = st.session_state.shap_vals[:n]
                    tbl = pd.DataFrame({
                        "Factor": SHAP_LABELS[:n],
                        "SHAP Value": [f"{v:+.4f}" for v in sv],
                        "Direction": ["↑ Risk" if v > 0 else "↓ Protective" for v in sv],
                        "|Impact|": [f"{abs(v):.4f}" for v in sv],
                    })
                    with st.expander("📋 Detailed Factor Table", expanded=False):
                        st.dataframe(tbl, use_container_width=True, hide_index=True)

                    st.markdown('</div>', unsafe_allow_html=True)  # close outer-box

                # ════  AI CLINICAL NOTE  ════
                if st.session_state.ai_note_text:
                    st.markdown("<div style='height:0.6rem'></div>", unsafe_allow_html=True)
                    _ai_box(st.session_state.ai_note_text)
                else:
                    api_key = os.getenv("open_router_api_key") or os.getenv("OPENROUTER_API_KEY")
                    if not api_key and st.session_state.t2_prob is not None:
                        st.markdown("""<div class="ai-note fade-in" style="opacity:0.7; margin-top:0.6rem">
                            <div class="ai-note-hdr">
                                <div class="ai-note-icon">🤖</div>
                                <h4>AI Clinical Note</h4>
                                <span class="ai-badge" style="background:var(--amber-100);color:var(--amber-600)">OFFLINE</span>
                            </div>
                            <div class="ai-note-body" style="color:var(--text-muted)">
                                Set the <code style="background:var(--bg-inner);padding:2px 6px;border-radius:4px">open_router_api_key</code>
                                environment variable to enable AI-generated clinical notes via Gemini 2.5 Flash.
                            </div>
                        </div>""", unsafe_allow_html=True)

            # ── Gray zone waiting message ──
            elif st.session_state.tier == "need_t2" and st.session_state.t2_prob is None:
                st.markdown("<div style='height:0.6rem'></div>", unsafe_allow_html=True)
                st.markdown("""<div class="outer-box fade-in empty-state"
                     style="background:linear-gradient(135deg, rgba(234,179,8,0.04), rgba(245,158,11,0.03))">
                    <div class="icon">🧪</div>
                    <h3 style="color:var(--gold-600) !important">Tier 2 Panel Active</h3>
                    <p>The lab panel has been revealed on the left. Enter results and click
                    <b>"Run Final Diagnosis"</b> to resolve the uncertainty.</p>
                </div>""", unsafe_allow_html=True)

    # ════  FOOTER  ════
    st.markdown("""<div class="app-ft">
        <strong>⚕️ Clinical Decision Support Only</strong><br>
        Not a diagnosis or replacement for licensed clinician judgment. Built on UCI Cleveland Heart Disease data.
    </div>""", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
