"""
Smart Clinic Triage Engine
Production Streamlit frontend for the two-tier heart disease triage workflow.

Run:
    streamlit run app.py
"""

from __future__ import annotations

import os
from html import escape
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

try:
    import plotly.graph_objects as go
    import plotly.express as px

    HAS_PLOTLY = True
except Exception:
    HAS_PLOTLY = False

try:
    import shap

    HAS_SHAP = True
except Exception:
    shap = None
    HAS_SHAP = False

try:
    from openai import OpenAI

    HAS_OPENAI = True
except Exception:
    OpenAI = None
    HAS_OPENAI = False

try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:
    pass


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "heart_disease_uci.csv"
TIER1_MODEL_PATH = BASE_DIR / "Tier_1_model.pkl"
TIER2_MODEL_PATH = BASE_DIR / "Tier_2_model.pkl"

LOW_THRESHOLD = 0.30
HIGH_THRESHOLD = 0.70

TIER1_COLS = [
    "age",
    "gender_Female",
    "gender_Male",
    "cp_asymptomatic",
    "cp_atypical angina",
    "cp_non-anginal",
    "cp_typical angina",
    "trestbps",
]

TIER2_FALLBACK_COLS = [
    "age",
    "trestbps",
    "chol",
    "thalch",
    "oldpeak",
    "ca_missing",
    "chol_missing_or_zero",
    "oldpeak_missing",
    "cp_asymptomatic",
    "cp_atypical angina",
    "cp_non-anginal",
    "cp_typical angina",
    "gender_Female",
    "gender_Male",
    "ca_-1.0",
    "ca_0.0",
    "ca_1.0",
    "ca_2.0",
    "ca_3.0",
    "restecg_-1.0",
    "restecg_0.0",
    "restecg_1.0",
    "restecg_2.0",
    "fbs_-1.0",
    "fbs_0.0",
    "fbs_1.0",
    "exang_-1.0",
    "exang_0.0",
    "exang_1.0",
    "slope_-1.0",
    "slope_0.0",
    "slope_1.0",
    "slope_2.0",
    "thal_-1.0",
    "thal_0.0",
    "thal_1.0",
    "thal_2.0",
]

CP_OPTIONS = ["asymptomatic", "atypical angina", "non-anginal", "typical angina"]
SEX_OPTIONS = ["Female", "Male"]
RESTECG_OPTIONS = ["normal", "st-t abnormality", "lv hypertrophy"]
SLOPE_OPTIONS = ["downsloping", "flat", "upsloping"]
THAL_OPTIONS = ["normal", "fixed defect", "reversable defect"]

RESTECG_MAP = {"normal": 0, "st-t abnormality": 1, "lv hypertrophy": 2}
SLOPE_MAP = {"downsloping": 0, "flat": 1, "upsloping": 2}
THAL_MAP = {"fixed defect": 0, "normal": 1, "reversable defect": 2}

FEATURE_LABELS = {
    "age": "Age",
    "trestbps": "Resting BP",
    "chol": "Cholesterol",
    "thalch": "Max Heart Rate",
    "oldpeak": "ST Depression",
    "ca_missing": "Vessels Missing",
    "chol_missing_or_zero": "Cholesterol Missing",
    "oldpeak_missing": "Oldpeak Missing",
    "cp_asymptomatic": "Chest Pain: Asymptomatic",
    "cp_atypical angina": "Chest Pain: Atypical",
    "cp_non-anginal": "Chest Pain: Non-anginal",
    "cp_typical angina": "Chest Pain: Typical",
    "gender_Female": "Sex: Female",
    "gender_Male": "Sex: Male",
    "restecg_0.0": "ECG: Normal",
    "restecg_1.0": "ECG: ST-T abnormality",
    "restecg_2.0": "ECG: LV hypertrophy",
    "fbs_0.0": "Fasting Blood Sugar: No",
    "fbs_1.0": "Fasting Blood Sugar: Yes",
    "exang_0.0": "Exercise Angina: No",
    "exang_1.0": "Exercise Angina: Yes",
    "slope_0.0": "Slope: Downsloping",
    "slope_1.0": "Slope: Flat",
    "slope_2.0": "Slope: Upsloping",
    "thal_0.0": "Thal: Fixed defect",
    "thal_1.0": "Thal: Normal",
    "thal_2.0": "Thal: Reversable defect",
    "ca_0.0": "Major Vessels: 0",
    "ca_1.0": "Major Vessels: 1",
    "ca_2.0": "Major Vessels: 2",
    "ca_3.0": "Major Vessels: 3",
}

FALLBACK_RANGES = {
    "age": {"min": 28.0, "max": 77.0},
    "trestbps": {"min": 80.0, "max": 200.0},
    "chol": {"min": 85.0, "max": 603.0},
    "thalch": {"min": 60.0, "max": 202.0},
    "oldpeak": {"min": -2.6, "max": 6.2},
}


st.set_page_config(
    page_title="Smart Clinic Triage Engine",
    page_icon="SC",
    layout="wide",
    initial_sidebar_state="collapsed",
)


def inject_css() -> None:
    """CSS injection: warm light theme, box-in-box layout, and Side Rays fallback."""
    st.markdown(
        """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

:root {
    --paper: #fffaf2;
    --paper-2: #fff4e7;
    --panel: rgba(255, 255, 255, 0.86);
    --panel-solid: #ffffff;
    --inner: #fff8ef;
    --ink: #28313a;
    --muted: #746c62;
    --quiet: #9c9286;
    --line: #eadfce;
    --line-strong: #dccab3;
    --gold: #eab308;
    --gold-dark: #a16207;
    --gold-soft: #fff1bd;
    --blue: #96c8ff;
    --blue-dark: #2563eb;
    --blue-soft: #e8f3ff;
    --green: #168a4a;
    --green-soft: #e6f7ed;
    --amber: #d97706;
    --amber-soft: #fff4d6;
    --red: #c2413b;
    --red-soft: #fee7e4;
    --violet: #7c3aed;
    --violet-soft: #f0e9ff;
    --shadow: 0 18px 55px rgba(91, 64, 31, 0.11);
    --shadow-soft: 0 8px 24px rgba(91, 64, 31, 0.08);
    --radius-xl: 28px;
    --radius-lg: 22px;
    --radius-md: 16px;
    --radius-sm: 12px;
}

html, body, .stApp, [data-testid="stAppViewContainer"] {
    font-family: Inter, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif !important;
    background: var(--paper) !important;
    color: var(--ink) !important;
}

.stApp::before {
    content: "";
    position: fixed;
    inset: 0;
    pointer-events: none;
    z-index: 0;
    background:
        radial-gradient(circle at 86% 6%, rgba(234, 179, 8, 0.22), transparent 22rem),
        radial-gradient(circle at 92% 18%, rgba(150, 200, 255, 0.22), transparent 24rem),
        linear-gradient(135deg, #fffaf2 0%, #fff6ea 48%, #f7fbff 100%);
}

.stApp::after {
    content: "";
    position: fixed;
    inset: 0;
    pointer-events: none;
    z-index: 0;
    opacity: .7;
    background-image:
        linear-gradient(118deg, transparent 0 61%, rgba(234,179,8,.13) 61.4%, transparent 64%),
        linear-gradient(126deg, transparent 0 58%, rgba(150,200,255,.15) 58.4%, transparent 61%),
        linear-gradient(136deg, transparent 0 54%, rgba(234,179,8,.09) 54.4%, transparent 57%),
        linear-gradient(145deg, transparent 0 52%, rgba(150,200,255,.10) 52.4%, transparent 55%);
    animation: raysDrift 11s ease-in-out infinite alternate;
}

@keyframes raysDrift {
    from { transform: translate3d(0, 0, 0); opacity: .58; }
    to { transform: translate3d(-16px, 10px, 0); opacity: .9; }
}

[data-testid="stHeader"] {
    background: rgba(255, 250, 242, .72) !important;
    backdrop-filter: blur(18px);
    border-bottom: 1px solid rgba(234, 223, 206, .74);
}

#MainMenu,
footer,
[data-testid="stDecoration"],
[data-testid="stStatusWidget"],
[data-testid="stToolbar"],
[data-testid="stDeployButton"],
[data-testid="stHeaderActionElements"] {
    visibility: hidden !important;
    display: none !important;
}

.block-container {
    max-width: 1460px !important;
    padding: 1.25rem 2rem 3rem !important;
    position: relative;
    z-index: 2;
}

h1, h2, h3, h4, p, span, label, div {
    color: var(--ink);
}

.hero {
    border: 1px solid rgba(220, 202, 179, .86);
    border-radius: var(--radius-xl);
    padding: 1.35rem;
    background: linear-gradient(145deg, rgba(255,255,255,.86), rgba(255,246,234,.82));
    box-shadow: var(--shadow);
    margin-bottom: 1.05rem;
}

.hero-inner {
    border: 1px solid rgba(234, 223, 206, .94);
    border-radius: 22px;
    padding: 1.8rem 2rem;
    background:
        linear-gradient(115deg, rgba(255,255,255,.92), rgba(255,248,239,.76)),
        radial-gradient(circle at top right, rgba(234,179,8,.18), transparent 22rem);
}

.eyebrow {
    display: inline-flex;
    align-items: center;
    gap: .55rem;
    color: var(--gold-dark);
    font-weight: 800;
    font-size: .8rem;
    letter-spacing: .11em;
    text-transform: uppercase;
    margin-bottom: .85rem;
}

.pulse-dot {
    width: 10px;
    height: 10px;
    border-radius: 999px;
    background: var(--gold);
    box-shadow: 0 0 0 7px rgba(234,179,8,.16);
}

.hero-title {
    margin: 0;
    font-size: clamp(2.15rem, 5vw, 4.35rem);
    line-height: .98;
    font-weight: 800;
    letter-spacing: 0;
    color: var(--ink) !important;
}

.hero-copy {
    max-width: 900px;
    margin-top: .9rem;
    font-size: 1.03rem;
    line-height: 1.65;
    color: var(--muted);
}

.outer-card {
    border: 1px solid rgba(220, 202, 179, .9);
    border-radius: var(--radius-xl);
    padding: .85rem;
    background: rgba(255, 255, 255, .70);
    box-shadow: var(--shadow-soft);
    margin-bottom: 1rem;
}

.inner-card {
    border: 1px solid rgba(234, 223, 206, .94);
    border-radius: var(--radius-lg);
    background: rgba(255, 250, 244, .92);
    padding: 1.15rem;
}

div[data-testid="stVerticalBlockBorderWrapper"] {
    border: 1px solid rgba(220, 202, 179, .9) !important;
    border-radius: var(--radius-xl) !important;
    padding: .9rem !important;
    background: rgba(255, 255, 255, .72) !important;
    box-shadow: var(--shadow-soft) !important;
}

div[data-testid="stVerticalBlockBorderWrapper"] > div {
    border-radius: var(--radius-lg) !important;
}

div[data-testid="stVerticalBlockBorderWrapper"] div[data-testid="stVerticalBlock"] {
    gap: .8rem;
}

.section-head {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 1rem;
    margin-bottom: .75rem;
}

.section-title {
    display: flex;
    align-items: center;
    gap: .7rem;
    margin: 0;
    font-size: 1.1rem;
    font-weight: 800;
}

.section-icon {
    width: 38px;
    height: 38px;
    display: grid;
    place-items: center;
    border-radius: 13px;
    background: linear-gradient(135deg, var(--gold-soft), var(--blue-soft));
    border: 1px solid var(--line);
    color: var(--gold-dark);
    font-weight: 900;
}

.mini-pill {
    border: 1px solid var(--line);
    background: rgba(255,255,255,.78);
    border-radius: 999px;
    color: var(--muted);
    font-size: .75rem;
    font-weight: 700;
    padding: .42rem .7rem;
}

.metric-card {
    border: 1px solid var(--line);
    border-radius: 20px;
    padding: 1rem;
    background: rgba(255,255,255,.84);
    box-shadow: 0 8px 22px rgba(91,64,31,.055);
    height: 100%;
}

.metric-label {
    color: var(--muted);
    font-size: .74rem;
    font-weight: 800;
    letter-spacing: .07em;
    text-transform: uppercase;
}

.metric-value {
    color: var(--ink);
    font-size: 1.55rem;
    line-height: 1.05;
    font-weight: 800;
    margin-top: .28rem;
}

.metric-help {
    color: var(--quiet);
    font-size: .78rem;
    margin-top: .32rem;
}

.decision {
    border-radius: 22px;
    padding: 1.15rem;
    border: 1px solid var(--line);
    background: #fff;
}

.decision.low { background: linear-gradient(145deg, #fff, var(--green-soft)); border-color: #b9e8ca; }
.decision.gray { background: linear-gradient(145deg, #fff, var(--amber-soft)); border-color: #f4d58b; }
.decision.high { background: linear-gradient(145deg, #fff, var(--red-soft)); border-color: #f4b3ad; }

.decision-top {
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    gap: 1rem;
}

.decision-kicker {
    color: var(--muted);
    font-size: .74rem;
    font-weight: 800;
    text-transform: uppercase;
    letter-spacing: .08em;
}

.decision-title {
    margin-top: .35rem;
    font-size: 1.25rem;
    font-weight: 800;
}

.decision-score {
    min-width: 110px;
    text-align: right;
    font-size: 2rem;
    font-weight: 800;
}

.decision p {
    margin: .75rem 0 0;
    color: var(--muted);
    line-height: 1.55;
}

.gate-row {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: .5rem;
    margin-top: .85rem;
}

.gate {
    border-radius: 14px;
    border: 1px solid var(--line);
    background: rgba(255,255,255,.72);
    padding: .62rem .7rem;
    color: var(--muted);
    font-size: .78rem;
    font-weight: 700;
}

.gate.active-low { color: var(--green); border-color: #aee3c0; background: var(--green-soft); }
.gate.active-gray { color: var(--amber); border-color: #f1c66e; background: var(--amber-soft); }
.gate.active-high { color: var(--red); border-color: #f0aaa4; background: var(--red-soft); }

.note-card {
    border: 1px solid #d7c5ad;
    border-radius: 24px;
    background: linear-gradient(145deg, #fff, #fff8ed);
    box-shadow: var(--shadow-soft);
    padding: 1rem;
}

.note-inner {
    border: 1px dashed #d9c5aa;
    border-radius: 18px;
    padding: 1.1rem;
    background: rgba(255,255,255,.78);
}

.note-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 1rem;
    margin-bottom: .75rem;
}

.note-title {
    font-weight: 800;
    font-size: 1.05rem;
}

.note-badge {
    border-radius: 999px;
    padding: .28rem .58rem;
    background: var(--blue-soft);
    color: var(--blue-dark);
    font-size: .72rem;
    font-weight: 800;
}

.note-body {
    color: var(--muted);
    line-height: 1.7;
    font-size: .95rem;
}

.empty-state {
    text-align: center;
    padding: 2.7rem 1.5rem;
}

.empty-icon {
    width: 72px;
    height: 72px;
    border-radius: 24px;
    display: grid;
    place-items: center;
    margin: 0 auto 1rem;
    background: linear-gradient(135deg, var(--gold-soft), var(--blue-soft));
    border: 1px solid var(--line);
    font-size: 1.5rem;
    font-weight: 900;
    color: var(--gold-dark);
}

.empty-state h3 {
    margin: .2rem 0;
    font-size: 1.25rem;
}

.empty-state p {
    margin: .4rem auto 0;
    max-width: 440px;
    color: var(--muted);
}

.footer {
    color: var(--quiet);
    text-align: center;
    font-size: .78rem;
    padding: 1.4rem 0 .6rem;
}

.stButton > button {
    border: 0 !important;
    border-radius: 15px !important;
    min-height: 3rem;
    background: linear-gradient(135deg, var(--gold), #f59e0b) !important;
    color: #2b2110 !important;
    font-weight: 800 !important;
    box-shadow: 0 12px 26px rgba(234,179,8,.21) !important;
    transition: transform .18s ease, box-shadow .18s ease, filter .18s ease;
}

.stButton > button:hover {
    transform: translateY(-1px);
    filter: saturate(1.08);
    box-shadow: 0 16px 32px rgba(234,179,8,.28) !important;
}

.stDownloadButton > button {
    border-radius: 15px !important;
    border: 1px solid var(--line) !important;
    background: #fff !important;
    color: var(--ink) !important;
    font-weight: 800 !important;
}

div[data-testid="stAlert"] {
    border-radius: 16px !important;
    border: 1px solid var(--line) !important;
    background: rgba(255,255,255,.82) !important;
}

[data-testid="stMetric"] {
    background: transparent !important;
}

input, textarea, [data-baseweb="select"] > div {
    border-radius: 14px !important;
    background-color: #fff !important;
    color: var(--ink) !important;
    border-color: var(--line-strong) !important;
}

input, textarea {
    color: var(--ink) !important;
    -webkit-text-fill-color: var(--ink) !important;
}

.stNumberInput div[data-baseweb="input"],
.stNumberInput div[data-baseweb="input"] > div,
.stNumberInput div[data-baseweb="input"] input {
    background: #fff !important;
    color: var(--ink) !important;
    -webkit-text-fill-color: var(--ink) !important;
    border-color: var(--line-strong) !important;
}

.stNumberInput button,
.stNumberInput [role="button"] {
    background: #fff8ef !important;
    color: var(--gold-dark) !important;
    border-color: var(--line-strong) !important;
    box-shadow: none !important;
}

.stNumberInput button svg,
.stNumberInput [role="button"] svg {
    color: var(--gold-dark) !important;
    fill: var(--gold-dark) !important;
}

[data-baseweb="select"] span,
[data-baseweb="select"] div,
[data-baseweb="select"] input,
[data-baseweb="select"] p {
    color: var(--ink) !important;
    -webkit-text-fill-color: var(--ink) !important;
}

[data-baseweb="popover"],
[data-baseweb="popover"] > div,
ul[role="listbox"],
li[role="option"] {
    background: #fff !important;
    color: var(--ink) !important;
}

li[role="option"]:hover,
li[role="option"][aria-selected="true"] {
    background: var(--gold-soft) !important;
    color: var(--ink) !important;
}

.stSlider [data-baseweb="slider"] > div {
    color: var(--gold) !important;
}

.stSlider [role="slider"] {
    background: var(--gold) !important;
    border-color: var(--gold-dark) !important;
}

label, .stCaptionContainer, .stMarkdown {
    color: var(--muted) !important;
}

[data-testid="stDataFrame"] {
    border-radius: 16px;
    overflow: hidden;
}

@media (max-width: 900px) {
    .block-container { padding: 1rem .85rem 2rem !important; }
    .hero-inner { padding: 1.3rem; }
    .gate-row { grid-template-columns: 1fr; }
    .decision-top { display: block; }
    .decision-score { text-align: left; margin-top: .5rem; }
}

@media (prefers-reduced-motion: reduce) {
    .stApp::after { animation: none; }
    .stButton > button { transition: none; }
}
</style>
        """,
        unsafe_allow_html=True,
    )


def inject_reactbits_side_rays() -> None:
    """Inject a React Bits SideRays-style WebGL shader as the full-page background."""
    components.html(
        """
<script>
(function () {
    let hostWindow = window;
    let doc = document;
    let useParentLayer = false;

    try {
        if (window.parent && window.parent.document) {
            hostWindow = window.parent;
            doc = window.parent.document;
            useParentLayer = true;
        }
    } catch (error) {
        hostWindow = window;
        doc = document;
        useParentLayer = false;
    }

    const oldCleanup = hostWindow.__smartClinicSideRaysCleanup;
    if (typeof oldCleanup === "function") oldCleanup();

    const oldLayer = doc.getElementById("reactbits-side-rays-bg");
    if (oldLayer) oldLayer.remove();

    const styleId = "reactbits-side-rays-style";
    const oldStyle = doc.getElementById(styleId);
    if (oldStyle) oldStyle.remove();

    const style = doc.createElement("style");
    style.id = styleId;
    style.textContent = `
        #reactbits-side-rays-bg {
            position: fixed;
            inset: 0;
            width: 100vw;
            height: 100vh;
            z-index: 0;
            pointer-events: none;
            overflow: hidden;
            opacity: .82;
        }
        #reactbits-side-rays-bg canvas {
            width: 100%;
            height: 100%;
            display: block;
        }
        @media (prefers-reduced-motion: reduce) {
            #reactbits-side-rays-bg { display: none; }
        }
    `;
    doc.head.appendChild(style);

    const layer = doc.createElement("div");
    layer.id = "reactbits-side-rays-bg";
    if (useParentLayer) {
        doc.body.prepend(layer);
    } else {
        const frame = window.frameElement;
        if (frame) {
            frame.style.position = "fixed";
            frame.style.inset = "0";
            frame.style.width = "100vw";
            frame.style.height = "100vh";
            frame.style.border = "0";
            frame.style.zIndex = "0";
            frame.style.pointerEvents = "none";
            frame.style.background = "transparent";
        }
        doc.documentElement.style.margin = "0";
        doc.body.style.margin = "0";
        doc.body.style.overflow = "hidden";
        doc.body.appendChild(layer);
    }

    const canvas = doc.createElement("canvas");
    layer.appendChild(canvas);

    const gl = canvas.getContext("webgl", { alpha: true, antialias: true, premultipliedAlpha: false });
    if (!gl) return;

    const settings = {
        speed: 2.5,
        rayColor1: "#EAB308",
        rayColor2: "#96c8ff",
        intensity: 2.0,
        spread: 2.0,
        origin: "top-right",
        tilt: 0.0,
        saturation: 1.5,
        blend: 0.75,
        falloff: 1.6,
        opacity: 0.78
    };

    function hexToRgb(hex) {
        const m = /^#?([a-f\\d]{2})([a-f\\d]{2})([a-f\\d]{2})$/i.exec(hex);
        return m ? [
            parseInt(m[1], 16) / 255,
            parseInt(m[2], 16) / 255,
            parseInt(m[3], 16) / 255
        ] : [1, 1, 1];
    }

    function originToFlip(origin) {
        switch (origin) {
            case "top-left": return [1, 0];
            case "bottom-right": return [0, 1];
            case "bottom-left": return [1, 1];
            default: return [0, 0];
        }
    }

    const vertexSource = `
        attribute vec2 position;
        void main() {
            gl_Position = vec4(position, 0.0, 1.0);
        }
    `;

    const fragmentSource = `
        precision highp float;

        uniform float iTime;
        uniform vec2 iResolution;
        uniform float iSpeed;
        uniform vec3 iRayColor1;
        uniform vec3 iRayColor2;
        uniform float iIntensity;
        uniform float iSpread;
        uniform float iFlipX;
        uniform float iFlipY;
        uniform float iTilt;
        uniform float iSaturation;
        uniform float iBlend;
        uniform float iFalloff;
        uniform float iOpacity;

        float rayStrength(vec2 raySource, vec2 rayRefDirection, vec2 coord, float seedA, float seedB, float speed) {
            vec2 sourceToCoord = coord - raySource;
            float cosAngle = dot(normalize(sourceToCoord), rayRefDirection);
            return clamp(
                (0.45 + 0.15 * sin(cosAngle * seedA + iTime * speed)) +
                (0.3 + 0.2 * cos(-cosAngle * seedB + iTime * speed)),
                0.0, 1.0
            ) * clamp((iResolution.x - length(sourceToCoord)) / iResolution.x, 0.5, 1.0);
        }

        void main() {
            vec2 fragCoord = gl_FragCoord.xy;
            if (iFlipX > 0.5) fragCoord.x = iResolution.x - fragCoord.x;
            if (iFlipY > 0.5) fragCoord.y = iResolution.y - fragCoord.y;

            vec2 coord = vec2(fragCoord.x, iResolution.y - fragCoord.y);
            vec2 rayPos = vec2(iResolution.x * 1.1, -0.5 * iResolution.y);

            float tiltRad = iTilt * 3.14159265 / 180.0;
            float cs = cos(tiltRad);
            float sn = sin(tiltRad);
            vec2 rel = coord - rayPos;
            vec2 tiltedCoord = vec2(rel.x * cs - rel.y * sn, rel.x * sn + rel.y * cs) + rayPos;

            float halfSpread = iSpread * 0.275;
            vec2 rayRefDir1 = normalize(vec2(cos(0.785398 + halfSpread), sin(0.785398 + halfSpread)));
            vec2 rayRefDir2 = normalize(vec2(cos(0.785398 - halfSpread), sin(0.785398 - halfSpread)));

            vec4 rays1 = vec4(iRayColor1, 1.0) * rayStrength(rayPos, rayRefDir1, tiltedCoord, 36.2214, 21.11349, iSpeed);
            vec4 rays2 = vec4(iRayColor2, 1.0) * rayStrength(rayPos, rayRefDir2, tiltedCoord, 22.3991, 18.0234, iSpeed * 0.2);

            vec4 color = rays1 * (1.0 - iBlend) * 0.9 + rays2 * iBlend * 0.9;

            float distanceToLight = length(fragCoord.xy - vec2(rayPos.x, iResolution.y - rayPos.y)) / iResolution.y;
            float brightness = iIntensity * 0.4 / pow(max(distanceToLight, 0.001), iFalloff);
            color.rgb *= brightness;

            float gray = dot(color.rgb, vec3(0.299, 0.587, 0.114));
            color.rgb = mix(vec3(gray), color.rgb, iSaturation);

            color.a = max(color.r, max(color.g, color.b)) * iOpacity;
            gl_FragColor = color;
        }
    `;

    function compile(type, source) {
        const shader = gl.createShader(type);
        gl.shaderSource(shader, source);
        gl.compileShader(shader);
        if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
            console.warn(gl.getShaderInfoLog(shader));
            gl.deleteShader(shader);
            return null;
        }
        return shader;
    }

    const vertexShader = compile(gl.VERTEX_SHADER, vertexSource);
    const fragmentShader = compile(gl.FRAGMENT_SHADER, fragmentSource);
    if (!vertexShader || !fragmentShader) return;

    const program = gl.createProgram();
    gl.attachShader(program, vertexShader);
    gl.attachShader(program, fragmentShader);
    gl.linkProgram(program);
    if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
        console.warn(gl.getProgramInfoLog(program));
        return;
    }
    gl.useProgram(program);

    const buffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 3, -1, -1, 3]), gl.STATIC_DRAW);

    const positionLocation = gl.getAttribLocation(program, "position");
    gl.enableVertexAttribArray(positionLocation);
    gl.vertexAttribPointer(positionLocation, 2, gl.FLOAT, false, 0, 0);

    const uniforms = {
        iTime: gl.getUniformLocation(program, "iTime"),
        iResolution: gl.getUniformLocation(program, "iResolution"),
        iSpeed: gl.getUniformLocation(program, "iSpeed"),
        iRayColor1: gl.getUniformLocation(program, "iRayColor1"),
        iRayColor2: gl.getUniformLocation(program, "iRayColor2"),
        iIntensity: gl.getUniformLocation(program, "iIntensity"),
        iSpread: gl.getUniformLocation(program, "iSpread"),
        iFlipX: gl.getUniformLocation(program, "iFlipX"),
        iFlipY: gl.getUniformLocation(program, "iFlipY"),
        iTilt: gl.getUniformLocation(program, "iTilt"),
        iSaturation: gl.getUniformLocation(program, "iSaturation"),
        iBlend: gl.getUniformLocation(program, "iBlend"),
        iFalloff: gl.getUniformLocation(program, "iFalloff"),
        iOpacity: gl.getUniformLocation(program, "iOpacity")
    };

    const [flipX, flipY] = originToFlip(settings.origin);
    const color1 = hexToRgb(settings.rayColor1);
    const color2 = hexToRgb(settings.rayColor2);
    let animationId = null;

    function resize() {
        const dpr = Math.min(hostWindow.devicePixelRatio || 1, 2);
        const width = Math.max(1, Math.floor(hostWindow.innerWidth * dpr));
        const height = Math.max(1, Math.floor(hostWindow.innerHeight * dpr));
        if (canvas.width !== width || canvas.height !== height) {
            canvas.width = width;
            canvas.height = height;
            gl.viewport(0, 0, width, height);
        }
        return [width, height];
    }

    function render(time) {
        const [width, height] = resize();
        gl.clearColor(0, 0, 0, 0);
        gl.clear(gl.COLOR_BUFFER_BIT);
        gl.useProgram(program);
        gl.uniform1f(uniforms.iTime, time * 0.001);
        gl.uniform2f(uniforms.iResolution, width, height);
        gl.uniform1f(uniforms.iSpeed, settings.speed);
        gl.uniform3f(uniforms.iRayColor1, color1[0], color1[1], color1[2]);
        gl.uniform3f(uniforms.iRayColor2, color2[0], color2[1], color2[2]);
        gl.uniform1f(uniforms.iIntensity, settings.intensity);
        gl.uniform1f(uniforms.iSpread, settings.spread);
        gl.uniform1f(uniforms.iFlipX, flipX);
        gl.uniform1f(uniforms.iFlipY, flipY);
        gl.uniform1f(uniforms.iTilt, settings.tilt);
        gl.uniform1f(uniforms.iSaturation, settings.saturation);
        gl.uniform1f(uniforms.iBlend, settings.blend);
        gl.uniform1f(uniforms.iFalloff, settings.falloff);
        gl.uniform1f(uniforms.iOpacity, settings.opacity);
        gl.drawArrays(gl.TRIANGLES, 0, 3);
        animationId = hostWindow.requestAnimationFrame(render);
    }

    hostWindow.addEventListener("resize", resize);
    animationId = hostWindow.requestAnimationFrame(render);

    hostWindow.__smartClinicSideRaysCleanup = function () {
        if (animationId) hostWindow.cancelAnimationFrame(animationId);
        hostWindow.removeEventListener("resize", resize);
        const ext = gl.getExtension("WEBGL_lose_context");
        if (ext) ext.loseContext();
        if (layer && layer.parentNode) layer.parentNode.removeChild(layer);
        if (style && style.parentNode) style.parentNode.removeChild(style);
    };
})();
</script>
        """,
        height=0,
        width=0,
    )


@st.cache_resource(show_spinner=False)
def load_model(path: Path) -> Any | None:
    return joblib.load(path) if path.exists() else None


@st.cache_data(show_spinner=False)
def load_dataset() -> pd.DataFrame:
    if not DATA_PATH.exists():
        return pd.DataFrame()

    df = pd.read_csv(DATA_PATH)
    if "target" in df.columns:
        df["target_binary"] = (pd.to_numeric(df["target"], errors="coerce").fillna(0) > 0).astype(int)
    return df


@st.cache_data(show_spinner=False)
def reference_ranges(df: pd.DataFrame) -> dict[str, dict[str, float]]:
    if df.empty:
        return FALLBACK_RANGES

    try:
        from sklearn.model_selection import train_test_split

        y = (pd.to_numeric(df["target"], errors="coerce").fillna(0) > 0).astype(int)
        train_df, _ = train_test_split(df, test_size=0.2, stratify=y, random_state=43)
    except Exception:
        train_df = df

    ranges: dict[str, dict[str, float]] = {}
    for col, fallback in FALLBACK_RANGES.items():
        series = pd.to_numeric(train_df.get(col), errors="coerce")
        if col == "chol":
            series = series.replace(0, np.nan)
        if col == "trestbps":
            series = series.replace(0, np.nan)
        series = series.dropna()
        if series.empty:
            ranges[col] = fallback
        else:
            ranges[col] = {"min": float(series.min()), "max": float(series.max())}
    return ranges


def scale_value(value: float | int | None, column: str, refs: dict[str, dict[str, float]]) -> float:
    if value is None:
        return 0.0
    bounds = refs.get(column, FALLBACK_RANGES[column])
    lo = bounds["min"]
    hi = bounds["max"]
    if hi <= lo:
        return 0.0
    return float(np.clip((float(value) - lo) / (hi - lo), 0.0, 1.0))


def tier1_vector(age: int, sex: str, cp: str, bp: int, refs: dict[str, dict[str, float]]) -> pd.DataFrame:
    row = {name: 0.0 for name in TIER1_COLS}
    row["age"] = scale_value(age, "age", refs)
    row["trestbps"] = scale_value(bp, "trestbps", refs)
    row[f"gender_{sex}"] = 1.0
    cp_col = f"cp_{cp}"
    if cp_col in row:
        row[cp_col] = 1.0
    return pd.DataFrame([row], columns=TIER1_COLS)


def tier2_columns(model: Any | None) -> list[str]:
    if model is not None and hasattr(model, "feature_names_in_"):
        return list(model.feature_names_in_)
    return TIER2_FALLBACK_COLS


def set_one_hot(row: dict[str, float], key: str) -> None:
    if key in row:
        row[key] = 1.0


def tier2_vector(
    model: Any | None,
    refs: dict[str, dict[str, float]],
    *,
    age: int,
    sex: str,
    cp: str,
    bp: int,
    chol: int,
    fbs: bool,
    restecg: str,
    thalch: int,
    exang: bool,
    oldpeak: float,
    slope: str,
    ca: int,
    thal: str,
) -> pd.DataFrame:
    cols = tier2_columns(model)
    row = {col: 0.0 for col in cols}

    row["age"] = scale_value(age, "age", refs) if "age" in row else 0.0
    row["trestbps"] = scale_value(bp, "trestbps", refs) if "trestbps" in row else 0.0
    row["chol"] = scale_value(chol, "chol", refs) if "chol" in row else 0.0
    row["thalch"] = scale_value(thalch, "thalch", refs) if "thalch" in row else 0.0
    row["oldpeak"] = scale_value(oldpeak, "oldpeak", refs) if "oldpeak" in row else 0.0

    if "ca_missing" in row:
        row["ca_missing"] = 0.0
    if "chol_missing_or_zero" in row:
        row["chol_missing_or_zero"] = 1.0 if chol == 0 else 0.0
    if "oldpeak_missing" in row:
        row["oldpeak_missing"] = 0.0

    set_one_hot(row, f"gender_{sex}")
    set_one_hot(row, f"cp_{cp}")
    set_one_hot(row, f"restecg_{float(RESTECG_MAP[restecg])}")
    set_one_hot(row, f"fbs_{1.0 if fbs else 0.0}")
    set_one_hot(row, f"exang_{1.0 if exang else 0.0}")
    set_one_hot(row, f"slope_{float(SLOPE_MAP[slope])}")
    set_one_hot(row, f"ca_{float(ca)}")
    set_one_hot(row, f"thal_{float(THAL_MAP[thal])}")

    return pd.DataFrame([row], columns=cols)


def boolish(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)


def tier2_background_frame(
    model: Any | None,
    refs: dict[str, dict[str, float]],
    dataset: pd.DataFrame,
    limit: int = 24,
) -> pd.DataFrame:
    cols = tier2_columns(model)
    fallback = pd.DataFrame([{col: 0.0 for col in cols}], columns=cols)

    required = [
        "age",
        "gender",
        "cp",
        "trestbps",
        "chol",
        "fbs",
        "restecg",
        "thalch",
        "exang",
        "oldpeak",
        "slope",
        "ca",
        "thal",
    ]
    if dataset.empty or any(col not in dataset.columns for col in required):
        return fallback

    rows = dataset.dropna(subset=required)
    if rows.empty:
        return fallback
    if len(rows) > limit:
        rows = rows.sample(n=limit, random_state=43)

    frames = []
    for row in rows.itertuples(index=False):
        values = row._asdict()
        try:
            frames.append(
                tier2_vector(
                    model,
                    refs,
                    age=int(values["age"]),
                    sex=str(values["gender"]),
                    cp=str(values["cp"]),
                    bp=int(values["trestbps"]),
                    chol=int(values["chol"]),
                    fbs=boolish(values["fbs"]),
                    restecg=str(values["restecg"]),
                    thalch=int(values["thalch"]),
                    exang=boolish(values["exang"]),
                    oldpeak=float(values["oldpeak"]),
                    slope=str(values["slope"]),
                    ca=int(float(values["ca"])),
                    thal=str(values["thal"]),
                )
            )
        except Exception:
            continue

    if not frames:
        return fallback
    return pd.concat(frames, ignore_index=True).reindex(columns=cols, fill_value=0.0)


def predict_probability(model: Any, frame: pd.DataFrame) -> float:
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(frame)
        classes = list(getattr(model, "classes_", range(probabilities.shape[1])))
        positive_index = classes.index(1) if 1 in classes else probabilities.shape[1] - 1
        return float(probabilities[0, positive_index])
    return float(model.predict(frame)[0])


def route_for_probability(probability: float, final: bool = False) -> dict[str, str]:
    if final:
        if probability >= 0.5:
            return {
                "status": "high",
                "title": "High cardiac risk",
                "action": "The full diagnostic profile indicates elevated cardiac risk. Prioritize clinician review and confirmatory cardiac workup.",
                "next": "Admit or escalate according to local protocol.",
            }
        return {
            "status": "low",
            "title": "Low cardiac risk",
            "action": "The full diagnostic profile is below the positive-risk threshold. Routine follow-up is still recommended.",
            "next": "Discharge planning may be considered after clinician review.",
        }

    if probability <= LOW_THRESHOLD:
        return {
            "status": "low",
            "title": "Low-risk discharge candidate",
            "action": "Tier 1 vitals sit below the low-risk gate. No automatic lab escalation is required.",
            "next": "Review clinically before discharge.",
        }
    if probability >= HIGH_THRESHOLD:
        return {
            "status": "high",
            "title": "Danger zone detected",
            "action": "Tier 1 vitals exceed the high-risk gate. The full Tier 2 diagnostic workflow is now active.",
            "next": "Complete the lab panel immediately.",
        }
    return {
        "status": "gray",
        "title": "Uncertainty gate activated",
        "action": "Tier 1 is inconclusive. The system requests additional labs before making a final routing decision.",
        "next": "Complete the Tier 2 panel.",
    }


def format_pct(value: float | None) -> str:
    return "--" if value is None else f"{value:.1%}"


def status_color(status: str) -> str:
    return {"low": "#168a4a", "gray": "#d97706", "high": "#c2413b"}.get(status, "#746c62")


def metric_card(label: str, value: str, help_text: str) -> None:
    st.markdown(
        f"""
<div class="metric-card">
    <div class="metric-label">{escape(label)}</div>
    <div class="metric-value">{escape(value)}</div>
    <div class="metric-help">{escape(help_text)}</div>
</div>
        """,
        unsafe_allow_html=True,
    )


def section(title: str, pill: str) -> None:
    st.markdown(
        f"""
<div class="section-head">
    <div class="section-title"><div class="section-icon">+</div>{escape(title)}</div>
    <div class="mini-pill">{escape(pill)}</div>
</div>
        """,
        unsafe_allow_html=True,
    )


def decision_card(probability: float, route: dict[str, str], label: str) -> None:
    status = route["status"]
    color = status_color(status)
    st.markdown(
        f"""
<div class="decision {status}">
    <div class="decision-top">
        <div>
            <div class="decision-kicker">{escape(label)}</div>
            <div class="decision-title">{escape(route["title"])}</div>
        </div>
        <div class="decision-score" style="color:{color}">{probability:.1%}</div>
    </div>
    <p>{escape(route["action"])}</p>
    <div class="gate-row">
        <div class="gate {'active-low' if status == 'low' else ''}">0-30% Low risk</div>
        <div class="gate {'active-gray' if status == 'gray' else ''}">30-70% Gray zone</div>
        <div class="gate {'active-high' if status == 'high' else ''}">70-100% Danger zone</div>
    </div>
</div>
        """,
        unsafe_allow_html=True,
    )


def gauge(probability: float, title: str) -> None:
    if not HAS_PLOTLY:
        st.progress(float(np.clip(probability, 0, 1)))
        return

    color = status_color(route_for_probability(probability)["status"])
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=probability * 100,
            number={"suffix": "%", "font": {"size": 34, "color": "#28313a"}},
            title={"text": title, "font": {"size": 15, "color": "#746c62"}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": "#9c9286"},
                "bar": {"color": color, "thickness": 0.28},
                "bgcolor": "rgba(255,255,255,0.65)",
                "borderwidth": 1,
                "bordercolor": "#eadfce",
                "steps": [
                    {"range": [0, 30], "color": "#e6f7ed"},
                    {"range": [30, 70], "color": "#fff4d6"},
                    {"range": [70, 100], "color": "#fee7e4"},
                ],
                "threshold": {"line": {"color": "#28313a", "width": 3}, "thickness": 0.75, "value": probability * 100},
            },
        )
    )
    fig.update_layout(
        height=265,
        margin=dict(l=18, r=18, t=48, b=12),
        paper_bgcolor="rgba(0,0,0,0)",
        font_family="Inter",
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def shap_values_for(model: Any, frame: pd.DataFrame, background: pd.DataFrame | None = None) -> pd.DataFrame | None:
    if not HAS_SHAP:
        return None

    try:
        columns = list(frame.columns)
        classes = list(getattr(model, "classes_", []))

        def predict_positive(data: Any) -> np.ndarray:
            data_frame = pd.DataFrame(data, columns=columns)
            probabilities = model.predict_proba(data_frame)
            positive_index = classes.index(1) if 1 in classes else probabilities.shape[1] - 1
            return probabilities[:, positive_index]

        if background is None or background.empty:
            background = pd.DataFrame([{col: 0.0 for col in columns}], columns=columns)
        else:
            background = background.reindex(columns=columns, fill_value=0.0)

        explainer = shap.Explainer(predict_positive, background, algorithm="permutation")
        values = explainer(frame, max_evals=(2 * len(columns)) + 1)
        raw = np.asarray(values.values)[0].astype(float)

        result = pd.DataFrame(
            {
                "feature": frame.columns,
                "label": [FEATURE_LABELS.get(col, col) for col in frame.columns],
                "value": raw,
            }
        )
        result["impact"] = result["value"].abs()
        result["direction"] = np.where(result["value"] >= 0, "Raises risk", "Lowers risk")
        return result.sort_values("impact", ascending=False).reset_index(drop=True)
    except Exception:
        return None


def plot_shap_bars(shap_df: pd.DataFrame) -> None:
    top = shap_df.head(12).sort_values("impact", ascending=True)
    colors = np.where(top["value"] >= 0, "#c2413b", "#168a4a")

    if HAS_PLOTLY:
        fig = go.Figure(
            go.Bar(
                x=top["value"],
                y=top["label"],
                orientation="h",
                marker_color=colors,
                hovertemplate="%{y}<br>Contribution: %{x:.4f}<extra></extra>",
            )
        )
        fig.add_vline(x=0, line_width=1, line_color="#9c9286")
        fig.update_layout(
            height=380,
            margin=dict(l=8, r=8, t=14, b=22),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(255,255,255,.62)",
            font_family="Inter",
            xaxis_title="Local contribution",
            yaxis_title=None,
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    else:
        st.bar_chart(top.set_index("label")["value"])


def plot_factor_balance(shap_df: pd.DataFrame) -> None:
    raising = float(shap_df.loc[shap_df["value"] > 0, "impact"].sum())
    lowering = float(shap_df.loc[shap_df["value"] < 0, "impact"].sum())
    total = raising + lowering

    if total <= 0:
        st.info("No factor balance available for this patient.")
        return

    if HAS_PLOTLY:
        fig = go.Figure(
            go.Pie(
                labels=["Risk-raising factors", "Protective factors"],
                values=[raising, lowering],
                hole=.58,
                marker_colors=["#c2413b", "#168a4a"],
                textinfo="label+percent",
            )
        )
        fig.update_layout(
            height=315,
            margin=dict(l=8, r=8, t=14, b=14),
            paper_bgcolor="rgba(0,0,0,0)",
            font_family="Inter",
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    else:
        st.write(pd.DataFrame({"group": ["Risk-raising", "Protective"], "impact": [raising, lowering]}))


def plot_patient_profile(inputs: dict[str, Any], refs: dict[str, dict[str, float]]) -> None:
    if not HAS_PLOTLY:
        return

    rows = []
    for key, label in [
        ("age", "Age"),
        ("bp", "Resting BP"),
        ("chol", "Cholesterol"),
        ("thalch", "Max Heart Rate"),
        ("oldpeak", "ST Depression"),
    ]:
        if key not in inputs:
            continue
        ref_key = "trestbps" if key == "bp" else key
        rows.append({"Metric": label, "Scaled value": scale_value(inputs[key], ref_key, refs)})

    if not rows:
        return

    fig = px.line_polar(
        pd.DataFrame(rows),
        r="Scaled value",
        theta="Metric",
        line_close=True,
        range_r=[0, 1],
    )
    fig.update_traces(fill="toself", line_color="#eab308", fillcolor="rgba(234,179,8,.18)")
    fig.update_layout(
        height=330,
        margin=dict(l=18, r=18, t=18, b=18),
        paper_bgcolor="rgba(0,0,0,0)",
        polar=dict(
            bgcolor="rgba(255,255,255,.65)",
            radialaxis=dict(showticklabels=False, ticks="", gridcolor="#eadfce"),
            angularaxis=dict(gridcolor="#eadfce"),
        ),
        font_family="Inter",
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def local_clinical_note(probability: float, shap_df: pd.DataFrame | None) -> str:
    risk_text = "low" if probability < 0.5 else "elevated"
    base = (
        f"Your overall cardiac risk estimate is {probability:.1%}, which falls in the {risk_text} range for this screening workflow. "
    )

    if shap_df is None or shap_df.empty:
        return (
            base
            + "The result should be interpreted by a licensed clinician alongside symptoms, examination findings, and local care protocols. "
            + "Maintain heart-healthy habits and arrange follow-up if symptoms persist or worsen."
        )

    top_risk = shap_df[shap_df["value"] > 0].head(2)["label"].tolist()
    top_protective = shap_df[shap_df["value"] < 0].head(2)["label"].tolist()

    parts = [base]
    if top_risk:
        parts.append("The main factors pushing risk upward are " + ", ".join(top_risk).lower() + ". ")
    if top_protective:
        parts.append("Reassuring factors include " + ", ".join(top_protective).lower() + ". ")
    parts.append(
        "This is clinical decision support, not a diagnosis. A clinician should confirm the result and decide whether additional testing or treatment is needed."
    )
    return "".join(parts)


def ai_clinical_note(probability: float, shap_df: pd.DataFrame | None) -> str:
    if not HAS_OPENAI:
        return local_clinical_note(probability, shap_df)

    api_key = os.getenv("open_router_api_key") or os.getenv("OPENROUTER_API_KEY")
    if not api_key or shap_df is None or shap_df.empty:
        return local_clinical_note(probability, shap_df)

    factors = "\n".join(
        f"- {row.label}: {row.value:.4f}"
        for row in shap_df.head(8).itertuples(index=False)
    )
    prompt = f"""
You are a highly experienced Consultant Cardiologist (20+ years of clinical practice) conducting an in-person consultation with a patient after reviewing their cardiovascular assessment.

Your responsibility is to explain the assessment exactly as an experienced physician would during a hospital consultation.

The patient is NOT a medical professional.

--------------------------------------------------
PATIENT ASSESSMENT
--------------------------------------------------

Estimated Heart Disease Risk:
{probability * 100:.1f}%

Clinical Findings:
{factors}

Interpretation of Findings:
• Positive contribution → increases the estimated risk.
• Negative contribution → decreases the estimated risk.

--------------------------------------------------
YOUR TASK
--------------------------------------------------

Write ONE natural clinical explanation for the patient.

The explanation should sound exactly like a compassionate, experienced cardiologist speaking face-to-face with the patient.

The patient should finish reading it feeling:

• informed
• reassured
• educated
• aware of the important findings

without feeling frightened or confused.

--------------------------------------------------
IMPORTANT MEDICAL INSTRUCTIONS
--------------------------------------------------

Explain ONLY the 3–4 findings with the greatest influence.

For every important finding:

1. State WHAT was observed.
2. Explain WHY that finding matters medically.
3. Explain HOW it influences the patient's estimated heart risk.
4. Explain whether it is reassuring or concerning.

Always balance positive and negative findings.

Never exaggerate.

Never create unnecessary alarm.

Never guarantee that the patient has or does not have heart disease.

Present the assessment as a probability—not a diagnosis.

Whenever possible, explain medical concepts using everyday language.

For example:

Instead of

"Flat ST slope"

say something similar to

"The electrical changes seen during your exercise test suggest that your heart may not respond to physical activity as efficiently as expected."

Likewise,

instead of

"Exercise Angina: No"

say

"It's reassuring that you do not experience chest discomfort during physical activity, which lowers the likelihood of significant heart-related symptoms."

Translate EVERY technical feature into language an average patient can understand.

--------------------------------------------------
STRICT RULES
--------------------------------------------------

DO NOT mention:

• SHAP
• Machine Learning
• Artificial Intelligence
• AI
• Algorithm
• Model
• Feature importance
• Contribution values
• Weights
• Data analysis
• Prediction engine
• Probability model

Never mention any numerical contribution values.

Never mention internal feature names.

Never write raw labels such as

"Chest Pain: Asymptomatic"

"Slope: Flat"

"Major Vessels"

"Oldpeak"

"Thal"

"CA"

Translate everything into natural medical language.

Never list raw medical variables.

Never sound like software.

Never sound like ChatGPT.

Never sound like an AI assistant.

Never say:

"Based on our assessment..."

"Let's break this down..."

"The model predicts..."

"The analysis indicates..."

"Our assessment suggests..."

Instead, speak naturally as a doctor.

--------------------------------------------------
STYLE
--------------------------------------------------

Use "you" throughout.

Professional.

Warm.

Confident.

Reassuring.

Human.

Conversational.

Natural.

Avoid repetitive sentence structures.

Vary sentence length.

Avoid generic advice.

Every recommendation should relate to the patient's findings whenever possible.

--------------------------------------------------
STRUCTURE
--------------------------------------------------

Paragraph 1
Briefly explain the estimated risk in one natural sentence.

Paragraph 2

Provide 3–4 bullet points.

Each bullet should contain:

• an appropriate emoji
• the observation
• why it matters
• whether it increases or decreases risk

Do NOT include technical terminology unless immediately explained.

Paragraph 3

Give one personalized lifestyle recommendation based on the findings.

Avoid generic advice.

Instead of saying

"eat healthy"

say something meaningful such as

"Because your exercise test showed changes that deserve attention, maintaining regular moderate physical activity, together with controlling blood pressure and cholesterol, can help reduce future cardiovascular risk."

Paragraph 4

End with a reassuring conclusion that encourages regular follow-up without creating fear.

--------------------------------------------------
OUTPUT REQUIREMENTS
--------------------------------------------------

Length:
140–180 words.

Formatting:
One opening paragraph.
3–4 bullet points.
One recommendation paragraph.
One closing paragraph.

Output ONLY the final explanation.

No headings.

No markdown.

No code fences.

No introductory text.

The response must be indistinguishable from what an experienced cardiologist would say during a real clinical consultation.

"""

    try:
        client = OpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1")
        response = client.chat.completions.create(
            model="google/gemini-2.5-flash",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=350,
            temperature=0.45,
            frequency_penalty=0.15,
            presence_penalty=0.05,
            top_p=0.92,
        )
        return response.choices[0].message.content.strip()
    except Exception:
        return local_clinical_note(probability, shap_df)


def note_box(text: str, live: bool = False) -> None:
    badge = "GENAI" if live else "LOCAL NOTE"
    st.markdown(
        f"""
<div class="note-card">
    <div class="note-inner">
        <div class="note-header">
            <div class="note-title">AI Clinical Note</div>
            <div class="note-badge">{badge}</div>
        </div>
        <div class="note-body">{escape(text)}</div>
    </div>
</div>
        """,
        unsafe_allow_html=True,
    )


def initialize_state() -> None:
    defaults = {
        "tier_state": "idle",
        "t1_prob": None,
        "t1_route": None,
        "t2_prob": None,
        "t2_route": None,
        "t1_inputs": {},
        "t2_inputs": {},
        "shap_df": None,
        "clinical_note": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def reset_downstream() -> None:
    st.session_state.t2_prob = None
    st.session_state.t2_route = None
    st.session_state.t2_inputs = {}
    st.session_state.shap_df = None
    st.session_state.clinical_note = None


def main() -> None:
    inject_css()
    inject_reactbits_side_rays()
    initialize_state()

    tier1_model = load_model(TIER1_MODEL_PATH)
    tier2_model = load_model(TIER2_MODEL_PATH)
    dataset = load_dataset()
    refs = reference_ranges(dataset)

    def run_tier1() -> None:
        if tier1_model is None:
            st.session_state.t1_route = {
                "status": "gray",
                "title": "Tier 1 model missing",
                "action": "Tier_1_model.pkl was not found. Add the model file and run again.",
                "next": "Model unavailable.",
            }
            return

        inputs = {
            "age": int(st.session_state.age),
            "sex": st.session_state.sex,
            "cp": st.session_state.cp,
            "bp": int(st.session_state.bp),
        }
        frame = tier1_vector(inputs["age"], inputs["sex"], inputs["cp"], inputs["bp"], refs)
        probability = predict_probability(tier1_model, frame)
        route = route_for_probability(probability)

        st.session_state.t1_inputs = inputs
        st.session_state.t1_prob = probability
        st.session_state.t1_route = route
        reset_downstream()

        if route["status"] in {"gray", "high"}:
            st.session_state.tier_state = "tier2_required"
        else:
            st.session_state.tier_state = "tier1_complete"

    def run_tier2() -> None:
        if tier2_model is None:
            st.session_state.t2_route = {
                "status": "gray",
                "title": "Tier 2 model missing",
                "action": "Tier_2_model.pkl was not found. Add the model file and run again.",
                "next": "Model unavailable.",
            }
            return

        if not st.session_state.t1_inputs:
            run_tier1()

        t1 = st.session_state.t1_inputs
        t2 = {
            "chol": int(st.session_state.chol),
            "fbs": bool(st.session_state.fbs),
            "restecg": st.session_state.restecg,
            "thalch": int(st.session_state.thalch),
            "exang": bool(st.session_state.exang),
            "oldpeak": float(st.session_state.oldpeak),
            "slope": st.session_state.slope,
            "ca": int(st.session_state.ca),
            "thal": st.session_state.thal,
        }
        frame = tier2_vector(tier2_model, refs, **t1, **t2)
        probability = predict_probability(tier2_model, frame)
        route = route_for_probability(probability, final=True)
        background = tier2_background_frame(tier2_model, refs, dataset)
        shap_df = shap_values_for(tier2_model, frame, background)
        note = ai_clinical_note(probability, shap_df)

        st.session_state.t2_inputs = t2
        st.session_state.t2_prob = probability
        st.session_state.t2_route = route
        st.session_state.shap_df = shap_df
        st.session_state.clinical_note = note
        st.session_state.tier_state = "tier2_complete"

    st.markdown(
        """
<div class="hero">
    <div class="hero-inner">
        <div class="eyebrow"><span class="pulse-dot"></span>Smart Clinic Assistant</div>
        <h1 class="hero-title">Cost-aware cardiac triage, built for clinical flow.</h1>
        <div class="hero-copy">
            A two-tier machine learning cascade screens with basic vitals first, then unlocks the full diagnostic panel only when the patient enters the gray zone or danger zone.
        </div>
    </div>
</div>
        """,
        unsafe_allow_html=True,
    )

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        metric_card("Patient Records", f"{len(dataset):,}" if not dataset.empty else "--", "UCI heart disease records")
    with k2:
        if "target_binary" in dataset:
            metric_card("Dataset Prevalence", f"{dataset['target_binary'].mean():.1%}", "Observed positive target rate")
        else:
            metric_card("Dataset Prevalence", "--", "Dataset unavailable")
    with k3:
        metric_card("Tier 1 Model", "Ready" if tier1_model is not None else "Missing", "Vitals gatekeeper")
    with k4:
        metric_card("Tier 2 Model", "Ready" if tier2_model is not None else "Missing", "Full diagnostic panel")

    st.markdown("<div style='height:.85rem'></div>", unsafe_allow_html=True)

    left, right = st.columns([1.0, 1.12], gap="large")

    with left:
        section("Tier 1 intake", "Free vitals")
        with st.container(border=True):
            c1, c2 = st.columns(2)
            with c1:
                st.number_input("Age", min_value=18, max_value=100, value=54, step=1, key="age")
                st.selectbox("Chest pain type", CP_OPTIONS, index=0, key="cp")
            with c2:
                st.selectbox("Sex", SEX_OPTIONS, index=1, key="sex")
                st.slider("Resting blood pressure", min_value=80, max_value=220, value=130, step=1, key="bp")

            if tier1_model is None:
                st.error("Tier_1_model.pkl is missing from the project folder.")

            if st.button("Run initial triage", use_container_width=True):
                run_tier1()
                st.rerun()

        if st.session_state.tier_state in {"tier2_required", "tier2_complete"}:
            st.markdown("<div style='height:.25rem'></div>", unsafe_allow_html=True)
            section("Tier 2 diagnostics", "Auto-launched")
            with st.container(border=True):
                if st.session_state.t1_route and st.session_state.t1_route["status"] == "high":
                    st.warning("Danger zone detected from Tier 1. The Tier 2 diagnostic panel has been launched automatically.")
                else:
                    st.info("Gray zone detected from Tier 1. Add labs and stress-test findings for the final decision.")

                d1, d2 = st.columns(2)
                with d1:
                    st.number_input("Serum cholesterol", min_value=50, max_value=700, value=220, step=1, key="chol")
                    st.selectbox(
                        "Fasting blood sugar > 120 mg/dL",
                        [False, True],
                        format_func=lambda value: "Yes" if value else "No",
                        key="fbs",
                    )
                    st.selectbox("Resting ECG", RESTECG_OPTIONS, index=0, key="restecg")
                    st.number_input("Maximum heart rate", min_value=50, max_value=250, value=150, step=1, key="thalch")
                with d2:
                    st.selectbox(
                        "Exercise-induced angina",
                        [False, True],
                        format_func=lambda value: "Yes" if value else "No",
                        key="exang",
                    )
                    st.number_input("ST depression", min_value=-3.0, max_value=7.0, value=1.0, step=0.1, key="oldpeak")
                    st.selectbox("Slope of peak exercise ST", SLOPE_OPTIONS, index=1, key="slope")
                    st.selectbox("Major vessels colored", [0, 1, 2, 3], index=0, key="ca")
                st.selectbox("Thalassemia", THAL_OPTIONS, index=0, key="thal")

                if tier2_model is None:
                    st.error("Tier_2_model.pkl is missing from the project folder.")

                if st.button("Run final diagnosis", use_container_width=True):
                    run_tier2()
                    st.rerun()

    with right:
        section("Decision dashboard", "Live output")

        if st.session_state.t1_prob is None:
            st.markdown(
                """
<div class="outer-card empty-state">
    <div class="empty-icon">SC</div>
    <h3>Awaiting patient intake</h3>
    <p>Enter Tier 1 vitals and run the initial triage. The lab panel will open automatically for gray-zone and danger-zone cases.</p>
</div>
                """,
                unsafe_allow_html=True,
            )
        else:
            with st.container(border=True):
                decision_card(st.session_state.t1_prob, st.session_state.t1_route, "Tier 1 gatekeeper")
                gauge(st.session_state.t1_prob, "Tier 1 cardiac risk")

            if st.session_state.tier_state == "tier2_required":
                st.markdown(
                    """
<div class="outer-card empty-state">
    <div class="empty-icon">T2</div>
    <h3>Tier 2 panel is active</h3>
    <p>The patient requires full diagnostic review. Complete the Tier 2 inputs on the left and run the final diagnosis.</p>
</div>
                    """,
                    unsafe_allow_html=True,
                )

            if st.session_state.t2_prob is not None:
                with st.container(border=True):
                    decision_card(st.session_state.t2_prob, st.session_state.t2_route, "Tier 2 final diagnosis")
                    gauge(st.session_state.t2_prob, "Tier 2 final risk")

                profile_inputs = {**st.session_state.t1_inputs, **st.session_state.t2_inputs}
                with st.container(border=True):
                    st.caption("Patient profile scaled against the training reference ranges.")
                    plot_patient_profile(profile_inputs, refs)

                if isinstance(st.session_state.shap_df, pd.DataFrame) and not st.session_state.shap_df.empty:
                    with st.container(border=True):
                        st.caption("Top local drivers. Red raises risk; green lowers risk.")
                        plot_shap_bars(st.session_state.shap_df)

                    s_col1, s_col2 = st.columns([.9, 1.1], gap="medium")
                    with s_col1:
                        with st.container(border=True):
                            st.caption("Risk vs protective contribution balance.")
                            plot_factor_balance(st.session_state.shap_df)
                    with s_col2:
                        with st.container(border=True):
                            display_df = st.session_state.shap_df.head(10).copy()
                            display_df = display_df[["label", "value", "direction", "impact"]]
                            display_df.columns = ["Factor", "Contribution", "Direction", "Impact"]
                            st.dataframe(
                                display_df,
                                use_container_width=True,
                                hide_index=True,
                                column_config={
                                    "Contribution": st.column_config.NumberColumn(format="%.4f"),
                                    "Impact": st.column_config.NumberColumn(format="%.4f"),
                                },
                            )
                else:
                    with st.container(border=True):
                        st.info("SHAP is unavailable for this model/runtime. The final risk score is still shown above.")

                if st.session_state.clinical_note:
                    note_box(
                        st.session_state.clinical_note,
                        live=bool(os.getenv("open_router_api_key") or os.getenv("OPENROUTER_API_KEY")),
                    )

    st.markdown(
        """
<div class="footer">
    Clinical decision support only. This tool is not a diagnosis and does not replace licensed clinician judgment.
</div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
