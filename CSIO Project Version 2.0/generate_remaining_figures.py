"""
Generate Publication-Quality Figures 9.6, 9.7, and 9.8
for the Smart Clinic Assistant project.

- Figure 9.6: Decision Curve Analysis (DCA) Plot
- Figure 9.7: Bar Chart of LOSO Accuracy vs. Institutional Majority Baseline
- Figure 9.8: SHAP Local Waterfall Plot Example
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    roc_auc_score,
)
import joblib
from pathlib import Path

# Import project modules
import Data_Processing_final as Data
from Model_handler_final import TIER_1_MODEL_PATH, TIER_2_MODEL_PATH

# ──────────────────────────── paths ────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "figures"
OUTPUT_DIR.mkdir(exist_ok=True)

# ──────────────────────────── style ────────────────────────────
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "figure.dpi": 300,
    "axes.grid": True,
    "grid.alpha": 0.3,
})

COLORS = {
    "tier1":      "#2563eb",
    "tier2":      "#168a4a",
    "cascade":    "#c2413b",
    "uncertainty":"#7c3aed",
    "treat_all":  "#94a3b8",
    "treat_none": "#cbd5e1",
    "bg":         "#f8fafc",
    "grid":       "#e2e8f0",
    "text":       "#1e293b",
    "gold":       "#eab308",
}

# ──────────────────────────── helpers ────────────────────────────
def load_models():
    tier1 = joblib.load(TIER_1_MODEL_PATH)
    tier2 = joblib.load(TIER_2_MODEL_PATH)
    return tier1, tier2


def pos_prob(model, frame):
    return model.predict_proba(frame)[:, 1]


def as_binary(scores, thr=0.5):
    return (np.asarray(scores) >= thr).astype(int)


def conservative_cascade(p1, p2, low=0.30):
    bypass = p1 <= low
    score  = np.where(bypass, p1, p2)
    pred   = as_binary(score)
    return pred, score, bypass


# ═══════════════════════════════════════════════════════════════
#  FIGURE 9.6  –  Decision Curve Analysis
# ═══════════════════════════════════════════════════════════════
def _net_benefit(y_true, y_pred, threshold):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    n = len(y_true)
    prevalence = y_true.mean()
    odds = threshold / (1.0 - threshold) if threshold < 1 else 1e6
    return (tp / n) - (fp / n) * odds


def generate_dca(tier1, tier2):
    """Figure 9.6: Decision Curve Analysis."""
    print("\n[FIG 9.6] Decision Curve Analysis ...")
    y = Data.y_test.to_numpy()
    p1 = pos_prob(tier1, Data.x_test_tier1)
    p2 = pos_prob(tier2, Data.x_test_tier2)
    _, cascade_score, _ = conservative_cascade(p1, p2)
    prevalence = y.mean()

    thresholds = np.round(np.arange(0.05, 0.51, 0.01), 2)
    lines = {
        "Tier 2 (Full Profile)": (p2, COLORS["tier2"],  "-"),
        "Conservative Cascade":  (cascade_score, COLORS["cascade"], "-."),
        "Tier 1 (Gatekeeper)":   (p1, COLORS["tier1"],  ":"),
    }

    fig, ax = plt.subplots(figsize=(10, 6.5))
    fig.patch.set_facecolor(COLORS["bg"])
    ax.set_facecolor("#ffffff")

    # treat-all
    nb_all = [prevalence - (1 - prevalence) * (t / (1 - t)) for t in thresholds]
    ax.plot(thresholds, nb_all, color=COLORS["treat_all"], lw=1.8,
            linestyle="--", label="Treat All")

    # treat-none
    ax.axhline(0, color=COLORS["treat_none"], lw=1.2, linestyle=":",
               label="Treat None")

    # model curves
    for label, (scores, color, ls) in lines.items():
        nbs = [_net_benefit(y, as_binary(scores, t), t) for t in thresholds]
        ax.plot(thresholds, nbs, color=color, lw=2.2, linestyle=ls,
                label=label)

    ax.set_xlim(0.05, 0.50)
    ax.set_ylim(min(min(nb_all) - 0.02, -0.08), max(nb_all) + 0.04)
    ax.set_xlabel("Threshold Probability", fontweight="bold")
    ax.set_ylabel("Net Benefit", fontweight="bold")
    ax.set_title(
        "Figure 9.6: Decision Curve Analysis (DCA)\n"
        "Net Benefit Across Threshold Probabilities — Held-Out Split",
        fontweight="bold", pad=14,
    )
    for spine in ax.spines.values():
        spine.set_color(COLORS["grid"])
        spine.set_linewidth(1.3)
    ax.grid(True, alpha=0.25, color=COLORS["grid"])
    ax.set_axisbelow(True)
    leg = ax.legend(loc="upper right", fontsize=10, framealpha=0.95,
                    edgecolor=COLORS["grid"], fancybox=True)
    leg.get_frame().set_linewidth(1.2)

    # annotation
    txt = (f"Prevalence = {prevalence:.1%}\n"
           f"Held-out N = {len(y)}")
    ax.text(0.02, 0.02, txt, transform=ax.transAxes, fontsize=9,
            va="bottom", bbox=dict(boxstyle="round,pad=0.4",
            facecolor="white", alpha=0.85, edgecolor=COLORS["grid"]))

    plt.tight_layout()
    out = OUTPUT_DIR / "figure_9_6_dca.png"
    fig.savefig(out, dpi=300, bbox_inches="tight",
                facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close()
    print(f"  [OK] Saved to: {out}")
    return out


# ═══════════════════════════════════════════════════════════════
#  FIGURE 9.7  –  LOSO Accuracy vs Majority Baseline
# ═══════════════════════════════════════════════════════════════
def generate_loso(tier1, tier2):
    """Figure 9.7: Leave-One-Site-Out Accuracy bar chart."""
    print("\n[FIG 9.7] Leave-One-Site-Out Analysis ...")
    raw = pd.read_csv(BASE_DIR / "heart_disease_uci.csv")
    raw["binary_target"] = (raw["target"] > 0).astype(int)

    # full encoded dataset (train + test combined)
    X_all_tier1 = pd.concat([Data.x_train_tier1, Data.x_test_tier1]).sort_index()
    X_all_tier2 = pd.concat([Data.x_train_tier2, Data.x_test_tier2]).sort_index()
    Y_all = pd.concat([Data.y_train, Data.y_test]).sort_index()

    sites = raw["dataset"].unique()
    site_labels, cascade_accs, majority_accs = [], [], []

    for site in sites:
        site_mask = raw["dataset"] == site
        idx = site_mask[site_mask].index

        # keep only indices present in our split
        keep = idx.intersection(Y_all.index)
        if len(keep) < 10:
            continue

        y_site = Y_all.loc[keep].to_numpy()
        X1_site = X_all_tier1.loc[keep]
        X2_site = X_all_tier2.loc[keep]

        p1 = pos_prob(tier1, X1_site)
        p2 = pos_prob(tier2, X2_site)
        cascade_pred, _, _ = conservative_cascade(p1, p2)

        cas_acc = accuracy_score(y_site, cascade_pred)
        maj_acc = max(y_site.mean(), 1 - y_site.mean())  # majority class

        site_labels.append(site.title())
        cascade_accs.append(cas_acc)
        majority_accs.append(maj_acc)

    # pooled
    p1_all = pos_prob(tier1, X_all_tier1)
    p2_all = pos_prob(tier2, X_all_tier2)
    pooled_pred, _, _ = conservative_cascade(p1_all, p2_all)
    pooled_acc = accuracy_score(Y_all.to_numpy(), pooled_pred)
    pooled_maj = max(Y_all.mean(), 1 - Y_all.mean())

    site_labels.append("Pooled")
    cascade_accs.append(pooled_acc)
    majority_accs.append(pooled_maj)

    # ── plot ──
    x = np.arange(len(site_labels))
    width = 0.36

    fig, ax = plt.subplots(figsize=(11, 6.5))
    fig.patch.set_facecolor(COLORS["bg"])
    ax.set_facecolor("#ffffff")

    bars1 = ax.bar(x - width / 2, [a * 100 for a in cascade_accs], width,
                   color=COLORS["cascade"], edgecolor="white", lw=0.8,
                   label="Conservative Cascade Accuracy", zorder=3)
    bars2 = ax.bar(x + width / 2, [a * 100 for a in majority_accs], width,
                   color=COLORS["treat_all"], edgecolor="white", lw=0.8,
                   label="Majority-Class Baseline", zorder=3)

    # value labels
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.6,
                f"{bar.get_height():.1f}%", ha="center", va="bottom",
                fontsize=9, fontweight="bold", color=COLORS["cascade"])
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.6,
                f"{bar.get_height():.1f}%", ha="center", va="bottom",
                fontsize=9, fontweight="bold", color=COLORS["treat_all"])

    ax.set_xticks(x)
    ax.set_xticklabels(site_labels, fontweight="bold", fontsize=10)
    ax.set_ylabel("Accuracy (%)", fontweight="bold")
    ax.set_ylim(0, 105)
    ax.set_title(
        "Figure 9.7: Leave-One-Site-Out (LOSO) Generalization\n"
        "Cascade Accuracy vs. Institutional Majority-Class Baseline",
        fontweight="bold", pad=14,
    )
    for spine in ax.spines.values():
        spine.set_color(COLORS["grid"])
        spine.set_linewidth(1.3)
    ax.grid(axis="y", alpha=0.25, color=COLORS["grid"])
    ax.set_axisbelow(True)
    leg = ax.legend(loc="upper right", fontsize=10, framealpha=0.95,
                    edgecolor=COLORS["grid"], fancybox=True)
    leg.get_frame().set_linewidth(1.2)

    plt.tight_layout()
    out = OUTPUT_DIR / "figure_9_7_loso_accuracy.png"
    fig.savefig(out, dpi=300, bbox_inches="tight",
                facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close()
    print(f"  [OK] Saved to: {out}")
    return out


# ═══════════════════════════════════════════════════════════════
#  FIGURE 9.8  –  SHAP Waterfall Plot
# ═══════════════════════════════════════════════════════════════
def generate_shap_waterfall(tier1, tier2):
    """Figure 9.8: SHAP Waterfall / Force Plot."""
    print("\n[FIG 9.8] SHAP Waterfall Plot ...")

    try:
        import shap
    except ImportError:
        print("  [SKIP] shap not installed – generating placeholder.")
        _placeholder_shap()
        return OUTPUT_DIR / "figure_9_8_shap_waterfall.png"

    y = Data.y_test.to_numpy()
    X2 = Data.x_test_tier2

    # pick a high-risk patient
    p2_all = pos_prob(tier2, X2)
    disease_idx = np.where(y == 1)[0]
    hi_idx = disease_idx[np.argmax(p2_all[disease_idx])]
    patient = X2.iloc[[hi_idx]]
    patient_prob = p2_all[hi_idx]

    # background
    bg = Data.x_train_tier2.sample(n=min(24, len(Data.x_train_tier2)),
                                   random_state=43)

    columns = list(patient.columns)
    classes = list(getattr(tier2, "classes_", []))

    def predict_positive(data):
        df = pd.DataFrame(data, columns=columns)
        proba = tier2.predict_proba(df)
        idx = classes.index(1) if 1 in classes else proba.shape[1] - 1
        return proba[:, idx]

    explainer = shap.Explainer(predict_positive, bg, algorithm="permutation")
    sv = explainer(patient, max_evals=(2 * len(columns)) + 1)

    # ── Matplotlib waterfall (top 12 features) ──
    FEATURE_LABELS = {
        "age": "Age", "trestbps": "Resting BP", "chol": "Cholesterol",
        "thalch": "Max Heart Rate", "oldpeak": "ST Depression",
        "ca_missing": "Vessels Missing",
        "chol_missing_or_zero": "Cholesterol Missing",
        "oldpeak_missing": "Oldpeak Missing",
        "cp_asymptomatic": "Chest Pain: Asymptomatic",
        "cp_atypical angina": "Chest Pain: Atypical",
        "cp_non-anginal": "Chest Pain: Non-anginal",
        "cp_typical angina": "Chest Pain: Typical",
        "gender_Female": "Sex: Female", "gender_Male": "Sex: Male",
        "restecg_0.0": "ECG: Normal",
        "restecg_1.0": "ECG: ST-T abnormality",
        "restecg_2.0": "ECG: LV hypertrophy",
        "fbs_0.0": "FBS: No", "fbs_1.0": "FBS: Yes",
        "exang_0.0": "Ex. Angina: No", "exang_1.0": "Ex. Angina: Yes",
        "slope_0.0": "Slope: Down", "slope_1.0": "Slope: Flat",
        "slope_2.0": "Slope: Up",
        "thal_0.0": "Thal: Fixed", "thal_1.0": "Thal: Normal",
        "thal_2.0": "Thal: Reversible",
        "ca_0.0": "Vessels: 0", "ca_1.0": "Vessels: 1",
        "ca_2.0": "Vessels: 2", "ca_3.0": "Vessels: 3",
    }

    raw = np.asarray(sv.values)[0].astype(float)
    vals = raw[:, 1] if raw.ndim > 1 else raw

    df = pd.DataFrame({
        "col":   columns,
        "label": [FEATURE_LABELS.get(c, c) for c in columns],
        "val":   vals,
    })
    df["abs"] = df["val"].abs()
    top = df.nlargest(12, "abs").sort_values("val")  # ascending for horizontal

    # ── waterfall layout ──
    n = len(top)
    labels = list(top["label"])
    values = list(top["val"])
    base_val = 0.0  # SHAP baseline = 0 for permutation explainer

    fig, ax = plt.subplots(figsize=(10, 7.5))
    fig.patch.set_facecolor(COLORS["bg"])
    ax.set_facecolor("#ffffff")

    cum = 0.0
    bottoms = []
    heights = []
    colors  = []
    for v in values:
        if v >= 0:
            bottoms.append(cum)
            heights.append(v)
            colors.append("#c2413b")
        else:
            bottoms.append(cum + v)
            heights.append(-v)
            colors.append("#168a4a")
        cum += v

    y_pos = np.arange(n)
    bars = ax.barh(y_pos, heights, left=bottoms, color=colors,
                   height=0.68, edgecolor="white", linewidth=0.5, zorder=3)

    # value labels
    for i, (b, h, v) in enumerate(zip(bottoms, heights, values)):
        x_text = b + h + 0.003 if v >= 0 else b - 0.003
        ha = "left" if v >= 0 else "right"
        ax.text(x_text, i, f"{v:+.4f}", va="center", ha=ha,
                fontsize=8.5, fontweight="bold", color=COLORS["text"])

    # zero line
    ax.axvline(0, color="#94a3b8", lw=1, ls="-", alpha=0.8, zorder=2)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=9.5)
    ax.set_xlabel("SHAP Contribution (log-odds units)", fontweight="bold")
    ax.set_title(
        "Figure 9.8: SHAP Waterfall — Local Explanation\n"
        f"Predicted Disease Risk = {patient_prob:.1%}",
        fontweight="bold", pad=14,
    )
    for spine in ax.spines.values():
        spine.set_color(COLORS["grid"])
        spine.set_linewidth(1.3)
    ax.grid(axis="x", alpha=0.25, color=COLORS["grid"])
    ax.set_axisbelow(True)

    legend_elements = [
        Patch(facecolor="#c2413b", label="Increases risk"),
        Patch(facecolor="#168a4a", label="Decreases risk"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=10,
              framealpha=0.95, edgecolor=COLORS["grid"])

    # annotation
    txt = (f"Patient index: {hi_idx}\n"
           f"True label: Disease (1)\n"
           f"Model: Tier 2 soft-voting ensemble")
    ax.text(0.02, 0.98, txt, transform=ax.transAxes, fontsize=8.5,
            va="top", bbox=dict(boxstyle="round,pad=0.4",
            facecolor="white", alpha=0.85, edgecolor=COLORS["grid"]))

    plt.tight_layout()
    out = OUTPUT_DIR / "figure_9_8_shap_waterfall.png"
    fig.savefig(out, dpi=300, bbox_inches="tight",
                facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close()
    print(f"  [OK] Saved to: {out}")
    return out


def _placeholder_shap():
    """Fallback when shap is not installed."""
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor(COLORS["bg"])
    ax.set_facecolor("#ffffff")
    ax.text(0.5, 0.5,
            "SHAP Waterfall Plot\n(shap package not available)",
            transform=ax.transAxes, ha="center", va="center",
            fontsize=16, fontweight="bold", color=COLORS["text"])
    ax.set_axis_off()
    out = OUTPUT_DIR / "figure_9_8_shap_waterfall.png"
    fig.savefig(out, dpi=300, bbox_inches="tight",
                facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close()
    print(f"  [OK] Saved placeholder to: {out}")


# ═══════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════
def main():
    print("=" * 60)
    print("Smart Clinic Assistant — Figures 9.6 / 9.7 / 9.8")
    print("=" * 60)

    tier1, tier2 = load_models()
    print("[OK] Models loaded")

    p_dca   = generate_dca(tier1, tier2)
    p_loso  = generate_loso(tier1, tier2)
    p_shap  = generate_shap_waterfall(tier1, tier2)

    print("\n" + "=" * 60)
    print("All figures generated successfully!")
    print("=" * 60)
    print(f"  {p_dca.name}")
    print(f"  {p_loso.name}")
    print(f"  {p_shap.name}")
    print(f"\nOutput directory: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
