"""
Generate publication-quality figures for the Smart Clinic Assistant project.
Generates:
- Figure 9.1: ROC Curve (Tier 1, Tier 2, and Cascade models)
- Figure 9.2: Confusion Matrix (Tier 1, Tier 2, and Cascade models)
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for Windows
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_curve,
    auc,
    confusion_matrix,
    accuracy_score,
)
import joblib
from pathlib import Path

# Import project modules
import Data_Processing_final as Data
from Model_handler_final import TIER_1_MODEL_PATH as TIER1_MODEL_PATH, TIER_2_MODEL_PATH as TIER2_MODEL_PATH

# Set up paths
BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "figures"
OUTPUT_DIR.mkdir(exist_ok=True)

# Configure matplotlib for high-quality output
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['figure.dpi'] = 300

# Color palette (professional medical journal style)
COLORS = {
    'tier1': '#2563eb',      # Blue
    'tier2': '#168a4a',      # Green
    'cascade': '#c2413b',    # Red
    'background': '#f8fafc',
    'grid': '#e2e8f0',
    'text': '#1e293b',
}

def load_models():
    """Load trained models from disk."""
    tier1_model = joblib.load(TIER1_MODEL_PATH)
    tier2_model = joblib.load(TIER2_MODEL_PATH)
    return tier1_model, tier2_model

def conservative_cascade(p1, p2, low_threshold=0.30):
    """Apply conservative cascade policy."""
    bypass = p1 <= low_threshold
    score = np.where(bypass, p1, p2)
    pred = (score >= 0.5).astype(int)
    return pred, score, bypass

def generate_roc_curve(tier1_model, tier2_model):
    """
    Figure 9.1: ROC Curve
    Shows ROC curves for Tier 1, Tier 2, and Cascade models.
    """
    print("Generating Figure 9.1: ROC Curve...")
    
    # Get predictions
    y_test = Data.y_test.to_numpy()
    p1 = tier1_model.predict_proba(Data.x_test_tier1)[:, 1]
    p2 = tier2_model.predict_proba(Data.x_test_tier2)[:, 1]
    cascade_pred, cascade_score, _ = conservative_cascade(p1, p2)
    
    # Calculate ROC curves
    fpr1, tpr1, _ = roc_curve(y_test, p1)
    fpr2, tpr2, _ = roc_curve(y_test, p2)
    fpr_c, tpr_c, _ = roc_curve(y_test, cascade_score)
    
    # Calculate AUC
    auc1 = auc(fpr1, tpr1)
    auc2 = auc(fpr2, tpr2)
    auc_c = auc(fpr_c, tpr_c)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Set background
    fig.patch.set_facecolor(COLORS['background'])
    ax.set_facecolor('#ffffff')
    
    # Plot ROC curves
    ax.plot(fpr1, tpr1, color=COLORS['tier1'], linewidth=2.5, 
            label=f'Tier 1 (Gatekeeper) AUC = {auc1:.3f}', linestyle='-')
    ax.plot(fpr2, tpr2, color=COLORS['tier2'], linewidth=2.5, 
            label=f'Tier 2 (Full Profile) AUC = {auc2:.3f}', linestyle='--')
    ax.plot(fpr_c, tpr_c, color=COLORS['cascade'], linewidth=2.5, 
            label=f'Cascade (Combined) AUC = {auc_c:.3f}', linestyle='-.')
    
    # Plot diagonal reference line
    ax.plot([0, 1], [0, 1], color='#94a3b8', linewidth=1, linestyle=':', 
            label='Random Classifier')
    
    # Formatting
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate (1 - Specificity)', fontweight='bold')
    ax.set_ylabel('True Positive Rate (Sensitivity)', fontweight='bold')
    ax.set_title('Figure 9.1: Receiver Operating Characteristic (ROC) Curve\n'
                 'Smart Clinic Assistant - Heart Disease Triage System', 
                 fontweight='bold', pad=20)
    
    # Grid
    ax.grid(True, alpha=0.3, color=COLORS['grid'], linestyle='-')
    ax.set_axisbelow(True)
    
    # Legend
    legend = ax.legend(loc='lower right', fontsize=11, framealpha=0.95, 
                      edgecolor=COLORS['grid'], fancybox=True)
    legend.get_frame().set_linewidth(1.5)
    
    # Add text box with performance metrics
    textstr = (f'Test Set: N = {len(y_test)}\n'
               f'Prevalence: {y_test.mean():.1%}')
    props = dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8, 
                edgecolor=COLORS['grid'])
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)
    
    # Spine styling
    for spine in ax.spines.values():
        spine.set_color(COLORS['grid'])
        spine.set_linewidth(1.5)
    
    plt.tight_layout()
    
    # Save figure
    output_path = OUTPUT_DIR / "figure_9_1_roc_curve.png"
    fig.savefig(output_path, dpi=300, bbox_inches='tight', 
                facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close()
    
    print("[OK] Saved ROC Curve to: " + str(output_path))
    return output_path

def generate_confusion_matrices(tier1_model, tier2_model):
    """
    Figure 9.2: Confusion Matrices
    Shows confusion matrices for Tier 1, Tier 2, and Cascade models.
    """
    print("Generating Figure 9.2: Confusion Matrices...")
    
    # Get predictions
    y_test = Data.y_test.to_numpy()
    p1 = tier1_model.predict_proba(Data.x_test_tier1)[:, 1]
    p2 = tier2_model.predict_proba(Data.x_test_tier2)[:, 1]
    
    y_pred1 = (p1 >= 0.5).astype(int)
    y_pred2 = (p2 >= 0.5).astype(int)
    cascade_pred, _, _ = conservative_cascade(p1, p2)
    
    # Calculate confusion matrices
    cm1 = confusion_matrix(y_test, y_pred1, labels=[0, 1])
    cm2 = confusion_matrix(y_test, y_pred2, labels=[0, 1])
    cm_c = confusion_matrix(y_test, cascade_pred, labels=[0, 1])
    
    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Set background
    fig.patch.set_facecolor(COLORS['background'])
    
    # Model data for iteration
    models_data = [
        ('Tier 1 (Gatekeeper)', cm1, y_pred1, COLORS['tier1']),
        ('Tier 2 (Full Profile)', cm2, y_pred2, COLORS['tier2']),
        ('Cascade (Combined)', cm_c, cascade_pred, COLORS['cascade']),
    ]
    
    for idx, (title, cm, y_pred, color) in enumerate(models_data):
        ax = axes[idx]
        ax.set_facecolor('#ffffff')
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        # Plot with custom colormap
        im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues, vmin=0)
        
        # Add text annotations
        fmt = 'd'
        thresh = cm.max() / 2.
        for i, j in np.ndindex(cm.shape):
            ax.text(j, i, format(cm[i, j], fmt),
                   ha="center", va="center",
                   color="white" if cm[i, j] > thresh else "black",
                   fontsize=16, fontweight='bold')
        
        # Labels
        ax.set_title(f'{title}\nAccuracy: {accuracy:.1%}', 
                    fontweight='bold', fontsize=13, pad=10)
        ax.set_xlabel('Predicted Label', fontweight='bold', fontsize=11)
        ax.set_ylabel('True Label', fontweight='bold', fontsize=11)
        
        # Set tick labels
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['No Disease', 'Disease'], fontsize=10)
        ax.set_yticks([0, 1])
        ax.set_yticklabels(['No Disease', 'Disease'], fontsize=10)
        
        # Add metrics text box
        textstr = (f'Sensitivity: {sensitivity:.1%}\n'
                  f'Specificity: {specificity:.1%}\n'
                  f'PPV: {tp/(tp+fp):.1%}\n'
                  f'NPV: {tn/(tn+fn):.1%}')
        props = dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8,
                    edgecolor=COLORS['grid'])
        ax.text(0.98, 0.02, textstr, transform=ax.transAxes, fontsize=9,
               verticalalignment='bottom', horizontalalignment='right', bbox=props)
        
        # Grid styling
        ax.grid(False)
        for spine in ax.spines.values():
            spine.set_color(COLORS['grid'])
            spine.set_linewidth(1.5)
    
    # Main title
    fig.suptitle('Figure 9.2: Confusion Matrices\nSmart Clinic Assistant - Heart Disease Triage System',
                fontweight='bold', fontsize=15, y=1.02)
    
    plt.tight_layout()
    
    # Save figure
    output_path = OUTPUT_DIR / "figure_9_2_confusion_matrix.png"
    fig.savefig(output_path, dpi=300, bbox_inches='tight',
                facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close()
    
    print("[OK] Saved Confusion Matrices to: " + str(output_path))
    return output_path

def main():
    """Main function to generate all figures."""
    print("=" * 60)
    print("Smart Clinic Assistant - Figure Generation")
    print("=" * 60)
    print()
    
    # Load models
    print("Loading models...")
    tier1_model, tier2_model = load_models()
    print("[OK] Models loaded successfully")
    print()
    
    # Generate figures
    roc_path = generate_roc_curve(tier1_model, tier2_model)
    cm_path = generate_confusion_matrices(tier1_model, tier2_model)
    
    print()
    print("=" * 60)
    print("Figure Generation Complete!")
    print("=" * 60)
    print("Output directory: " + str(OUTPUT_DIR))
    print("  - " + roc_path.name)
    print("  - " + cm_path.name)
    print()
    print("These figures are publication-ready at 300 DPI.")
    print("Use them in your report as Figure 9.1 and Figure 9.2.")

if __name__ == "__main__":
    main()
