"""
Generate SHAP Explanation Screenshot for Figure 9.3
This script runs the Tier 2 model on a sample patient and visualizes the SHAP contributions.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import joblib
from pathlib import Path
import shap

# Import project modules
import Data_Processing_final as Data
from Model_handler_final import TIER_1_MODEL_PATH as TIER1_MODEL_PATH, TIER_2_MODEL_PATH as TIER2_MODEL_PATH

# Set up paths
BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "figures"
OUTPUT_DIR.mkdir(exist_ok=True)

# Configure matplotlib
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 11
plt.rcParams['figure.dpi'] = 300

# Feature labels for the Tier 2 model (matching app.py)
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

def load_models():
    """Load trained models from disk."""
    tier1_model = joblib.load(TIER1_MODEL_PATH)
    tier2_model = joblib.load(TIER2_MODEL_PATH)
    return tier1_model, tier2_model

def get_shap_values(tier2_model, patient_frame, background_frame):
    """Calculate SHAP values for a patient."""
    columns = list(patient_frame.columns)
    classes = list(getattr(tier2_model, "classes_", []))
    
    def predict_positive(data):
        data_frame = pd.DataFrame(data, columns=columns)
        probabilities = tier2_model.predict_proba(data_frame)
        positive_index = classes.index(1) if 1 in classes else probabilities.shape[1] - 1
        return probabilities[:, positive_index]
    
    explainer = shap.Explainer(predict_positive, background_frame, algorithm="permutation")
    shap_values = explainer(patient_frame, max_evals=(2 * len(columns)) + 1)
    return shap_values

def generate_shap_figure(tier1_model, tier2_model):
    """
    Figure 9.3: SHAP Explanation Screenshot
    Shows SHAP waterfall/bar plot for a sample high-risk patient.
    """
    print("Generating Figure 9.3: SHAP Explanation Screenshot...")
    
    # Select a high-risk patient from test set (one with disease)
    y_test = Data.y_test.to_numpy()
    X_test_tier2 = Data.x_test_tier2
    
    # Find patients with disease (target=1)
    disease_indices = np.where(y_test == 1)[0]
    
    # Get probabilities to find a high-confidence case
    p2_all = tier2_model.predict_proba(X_test_tier2)[:, 1]
    high_risk_idx = disease_indices[np.argmax(p2_all[disease_indices])]
    
    # Get patient data
    patient_frame = X_test_tier2.iloc[[high_risk_idx]]
    patient_prob = p2_all[high_risk_idx]
    
    print(f"Selected patient at index {high_risk_idx} with predicted probability: {patient_prob:.2%}")
    
    # Create background frame (sample from training data)
    background_frame = Data.x_train_tier2.sample(n=min(24, len(Data.x_train_tier2)), random_state=43)
    
    # Calculate SHAP values
    print("Calculating SHAP values...")
    shap_values = get_shap_values(tier2_model, patient_frame, background_frame)
    
    # Extract contributions for positive class
    raw = np.asarray(shap_values.values)[0].astype(float)
    
    # Create DataFrame with feature contributions
    shap_df = pd.DataFrame({
        'feature': patient_frame.columns,
        'label': [FEATURE_LABELS.get(col, col) for col in patient_frame.columns],
        'value': raw[:, 1] if raw.ndim > 1 else raw,
    })
    shap_df['impact'] = shap_df['value'].abs()
    shap_df['direction'] = np.where(shap_df['value'] >= 0, 'Raises risk', 'Lowers risk')
    shap_df = shap_df.sort_values('impact', ascending=False).reset_index(drop=True)
    
    # Take top 12 features for visualization
    top_features = shap_df.head(12).sort_values('impact', ascending=True)
    
    # Create the figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Set background
    fig.patch.set_facecolor('#f8fafc')
    ax.set_facecolor('#ffffff')
    
    # Create horizontal bar chart
    colors = ['#c2413b' if v >= 0 else '#168a4a' for v in top_features['value']]
    bars = ax.barh(top_features['label'], top_features['value'], color=colors, 
                   height=0.7, edgecolor='white', linewidth=0.5)
    
    # Add value labels on bars
    for bar, val in zip(bars, top_features['value']):
        width = bar.get_width()
        label_x = width + 0.002 if width >= 0 else width - 0.002
        ha = 'left' if width >= 0 else 'right'
        ax.text(label_x, bar.get_y() + bar.get_height()/2, 
                f'{val:.4f}', va='center', ha=ha, fontsize=9, fontweight='bold')
    
    # Vertical line at zero
    ax.axvline(x=0, color='#94a3b8', linewidth=1, linestyle='-', alpha=0.8)
    
    # Labels and title
    ax.set_xlabel('SHAP Contribution to Disease Risk', fontweight='bold', fontsize=12)
    ax.set_title(f'Figure 9.3: SHAP Feature Contributions - Local Explanation\n'
                 f'Predicted Heart Disease Risk: {patient_prob:.1%}', 
                 fontweight='bold', fontsize=13, pad=15)
    
    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#c2413b', label='Increases risk'),
        Patch(facecolor='#168a4a', label='Decreases risk')
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=10, 
              framealpha=0.95, edgecolor='#e2e8f0')
    
    # Grid
    ax.grid(axis='x', alpha=0.3, color='#e2e8f0')
    ax.set_axisbelow(True)
    
    # Spine styling
    for spine in ax.spines.values():
        spine.set_color('#e2e8f0')
        spine.set_linewidth(1.5)
    
    # Add annotation box
    textstr = (f'Patient Profile\n'
               f'True Label: Disease\n'
               f'N features: {len(patient_frame.columns)}')
    props = dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8, 
                edgecolor='#e2e8f0')
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    
    # Save figure
    output_path = OUTPUT_DIR / "figure_9_3_shap_explanation.png"
    fig.savefig(output_path, dpi=300, bbox_inches='tight', 
                facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close()
    
    print("[OK] Saved SHAP Explanation to: " + str(output_path))
    return output_path

def main():
    """Main function to generate SHAP explanation figure."""
    print("=" * 60)
    print("Smart Clinic Assistant - SHAP Figure Generation")
    print("=" * 60)
    print()
    
    # Load models
    print("Loading models...")
    tier1_model, tier2_model = load_models()
    print("[OK] Models loaded successfully")
    print()
    
    # Generate SHAP figure
    shap_path = generate_shap_figure(tier1_model, tier2_model)
    
    print()
    print("=" * 60)
    print("SHAP Figure Generation Complete!")
    print("=" * 60)
    print("Output: " + str(shap_path))
    print()
    print("Use this as Figure 9.3 in your report.")

if __name__ == "__main__":
    main()
