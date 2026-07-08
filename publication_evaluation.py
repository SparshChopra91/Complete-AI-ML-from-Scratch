from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

import Data_Processing_final as Data
from Model_handler_final import (
    TIER_1_MODEL_PATH,
    TIER_2_MODEL_PATH,
    build_tier1_model,
)


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "publication_outputs"
LOW_THRESHOLD = 0.30
HIGH_THRESHOLD = 0.70


@dataclass
class ClassificationSummary:
    name: str
    n: int
    accuracy: float
    balanced_accuracy: float
    precision: float
    recall: float
    specificity: float
    f1: float
    mcc: float
    auroc: float
    brier: float
    tn: int
    fp: int
    fn: int
    tp: int


def positive_probability(model, frame: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(frame)[:, 1]


def as_binary(probabilities: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    return (probabilities >= threshold).astype(int)


def metric_summary(name: str, y_true, y_pred, y_score) -> ClassificationSummary:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if (tn + fp) else float("nan")
    return ClassificationSummary(
        name=name,
        n=int(len(y_true)),
        accuracy=float(accuracy_score(y_true, y_pred)),
        balanced_accuracy=float(balanced_accuracy_score(y_true, y_pred)),
        precision=float(precision_score(y_true, y_pred, zero_division=0)),
        recall=float(recall_score(y_true, y_pred, zero_division=0)),
        specificity=float(specificity),
        f1=float(f1_score(y_true, y_pred, zero_division=0)),
        mcc=float(matthews_corrcoef(y_true, y_pred)),
        auroc=float(roc_auc_score(y_true, y_score)),
        brier=float(brier_score_loss(y_true, y_score)),
        tn=int(tn),
        fp=int(fp),
        fn=int(fn),
        tp=int(tp),
    )


def expected_calibration_error(y_true, probabilities, n_bins: int = 10) -> float:
    y_true = np.asarray(y_true)
    probabilities = np.asarray(probabilities)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(probabilities, bins[1:-1], right=True)
    ece = 0.0
    for bin_id in range(n_bins):
        mask = bin_ids == bin_id
        if not np.any(mask):
            continue
        observed = float(np.mean(y_true[mask]))
        predicted = float(np.mean(probabilities[mask]))
        ece += float(np.mean(mask)) * abs(observed - predicted)
    return ece


def reliability_bins(y_true, probabilities, model_name: str, n_bins: int = 10) -> pd.DataFrame:
    y_true = np.asarray(y_true)
    probabilities = np.asarray(probabilities)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(probabilities, bins[1:-1], right=True)
    rows = []
    for bin_id in range(n_bins):
        mask = bin_ids == bin_id
        rows.append(
            {
                "model": model_name,
                "bin": bin_id + 1,
                "bin_low": bins[bin_id],
                "bin_high": bins[bin_id + 1],
                "n": int(mask.sum()),
                "mean_predicted": float(np.mean(probabilities[mask])) if np.any(mask) else np.nan,
                "observed_rate": float(np.mean(y_true[mask])) if np.any(mask) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def conservative_cascade(p1, p2):
    bypass = p1 <= LOW_THRESHOLD
    score = np.where(bypass, p1, p2)
    pred = np.where(bypass, as_binary(p1), as_binary(p2))
    return pred, score, bypass


def uncertainty_band_policy(p1, p2):
    tier2 = (p1 > LOW_THRESHOLD) & (p1 < HIGH_THRESHOLD)
    score = np.where(tier2, p2, p1)
    pred = np.where(tier2, as_binary(p2), as_binary(p1))
    return pred, score, ~tier2


def bypass_safety(y_true, policy_name: str, bypass_mask, y_pred, route_type: str = "all_bypass") -> dict:
    bypass_y = np.asarray(y_true)[bypass_mask]
    bypass_pred = np.asarray(y_pred)[bypass_mask]
    bypass_n = int(bypass_mask.sum())
    bypass_positives = int(bypass_y.sum()) if bypass_n else 0
    missed_positives = int(((bypass_y == 1) & (bypass_pred == 0)).sum()) if bypass_n else 0
    bypass_negatives = int((bypass_y == 0).sum()) if bypass_n else 0
    predicted_low = int((bypass_pred == 0).sum()) if bypass_n else 0
    true_low_among_predicted_low = int(((bypass_y == 0) & (bypass_pred == 0)).sum()) if bypass_n else 0
    return {
        "policy": policy_name,
        "route_type": route_type,
        "bypass_n": bypass_n,
        "bypass_rate": float(bypass_n / len(y_true)),
        "bypass_prevalence": float(bypass_y.mean()) if bypass_n else np.nan,
        "bypass_positives": bypass_positives,
        "missed_positives": missed_positives,
        "bypass_false_negative_rate": float(missed_positives / bypass_positives) if bypass_positives else 0.0,
        "bypass_npv": float(true_low_among_predicted_low / predicted_low) if predicted_low else np.nan,
        "bypass_negatives": bypass_negatives,
    }


def threshold_sweep(y_true, p1, p2) -> pd.DataFrame:
    rows = []
    for threshold in np.round(np.arange(0.05, 0.61, 0.05), 2):
        bypass = p1 <= threshold
        score = np.where(bypass, p1, p2)
        pred = np.where(bypass, as_binary(p1), as_binary(p2))
        bypass_y = np.asarray(y_true)[bypass]
        missed = int(((bypass_y == 1) & (pred[bypass] == 0)).sum()) if bypass.any() else 0
        positives = int(bypass_y.sum()) if bypass.any() else 0
        rows.append(
            {
                "low_threshold": threshold,
                "tier2_avoided_rate": float(bypass.mean()),
                "accuracy": float(accuracy_score(y_true, pred)),
                "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
                "auroc": float(roc_auc_score(y_true, score)),
                "brier": float(brier_score_loss(y_true, score)),
                "bypass_n": int(bypass.sum()),
                "bypass_positives": positives,
                "missed_positives": missed,
                "bypass_false_negative_rate": float(missed / positives) if positives else 0.0,
            }
        )
    return pd.DataFrame(rows)


def decision_curve(y_true, scores: dict[str, np.ndarray]) -> pd.DataFrame:
    rows = []
    y_true = np.asarray(y_true)
    n = len(y_true)
    prevalence = float(y_true.mean())
    for threshold in np.round(np.arange(0.05, 0.51, 0.05), 2):
        odds = threshold / (1.0 - threshold)
        rows.append(
            {
                "model": "treat_all",
                "threshold": threshold,
                "net_benefit": prevalence - (1.0 - prevalence) * odds,
            }
        )
        rows.append({"model": "treat_none", "threshold": threshold, "net_benefit": 0.0})
        for name, score in scores.items():
            pred = as_binary(score, threshold)
            tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
            net_benefit = (tp / n) - (fp / n) * odds
            rows.append({"model": name, "threshold": threshold, "net_benefit": float(net_benefit)})
    return pd.DataFrame(rows)


def subgroup_rows(y_true, y_pred, y_score, meta: pd.DataFrame, group_column: str) -> list[dict]:
    rows = []
    for group_value, group_frame in meta.groupby(group_column, observed=False):
        positions = group_frame["_position"].to_numpy()
        summary = metric_summary(
            str(group_value),
            np.asarray(y_true)[positions],
            np.asarray(y_pred)[positions],
            np.asarray(y_score)[positions],
        )
        row = asdict(summary)
        row["group_type"] = group_column
        row["group"] = str(group_value)
        row["prevalence"] = float(np.asarray(y_true)[positions].mean())
        rows.append(row)
    return rows


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)

    y_test = Data.y_test.to_numpy()
    tier1_model = joblib.load(TIER_1_MODEL_PATH)
    tier2_model = joblib.load(TIER_2_MODEL_PATH)

    p1 = positive_probability(tier1_model, Data.x_test_tier1)
    p2 = positive_probability(tier2_model, Data.x_test_tier2)

    y1 = as_binary(p1)
    y2 = as_binary(p2)
    cascade_pred, cascade_score, cascade_bypass = conservative_cascade(p1, p2)
    band_pred, band_score, band_bypass = uncertainty_band_policy(p1, p2)

    heldout = pd.DataFrame(
        [
            asdict(metric_summary("tier1_only", y_test, y1, p1)),
            asdict(metric_summary("tier2_full_profile", y_test, y2, p2)),
            asdict(metric_summary("conservative_cascade", y_test, cascade_pred, cascade_score)),
            asdict(metric_summary("uncertainty_band_policy", y_test, band_pred, band_score)),
        ]
    )
    heldout.to_csv(OUTPUT_DIR / "heldout_performance.csv", index=False)

    bypass = pd.DataFrame(
        [
            bypass_safety(
                y_test,
                "conservative_cascade",
                cascade_bypass,
                cascade_pred,
                "low_risk_bypass",
            ),
            bypass_safety(
                y_test,
                "uncertainty_band_policy",
                p1 <= LOW_THRESHOLD,
                band_pred,
                "low_risk_bypass",
            ),
            bypass_safety(
                y_test,
                "uncertainty_band_policy",
                p1 >= HIGH_THRESHOLD,
                band_pred,
                "high_risk_bypass",
            ),
        ]
    )
    bypass.to_csv(OUTPUT_DIR / "bypass_safety.csv", index=False)

    calibrated_tier1 = CalibratedClassifierCV(build_tier1_model(), method="isotonic", cv=5)
    calibrated_tier1.fit(Data.x_train_tier1, Data.y_train)
    p1_calibrated = positive_probability(calibrated_tier1, Data.x_test_tier1)
    calibration = pd.DataFrame(
        [
            {
                "model": "tier1_raw",
                "brier": brier_score_loss(y_test, p1),
                "ece_10_bins": expected_calibration_error(y_test, p1),
                "auroc": roc_auc_score(y_test, p1),
            },
            {
                "model": "tier1_isotonic_cv5",
                "brier": brier_score_loss(y_test, p1_calibrated),
                "ece_10_bins": expected_calibration_error(y_test, p1_calibrated),
                "auroc": roc_auc_score(y_test, p1_calibrated),
            },
            {
                "model": "tier2_raw",
                "brier": brier_score_loss(y_test, p2),
                "ece_10_bins": expected_calibration_error(y_test, p2),
                "auroc": roc_auc_score(y_test, p2),
            },
        ]
    )
    calibration.to_csv(OUTPUT_DIR / "calibration_summary.csv", index=False)
    pd.concat(
        [
            reliability_bins(y_test, p1, "tier1_raw"),
            reliability_bins(y_test, p1_calibrated, "tier1_isotonic_cv5"),
            reliability_bins(y_test, p2, "tier2_raw"),
        ],
        ignore_index=True,
    ).to_csv(OUTPUT_DIR / "reliability_bins.csv", index=False)

    threshold_sweep(y_test, p1, p2).to_csv(OUTPUT_DIR / "threshold_sweep.csv", index=False)
    decision_curve(
        y_test,
        {
            "tier1_only": p1,
            "tier2_full_profile": p2,
            "conservative_cascade": cascade_score,
            "uncertainty_band_policy": band_score,
        },
    ).to_csv(OUTPUT_DIR / "decision_curve.csv", index=False)

    raw = pd.read_csv(BASE_DIR / "heart_disease_uci.csv")
    meta = raw.loc[Data.y_test.index, ["dataset", "gender", "age"]].copy()
    meta["_position"] = np.arange(len(meta))
    meta["age_band"] = pd.cut(
        meta["age"],
        bins=[0, 49, 65, 200],
        labels=["<50", "50-65", ">65"],
        include_lowest=True,
    )
    subgroup = pd.DataFrame(
        subgroup_rows(y_test, cascade_pred, cascade_score, meta, "gender")
        + subgroup_rows(y_test, cascade_pred, cascade_score, meta, "age_band")
        + subgroup_rows(y_test, cascade_pred, cascade_score, meta, "dataset")
    )
    subgroup.to_csv(OUTPUT_DIR / "heldout_subgroup_and_site_metrics.csv", index=False)

    site_prevalence = (
        raw.assign(binary_target=(raw["target"] > 0).astype(int))
        .groupby("dataset", as_index=False)
        .agg(n=("binary_target", "size"), prevalence=("binary_target", "mean"))
    )
    site_prevalence.to_csv(OUTPUT_DIR / "site_prevalence.csv", index=False)

    manifest = {
        "low_threshold": LOW_THRESHOLD,
        "high_threshold": HIGH_THRESHOLD,
        "notes": [
            "This script is a post-hoc publication analysis and does not modify saved models or app behavior.",
            "Cascade AUROC/Brier use the probability from the stage that determined each decision.",
            "Isotonic calibration is reported for Tier 1 threshold analysis only; it is not used by the deployed app.",
        ],
        "outputs": sorted(path.name for path in OUTPUT_DIR.glob("*.csv")),
    }
    (OUTPUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote publication evaluation outputs to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
