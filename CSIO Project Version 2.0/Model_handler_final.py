from pathlib import Path

import joblib
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier

import Data_Processing_final as Data


BASE_DIR = Path(__file__).resolve().parent
TIER_1_MODEL_PATH = BASE_DIR / "Tier_1_model.pkl"
TIER_2_MODEL_PATH = BASE_DIR / "Tier_2_model.pkl"


def build_tier1_model() -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=600,
        max_depth=4,
        min_samples_split=2,
        random_state=43,
    )


def build_tier2_model() -> VotingClassifier:
    return VotingClassifier(
        estimators=[
            (
                "rf_best",
                RandomForestClassifier(
                    n_estimators=200,
                    max_depth=12,
                    min_samples_split=10,
                    random_state=7,
                ),
            ),
            (
                "rf_current",
                RandomForestClassifier(
                    n_estimators=1000,
                    max_depth=5,
                    min_samples_split=10,
                    random_state=43,
                ),
            ),
            ("logreg", LogisticRegression(max_iter=5000)),
            ("knn7", KNeighborsClassifier(n_neighbors=7)),
        ],
        voting="soft",
    )


def load_or_train_tier1_model():
    try:
        return joblib.load(TIER_1_MODEL_PATH)
    except FileNotFoundError:
        model = build_tier1_model()
        model.fit(Data.x_train_tier1, Data.y_train)
        joblib.dump(model, TIER_1_MODEL_PATH)
        return model


def load_or_train_tier2_model():
    try:
        return joblib.load(TIER_2_MODEL_PATH)
    except FileNotFoundError:
        model = build_tier2_model()
        model.fit(Data.x_train_tier2, Data.y_train)
        joblib.dump(model, TIER_2_MODEL_PATH)
        return model


tier1_model = load_or_train_tier1_model()
tier2_model = load_or_train_tier2_model()


if __name__ == "__main__":
    y_pred_tier1 = tier1_model.predict(Data.x_test_tier1)
    tier1_accuracy = accuracy_score(Data.y_test, y_pred_tier1)
    print("Tier 1 model path:", TIER_1_MODEL_PATH)
    print("The accuracy of the Tier 1 model is :-", tier1_accuracy)

    y_pred_tier2 = tier2_model.predict(Data.x_test_tier2)
    tier2_accuracy = accuracy_score(Data.y_test, y_pred_tier2)
    print("Tier 2 model path:", TIER_2_MODEL_PATH)
    print("The accuracy of the Tier 2 model is :-", tier2_accuracy)
