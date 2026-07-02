import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import Data_Processing_final as Data


try:
    tier1_model = joblib.load("C:\\Users\\admin\\Desktop\\Healthcare\\Healthcare Project 2.0\\Tier_1_model.pkl")
    print("✅ Tier 1 Model loaded successfully")
except FileNotFoundError:
    print("⚠️  Tier 1 Model file not found. Training a new model...")
    tier1_model = RandomForestClassifier(
        n_estimators=600,
        max_depth=4,
        min_samples_split=2,
        random_state=43
    )
    tier1_model.fit(Data.x_train_tier1, Data.y_train)
    joblib.dump(tier1_model, "C:\\Users\\admin\\Desktop\\Healthcare\\Healthcare Project 2.0\\Tier_1_model.pkl")
    print("✅ Tier 1 Model Trained and saved successfully")


y_pred_tier1 = tier1_model.predict(Data.x_test_tier1)
tier1_accuracy = accuracy_score(Data.y_test, y_pred_tier1)

print("The accuracy of the Tier 1 model is :-", tier1_accuracy)

# confusion_matrix_binary = confusion_matrix(Data.y_test, y_pred_tier1)
# print(confusion_matrix_binary)


try:
    tier2_model = joblib.load("C:\\Users\\admin\\Desktop\\Healthcare\\Healthcare Project 2.0\\Tier_2_model.pkl")
    print("✅ Tier 2 Model loaded successfully")
except FileNotFoundError:
    print("⚠️  Tier 2 Model file not found. Training a new model...")
    tier2_model = VotingClassifier(
        estimators=[
            ('rf_best', RandomForestClassifier(
                n_estimators=200,
                max_depth=12,
                min_samples_split=10,
                random_state=7
            )),
            ('rf_current', RandomForestClassifier(
                n_estimators=1000,
                max_depth=5,
                min_samples_split=10,
                random_state=43
            )),
            ('logreg', LogisticRegression(max_iter=5000)),
            ('knn7', KNeighborsClassifier(n_neighbors=7))
        ],
        voting='soft'
    )
    tier2_model.fit(Data.x_train_tier2, Data.y_train)
    joblib.dump(tier2_model, "C:\\Users\\admin\\Desktop\\Healthcare\\Healthcare Project 2.0\\Tier_2_model.pkl")
    print("✅ Tier 2 Model Trained and saved successfully")


y_pred_tier2 = tier2_model.predict(Data.x_test_tier2)
tier2_accuracy = accuracy_score(Data.y_test, y_pred_tier2)

print("The accuracy of the Tier 2 model is :-", tier2_accuracy)




