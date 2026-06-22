import pandas as pd
import numpy as np 
from sklearn.model_selection import train_test_split
import data_preprocessing as datafile 
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score , classification_report , confusion_matrix
import joblib


try :
    Tier1_model = joblib.load("Tier1_model_trained.pkl")
    print("✅ Tier 1 Model loaded successfully")
except FileNotFoundError:
    print("⚠️  Tier 1 Model file not found. Training a new model...")
    Tier1_model = RandomForestClassifier(n_estimators=500 , max_depth=5)
    Tier1_model.fit(datafile.final_x_train,datafile.y_train)
    joblib.dump(Tier1_model , "Tier1_model_trained.pkl")
    print("✅ Tier 1 Model Trained and saved Successfully")


y_pred = Tier1_model.predict(datafile.final_x_test)
accuracy = accuracy_score(datafile.y_test, y_pred)
print(accuracy)

#print(y.unique())
np.set_printoptions(suppress=True)
probabilities = np.round((Tier1_model.predict_proba(datafile.x_test)) * 100 , 2)
#print(probabilities)

#print(Tier1_model.predict(x_test))
#print(y_test)
#print("the accuarcy of the model trained is :----- " , accuracy)

#confusion matrix 
#confusion_matrix_binary = confusion_matrix(y_test,y_pred)
#print(confusion_matrix_binary)