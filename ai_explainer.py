import pandas as pd 
import numpy as np 
import shap 
import joblib
import data_processor as datafile 
import model_handler as model 
import openai
from google import genai
import os 




patient_final_np = np.array(model.patient_final)
explainer = shap.Explainer(model.tier2_model)
shap_values = explainer(patient_final_np)
contributions = shap_values.values[0]

#print("the contributions that are there for the patients are :-----  " , contributions)
disease_contributions = np.round(contributions[:,1] , 4)
#print(disease_contributions)

feature_names = [
    "Age",
    "Gender",
    "Chest Pain Type",
    "Resting Blood Pressure",
    "Serum Cholesterol Level",
    "Fasting Blood Sugar",
    "Resting Electrocardiogram Result",
    "Maximum Heart Rate Achieved"
]

for i in range(len(feature_names)):
    print(feature_names[i] + " : " + str(disease_contributions[i]*100))
