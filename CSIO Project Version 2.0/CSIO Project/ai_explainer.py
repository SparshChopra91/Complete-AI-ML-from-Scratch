import pandas as pd 
import numpy as np 
import shap 
import joblib
import data_processor as datafile 
import model_handler as model 
import openai
from google import genai
from openai import OpenAI
import os 
from dotenv import load_dotenv



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


load_dotenv()

api_key = os.getenv("open_router_api_key")
#print(api_key)
client = OpenAI(
    api_key=api_key,
    base_url="https://openrouter.ai/api/v1"
)

heart_disease_percentage = model.sick_percentage_2
shap_age = shap_values.values[0][0 , 1]
shap_gender = shap_values.values[0][1 , 1]
shap_chest_pain = shap_values.values[0][2 , 1]
shap_resting_blood_pressure = shap_values.values[0][3 , 1]
shap_serum_cholesterol_level = shap_values.values[0][4 , 1]
shap_blood_sugar = shap_values.values[0][5 , 1]
shap_Resting_Electrocardiogram_Result = shap_values.values[0][6 , 1]
shap_Maximum_heart_rate_achieved = shap_values.values[0][7 , 1]

prompt = f"""
You are a senior cardiologist with more than 20 years of clinical experience explaining a heart health assessment to a patient during a consultation.

Patient Information:

* Predicted Heart Disease Risk: {heart_disease_percentage:.1f}%

Factors considered in the assessment:

* Age: {shap_age}
* Gender: {shap_gender}
* Chest Pain Type: {shap_chest_pain}
* Resting Blood Pressure: {shap_resting_blood_pressure}
* Serum Cholesterol Level: {shap_serum_cholesterol_level}
* Fasting Blood Sugar: {shap_blood_sugar}
* Resting Electrocardiogram Result: {shap_Resting_Electrocardiogram_Result}
* Maximum Heart Rate Achieved: {shap_Maximum_heart_rate_achieved}

Interpretation Rules:

* Positive values indicate factors that increased the patient's risk.
* Negative values indicate factors that reduced the patient's risk.
* Focus primarily on the 3-4 strongest factors affecting the result.
* Explain the findings naturally as a doctor would during a consultation.
* Mention both concerning factors and reassuring factors when applicable.
* Explain why each factor may matter for heart health in simple language.
* Avoid technical jargon, statistics, machine learning terms, SHAP values, feature importance, model predictions, probabilities, algorithms, or data science terminology.
* Do not mention that a computer, AI, model, or algorithm generated the assessment.

Writing Style:

* Warm, professional, and reassuring.
* Speak directly to the patient using "you" and "your".
* Sound like a real cardiologist discussing results in a clinic.
* Do not exaggerate risk or create unnecessary fear.
* Provide balanced medical context.
* Include one short recommendation for lifestyle improvement or follow-up if appropriate.

Output Requirements:

* Write 1 short paragraph of approximately 120-180 words.
* Make the explanation personalized.
* Mention the overall risk percentage naturally within the explanation.
* Return only the patient explanation.

"""


response = client.chat.completions.create(
    model="google/gemini-2.5-flash",
    messages=[
        {
            "role": "user",
            "content": prompt
        }
    ],
    max_tokens=500,
    top_p= 0.9,
    temperature=0.3
)

ai_explanation = response.choices[0].message.content

print(ai_explanation)







