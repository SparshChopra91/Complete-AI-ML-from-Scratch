import pandas as pd
import numpy as np 
from sklearn.model_selection import train_test_split
import data_processor as datafile 
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score , classification_report , confusion_matrix
import joblib

###----------------------- TIER 1 MODEL ------------------------###


x = datafile.tier1_dataset[['age' , 'gender' , 'cp' , 'trestbps']]
y = (datafile.tier1_dataset['target'] > 0).astype(int)
x_train , x_test , y_train , y_test = train_test_split(x ,y , test_size=0.1 , random_state=50 , stratify=y)

trestbps_meadian = x_train['trestbps'].median()
#print(str(trestbps_meadian))
x_train['trestbps'] = x_train['trestbps'].fillna(trestbps_meadian )
x_test['trestbps'] = x_test['trestbps'].fillna(trestbps_meadian )

#print("Train nulls:", x_train['trestbps'].isnull().sum())
#print("Test nulls:", x_test['trestbps'].isnull().sum())
#print("the unique values in the y are :- " , y.unique())

try :
    Tier1_model = joblib.load("Tier1_model_trained.pkl")
    print("✅ Tier 1 Model loaded successfully")
except FileNotFoundError:
    print("⚠️  Tier 1 Model file not found. Training a new model...")
    Tier1_model = RandomForestClassifier(n_estimators=10000)
    Tier1_model.fit(x_train,y_train)
    joblib.dump(Tier1_model , "Tier1_model_trained.pkl")
    print("✅ Tier 1 Model Trained and saved Successfully")


#y_pred = Tier1_model.predict(x_test)
#accuracy = accuracy_score(y_test, y_pred)

#print(y.unique())
np.set_printoptions(suppress=True)
probabilities = np.round((Tier1_model.predict_proba(x_test)) * 100 , 2)
#print(probabilities)

#print(Tier1_model.predict(x_test))
#print(y_test)
#print("the accuarcy of the model trained is :----- " , accuracy)

#confusion matrix 
#confusion_matrix_binary = confusion_matrix(y_test,y_pred)
#print(confusion_matrix_binary)


patient_full = [[
    1111,
    68,
    1,
    4,
    180
]]
patient = [patient_full[0][1:]]

prediction1 = Tier1_model.predict_proba(patient)
sick_percentage_1 = np.round(prediction1[0][1] * 100 , 2)
print("the probability of the patient needs further medical assistance is : - " , sick_percentage_1)



###-------------------  DECESION TIME ---------------###
if (sick_percentage_1 < 30):
    print("the patient is free from any heart disease \nNo further medical assistance is needed in this case \n")
    exit(0)
else:
    print("the patient health issue comes in the gray zone so , \nThe patient is needed to get further tests to get precise answer")
    print("executing the Tier-2 Model :---->  ")




###---------------------  TIER 2 MODEL -----------------###

x2 = datafile.tier2_dataset[['age' , 'gender' , 'cp' , 'trestbps', 'chol', 'fbs' , 'restecg' , 'thalch']]
y2 = (datafile.tier1_dataset['target'] > 0).astype(int)
x2_train , x2_test , y2_train , y2_test = train_test_split(x2 , y2 , test_size=.1 , random_state=50)

chol_median = datafile.tier2_dataset['chol'].median()
fbs_median = datafile.tier2_dataset['fbs'].median()
restecg_median = datafile.tier2_dataset['restecg'].median()
thalch_median = datafile.tier2_dataset['thalch'].median()

x2_train['chol'] = x2_train['chol'].fillna(chol_median)
x2_test['chol'] = x2_test['chol'].fillna(chol_median)

x2_train['fbs'] = x2_train['fbs'].fillna(fbs_median)
x2_test['fbs'] = x2_test['fbs'].fillna(fbs_median)

x2_train['restecg'] = x2_train['restecg'].fillna(restecg_median)
x2_test['restecg'] = x2_test['restecg'].fillna(restecg_median)

x2_train['thalch'] = x2_train['thalch'].fillna(thalch_median)
x2_test['thalch'] = x2_test['thalch'].fillna(thalch_median)

try :
    tier2_model = joblib.load("Tier2_model_trained.pkl")
    print("✅ Tier 2 Model loaded successfully")
except FileNotFoundError :
    print("⚠️  Tier 2 Model file not found. Training a new model...")
    tier2_model = RandomForestClassifier(n_estimators=10000)
    tier2_model.fit(x2_train,y2_train)
    joblib.dump( tier2_model, "Tier2_model_trained.pkl")
    print("✅ Tier 2 Model trained and saved successfully")

patient_futher_details = [[1111 , 233 , 1 , 2 , 150]]
patient_final = [patient_full[0][1:] + patient_futher_details[0][1:]]


prediction2 = tier2_model.predict_proba(patient_final)
sick_percentage_2 = np.round(prediction2[0][1] * 100 , 2)
print("the final prediction that the patient is having the heart disease is  : - " + str(sick_percentage_2) + "%")


#accuracy_model2 = accuracy_score(y2_test, y2_pred)
#print("the accuracy score of the tier 2 model is :--- " , accuracy_model2)