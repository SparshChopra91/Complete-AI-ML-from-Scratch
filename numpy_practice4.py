import numpy as np 
import matplotlib.pyplot as plt


patients = np.array([
# ID  Age  BP   HR   Sugar  Cholesterol
[1,   45,  120, 72, 110,   180],
[2,   62,  145, 88, 140,   240],
[3,   55,  130, 76, 125,   210],
[4,   70,  160, 95, 180,   280],
[5,   35,  110, 68, 90,    170],
[6,   52,  135, 82, 135,   220],
[7,   65,  150, 90, 160,   260],
[8,   40,  118, 70, 100,   175]
])

"""



print("the shape of the original matrix is :- " , patients.shape)
# cleaning of the data
cleaned_data = patients[:,1:]
print("the cleaned data to use is :- \n" , cleaned_data)
print("the shape of the cleaned data is :- " , cleaned_data.shape)

# finding the average of the data 
average_total  = np.mean(cleaned_data[:,:] , axis=0)
print("the average of all the data combined is :- " , average_total)

# finding average of only the age 
average_age = np.mean(patients[:,1] , axis=0)
print("the average age of the patients in the data is the :- " , average_age)

#finding the patients under risk BP > 140 AND Sugar > 130 AND Cholesterol > 220
risk_factor = np.where((patients[:,2] > 140) & (patients[:,4] > 130) & (patients[:,5] > 220))
print("the patients that are under risk of the heart are :- " , patients[risk_factor])

#finding the patient with the highest bp 
max_bp = np.argmax(cleaned_data[:,1] , axis=0)
print("the patient having the max bp is :-  " + str(max_bp+1))

#finding all the patients that are greater than the 50 
more_than_50 = np.where((patients[:,1] > 50))
print("the patients that have the age more than 50 are:- " , patients[more_than_50][:,0:2])

#creating a new feature called the risk score 
risk_score_old = np.array(((patients[:,1] * 0.4)) + (patients[:,4] * 0.3) + (patients[:,5] * 0.3) ) 
risk_score = risk_score_old.reshape(8,1)

print("the risk score array newly created is :- " , risk_score)
print("the shape of the new risk factor array is :- " , risk_score.shape)

#adding the new feature of the risk factor into the original patients matrix 
final_patients = np.hstack((patients , risk_score))
print("the final patients matrix with the risk factor included are :-  " , final_patients)


#find all the patients whose bp is above average
average_bp = np.mean(patients[:,2] , axis=0)
print("the average bp of the people are :- " + str(average_bp))
where_condition = np.where(patients[:,2] > average_bp)
print("the patients having the bp above average are :- " , patients[where_condition][:,0])


# replace the cholesterol above 250 to danger level
cholestrol_where = np.where((patients[:,5] > 250) , "Danger" , "ok")
print("the level of the cholestrol in the danger level are:-  " , cholestrol_where)
print("the shape of the cholestrol level are :- " , cholestrol_where.shape)
reshaped_cholesterol_where = cholestrol_where.reshape(8,1)
cholestrol_level_list = np.hstack((patients , reshaped_cholesterol_where))
print("the full and final list of the patients with the cholesterol level are :- \n " , cholestrol_level_list[:,(0,6)])



"""


#creating a label by checking the risk score
#creating a new feature called the risk score 
risk_score_old = np.array(((patients[:,1] * 0.4)) + (patients[:,4] * 0.3) + (patients[:,5] * 0.3) ) 
risk_score = risk_score_old.reshape(8,1)

print("the risk score array newly created is :- " , risk_score)
print("the shape of the new risk factor array is :- " , risk_score.shape)
patients_risk_score = np.hstack((patients , risk_score))
print("the array created with the risk score of the each patient are :-  \n" , patients_risk_score)
# if more than 130 then heart disease risk is seen 

risk_where = np.where(patients_risk_score[:,6] > 130 , "Yes" , "No")
reshaped_risk_where = risk_where.reshape(8,1)
risk_patients = np.hstack((patients_risk_score , reshaped_risk_where))
print("the patients with the high chances of heart disease are :-  " , risk_patients)


 


