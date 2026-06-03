import numpy as np 
import pandas as pd

patients = {
    "Patient_ID": [1, 2, 3, 4, 5, 6, 7, 8],
    "Age": [45, 62, 55, 70, 35, 52, 65, 40],
    "BP": [120, 145, 130, 160, 110, 135, 150, 118],
    "HR": [72, 88, 76, 95, 68, 82, 90, 70],
    "Sugar": [110, 140, 125, 180, 90, 135, 160, 100],
    "Cholesterol": [180, 240, 210, 280, 170, 220, 260, 175]
}

dataset = pd.DataFrame(patients)

"""

print(dataset)

dataset.to_csv("Patients Dataset")

print("the dataset starting from the top is :- \n " , dataset.head())
print("the dataset starting from the bottom is :- \n " , dataset.tail())


dataset.loc[0,"Age"] = 100
print(dataset["Age"][0])

print(dataset.columns)
dataset.rename(columns={'Age' : 'Patients Age'} , inplace=True)
print(dataset)



print("showing the BP column values" , dataset[['BP']])
print("showing all the rows of the dataset \n"  , dataset.loc[:]) 
high_BP = dataset.loc[dataset.BP > 120]
print("showing only those rows where the bp is higher than the 120 \n " , high_BP[['Patient_ID' ,'BP']])

print("now printing the values that are the where the cholesterol is higher than 200 \n")
high_cholesterol = dataset[dataset['Cholesterol'] > 200]
print(high_cholesterol[['Patient_ID' , 'Cholesterol']])

high_cholesterol_where = dataset.where(dataset["Cholesterol"] > 200 , other="None")
print(high_cholesterol_where)

dataset.loc[len(dataset)] = [9, 48, 145, 85, 300, 250]
print(dataset)



dataset['danger zone'] = (dataset['BP'] * 0.4) + (dataset['Sugar'] * 0.3) + (dataset['Cholesterol'] * 0.3)
print(dataset)


dataset.loc[len(dataset)] = [9, 48, 145, 85, 300, 250]
print(dataset)
dataset.loc[8,'Sugar'] = 290

dataset.loc[len(dataset)] = ['ABC', 48, 145, 85, 300, 250]
print(dataset)


dataset.drop(dataset[dataset['Patient_ID'] == 'ABC'].index , inplace=True)
print(dataset)


sorted_dataset = dataset.sort_values('Danger zone')
print(sorted_dataset)

sorted_dataset_desc = dataset.sort_values(('Danger zone') , ascending=False)
print(sorted_dataset_desc)



"""

dataset['danger zone'] = (dataset['BP'] * 0.4) + (dataset['Sugar'] * 0.3) + (dataset['Cholesterol'] * 0.3)
print(dataset)

dataset.rename(columns={'danger zone' : 'Danger zone'} , inplace=True)

dataset['Need care?'] = np.where((dataset['Danger zone'] > 160) , 'Yes immediately' , 'Not needed')
print(dataset)
sorted_dataset =  dataset.sort_values('Need care?' , ascending=False)
print(sorted_dataset[['Patient_ID' , 'Danger zone' , 'Need care?']]) 
