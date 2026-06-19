import pandas as pd
import numpy as np 

### ----------------------   DATA FOR THE TIER1 MODEL ----------------------###

tier1_dataset = pd.read_csv("heart_disease_uci.csv" , 
                            usecols=["id","age" , "gender" , "cp" , "trestbps" , "target"]
                            )

tier1_dataset["cp"] = (tier1_dataset["cp"].str.lower().str.strip())
tier1_dataset["gender"] = (tier1_dataset["gender"].str.lower().str.strip())

#print("the head of the tier 1 ml model is :-  \n" , tier1_dataset.head())

#print("the empty cells here in this dataset are :- \n" , tier1_dataset.isnull().sum())
tier1_dataset = tier1_dataset.replace(["" , " " ,"N/A" ] , np.nan)

cp_mapping = {
    "typical angina":   0,
    "atypical angina": 1,
    "non-anginal":   2,
    "asymptomatic":   3
}
gender_mapping = {
    "male" : 1,
    "female" :0
}

tier1_dataset['cp'] = tier1_dataset['cp'].map(cp_mapping)
tier1_dataset['gender'] = tier1_dataset['gender'].map(gender_mapping)

tier1_dataset = tier1_dataset.astype({
    'age' : 'Int8',
    'gender' : 'Int16',
    'cp' : 'Int8' , 
    'trestbps' : 'Int16'
})

#print(tier1_dataset[tier1_dataset.isnull().any(axis=1)])



### -----------------------   DATA FOR THE TIER 2 MODEL ------------------###

initial_tier2_dataset = pd.read_csv("heart_disease_uci.csv" , 
                                    usecols=['id', 'chol' , 'fbs' , 'restecg' , 'thalch' ])

#print((initial_tier2_dataset['restecg']).unique())

initial_tier2_dataset['restecg'] = (initial_tier2_dataset['restecg']).str.lower().str.strip()
initial_tier2_dataset[['chol' , 'fbs' , 'restecg' , 'thalch']] = initial_tier2_dataset[['chol' , 'fbs' , 'restecg' , 'thalch']].replace(["" ," " ,"N/A" ] , np.nan)

#print(initial_tier2_dataset.isnull().sum())

restecg_mapping = {
    "normal": 0,
    "st-t abnormality": 1,
    "lv hypertrophy": 2
}
fbs_mapping = {
    True : 1,
    False : 0
}

initial_tier2_dataset['restecg'] = initial_tier2_dataset['restecg'].map(restecg_mapping)
initial_tier2_dataset['fbs'] = initial_tier2_dataset['fbs'].map(fbs_mapping)
#print(initial_tier2_dataset.head())

tier2_dataset = pd.merge(tier1_dataset,initial_tier2_dataset, on="id")
#print(tier2_dataset.head())