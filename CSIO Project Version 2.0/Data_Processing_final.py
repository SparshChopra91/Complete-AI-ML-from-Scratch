import numpy as np 
import pandas as pd 
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer


#####---------------Importing The Whole Dataset -----------------#####
PROJECT_DIR = Path(__file__).resolve().parent

uci_dataset = pd.read_csv(PROJECT_DIR / "heart_disease_uci.csv" ,
                          usecols=[ 'age' , 'gender' ,'cp' ,'trestbps', 'chol' ,'fbs' ,'restecg','thalch' , 'exang' , 'oldpeak',
                                   'slope', 'ca' , 'thal','target'])

#print(uci_dataset.head())
#print(uci_dataset.isnull().sum())

X = uci_dataset[['age' , 'gender' ,'cp' ,'trestbps', 'chol' ,'fbs' ,'restecg','thalch' , 'exang' , 'oldpeak',
                                   'slope', 'ca' , 'thal']]
Y = (uci_dataset['target']>0).astype(int)
x_train , x_test , y_train , y_test = train_test_split(X,Y, test_size=0.2 ,stratify=Y , random_state=43)

#print(x_train.head())

##--------------- Mapping the categorical clinical data ----------- ##

fbs_mapping = {
    True : 1,
    False : 0
}

x_train['fbs'] = x_train['fbs'].map(fbs_mapping)
x_test['fbs'] = x_test['fbs'].map(fbs_mapping)

exang_mapping = {
    True : 1,
    False : 0
}

x_train['exang'] = x_train['exang'].map(exang_mapping)
x_test['exang'] = x_test['exang'].map(exang_mapping)

restecg_mapping = {
    "normal": 0,
    "st-t abnormality": 1,
    "lv hypertrophy": 2
}

x_train['restecg'] = x_train['restecg'].map(restecg_mapping)
x_test['restecg'] = x_test['restecg'].map(restecg_mapping)

slope_mapping = {
    "downsloping": 0,
    "flat": 1,
    "upsloping": 2
}

x_train['slope'] = x_train['slope'].map(slope_mapping)
x_test['slope'] = x_test['slope'].map(slope_mapping)

thal_mapping = {
    "fixed defect": 0,
    "normal": 1,
    "reversable defect": 2
}

x_train['thal'] = x_train['thal'].map(thal_mapping)
x_test['thal'] = x_test['thal'].map(thal_mapping)

##--------------- Replacing invalid cholesterol values ----------- ##

x_train['chol'] = x_train['chol'].replace(0 , np.nan)
x_test['chol'] = x_test['chol'].replace(0 , np.nan)

###-------------- Numeric Missing Indicators ------------- ###

x_train['ca_missing'] = x_train['ca'].isnull().astype(int)
x_test['ca_missing'] = x_test['ca'].isnull().astype(int)

x_train['chol_missing_or_zero'] = x_train['chol'].isnull().astype(int)
x_test['chol_missing_or_zero'] = x_test['chol'].isnull().astype(int)

x_train['oldpeak_missing'] = x_train['oldpeak'].isnull().astype(int)
x_test['oldpeak_missing'] = x_test['oldpeak'].isnull().astype(int)

####------------------- ENCODING THE DATA ---------------------####
cp_encoder = OneHotEncoder(sparse_output=False ,
                           handle_unknown='ignore')
cp_encoder.fit(x_train[['cp']])
train_encoded = cp_encoder.transform(x_train[['cp']])
test_encoded = cp_encoder.transform(x_test[['cp']])

cp_train_df = pd.DataFrame(
    train_encoded,
    columns=cp_encoder.get_feature_names_out(),
    index=x_train.index
)
cp_test_df = pd.DataFrame(
    test_encoded,
    columns=cp_encoder.get_feature_names_out(),
    index=x_test.index
)

x_train = x_train.drop(columns=['cp'])
x_test = x_test.drop(columns=['cp'])

x_train = pd.concat([x_train , cp_train_df] , axis=1)
x_test = pd.concat([x_test , cp_test_df] , axis=1)

#print(x_train.head())

##------------- gender encoding -------- ##

gender_encoder = OneHotEncoder(
    sparse_output=False,
    handle_unknown='ignore'
)

gender_encoder.fit(x_train[['gender']])

encoded_train_gender = gender_encoder.transform(x_train[['gender']])
encoded_test_gender = gender_encoder.transform(x_test[['gender']])

gender_train_df = pd.DataFrame(
    encoded_train_gender,
    columns=gender_encoder.get_feature_names_out(),
    index=x_train.index
)
gender_test_df = pd.DataFrame(
    encoded_test_gender,
    columns=gender_encoder.get_feature_names_out(),
    index=x_test.index
)

x_train = x_train.drop(columns=['gender'])
x_test = x_test.drop(columns=['gender'])

x_train = pd.concat([x_train , gender_train_df] , axis=1)
x_test = pd.concat([x_test , gender_test_df] , axis=1)

#print(x_train.head())

###-------------- Categorical Imputation ------------- ###

categorical_imputer = SimpleImputer(
    strategy='constant',
    fill_value=-1
)

categorical_columns = [
    'fbs',
    'restecg',
    'exang',
    'slope',
    'thal',
    'ca'
]
categorical_imputer.fit(x_train[categorical_columns])

x_train[categorical_columns] = categorical_imputer.transform(
    x_train[categorical_columns]
)

x_test[categorical_columns] = categorical_imputer.transform(
    x_test[categorical_columns]
)

x_train[categorical_columns] = x_train[categorical_columns].astype(float)
x_test[categorical_columns] = x_test[categorical_columns].astype(float)

###---------------- MICE -------------###

dataset_imputer = IterativeImputer(
    random_state=43,
    max_iter=15
)
mice_columns = ['age','trestbps', 'chol' ,'thalch'  , 'oldpeak']
dataset_imputer.fit(x_train[mice_columns])

x_train[mice_columns] = dataset_imputer.transform(x_train[mice_columns])
x_test[mice_columns] = dataset_imputer.transform(x_test[mice_columns])

#print(x_train.isnull().sum())


## ------ CA encoding ----------##

ca_encoder = OneHotEncoder(
    sparse_output=False,
    handle_unknown='ignore'
)

ca_encoder.fit(x_train[['ca']])

ca_train_encoded = ca_encoder.transform(x_train[['ca']])
ca_test_encoded = ca_encoder.transform(x_test[['ca']])

ca_df_train = pd.DataFrame(
    ca_train_encoded,
    columns=ca_encoder.get_feature_names_out(),
    index=x_train.index
)

ca_df_test = pd.DataFrame(
    ca_test_encoded,
    columns=ca_encoder.get_feature_names_out(),
    index=x_test.index
)

x_train = x_train.drop(columns=['ca'])
x_test = x_test.drop(columns=['ca'])

x_train = pd.concat([x_train, ca_df_train], axis=1)
x_test = pd.concat([x_test, ca_df_test], axis=1)


## ------  RESTECG encoding ----------##

restecg_encoder = OneHotEncoder(
    sparse_output=False,
    handle_unknown='ignore'
)
restecg_encoder.fit(x_train[['restecg']])

restecg_train_encoded = restecg_encoder.transform(x_train[['restecg']])
restecg_test_encoded = restecg_encoder.transform(x_test[['restecg']])

restecg_df_train = pd.DataFrame(
    restecg_train_encoded,
    columns=restecg_encoder.get_feature_names_out(),
    index = x_train.index
)
restecg_df_test = pd.DataFrame(
    restecg_test_encoded,
    columns=restecg_encoder.get_feature_names_out(),
    index = x_test.index
)

x_train = x_train.drop(columns=['restecg'])
x_test = x_test.drop(columns=['restecg'])

x_train = pd.concat([x_train,restecg_df_train] , axis=1)
x_test = pd.concat([x_test , restecg_df_test] , axis=1)

#print(x_train.head())

#print(x_train.head())


##------- fbs encoding --------##
fbs_encoder = OneHotEncoder(
    sparse_output=False,
    handle_unknown='ignore'
)

fbs_encoder.fit(x_train[['fbs']])

encoded_train_fbs = fbs_encoder.transform(x_train[['fbs']])
encoded_test_fbs = fbs_encoder.transform(x_test[['fbs']])

train_fbs_df = pd.DataFrame(
    encoded_train_fbs,
    columns=fbs_encoder.get_feature_names_out(),
    index=x_train.index
)
test_fbs_df = pd.DataFrame(
    encoded_test_fbs,
    columns=fbs_encoder.get_feature_names_out(),
    index=x_test.index
)

x_train = x_train.drop(columns=['fbs'])
x_test = x_test.drop(columns=['fbs'])

x_train = pd.concat([x_train , train_fbs_df] , axis=1)
x_test = pd.concat([x_test , test_fbs_df] , axis=1)

## ------ EXANG encoding ----------##

exang_encoder = OneHotEncoder(
    sparse_output=False,
    handle_unknown='ignore'
)

exang_encoder.fit(x_train[['exang']])

exang_train_encoded = exang_encoder.transform(x_train[['exang']])
exang_test_encoded = exang_encoder.transform(x_test[['exang']])

exang_df_train = pd.DataFrame(
    exang_train_encoded,
    columns=exang_encoder.get_feature_names_out(),
    index=x_train.index
)

exang_df_test = pd.DataFrame(
    exang_test_encoded,
    columns=exang_encoder.get_feature_names_out(),
    index=x_test.index
)

x_train = x_train.drop(columns=['exang'])
x_test = x_test.drop(columns=['exang'])

x_train = pd.concat([x_train, exang_df_train], axis=1)
x_test = pd.concat([x_test, exang_df_test], axis=1)


## ------ SLOPE encoding ----------##

slope_encoder = OneHotEncoder(
    sparse_output=False,
    handle_unknown='ignore'
)

slope_encoder.fit(x_train[['slope']])

slope_train_encoded = slope_encoder.transform(x_train[['slope']])
slope_test_encoded = slope_encoder.transform(x_test[['slope']])

slope_df_train = pd.DataFrame(
    slope_train_encoded,
    columns=slope_encoder.get_feature_names_out(),
    index=x_train.index
)

slope_df_test = pd.DataFrame(
    slope_test_encoded,
    columns=slope_encoder.get_feature_names_out(),
    index=x_test.index
)

x_train = x_train.drop(columns=['slope'])
x_test = x_test.drop(columns=['slope'])

x_train = pd.concat([x_train, slope_df_train], axis=1)
x_test = pd.concat([x_test, slope_df_test], axis=1)


## ------ THAL encoding ----------##

thal_encoder = OneHotEncoder(
    sparse_output=False,
    handle_unknown='ignore'
)

thal_encoder.fit(x_train[['thal']])

thal_train_encoded = thal_encoder.transform(x_train[['thal']])
thal_test_encoded = thal_encoder.transform(x_test[['thal']])

thal_df_train = pd.DataFrame(
    thal_train_encoded,
    columns=thal_encoder.get_feature_names_out(),
    index=x_train.index
)

thal_df_test = pd.DataFrame(
    thal_test_encoded,
    columns=thal_encoder.get_feature_names_out(),
    index=x_test.index
)

x_train = x_train.drop(columns=['thal'])
x_test = x_test.drop(columns=['thal'])

x_train = pd.concat([x_train, thal_df_train], axis=1)
x_test = pd.concat([x_test, thal_df_test], axis=1)

#print(x_train.head())
#print(x_train.columns)

#x_train.to_csv("Final_full_processed_x_train_data.csv")


####----------------- SCALING THE DATA ---------------------####
thalch_scaler = MinMaxScaler()
thalch_scaler.fit(x_train[['thalch']])

x_train['thalch'] = thalch_scaler.transform(x_train[['thalch']])
x_test['thalch'] = thalch_scaler.transform(x_test[['thalch']])

chol_scaler = MinMaxScaler()
chol_scaler.fit(x_train[['chol']])

x_train['chol'] = chol_scaler.transform(x_train[['chol']])
x_test['chol'] = chol_scaler.transform(x_test[['chol']])

trestbps_scaler = MinMaxScaler()
trestbps_scaler.fit(x_train[['trestbps']])

x_train['trestbps'] = trestbps_scaler.transform(x_train[['trestbps']])
x_test['trestbps'] = trestbps_scaler.transform(x_test[['trestbps']])

age_scaler = MinMaxScaler()
age_scaler.fit(x_train[['age']])

x_train['age'] = age_scaler.transform(x_train[['age']])
x_test['age'] = age_scaler.transform(x_test[['age']])

oldpeak_scaler = MinMaxScaler()
oldpeak_scaler.fit(x_train[['oldpeak']])

x_train['oldpeak'] = oldpeak_scaler.transform(x_train[['oldpeak']])
x_test['oldpeak'] = oldpeak_scaler.transform(x_test[['oldpeak']])

#x_train.to_csv("final_xtrain_data.csv" , index=False)

#####--------------------------- TIER 1 DATA --------------------------####
tier1_columns = [
    'age',
    'gender_Female',
    'gender_Male',
    'cp_asymptomatic',
    'cp_atypical angina',
    'cp_non-anginal',
    'cp_typical angina',
    'trestbps'
]

x_train_tier1 = x_train[tier1_columns]
x_test_tier1 = x_test[tier1_columns]

x_train_tier1 = x_train[tier1_columns]
x_test_tier1 = x_test[tier1_columns]


#####----------------------------- TIER 2 DATA --------------------#####    
x_train_tier2 = x_train
x_test_tier2 = x_test
