import numpy as np
import pandas as pd 
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer




uci_dataset = pd.read_csv("C:\\Users\\admin\\Desktop\\Healthcare\\Data Preprocessing\\heart_disease_uci.csv" , 
                          usecols=['id','age' , 'gender' , 'cp' , 'trestbps' , 'target'])

X = uci_dataset[['age' , 'gender' , 'cp' , 'trestbps'] ]
Y = (uci_dataset['target']>0).astype(int)
x_train , x_test , y_train , y_test = train_test_split(X , Y , test_size=0.2,stratify=Y , random_state=42)




#print(uci_dataset.head())

### -------------------   Filling The Missing Values -----------------###
#print(uci_dataset.isnull().sum())
#trestbps_meadian = x_train['trestbps'].median()
#x_train['trestbps']  = x_train['trestbps'].fillna(trestbps_meadian)
#x_test['trestbps'] = x_test['trestbps'].fillna(trestbps_meadian)
#print(x_train.isnull().sum())




###-------- Doing Encoding on the gender and the cp ---------###
Encoder_gender = OneHotEncoder(sparse_output=False , handle_unknown='ignore')
Encoder_gender.fit(x_train[['gender']])
encoded_x_train = Encoder_gender.transform(x_train[['gender']])
encoded_x_test = Encoder_gender.transform(x_test[['gender']])

gender_df_train = pd.DataFrame(
    encoded_x_train,
    index=x_train.index,
    columns=Encoder_gender.get_feature_names_out()
)

gender_df_test = pd.DataFrame(
    encoded_x_test,
    index=x_test.index,
    columns=Encoder_gender.get_feature_names_out()
)

x_train = x_train.drop(columns=['gender'])
x_test = x_test.drop(columns=['gender'])
x_train = pd.concat([x_train , gender_df_train] , axis=1)
x_test = pd.concat([x_test , gender_df_test] , axis=1)


Encoder_cp = OneHotEncoder(sparse_output=False , 
                           handle_unknown='ignore')
Encoder_cp.fit(x_train[['cp']])
encoded_cp_x_train = Encoder_cp.transform(x_train[['cp']])
encoded_cp_x_test = Encoder_cp.transform(x_test[['cp']])

encoded_cp_train = pd.DataFrame(
    encoded_cp_x_train,
    columns=Encoder_cp.get_feature_names_out(),
    index = x_train.index
)
encoded_cp_test = pd.DataFrame(
    encoded_cp_x_test,
    columns=Encoder_cp.get_feature_names_out(),
    index = x_test.index
)

x_train = x_train.drop(columns=['cp'])
x_test = x_test.drop(columns=['cp'])

x_train = pd.concat([x_train , encoded_cp_train] , axis=1)
x_test = pd.concat([x_test , encoded_cp_test] , axis=1)

#print(x_train.head())


### ------------------  Using KNN Imputation To Fill The Missing Values -----------###

"""
imputer_trestbps = KNNImputer(
    n_neighbors=5)
imputer_trestbps.fit(x_train[knn_columns])
x_train[knn_columns] = imputer_trestbps.transform(x_train[knn_columns])
x_test[knn_columns] = imputer_trestbps.transform(x_test[knn_columns])



"""
#print(x_test.isnull().sum())
knn_columns = [
    'age',
    'trestbps',
    'gender_Female',
    'gender_Male',
    'cp_asymptomatic',
    'cp_atypical angina',
    'cp_non-anginal',
    'cp_typical angina'
]



### -------------- Using MICE to Fill The Missing Data---------------###

imputer_trestbps = IterativeImputer(
    random_state=42,
    max_iter=15
)
imputer_trestbps.fit(x_train[knn_columns])
x_train = pd.DataFrame(
    imputer_trestbps.transform(x_train[knn_columns]),
    columns=x_train.columns,
    index=x_train.index
)
x_test = pd.DataFrame(
    imputer_trestbps.transform(x_test[knn_columns]),
    columns=x_test.columns,
    index=x_test.index
)




### -------  Doing Scaling on trestbps and age --------###
scaler_trestbps = MinMaxScaler()
scaler_trestbps.fit(x_train[['trestbps']])
x_train['trestbps'] = scaler_trestbps.transform(x_train[['trestbps']])
x_test['trestbps'] = scaler_trestbps.transform(x_test[['trestbps']])

scaler_age = MinMaxScaler()
scaler_age.fit(x_train[['age']])
x_train['age'] = scaler_age.transform(x_train[['age']])
x_test['age'] = scaler_age.transform(x_test[['age']])


#print(x_test.head())

final_x_train = x_train
final_x_test = x_test

#print(final_x_train.head())
