import numpy as np
import pandas as pd

from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
#from streamlit import input_user

folder_dir = "./data"

#Dataset 
def read_data(data_dir):
    '''
    Prepare training data
    Input:
        data_dir <str> : path to data file (csv), can be train set or test set
    Returns:
        Dataframe after clean
    '''
    # Load data
    df = pd.read_csv(data_dir)
    DT_bmi_pipe = Pipeline(steps=[('scale',StandardScaler()),
                            ('dtr',DecisionTreeRegressor(random_state=42))])
    X_BMI = df[['age','gender','bmi']].copy()
    X_BMI.gender = X_BMI.gender.replace({'Male':0,'Female':1,'Other':-1}).astype(np.uint8)

    # Handling NA data
    na_data = X_BMI[X_BMI.bmi.isna()]
    X_BMI = X_BMI[~X_BMI.bmi.isna()]
    Y_BMI = X_BMI.pop('bmi')
    DT_bmi_pipe.fit(X_BMI,Y_BMI)
    predicted_bmi = pd.Series(DT_bmi_pipe.predict(na_data[['age','gender']]),index=na_data.index)
    df.loc[na_data.index,'bmi'] = predicted_bmi
    df.drop(df[df['gender'] == "Other"].index, inplace = True)
    return df
   

def target(df):
    '''
    Input: Dataframe after clean data

    Return: 
        X: <5109,8>
        y: <5109,1>
    '''
    X = df.drop(["stroke","id", "ever_married", "work_type"], 1)
    y = df.stroke
    return X, y

patient = {"gender": ["Male"], "age": [67], "hypertension": ["No"],\
               "heart_disease": ["Yes"], "Residence_type": ["Urban"],\
               "avg_glucose_level": [228], "bmi": [37],\
               "smoking_status": ["formerly smoked"]}
patient = pd.DataFrame(patient)
def add_input(X, patient):
    frames = [X, patient]
    X = pd.concat(frames, ignore_index= True)
    cat_cols = X.select_dtypes(include = ['object']).columns.to_list()
    for i in cat_cols:
        le = LabelEncoder()
        X[i] = le.fit_transform(X[i].astype(str))
    return X


def scaler_data(data):
    sc = StandardScaler()
    data = sc.fit_transform(data)
    return data




        
