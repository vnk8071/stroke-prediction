import numpy as np
import pandas as pd
import os
import glob

from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline

folder_dir = "./data"

#Dataset 
def df():
    for filename in glob(os.path.join(folder_dir, "*.zip")):
        df = pd.read_csv(filename)
    return df
        
def preprocessing(data_dir):
    '''
    Prepare training data
    Input:
        data_dir <str> : path to data file (csv), can be train set or test set
    Returns:
        X <m x n>: data features of n samples
        y <1 x n>:  data labels
    '''
    # Load data
    df = pd.read_csv(data_dir)

    # Transform
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
    # Drop irrelevant features
    X = df.drop("stroke", axis= "columns")
    y = df.stroke

    num_cols = X.select_dtypes(include = ['int64', 'float64']).columns.to_list()
    cat_cols = X.select_dtypes(include = ['object']).columns.to_list()

    sc = StandardScaler()
    X[num_cols] = sc.fit_transform(X[num_cols])

    # Label encoding
    X = label_encoder(X, cat_cols)

    return X, y


def label_encoder(df, cat_cols):
    for i in cat_cols:
        le = LabelEncoder()
        df[i] = le.fit_transform(df[i])
    return df


        
