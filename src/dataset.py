import numpy as np
import pandas as pd
import os
import glob

from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split


folder_dir = "./data"
#Dataset 
def df():
    for filename in glob(os.path.join(folder_dir, "*.zip")):
        df = pd.read_csv(filename)
        def data_handlemissing(df):
            DT_bmi_pipe = Pipeline(steps=[('scale',StandardScaler()),
                                ('dtr',DecisionTreeRegressor(random_state=42))])
            X_BMI = df[['age','gender','bmi']].copy()
            X_BMI.gender = X_BMI.gender.replace({'Male':0,'Female':1,'Other':-1}).astype(np.uint8)

            Missing = X_BMI[X_BMI.bmi.isna()]
            X_BMI = X_BMI[~X_BMI.bmi.isna()]
            Y_BMI = X_BMI.pop('bmi')
            DT_bmi_pipe.fit(X_BMI,Y_BMI)
            predicted_bmi = pd.Series(DT_bmi_pipe.predict(Missing[['age','gender']]),index=Missing.index)
            df.loc[Missing.index,'bmi'] = predicted_bmi
            df.drop(df[df['gender'] == "Other"].index, inplace = True)
        return df



        