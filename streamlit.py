import streamlit as st
from src import model_train
from src import config
import numpy as np
import pandas as pd
from src.dataset import read_data, target, add_input, scaler_data
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings("ignore")

st.title("Stroke Prediction")
st.write("""
# The project to predict the risk of stroke \n
The data was collected from Kaggle Community. \n
Link: https://www.kaggle.com/fedesoriano/stroke-prediction-dataset
""")

st.write("Please set your profile to predict")
classifier_model = st.sidebar.selectbox("Select model", ("Logistic Regression",\
                        "LightGBM Classifier", "Random Forest", "XGB Classifier",\
                        "Adaboosting Classifier", "Decision Tree"))

def main(classifier_model):
    '''
    Input: Classifier model like Logistic Regression

    Return: The shape of the train dataset after use SMOTE method and model
    '''
    train_dir = config.TRAINING_FILE
    df = read_data(train_dir)
    X, y = target(df)
    X = add_input(X, patient)
    X = scaler_data(X)
    patient_input = X[-1].reshape(1, -1)
    X = X[:-1]

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.3, random_state= 0)
    oversample = SMOTE()
    X_train_balanced, y_train_balanced = oversample.fit_resample(X_train, y_train)

    # Fetch model from model_train.py
    model_select = model_train.models[classifier_model]

    # Fit model on training data
    model_select.fit(X_train_balanced, y_train_balanced)

    output = model_select.predict(patient_input)
    if output == 0:
        output = "No stroke"
    else:
        output = "Stroke" 
    return output 


def input_user():
    '''
    Input: The status of patient like gender, age, etc.

    Return: Type dataframe of patient healthcare status {"gender": Male, "age": 20, etc}
    '''

    # Input parameter
    gender = st.selectbox("Select gender", ("Male", "Female"))
    age = st.slider("Select age",1,100)
    hypertension = st.selectbox("Hypertension", ("Yes", "No"))
    heart_disease = st.selectbox("Heart Disease",("Yes", "No"))
    residence = st.selectbox("Type of residence", ("Rural", "Urban"))
    avr_glucose = st.slider("The glucose level",50,250)
    bmi_value = st.slider("The BMI value", 10, 50)
    smoking = st.selectbox("The smoking status", ("Formerly smoked", "Never smoked",\
                                                     "Smokes", "Unknow"))


    # Change into array
    patient = {"gender": [gender], "age": [age], "hypertension": [hypertension],\
               "heart_disease": [heart_disease], "Residence_type": [residence],\
               "avg_glucose_level": [avr_glucose], "bmi": [bmi_value],\
               "smoking_status": [smoking]}
    patient = pd.DataFrame(patient)
    return patient


if __name__ == '__main__':

    # Input of patient
    patient = input_user()
    st.write("The input of patient", patient)
    
    st.markdown("The model prediction: " + classifier_model)

    start = st.button("Start predict")
    if start == True:
        # Output predict
        output = main(classifier_model)
        st.write("The result of predict: ", output)
        st.write("Done")
    else:
        st.write("Set patient profile again")

