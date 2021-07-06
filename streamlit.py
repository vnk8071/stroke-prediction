import streamlit as st
import numpy as np
import pandas as pd
from src.dataset import label_encoder
from StrokeDetector import StrokeDectector
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


def input_user():
        '''
        Input: The status of patient like gender, age, etc.

        Return: Type ndarray of patient healthcare status [0, 1, etc]
        '''
        # Input parameter
        gender = st.selectbox("Select gender", ("Male", "Female"))
        age = float(st.slider("Select age",1,100))
        hypertension = st.selectbox("Hypertension", ("Yes", "No"))
        heart_disease = st.selectbox("Heart Disease",("Yes", "No"))
        residence = st.selectbox("Type of residence", ("Rural", "Urban"))
        avg_glucose = float(st.slider("The glucose level",50,250))
        bmi_value = float(st.slider("The BMI value", 10, 50))
        smoking = st.selectbox("The smoking status", ("formerly smoked", "never smoked",\
                                                        "smokes", "Unknown"))

        # Change into array
        patient = {"gender": [gender],"age": [age], "hypertension": [hypertension],\
                   "heart_disease": [heart_disease], "Residence_type": [residence],\
                   "avg_glucose_level": [avg_glucose], "bmi": [bmi_value],\
                   "smoking_status": [smoking]}
        patient = pd.DataFrame(patient)
        patient = label_encoder(patient)
        patient = np.array(patient)
        return patient


if __name__ == '__main__':
    
    # Input of patient
    patient = input_user()
    st.write("The input of patient", patient) 
    st.markdown("The model prediction: " + classifier_model)

    # Start to predict
    start = st.button("Start predict")
    if start == True:
        output = StrokeDectector(classifier_model)
        output = output.predict(patient)
        st.write("The result of predict: ", output)
        st.write("Done")
    else:
        st.write("Set patient profile again")

