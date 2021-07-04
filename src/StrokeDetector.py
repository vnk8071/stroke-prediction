import joblib
import config
from src.model_train import save_models
from src.param import mean, std
from streamlit import input_user, user_predict


class StrokeDetector:
    def __init__(self, modelpath):
        self.model = joblib.load(config.MODEL_OUTPUT + save_models[modelpath])
        self.mean = mean
        self.std = std
    
    def transform(self):
        # Transform input of patient into encoder and scaler
        patient = input_user()
        patient_scaler = user_predict(patient, self.mean, self.std)
        return patient_scaler

    def predict(self, patient):
        patient_scaler = self.transform(patient)
        output = self.model(patient_scaler)
        return output

