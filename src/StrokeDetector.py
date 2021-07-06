import joblib
from src import config
from src import model_train
from src.param import mean, std


class StrokeDectector:
    def __init__(self, classifier_model):
        self.model = joblib.load(config.MODEL_OUTPUT + model_train.save_models[classifier_model])
        self.mean = mean
        self.std = std 
    

    def user_predict(self, patient):
        '''
        Input: Patient status with label encoding

        Return: Type ndarray of patient status after using Standardization method
        '''
        patient_predict = (patient - self.mean) / self.std
        return patient_predict


    def predict(self, patient):
        '''
        Input: Paitent status 

        Return: Predict stroke or no stroke
        '''
        patient_predict = self.user_predict(patient)
        output = self.model.predict(patient_predict)
        
        if output == 0:
            output = "No stroke"
        else:
            output = "Stroke" 
        return output 

