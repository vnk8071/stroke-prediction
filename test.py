from src.model_train import save_models
from src.StrokeDetector import StrokeDetector
import numpy as np

MODEL_PATH = './models/' + save_models["Logistic Regression"]

patient_input = np.array([1,20,1,1,1,150,30,0])

detector = StrokeDetector(modeltype=MODEL_PATH)

out = detector.predict(patient_input)

print("Output predict :", out)