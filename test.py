from src.StrokeDetector import StrokeDectector
import numpy as np
import argparse

# Argparse
parser = argparse.ArgumentParser(
                        prog= 'test',
                        description= 'Choose model to predict'
                        )

parser.add_argument(
                    '-m', '--model', 
                    type= str,
                    default= "Logistic Regression",
                    choices= [
                              "Logistic Regression",
                              "LightGBM Classifier",
                              "Random Forest",
                              "XGB Classifier",
                              "Adaboosting Classifier",
                              "Decision Tree"
                             ]
                   )

parser.add_argument(
                    '-s', '--string',
                    type= str,
                    default= '0,0,0,0,0,0,0,0'
                   )
args = parser.parse_args()

# Input of patient
patient_input = np.array(args.string.split(','), float).reshape(1,-1)

# Predict input
detector = StrokeDectector(args.model)

# Output stroke or no stroke
output = detector.predict(patient_input)
print("Output predict :", output)