from src import config
from src import model_train
import pandas as pd
import os
import argparse
import joblib
from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report, roc_auc_score, accuracy_score
from src.dataset import read_data, target, add_input, scaler_data
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings("ignore")

# Train
def run(models):
    train_dir = config.TRAINING_FILE
    df = read_data(train_dir)
    X, y = target(df)
    
    patient = {"gender":["Male"], "age": [67], "hypertension": ["No"],\
               "heart_disease": ["Yes"], "Residence_type": ["Urban"],\
               "avg_glucose_level": [228], "bmi": [37], "smoking_status": ["formerly smoked"]}
    
    
    patient = pd.DataFrame(patient)
    X = add_input(X, patient)
    X = scaler_data(X)
    patient_input = X[-1].reshape(1, -1)
    X = X[:-1]
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    oversample = SMOTE()
    X_train_balanced, y_train_balanced = oversample.fit_resample(X_train, y_train)
    
    # Fetch model from model_train.py
    trainmodel = model_train.models.values()
 
    for models in trainmodel:
    # Fit model on training data
        models.fit(X_train_balanced, y_train_balanced)

    # Predictions
        y_pred = models.predict(X_test)
        y_prob = models.predict_proba(X_test)[:,1]
    
    # Metric
        print("------------------Model------------------")
        print(models)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Accuracy= {round(accuracy, 3)}")
        print(f'ROC AUC score= {round(roc_auc_score(y_test, y_prob), 3)}')
        print(classification_report(y_test, y_pred))


    # Predict
        print("Output predict: ", models.predict(patient_input))
    
    # Save model
        joblib.dump(
            models,
            os.path.join(config.MODEL_OUTPUT, str(models)[:4] + "model.pkl")
        )


if __name__ == "__main__":
    os.mkdir(config.MODEL_OUTPUT)
    parser = argparse.ArgumentParser()

    parser.add_argument('--model', type=str)
    
    args = parser.parse_args()

    run(models=args.model)


# Outputs
