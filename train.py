from src import config
from src import model_train
import os
import argparse
import joblib
from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report, roc_auc_score, accuracy_score
from src.dataset import preprocessing
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings("ignore")

# Train
def run(models):
    train_dir = config.TRAINING_FILE
    X, y = preprocessing(train_dir)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X.values, y.values, test_size=0.3, random_state=0)
    oversample = SMOTE()
    X_train_balanced, y_train_balanced = oversample.fit_resample(X_train, y_train)
    print(X_train_balanced.shape)
    # Fetch model from model.py
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
    # Save model
        joblib.dump(
            models,
            os.path.join(config.MODEL_OUTPUT, f"./models/models")
        )


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()

    parser.add_argument('--model', type=str)
    
    args = parser.parse_args()

    run(models=args.model)
# Outputs
