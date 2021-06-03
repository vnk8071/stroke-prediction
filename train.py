import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

import seaborn as sns
import matplotlib.pyplot as plt
import config
import model
import os
import argparse
import joblib

from sklearn.model_selection import train_test_split, cross_val_score

from sklearn.linear_model import LogisticRegression
#from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
#from xgboost import XGBClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import classification_report, roc_curve, auc, confusion_matrix
from sklearn.metrics import f1_score, log_loss, roc_auc_score, recall_score, accuracy_score, precision_score

import warnings
warnings.filterwarnings("ignore")



# Train
def run(df, model):
        df = pd.read_csv(config.TRAINING_FILE)

        # Preprocessing
        X = df.drop("stroke", axis= "columns").values
        y = df.stroke
        from sklearn.preprocessing import LabelEncoder

        num_cols = X.select_dtypes(include = ['int64', 'float64']).columns.to_list()
        cat_cols = X.select_dtypes(include = ['object']).columns.to_list()
        def label_encoder(df):
            for i in cat_cols:
                le = LabelEncoder()
                df[i] = le.fit_transform(df[i])
            return df

        sc = StandardScaler()
        X[num_cols] = sc.fit_transform(X[num_cols])

        # Label encoding
        X = label_encoder(X)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X.values, y.values, test_size=0.3, random_state=0)
        cv = 10

        # Fetch model from model.py
        train_model = model.models[model]

        # Fit model on training data
        train_model.fit(X_train,y_train)

        # Predictions
        y_pred = train_model.predict(X_test)
        y_prob = train_model.predict_proba(X_test)[:,1]
        # Metric
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Accuracy= {accuracy}")
        print(classification_report(y_test, y_pred))
        print(f'ROC AUC score: {round(roc_auc_score(y_test, y_prob), 3)}')
        # Save model
        joblib.dump(
            train_model,
            os.path.join(config.MODEL_OUTPUT, f"../models/train-model.bin")
        )

# Outputs
