from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier



models = {
    "Logistic Regression": LogisticRegression(
    C= 0.01, solver='newton-cg',  penalty='l2'
    ),
    "Random Forest": RandomForestClassifier(
        max_depth= 7
    ),
    "Adaboosting Classifier": AdaBoostClassifier(
    ),
    "Decision Tree": DecisionTreeClassifier(
    ),
    "LightGBM Classifier": LGBMClassifier(    
    ),
    "XGB Classifier": XGBClassifier(
    )
}

save_models = {
    "Logistic Regression": "Logimodel.pkl",
    "Random Forest": "Randmodel.pkl",
    "Adaboosting Classifier": "AdaBmodel.pkl",
    "Decision Tree": "Decimodel.pkl",
    "LightGBM Classifier": "LGBMmodel.pkl",
    "XGB Classifier": "XGBCmodel.pkl"
}

