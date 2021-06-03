from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

models = {
    "Logistic Regression": LogisticRegression(
    C= 0.01, solver='newton-cg',  penalty='l2',
    ),
    "RandomForest": RandomForestClassifier(
        max_depth= 7
    ),
    "AdaBoost": AdaBoostClassifier(
    ),
    "DecisionTree": DecisionTreeClassifier(
    )
}