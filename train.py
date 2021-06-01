import gc
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

import seaborn as sns
import matplotlib.pyplot as plt


from sklearn.linear_model import LogisticRegression
#from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
#from xgboost import XGBClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

from sklearn.metrics import classification_report, roc_curve, auc, confusion_matrix
from sklearn.metrics import f1_score, log_loss, roc_auc_score, recall_score, accuracy_score, precision_score

import warnings
warnings.filterwarnings("ignore")

from google.colab import drive
drive.mount('/content/gdrive')

gc.collect()
# Load data
df = pd.read_csv("/content/gdrive/MyDrive/Colab Notebooks/Stroke data/healthcare-dataset-stroke-data.csv")
df.head()

DT_bmi_pipe = Pipeline(steps=[('scale',StandardScaler()),
                               ('lr',DecisionTreeRegressor(random_state=42))])
X_BMI = df[['age','gender','bmi']].copy()
X_BMI.gender = X_BMI.gender.replace({'Male':0,'Female':1,'Other':-1}).astype(np.uint8)

# Handling misssing
Missing = X_BMI[X_BMI.bmi.isna()]
X_BMI = X_BMI[~X_BMI.bmi.isna()]
Y_BMI = X_BMI.pop('bmi')
DT_bmi_pipe.fit(X_BMI,Y_BMI)
predicted_bmi = pd.Series(DT_bmi_pipe.predict(Missing[['age','gender']]),index=Missing.index)
df.loc[Missing.index,'bmi'] = predicted_bmi
df.head()

df.drop(df[df['gender'] == "Other"].index, inplace = True)
df['gender'].value_counts()

# preprocessing
X = df.drop("stroke", axis= "columns")
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

# split data
from sklearn.model_selection import train_test_split, cross_val_score
from imblearn.over_sampling import SMOTE
cv = 10
X_train, X_test, y_train, y_test = train_test_split( X.values, y.values, test_size=0.3, random_state=0)
X_train_balanced, y_train_balanced = oversample.fit_resample(X_train, y_train)

# Train
    #LogisticRegression
lg = LogisticRegression(C=0.01,penalty='l2',solver= 'newton-cg', random_state = 17)
lg.fit(X_train_balanced, y_train_balanced)
y_pred = lg.predict(X_test)
y_prob = lg.predict_proba(X_test)[:,1]

    #LGBMClassifier
"""lgbm = LGBMClassifier(random_state = 17, max_depth = 8, num_leaves = 50, objective= 'binary', boosting_type= 'goss')
lgbm.fit(X_train_balanced, y_train_balanced)
y_pred = lgbm.predict(X_test)
y_prob = lgbm.predict_proba(X_test)[:,1]"""

    #RandomForest
rf = RandomForestClassifier(random_state = 17, max_depth = 5)
rf.fit(X_train_balanced, y_train_balanced)
y_pred = rf.predict(X_test)
y_prob = rf.predict_proba(X_test)[:,1]

    #XGB
"""xgb = XGBClassifier(random_state = 17, learning_rate= 0.3, max_depth = 8, objective = 'binary:logistic', eval_metric = 'logloss', early_stopping_rounds=10, verbose=True)
xgb.fit(X_train_balanced, y_train_balanced)
y_pred = xgb.predict(X_test)
y_prob = xgb.predict_proba(X_test)[:,1]"""

    #Adaboost
adb = AdaBoostClassifier(random_state = 17)
adb.fit(X_train_balanced, y_train_balanced)
y_pred = adb.predict(X_test)
y_prob = adb.predict_proba(X_test)[:,1]

    #Decision Tree
dtr = DecisionTreeClassifier(random_state = 17)
dtr.fit(X_train_balanced, y_train_balanced)
y_pred = dtr.predict(X_test)
y_prob = dtr.predict_proba(X_test)[:,1]   
# validate
results = pd.DataFrame(columns = ['LR', 'LGBM', 'RF', 'XGB', 'ADB', 'DT'], index = range(5))
results_cv = pd.DataFrame(columns = ['LR', 'LGBM', 'RF', 'XGB', 'ADB', 'DT'], index = range(5))
results.iloc[0, 0] = round(accuracy_score(y_test, y_pred), 3)
results.iloc[1, 0] = round(precision_score(y_test, y_pred), 2)
results.iloc[2, 0] = round(recall_score(y_test, y_pred), 2)
results.iloc[3, 0] = round(f1_score(y_test, y_pred), 2)
results.iloc[4, 0] = round(roc_auc_score(y_test, y_prob), 3)
lg_cm = confusion_matrix(y_test, y_pred)

print(classification_report(y_test, y_pred))
print(f'ROC AUC score: {round(roc_auc_score(y_test, y_prob), 3)}')

results_cv.iloc[0, 0] = round(cross_val_score(lg, X_train_balanced, y_train_balanced, cv = cv, scoring = 'accuracy').mean(), 3)
results_cv.iloc[1, 0] = round(cross_val_score(lg, X_train_balanced, y_train_balanced, cv = cv, scoring = 'precision').mean(), 2)
results_cv.iloc[2, 0] = round(cross_val_score(lg, X_train_balanced, y_train_balanced, cv = cv, scoring = 'recall').mean(), 2)
results_cv.iloc[3, 0] = round(cross_val_score(lg, X_train_balanced, y_train_balanced, cv = cv, scoring = 'f1').mean(), 2)
results_cv.iloc[4, 0] = round(cross_val_score(lg, X_train_balanced, y_train_balanced, cv = cv, scoring = 'roc_auc').mean(), 3)

# Visualize confusion matrix
plt.figure(figsize = (9, 6))
sns.heatmap(lg_cm, cmap = 'Blues', annot = True, fmt = 'd', linewidths = 5, cbar = False, annot_kws = {'fontsize': 15}, 
            yticklabels = ['No stroke', 'Stroke'], xticklabels = ['Predicted no stroke', 'Predicted stroke'])
plt.yticks(rotation = 0)
plt.show()

# Roc curve
false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(false_positive_rate, true_positive_rate)

sns.set_theme(style = 'white')
plt.figure(figsize = (10, 10))
plt.plot(false_positive_rate,true_positive_rate, color = '#b01717', label = 'AUC = %0.3f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1], linestyle = '--', color = '#174ab0')
plt.axis('tight')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()
# Outputs
