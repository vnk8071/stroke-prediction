# STROKE PREDICTION

Machine Learning tool to predict risk of having stroke.

The data collect from Kaggle: https://www.kaggle.com/fedesoriano/stroke-prediction-dataset

According to the World Health Organization (WHO) stroke is the 2nd leading cause of death globally, responsible for approximately 11% of total deaths.This dataset is used to predict whether a patient is likely to get stroke based on the input parameters like gender, age, various diseases, and smoking status. Each row in the data provides relavant information about the patient.

We use 6 models of Machine Learning (Logistic Regression, lightGBM, xgboost, Adaboost, Random Forest and Decision Tree) and compare them with each other. 

The output expected: Logistic Regreesion has area under curve (82%) higher than each other. 

## Install 
Create virtual environment
```bash
conda create -n myenv python=3.7
conda activate myenv
```

Change directory:
```bash
cd stroke-prediction/
```

Install dependencies:
```bash
pip install -r requirements.txt
```

Download and set up data by running
```bash
bash setup-data.sh
```

## Usage
Run
```bash
python train.py
```
## Try your best


