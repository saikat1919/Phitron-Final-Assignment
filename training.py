import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer, KBinsDiscretizer
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, StackingClassifier, VotingClassifier
from sklearn.svm import SVC

from sklearn.metrics import classification_report, confusion_matrix

df = pd.read_csv("diabetes.csv")
invalid_value_cols = ["Glucose", "BloodPressure",	"SkinThickness",	"Insulin",	"BMI"]
df[invalid_value_cols] = df[invalid_value_cols].replace(0, np.nan)

bin_features = ['Glucose']
num_features = [col for col in df.drop(columns=['Outcome']).select_dtypes(include='number').columns if col not in bin_features]

num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('log', FunctionTransformer(np.log1p, validate=False)),
    ("scaler", StandardScaler())
])

bin_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('bin', KBinsDiscretizer(
        n_bins=3,
        strategy='quantile',
        encode='onehot'
    ))
])

preprocessor = ColumnTransformer([
    ('num', num_pipeline, num_features),
    ('bin', bin_pipeline, bin_features)
])

X, y = df.drop(columns=["Outcome"]), df["Outcome"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=42
)

rf_model =  RandomForestClassifier(max_features='log2', min_samples_leaf=4,
                                        min_samples_split=5, n_estimators=500,
                                        random_state=42)

rf_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', rf_model)
])

rf_pipeline.fit(X_train, y_train)

y_pred = rf_pipeline.predict(X_test)

print(classification_report(y_test, y_pred))
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:\n",cm)

import pickle

with open("RandomForest.pkl", 'wb') as f:
  pickle.dump(rf_pipeline, f)




