"""
Titanic Survival Prediction
=============================
Build a model that predicts whether a passenger on the Titanic
survived or not, using passenger info: age, gender, ticket class,
fare, cabin, etc.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ----------------------------------------------------------------------
# 1. LOAD DATA
# ----------------------------------------------------------------------
df = pd.read_csv("titanic.csv")

# ----------------------------------------------------------------------
# 2. CLEAN DATA
# ----------------------------------------------------------------------
df["Age"] = df["Age"].fillna(df["Age"].median())
df["Fare"] = df["Fare"].fillna(df["Fare"].median())
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

# Cabin is mostly missing -> just use whether it's known or not
df["Has_Cabin"] = df["Cabin"].notna().astype(int)

# ----------------------------------------------------------------------
# 3. FEATURE ENGINEERING
# ----------------------------------------------------------------------
df["Sex"] = df["Sex"].map({"male": 0, "female": 1})
df["Embarked"] = df["Embarked"].map({"S": 0, "C": 1, "Q": 2})

# Family size from siblings/spouses + parents/children
df["Family_Size"] = df["SibSp"] + df["Parch"] + 1
df["Is_Alone"] = (df["Family_Size"] == 1).astype(int)

# ----------------------------------------------------------------------
# 4. TRAIN / TEST SPLIT
# ----------------------------------------------------------------------
features = ["Pclass", "Sex", "Age", "Fare", "Embarked", "Family_Size", "Is_Alone", "Has_Cabin"]
X = df[features]
y = df["Survived"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ----------------------------------------------------------------------
# 5. TRAIN CLASSIFICATION MODEL
# ----------------------------------------------------------------------
model = RandomForestClassifier(n_estimators=300, max_depth=6, min_samples_leaf=3, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

# ----------------------------------------------------------------------
# 6. EVALUATE
# ----------------------------------------------------------------------
preds = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, preds))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, preds))
print("\nClassification Report:\n", classification_report(y_test, preds))

# ----------------------------------------------------------------------
# 7. EXAMPLE PREDICTION
# ----------------------------------------------------------------------
sample = X_test.iloc[[0]]
print("Sample prediction:", model.predict(sample)[0], " | Actual:", y_test.iloc[0])
