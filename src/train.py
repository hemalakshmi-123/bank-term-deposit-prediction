"""
train.py
--------
Trains Logistic Regression, SVM, Random Forest, and XGBoost on the
bank marketing dataset using sklearn Pipelines (so preprocessing +
model are bundled together — this is what you'd actually deploy).

Handles class imbalance with `class_weight='balanced'` (LogReg, SVM, RF)
and `scale_pos_weight` (XGBoost) rather than SMOTE, to keep the pipeline
simple and avoid leaking synthetic samples across train/test.
"""

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from src.preprocess import build_preprocessed_data


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
    numeric_cols = X.select_dtypes(
        include=["int64", "float64"]).columns.tolist()

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_transformer, numeric_cols),
        ("cat", categorical_transformer, categorical_cols),
    ])
    return preprocessor


def get_models(y_train: pd.Series) -> dict:
    # for XGBoost scale_pos_weight
    pos_ratio = (y_train == 0).sum() / (y_train == 1).sum()

    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=42
        ),

        "Random Forest": RandomForestClassifier(
            n_estimators=300, class_weight="balanced", random_state=42, n_jobs=-1
        ),
        "XGBoost": XGBClassifier(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.1,
            scale_pos_weight=pos_ratio,
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1,
        ),
    }
    return models


def train_all(raw_path: str):
    X_train, X_test, y_train, y_test = build_preprocessed_data(raw_path)
    preprocessor = build_preprocessor(X_train)
    models = get_models(y_train)

    trained_pipelines = {}
    for name, model in models.items():
        print(f"Training {name}...")
        pipe = Pipeline(
            steps=[("preprocessor", preprocessor), ("classifier", model)])
        pipe.fit(X_train, y_train)
        trained_pipelines[name] = pipe

        # Save each model
        filename = f"models/{name.replace(' ', '_').replace('(', '').replace(')', '')}.pkl"
        joblib.dump(pipe, filename)
        print(f"Saved -> {filename}")

    return trained_pipelines, X_test, y_test


if __name__ == "__main__":
    trained_pipelines, X_test, y_test = train_all(
        "data/bank-additional-full.csv")
    joblib.dump((X_test, y_test), "models/test_data.pkl")
    print("\nAll models trained and saved.")
