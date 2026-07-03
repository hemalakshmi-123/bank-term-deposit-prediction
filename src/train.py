import joblib

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier

from src.preprocess import build_preprocessed_data


def create_preprocessor(X):
    cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
    num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    return ColumnTransformer([
        ("num", num_pipeline, num_cols),
        ("cat", cat_pipeline, cat_cols)
    ])


def get_models(y_train):
    weight = (y_train == 0).sum() / (y_train == 1).sum()

    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=42
        ),

        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        ),

        "XGBoost": XGBClassifier(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.1,
            scale_pos_weight=weight,
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1
        )
    }

    return models


def train_models(data_path):
    X_train, X_test, y_train, y_test = build_preprocessed_data(data_path)

    preprocessor = create_preprocessor(X_train)
    models = get_models(y_train)

    trained_models = {}

    for name, model in models.items():
        print(f"Training {name}...")

        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", model)
        ])

        pipeline.fit(X_train, y_train)

        trained_models[name] = pipeline

        file_name = (
            "models/"
            + name.replace(" ", "_").replace("(", "").replace(")", "")
            + ".pkl"
        )

        joblib.dump(pipeline, file_name)
        print(f"Saved -> {file_name}")

    return trained_models, X_test, y_test


if __name__ == "__main__":
    trained_models, X_test, y_test = train_models(
        "data/bank-additional-full.csv"
    )

    joblib.dump((X_test, y_test), "models/test_data.pkl")

    print("\nAll models trained and saved.")
