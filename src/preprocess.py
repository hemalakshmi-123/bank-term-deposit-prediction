"""
preprocess.py
--------------
Loads the raw UCI Bank Marketing dataset and returns clean,
model-ready features + target.

Key decisions (be ready to explain these in interviews):
1. `duration` is DROPPED — it's only known AFTER the call happens,
   so using it causes data leakage in a real deployment scenario.
2. `pdays == 999` means "never contacted before" — we convert this into
   a binary flag `was_contacted_before` instead of treating 999 as a
   real numeric distance.
3. Categorical columns are one-hot encoded.
4. Target `y` (yes/no) is converted to 1/0.
"""

import pandas as pd
from sklearn.model_selection import train_test_split


def load_data(path: str) -> pd.DataFrame:
    """The UCI file uses ';' as separator."""
    df = pd.read_csv(path, sep=";")
    return df


def clean_and_engineer(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # 1. Drop duration -> prevents data leakage
    if "duration" in df.columns:
        df = df.drop(columns=["duration"])

    # 2. Handle pdays=999 ("never contacted")
    df["was_contacted_before"] = (df["pdays"] != 999).astype(int)
    df["pdays"] = df["pdays"].replace(999, -1)  # keep column, but flagged

    # 3. Encode target
    df["y"] = df["y"].map({"yes": 1, "no": 0})

    # 4. Replace "unknown" with NaN so we can see missingness clearly
    df = df.replace("unknown", pd.NA)

    return df


def split_features_target(df: pd.DataFrame):
    y = df["y"]
    X = df.drop(columns=["y"])
    return X, y


def build_preprocessed_data(raw_path: str, test_size: float = 0.2, random_state: int = 42):
    df = load_data(raw_path)
    df = clean_and_engineer(df)
    X, y = split_features_target(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = build_preprocessed_data(
        "data/bank-additional-full.csv"
    )
    print("Train shape:", X_train.shape)
    print("Test shape:", X_test.shape)
    print("Positive class ratio (train):", y_train.mean().round(3))
