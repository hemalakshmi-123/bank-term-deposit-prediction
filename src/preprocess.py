import pandas as pd
from sklearn.model_selection import train_test_split


def load_data(file_path):
    return pd.read_csv(file_path, sep=";")


def preprocess_data(df):
    df = df.copy()

    if "duration" in df.columns:
        df.drop(columns=["duration"], inplace=True)

    df["was_contacted_before"] = (df["pdays"] != 999).astype(int)
    df["pdays"] = df["pdays"].replace(999, -1)

    df["y"] = df["y"].map({
        "yes": 1,
        "no": 0
    })

    df.replace("unknown", pd.NA, inplace=True)

    return df


def split_data(df):
    X = df.drop("y", axis=1)
    y = df["y"]
    return X, y


def build_preprocessed_data(file_path, test_size=0.2, random_state=42):
    df = load_data(file_path)
    df = preprocess_data(df)

    X, y = split_data(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = build_preprocessed_data(
        "data/bank-additional-full.csv"
    )

    print("Train Shape:", X_train.shape)
    print("Test Shape:", X_test.shape)
    print("Positive Class Ratio:", round(y_train.mean(), 3))
