import glob
import joblib
import pandas as pd

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
)


def evaluate_models(models_path="models"):
    X_test, y_test = joblib.load(f"{models_path}/test_data.pkl")

    results = []

    model_files = glob.glob(f"{models_path}/*.pkl")
    model_files = [file for file in model_files if "test_data" not in file]

    for file in model_files:
        model_name = file.split("/")[-1].replace(".pkl", "").replace("_", " ")

        model = joblib.load(file)

        predictions = model.predict(X_test)
        probabilities = model.predict_proba(X_test)[:, 1]

        results.append({
            "Model": model_name,
            "Precision": round(precision_score(y_test, predictions), 3),
            "Recall": round(recall_score(y_test, predictions), 3),
            "F1": round(f1_score(y_test, predictions), 3),
            "ROC-AUC": round(roc_auc_score(y_test, probabilities), 3),
            "PR-AUC": round(average_precision_score(y_test, probabilities), 3),
        })

        print(f"\n===== {model_name} =====")
        print(confusion_matrix(y_test, predictions))
        print(classification_report(y_test, predictions))

    result_df = pd.DataFrame(results)
    result_df = result_df.sort_values(by="F1", ascending=False)

    print("\nFinal Comparison")
    print(result_df.to_string(index=False))

    result_df.to_csv(f"{models_path}/comparison_results.csv", index=False)

    return result_df


if __name__ == "__main__":
    evaluate_models()
