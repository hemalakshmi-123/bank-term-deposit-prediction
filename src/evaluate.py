"""
evaluate.py
-----------
Compares all trained models using metrics that actually matter for
imbalanced data: Precision, Recall, F1, ROC-AUC, PR-AUC.
(Accuracy alone is misleading here since ~89% of customers say "no".)
"""

import glob
import joblib
import pandas as pd
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score,
    confusion_matrix, classification_report,
)


def evaluate_all(models_dir: str = "models"):
    X_test, y_test = joblib.load(f"{models_dir}/test_data.pkl")

    results = []
    model_files = glob.glob(f"{models_dir}/*.pkl")
    model_files = [f for f in model_files if "test_data" not in f]

    for file in model_files:
        name = file.split("/")[-1].replace(".pkl", "").replace("_", " ")
        pipe = joblib.load(file)

        y_pred = pipe.predict(X_test)
        y_proba = pipe.predict_proba(X_test)[:, 1]

        results.append({
            "Model": name,
            "Precision": round(precision_score(y_test, y_pred), 3),
            "Recall": round(recall_score(y_test, y_pred), 3),
            "F1": round(f1_score(y_test, y_pred), 3),
            "ROC-AUC": round(roc_auc_score(y_test, y_proba), 3),
            "PR-AUC": round(average_precision_score(y_test, y_proba), 3),
        })

        print(f"\n=== {name} ===")
        print(confusion_matrix(y_test, y_pred))
        print(classification_report(y_test, y_pred))

    results_df = pd.DataFrame(results).sort_values("F1", ascending=False)
    print("\n\n===== FINAL COMPARISON TABLE =====")
    print(results_df.to_string(index=False))
    results_df.to_csv("models/comparison_results.csv", index=False)
    return results_df


if __name__ == "__main__":
    evaluate_all()
