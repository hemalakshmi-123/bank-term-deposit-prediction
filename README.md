# 🏦 Bank Term Deposit Subscription Prediction

Predicts whether a bank customer will subscribe to a term deposit after a
telemarketing call, using the UCI Bank Marketing dataset (~41k records).

## Problem
Banks run expensive outbound calling campaigns to sell term deposits.
Only ~11% of contacted customers actually subscribe. This project builds
a classifier to prioritize customers most likely to say yes — reducing
wasted calls and improving campaign ROI.

## Dataset
[UCI Bank Marketing Dataset](https://archive.ics.uci.edu/dataset/222/bank+marketing)
— `bank-additional-full.csv` (41,188 rows, 20 features + target).

## Key Design Decisions
- **Dropped `duration`**: only known after a call happens → would leak
  information not available at prediction time.
- **`pdays == 999`** ("never contacted") converted into an explicit
  `was_contacted_before` binary flag instead of treated as a numeric distance.
- **Class imbalance** (~89% "no") handled via `class_weight="balanced"`
  (Logistic Regression, SVM, Random Forest) and `scale_pos_weight`
  (XGBoost) — avoided SMOTE to keep the deployed pipeline simple and
  prevent synthetic-sample leakage.
- Evaluated on **Precision, Recall, F1, ROC-AUC, PR-AUC** — not accuracy,
  since a model predicting "no" for everyone would already score ~89%.

## Models Compared
| Model | Notes |
|---|---|
| Logistic Regression | Interpretable baseline |
| SVM (linear kernel) | Slower on this dataset size, useful contrast point |
| Random Forest | Strong baseline, handles mixed feature types well |
| XGBoost | Best performer, tuned via grid/random search |

Results saved automatically to `models/comparison_results.csv` after running evaluation.

## Project Structure
```
bank-term-deposit/
├── data/                   # place bank-additional-full.csv here
├── notebooks/               # EDA.ipynb
├── src/
│   ├── preprocess.py        # cleaning + feature engineering
│   ├── train.py              # trains all 4 models, saves .pkl
│   └── evaluate.py           # compares models on proper metrics
├── app/
│   └── streamlit_app.py      # interactive demo
├── models/                   # saved trained pipelines
├── requirements.txt
└── README.md
```

## How to Run
```bash
pip install -r requirements.txt

# 1. Download data manually and place in data/bank-additional-full.csv

# 2. Train all models
cd src
python train.py

# 3. Evaluate + compare
python evaluate.py

# 4. Launch demo app
cd ..
streamlit run app/streamlit_app.py
```

## Results (fill in after running)
| Model | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|
| Logistic Regression | | | | | |
| SVM | | | | | |
| Random Forest | | | | | |
| XGBoost | | | | | |

## Future Improvements
- SHAP values for per-prediction explainability
- Hyperparameter tuning via Optuna
- Monitor model drift as economic indicators shift over time
