# 🏦 Bank Term Deposit Subscription Prediction — Complete Project Guide

## 1. What This Project Is
A **Machine Learning classification project** that predicts whether a bank
customer will subscribe to a term deposit after being contacted in a
telemarketing campaign. You compare 4 different ML algorithms, evaluate
them properly, and deploy the best one as an interactive app.

**Type:** Supervised Learning → Binary Classification (Yes/No)

---

## 2. Tools & Tech Stack Needed

| Category | Tool | Purpose |
|---|---|---|
| Language | Python 3.9+ | Core programming language |
| Data handling | pandas, numpy | Loading, cleaning, transforming data |
| ML Models | scikit-learn | Logistic Regression, SVM, Random Forest |
| ML Models | xgboost | Gradient boosting model |
| Model saving | joblib | Save/load trained models |
| Visualization | matplotlib, seaborn | EDA charts, confusion matrices |
| Notebook | Jupyter | Exploratory Data Analysis |
| Deployment | Streamlit | Interactive web demo |
| Version Control | Git + GitHub | Track code, host the project publicly |

Install everything with:
```bash
pip install -r requirements.txt
```

---

## 3. Dataset
- **Name:** UCI Bank Marketing Dataset
- **Link:** https://archive.ics.uci.edu/dataset/222/bank+marketing
- **File to use:** `bank-additional-full.csv` (41,188 rows, 20 features)
- **Target column:** `y` (yes/no — did the customer subscribe?)
- Place the downloaded file inside the `data/` folder.

---

## 4. Project Folder Structure
```
bank-term-deposit/
├── data/                   # dataset goes here (not pushed to GitHub if large)
├── notebooks/               # EDA.ipynb — exploratory analysis
├── src/
│   ├── preprocess.py        # cleaning + feature engineering
│   ├── train.py              # trains all 4 models
│   └── evaluate.py           # compares models, prints metrics
├── app/
│   └── streamlit_app.py      # interactive prediction demo
├── models/                   # saved trained models (.pkl files)
├── requirements.txt          # all dependencies
└── README.md                 # project documentation
```

---

## 5. Step-by-Step: How To Build & Run It

### Step 1 — Setup
```bash
git clone <your-repo-url>
cd bank-term-deposit
pip install -r requirements.txt
```

### Step 2 — Get the Data
Download `bank-additional-full.csv` from the UCI link above → place in `data/`.

### Step 3 — Explore the Data (EDA)
Open a Jupyter notebook and check:
- Class balance (~89% "no", ~11% "yes")
- Missing/unknown values in categorical columns
- Correlations between numeric features
- Distribution of `age`, `campaign`, `pdays`, etc.

### Step 4 — Preprocess
`src/preprocess.py` handles:
- **Dropping `duration`** — it's only known after a call happens, so
  using it would leak future information (a classic ML mistake to avoid).
- **Handling `pdays = 999`** ("never contacted") by converting it into a
  clear `was_contacted_before` flag instead of treating it as a real number.
- One-hot encoding categorical columns (job, education, marital status, etc.)
- Train/test split with stratification (keeps the 89/11 ratio consistent
  in both sets).

### Step 5 — Train the Models
```bash
cd src
python train.py
```
This trains and saves 4 models:
- **Logistic Regression** — simple, interpretable baseline
- **SVM (linear kernel)** — good for comparison, slower on large data
- **Random Forest** — ensemble of decision trees, handles mixed data well
- **XGBoost** — gradient boosting, usually the best performer

Since only ~11% of customers say "yes," **class imbalance** is handled using
`class_weight="balanced"` (for LogReg/SVM/RF) and `scale_pos_weight`
(for XGBoost) — this stops the model from just predicting "no" every time.

### Step 6 — Evaluate & Compare
```bash
python evaluate.py
```
Prints a comparison table using the right metrics for imbalanced data:
**Precision, Recall, F1-score, ROC-AUC, PR-AUC** — NOT plain accuracy
(a model saying "no" to everyone would already score ~89% accuracy,
which is meaningless here).

### Step 7 — Deploy the Demo
```bash
cd ..
streamlit run app/streamlit_app.py
```
Opens a browser form where you enter customer details and get a live
subscription probability prediction.

### Step 8 — Push to GitHub
```bash
git init
git add .
git commit -m "Initial commit: Bank Term Deposit Prediction project"
git branch -M main
git remote add origin <your-repo-url>
git push -u origin main
```

---

## 6. Key Concepts You Must Be Able to Explain

| Concept | Simple Explanation |
|---|---|
| Data leakage | Using info not available at prediction time (why we dropped `duration`) |
| Class imbalance | Rare positive class (~11%) — accuracy is misleading, use F1/AUC instead |
| class_weight="balanced" | Penalizes the model more for misclassifying the minority class |
| Precision vs Recall | Precision = "of predicted yes, how many actually yes"; Recall = "of actual yes, how many did we catch" |
| ROC-AUC vs PR-AUC | PR-AUC is more informative than ROC-AUC when classes are imbalanced |
| One-hot encoding | Converts categorical text columns into numeric 0/1 columns for models |
| Ensemble models (RF, XGBoost) | Combine many weak learners (decision trees) into a strong one |
| Pipeline | Bundles preprocessing + model together so it works the same way in deployment |

---

## 7. Resume Bullet Point (ready to use)
> Built an end-to-end ML classification pipeline predicting bank term
> deposit subscriptions, comparing Logistic Regression, SVM, Random
> Forest, and XGBoost; addressed class imbalance and data leakage,
> and deployed the best model via a Streamlit web app.

---

## 8. Future Improvements (optional, mention if asked "what would you add")
- SHAP values for explaining individual predictions
- Hyperparameter tuning with Optuna/GridSearchCV
- Model monitoring for drift as economic indicators change over time
