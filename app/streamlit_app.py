"""
streamlit_app.py
-----------------
Simple UI to demo the best model. Run with:
    streamlit run app/streamlit_app.py
"""

import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Term Deposit Predictor", page_icon="🏦")
st.title("🏦 Bank Term Deposit Subscription Predictor")
st.write("Enter customer details to predict likelihood of subscribing to a term deposit.")

# Load best model (change filename to whichever performed best in evaluate.py)
MODEL_PATH = "models/XGBoost.pkl"
model = joblib.load(MODEL_PATH)

with st.form("input_form"):
    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age", 18, 100, 35)
        job = st.selectbox("Job", [
            "admin.", "blue-collar", "entrepreneur", "housemaid",
            "management", "retired", "self-employed", "services",
            "student", "technician", "unemployed"
        ])
        marital = st.selectbox("Marital Status", ["married", "single", "divorced"])
        education = st.selectbox("Education", [
            "basic.4y", "basic.6y", "basic.9y", "high.school",
            "illiterate", "professional.course", "university.degree"
        ])
        default = st.selectbox("Has Credit Default?", ["no", "yes"])
        housing = st.selectbox("Has Housing Loan?", ["no", "yes"])
        loan = st.selectbox("Has Personal Loan?", ["no", "yes"])

    with col2:
        contact = st.selectbox("Contact Type", ["cellular", "telephone"])
        month = st.selectbox("Last Contact Month", [
            "jan", "feb", "mar", "apr", "may", "jun",
            "jul", "aug", "sep", "oct", "nov", "dec"
        ])
        day_of_week = st.selectbox("Day of Week", ["mon", "tue", "wed", "thu", "fri"])
        campaign = st.number_input("Number of Contacts This Campaign", 1, 50, 2)
        pdays = st.number_input("Days Since Last Contact (-1 if never)", -1, 999, -1)
        previous = st.number_input("Number of Previous Contacts", 0, 20, 0)
        poutcome = st.selectbox("Previous Campaign Outcome", ["nonexistent", "failure", "success"])

    emp_var_rate = st.number_input("Employment Variation Rate", -5.0, 5.0, 1.1)
    cons_price_idx = st.number_input("Consumer Price Index", 90.0, 100.0, 93.9)
    cons_conf_idx = st.number_input("Consumer Confidence Index", -60.0, 0.0, -36.4)
    euribor3m = st.number_input("Euribor 3 Month Rate", 0.0, 6.0, 4.86)
    nr_employed = st.number_input("Number of Employees (thousands)", 4900.0, 5300.0, 5191.0)

    submitted = st.form_submit_button("Predict")

if submitted:
    was_contacted_before = 1 if pdays != -1 else 0

    input_df = pd.DataFrame([{
        "age": age, "job": job, "marital": marital, "education": education,
        "default": default, "housing": housing, "loan": loan,
        "contact": contact, "month": month, "day_of_week": day_of_week,
        "campaign": campaign, "pdays": pdays, "previous": previous,
        "poutcome": poutcome, "emp.var.rate": emp_var_rate,
        "cons.price.idx": cons_price_idx, "cons.conf.idx": cons_conf_idx,
        "euribor3m": euribor3m, "nr.employed": nr_employed,
        "was_contacted_before": was_contacted_before,
    }])

    proba = model.predict_proba(input_df)[0][1]
    pred = model.predict(input_df)[0]

    st.subheader("Result")
    if pred == 1:
        st.success(f"✅ Likely to SUBSCRIBE — probability: {proba:.1%}")
    else:
        st.error(f"❌ Likely to NOT subscribe — probability: {proba:.1%}")

    st.progress(float(proba))
