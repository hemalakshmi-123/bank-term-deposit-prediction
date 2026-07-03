import joblib
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Term Deposit Predictor",
    page_icon="🏦"
)

st.title("🏦 Bank Term Deposit Subscription Predictor")
st.write("Enter customer details to predict whether a customer will subscribe.")

model = joblib.load("models/XGBoost.pkl")

with st.form("prediction_form"):

    left, right = st.columns(2)

    with left:
        age = st.number_input("Age", 18, 100, 35)

        job = st.selectbox("Job", [
            "admin.", "blue-collar", "entrepreneur", "housemaid",
            "management", "retired", "self-employed", "services",
            "student", "technician", "unemployed"
        ])

        marital = st.selectbox(
            "Marital Status",
            ["married", "single", "divorced"]
        )

        education = st.selectbox(
            "Education",
            [
                "basic.4y",
                "basic.6y",
                "basic.9y",
                "high.school",
                "illiterate",
                "professional.course",
                "university.degree"
            ]
        )

        default = st.selectbox(
            "Has Credit Default?",
            ["no", "yes"]
        )

        housing = st.selectbox(
            "Has Housing Loan?",
            ["no", "yes"]
        )

        loan = st.selectbox(
            "Has Personal Loan?",
            ["no", "yes"]
        )

    with right:

        contact = st.selectbox(
            "Contact Type",
            ["cellular", "telephone"]
        )

        month = st.selectbox(
            "Last Contact Month",
            [
                "jan", "feb", "mar", "apr",
                "may", "jun", "jul", "aug",
                "sep", "oct", "nov", "dec"
            ]
        )

        day = st.selectbox(
            "Day of Week",
            ["mon", "tue", "wed", "thu", "fri"]
        )

        campaign = st.number_input(
            "Number of Contacts This Campaign",
            1,
            50,
            2
        )

        pdays = st.number_input(
            "Days Since Last Contact (-1 if never)",
            -1,
            999,
            -1
        )

        previous = st.number_input(
            "Number of Previous Contacts",
            0,
            20,
            0
        )

        poutcome = st.selectbox(
            "Previous Campaign Outcome",
            ["nonexistent", "failure", "success"]
        )

    emp_rate = st.number_input(
        "Employment Variation Rate",
        -5.0,
        5.0,
        1.1
    )

    price_index = st.number_input(
        "Consumer Price Index",
        90.0,
        100.0,
        93.9
    )

    confidence = st.number_input(
        "Consumer Confidence Index",
        -60.0,
        0.0,
        -36.4
    )

    euribor = st.number_input(
        "Euribor 3 Month Rate",
        0.0,
        6.0,
        4.86
    )

    employees = st.number_input(
        "Number of Employees (thousands)",
        4900.0,
        5300.0,
        5191.0
    )

    predict = st.form_submit_button("Predict")

if predict:

    contacted = 1 if pdays != -1 else 0

    user_data = pd.DataFrame([{
        "age": age,
        "job": job,
        "marital": marital,
        "education": education,
        "default": default,
        "housing": housing,
        "loan": loan,
        "contact": contact,
        "month": month,
        "day_of_week": day,
        "campaign": campaign,
        "pdays": pdays,
        "previous": previous,
        "poutcome": poutcome,
        "emp.var.rate": emp_rate,
        "cons.price.idx": price_index,
        "cons.conf.idx": confidence,
        "euribor3m": euribor,
        "nr.employed": employees,
        "was_contacted_before": contacted
    }])

    prediction = model.predict(user_data)[0]
    probability = model.predict_proba(user_data)[0][1]

    st.subheader("Prediction")

    if prediction == 1:
        st.success(
            f"✅ Likely to SUBSCRIBE (Probability: {probability:.1%})"
        )
    else:
        st.error(
            f"❌ Likely to NOT Subscribe (Probability: {probability:.1%})"
        )

    st.progress(float(probability))
