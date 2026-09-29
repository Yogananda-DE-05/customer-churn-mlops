import streamlit as st
import joblib
import pandas as pd

model = joblib.load("models/churn_model.pkl")
feature_names = joblib.load("models/feature_names.pkl")

st.title("Customer Churn Prediction")

senior = st.selectbox("Senior Citizen", [0, 1])
tenure = st.number_input("Tenure", min_value=0, value=12)
monthly = st.number_input("Monthly Charges", min_value=0.0, value=50.0)
total = st.number_input("Total Charges", min_value=0.0, value=600.0)

if st.button("Predict"):

    row = {feature: 0 for feature in feature_names}

    row["SeniorCitizen"] = senior
    row["tenure"] = tenure
    row["MonthlyCharges"] = monthly
    row["TotalCharges"] = total

    input_df = pd.DataFrame([row])

    prediction = model.predict(input_df)[0]

    if prediction == 1:
        st.error("Customer Will Churn")
    else:
        st.success("Customer Will Not Churn")