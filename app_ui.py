import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load("api/model.pkl")
features = joblib.load("api/features.pkl")

# Page config
st.set_page_config(page_title="Churn Prediction", layout="centered")

# Title
st.title("📉 Customer Churn Prediction System")
st.markdown("Predict customer churn risk and take action.")

# --- INPUT SECTION ---
st.subheader("Customer Details")

tenure = st.slider("Tenure (months)", 0, 72, 12)
monthly_charges = st.number_input("Monthly Charges", 0.0, 150.0, 70.0)
total_charges = st.number_input("Total Charges", 0.0, 10000.0, 1000.0)

contract = st.selectbox(
    "Contract Type",
    ["Month-to-month", "One year", "Two year"]
)

# --- ENCODING ---
contract_one_year = 1 if contract == "One year" else 0
contract_two_year = 1 if contract == "Two year" else 0

# --- PREDICT BUTTON ---
if st.button("🔍 Predict Churn Risk"):

    payload = {
        "tenure": tenure,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
        "Contract_One year": contract_one_year,
        "Contract_Two year": contract_two_year
    }

    df = pd.DataFrame([payload])

    # Ensure all features exist
    for col in features:
        if col not in df:
            df[col] = 0

    df = df[features]

    prob = model.predict_proba(df)[0][1]
    score = round(prob * 100, 1)

    # Risk tier
    if score < 25:
        tier = "Low Risk"
    elif score < 50:
        tier = "Medium Risk"
    elif score < 75:
        tier = "High Risk"
    else:
        tier = "Critical Risk"

    st.success("Prediction Successful!")

    # --- OUTPUT CARDS ---
    col1, col2 = st.columns(2)

    with col1:
        st.metric("Risk Score", f"{score}")

    with col2:
        st.metric("Risk Tier", tier)

    # --- ACTION ---
    st.subheader("Recommended Action")

    if tier == "Low Risk":
        st.info("Maintain engagement. No immediate action needed.")
    elif tier == "Medium Risk":
        st.warning("Offer loyalty benefits or upgrade plan.")
    elif tier == "High Risk":
        st.warning("Proactive outreach with personalized offers.")
    else:
        st.error("Immediate action required! Offer discounts or escalate.")
