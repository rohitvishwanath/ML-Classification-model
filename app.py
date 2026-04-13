import streamlit as st
import numpy as np
import joblib

# Load model & scaler
model = joblib.load("fraud_model.pkl")
scaler = joblib.load("scaler.pkl")

# Page config
st.set_page_config(page_title="Fraud Detection", page_icon="💳", layout="wide")

# Title
st.markdown("<h1 style='text-align: center; color: #4CAF50;'>💳 Credit Card Fraud Detection</h1>", unsafe_allow_html=True)
st.markdown("---")

# Sidebar
st.sidebar.header("ℹ️ About")
st.sidebar.info("This app predicts whether a transaction is Fraud or Normal using Machine Learning.")

# Input section
st.subheader("📥 Enter Transaction Details")

col1, col2, col3 = st.columns(3)

# Example inputs (you can expand all features)
with col1:
    v1 = st.number_input("V1", value=0.0)
    v2 = st.number_input("V2", value=0.0)
    v3 = st.number_input("V3", value=0.0)

with col2:
    v4 = st.number_input("V4", value=0.0)
    v5 = st.number_input("V5", value=0.0)
    v6 = st.number_input("V6", value=0.0)

with col3:
    amount = st.number_input("Amount", value=0.0)

# Create input array (IMPORTANT: match training features count)
input_data = np.array([[v1, v2, v3, v4, v5, v6] + [0]*24])  # fill remaining features

# Scale
input_data = scaler.transform(input_data)

# Predict button
if st.button("🔍 Predict Transaction"):
    prediction = model.predict(input_data)

    st.markdown("---")

    if prediction[0] == 1:
        st.error("🚨 Fraudulent Transaction Detected!")
    else:
        st.success("✅ Normal Transaction")

    st.balloons()