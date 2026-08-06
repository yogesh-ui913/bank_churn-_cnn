
import streamlit as st
import pandas as pd
import numpy as np
import pickle
from tensorflow.keras.models import load_model

# Set page config
st.set_page_config(page_title="Customer Churn Predictor", layout="centered")

# Load the saved model and scaler
@st.cache_resource
def load_resources():
    model = load_model('base_smote_ann_model.keras')
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    return model, scaler

model, scaler = load_resources()

st.title("🏦 Bank Customer Churn Prediction")
st.write("Enter customer details below to predict the likelihood of churning.")

# Layout for inputs
col1, col2 = st.columns(2)

with col1:
    credit_score = st.number_input("Credit Score", min_value=300, max_value=850, value=600)
    geography = st.selectbox("Geography", ["France", "Germany", "Spain"])
    gender = st.selectbox("Gender", ["Female", "Male"])
    age = st.number_input("Age", min_value=18, max_value=100, value=40)
    tenure = st.slider("Tenure (Years)", 0, 10, 5)

with col2:
    balance = st.number_input("Balance", min_value=0.0, value=0.0, format="%.2f")
    num_products = st.selectbox("Number of Products", [1, 2, 3, 4])
    has_cr_card = st.selectbox("Has Credit Card?", ["Yes", "No"])
    is_active = st.selectbox("Is Active Member?", ["Yes", "No"])
    salary = st.number_input("Estimated Salary", min_value=0.0, value=50000.0, format="%.2f")

# Preprocess user input
if st.button("Predict Churn"):
    # Encode categorical inputs (Matching training LabelEncoder logic)
    geo_map = {"France": 0, "Germany": 1, "Spain": 2}
    gen_map = {"Female": 0, "Male": 1}
    bin_map = {"Yes": 1, "No": 0}

    # Prepare the raw feature array
    # Order must match training: ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary', 'Geography', 'Gender']
    input_data = pd.DataFrame({
        'CreditScore': [credit_score],
        'Age': [np.log1p(age)], # Log transform as applied in training
        'Tenure': [tenure],
        'Balance': [balance],
        'NumOfProducts': [np.log1p(num_products)], # Log transform as applied in training
        'HasCrCard': [bin_map[has_cr_card]],
        'IsActiveMember': [bin_map[is_active]],
        'EstimatedSalary': [salary],
        'Geography': [geo_map[geography]],
        'Gender': [gen_map[gender]]
    })

    # Scale features
    input_scaled = scaler.transform(input_data)

    # Make prediction
    prediction_prob = model.predict(input_scaled)[0][0]
    prediction = (prediction_prob > 0.5).astype(int)

    # Results display
    st.divider()
    if prediction == 1:
        st.error(f"🚨 Alert: High probability of Churn ({prediction_prob:.2%})")
        st.write("Consider offering retention incentives to this customer.")
    else:
        st.success(f"✅ Low Risk: Customer is likely to Stay ({1 - prediction_prob:.2%})")
