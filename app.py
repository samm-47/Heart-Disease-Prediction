import streamlit as st
import pandas as pd
import joblib
import numpy as np

# 1. Load the saved model and scaler
model = joblib.load('heart_disease_model.pkl')
scaler = joblib.load('heart_scaler.pkl')

st.title("❤️ Heart Disease Risk Predictor")
st.write("Enter patient clinical data to calculate cardiac risk.")

# 2. Create Input Fields
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", 1, 100, 50)
    sex = st.selectbox("Sex (1=Male, 0=Female)", [1, 0])
    cp = st.selectbox("Chest Pain Type (0-3)", [0, 1, 2, 3])
    trestbps = st.number_input("Resting Blood Pressure", 80, 200, 120)
    chol = st.number_input("Cholesterol", 100, 600, 200)
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl (1=True, 0=False)", [0, 1])

with col2:
    restecg = st.selectbox("Resting ECG Results (0-2)", [0, 1, 2])
    thalach = st.number_input("Max Heart Rate Achieved", 60, 220, 150)
    exang = st.selectbox("Exercise Induced Angina (1=Yes, 0=No)", [0, 1])
    oldpeak = st.number_input("ST Depression (Oldpeak)", 0.0, 6.0, 1.0)
    slope = st.selectbox("Slope of Peak Exercise ST Segment (0-2)", [0, 1, 2])
    ca = st.selectbox("Major Vessels (0-4) Colored by Fluoroscopy", [0, 1, 2, 3, 4])
    thal = st.selectbox("Thalassemia (1=Normal, 2=Fixed, 3=Reversable)", [1, 2, 3])

# 3. Prediction Logic
if st.button("Calculate Risk Score"):
    # Organize input into a DataFrame
    input_data = pd.DataFrame([[age, sex, cp, trestbps, chol, fbs, restecg, 
                                thalach, exang, oldpeak, slope, ca, thal]],
                              columns=['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
                                       'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'])
    
    # Scale and Predict
    input_scaled = scaler.transform(input_data)
    prediction_proba = model.predict_proba(input_scaled)[0][1]
    
    # 4. Display Results
    st.subheader(f"Calculated Risk: {prediction_proba:.2%}")
    if prediction_proba > 0.5:
        st.error("⚠️ Status: HIGH RISK")
    else:
        st.success("✅ Status: LOW RISK")