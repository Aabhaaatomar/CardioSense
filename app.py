import streamlit as st
import joblib
import numpy as np
import matplotlib.pyplot as plt

# Load trained model
model = joblib.load("model.pkl")

st.title("🫀 CardioSense: Cardiovascular Disease Risk Prediction")

# Input form
age = st.number_input("Age", 18, 100, 30)
gender = st.selectbox("Gender", ["Male", "Female"])
height = st.number_input("Height (cm)", 100, 250, 170)
weight = st.number_input("Weight (kg)", 30, 200, 70)
ap_hi = st.number_input("Systolic BP (ap_hi)", 80, 200, 120)
ap_lo = st.number_input("Diastolic BP (ap_lo)", 50, 130, 80)
cholesterol = st.selectbox("Cholesterol Level", [1, 2, 3])
glucose = st.selectbox("Glucose Level", [1, 2, 3])
smoke = st.selectbox("Smokes?", ["No", "Yes"])
alco = st.selectbox("Alcohol Intake?", ["No", "Yes"])
active = st.selectbox("Physically Active?", ["Yes", "No"])

# Encode categorical inputs
gender = 1 if gender == "Male" else 0
smoke = 1 if smoke == "Yes" else 0
alco = 1 if alco == "Yes" else 0
active = 1 if active == "Yes" else 0

# Prepare input for model
input_data = np.array([[age, gender, height, weight, ap_hi, ap_lo,
                        cholesterol, glucose, smoke, alco, active]])

# Prediction
if st.button("Predict Risk"):
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    if prediction == 1:
        st.error(f"⚠ High Risk of Cardiovascular Disease (Prob: {probability:.2f})")
    else:
        st.success(f"✅ Low Risk of Cardiovascular Disease (Prob: {probability:.2f})")

    # Progress bar
    st.write("### Risk Probability")
    st.progress(int(probability * 100))

# Global Feature Importance (Bar Chart)
if st.checkbox("Show Important Features (Global)"):
    importances = model.feature_importances_
    features = ["age", "gender", "height", "weight", "ap_hi", "ap_lo",
                "cholesterol", "glucose", "smoke", "alco", "active"]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.barh(features, importances, color="skyblue")
    ax.set_xlabel("Importance")
    ax.set_title("Top Features Affecting Prediction")
    st.pyplot(fig)