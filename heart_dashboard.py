import streamlit as st
import numpy as np
import pickle  # or joblib
from sklearn.linear_model import LinearRegression

# === Load your trained model ===
# If saved earlier with pickle
# model = pickle.load(open('heart_risk_model.pkl', 'rb'))

# Or define manually if model was trained inline
# (Use the same coefficients you trained)
coefficients = [0.1585, 0.0340, 0.0958, 0.0242]  # Example values
intercept = 0.0615  # Replace with your model's intercept


def predict_risk(inputs):
    return float(np.dot(inputs, coefficients) + intercept)

# === Streamlit UI ===
st.title("🫀 Heart Attack Risk Calculator")
st.markdown("Estimate the probability of a heart attack based on your health profile.")

# Input sliders
health = st.slider("Health Condition Score (0–4)", 0, 4, 2)
metabolic = st.slider("Metabolic Risk Score (0–3)", 0.0, 3.0, 1.5, step=0.1)
clinical = st.slider("Clinical Heart Risk Flag (0–3)", 0, 3, 1)
lifestyle = st.slider("Lifestyle & Behavioral Risk Score (0–8)", 0, 8, 3)

# Prediction
if st.button("Predict Risk"):
    features = np.array([health, metabolic, clinical, lifestyle])
    probability = predict_risk(features)
    probability = np.clip(probability, 0, 1)  # keep between 0 and 1

    st.subheader("🔍 Predicted Heart Attack Probability:")
    st.metric(label="Risk Level", value=f"{probability*100:.2f} %")

    if probability >= 0.7:
        st.warning("⚠️ High Risk! Consider clinical evaluation.")
    elif probability >= 0.4:
        st.info("🧐 Moderate Risk. Consider lifestyle improvements.")
    else:
        st.success("✅ Low Risk. Keep maintaining good health!")


