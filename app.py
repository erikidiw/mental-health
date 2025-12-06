import streamlit as st
import pandas as pd
import pickle

# Load model
with open("model.pkl", "rb") as file:
    model = pickle.load(file)

st.set_page_config(page_title="Depression Risk Prediction", layout="centered")

st.title("🧠 Student Depression Risk Prediction")
st.write("Isi form berikut untuk memprediksi tingkat risiko depresi mahasiswa.")

# Input features
academic_pressure = st.slider("Academic Pressure (0-5)", 0, 5, 2)
cgpa = st.number_input("CGPA (2.0 - 10.0)", min_value=2.0, max_value=10.0, value=7.5, step=0.1)
study_satisfaction = st.slider("Study Satisfaction (0-5)", 0, 5, 3)

# Prediction button
if st.button("Predict"):
    data = pd.DataFrame({
        "Academic Pressure": [academic_pressure],
        "CGPA": [cgpa],
        "Study Satisfaction": [study_satisfaction]
    })

    result = model.predict(data)[0]

    risk_map = {
        0: ("Low Risk", "🟢"),
        1: ("Medium Risk", "🟡"),
        2: ("High Risk", "🔴")
    }

    label, icon = risk_map[result]

    st.subheader(f"Prediction Result: {icon} {label}")
    st.success("Prediksi berhasil!")
