import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# CSS styling
st.markdown("""
    <style>
    .main {
        background-color: #ffffff;
        padding: 2rem;
        border-radius: 12px;
        box-shadow: 0px 2px 10px rgba(0,0,0,0.1);
    }
    .title {
        font-size: 32px;
        font-weight: bold;
        color: #333333;
        margin-bottom: 10px;
    }
    .prediction-box {
        background-color: #e3f2fd;
        padding: 1rem;
        border-left: 5px solid #2196f3;
        font-weight: bold;
        font-size: 18px;
        border-radius: 8px;
        margin-top: 20px;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="main">', unsafe_allow_html=True)
st.markdown('<div class="title">🩺 Prediksi Diabetes</div>', unsafe_allow_html=True)

# Data dummy training (dari dataset diabetes.csv)
@st.cache_data
def load_model():
    url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
    col_names = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin",
                 "BMI", "DiabetesPedigreeFunction", "Age", "Outcome"]
    df = pd.read_csv(url, names=col_names)

    X = df.drop("Outcome", axis=1)
    y = df["Outcome"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_scaled, y)
    return model, scaler

model, scaler = load_model()

# Form input
pregnancies = st.number_input("Jumlah Kehamilan", min_value=0, max_value=20, value=1)
glucose = st.number_input("Glukosa", min_value=0, max_value=300, value=120)
blood_pressure = st.number_input("Tekanan Darah", min_value=0, max_value=200, value=70)
skin_thickness = st.number_input("Ketebalan Kulit", min_value=0, max_value=100, value=20)
insulin = st.number_input("Insulin", min_value=0, max_value=900, value=80)
bmi = st.number_input("BMI", min_value=0.0, max_value=70.0, value=25.0)
dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=2.5, value=0.5)
age = st.number_input("Usia", min_value=1, max_value=120, value=30)

if st.button("Prediksi"):
    input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness,
                            insulin, bmi, dpf, age]])
    scaled_data = scaler.transform(input_data)
    pred = model.predict(scaled_data)[0]
    result = "🟢 Tidak Diabetes" if pred == 0 else "🔴 Diabetes"
    st.markdown(f'<div class="prediction-box">Hasil Prediksi: {result}</div>', unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)
