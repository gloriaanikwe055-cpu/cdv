"""app.py — Stage 06: clinician-facing GUI (Streamlit).

Loads the FROZEN best model from Stage 04 and returns a CVD-risk prediction plus
a plain-language explanation. Built for non-technical clinicians (proposal Sec 3
objective 4). Run: streamlit run 06_gui/app.py
"""
import sys
from pathlib import Path
import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cvd.config import load_config  # noqa: E402

st.set_page_config(page_title="CVD Risk Detection", page_icon="heart", layout="centered")
cfg = load_config()

st.title("Cardiovascular Disease Risk — Decision Support")
st.caption("Research prototype. Trained on SYNTHETIC data. Not for clinical use.")

model_path = ROOT / cfg["gui"]["model_path"]
if not model_path.exists():
    st.error("No trained model found. Run `python -m cvd.run --stage all` first.")
    st.stop()
pipe = joblib.load(model_path)

st.subheader("Patient inputs")
c1, c2 = st.columns(2)
with c1:
    age = st.slider("Age", 18, 90, 55)
    sex = st.selectbox("Sex", ["female", "male"])
    systolic_bp = st.slider("Systolic BP (mmHg)", 90, 220, 135)
    diastolic_bp = st.slider("Diastolic BP (mmHg)", 55, 130, 85)
    resting_heart_rate = st.slider("Resting HR (bpm)", 45, 130, 74)
    total_cholesterol = st.slider("Total cholesterol (mg/dL)", 120, 400, 210)
    hdl_cholesterol = st.slider("HDL (mg/dL)", 20, 100, 48)
with c2:
    fasting_glucose = st.slider("Fasting glucose (mg/dL)", 65, 260, 100)
    bmi = st.slider("BMI", 16.0, 55.0, 27.5)
    smoker = st.selectbox("Smoker", ["no", "yes"])
    physical_activity = st.selectbox("Activity", ["active", "moderate", "sedentary"])
    sleep_hours = st.slider("Sleep (hrs/night)", 2.0, 11.0, 6.5)
    alcohol_units_week = st.slider("Alcohol (units/wk)", 0, 80, 8)
    st_depression = st.slider("ECG ST-depression (mm)", 0.0, 7.0, 1.0)
    max_heart_rate_achieved = st.slider("Max HR achieved (bpm)", 70, 210, 150)

family_history_cvd = st.selectbox("Family history of CVD", ["no", "yes"])
diabetes = st.selectbox("Diabetes", ["no", "yes"])

row = pd.DataFrame([{
    "age": age, "sex": sex, "systolic_bp": systolic_bp, "diastolic_bp": diastolic_bp,
    "resting_heart_rate": resting_heart_rate, "total_cholesterol": total_cholesterol,
    "hdl_cholesterol": hdl_cholesterol, "fasting_glucose": fasting_glucose, "bmi": bmi,
    "smoker": smoker, "physical_activity": physical_activity, "sleep_hours": sleep_hours,
    "alcohol_units_week": alcohol_units_week, "st_depression": st_depression,
    "max_heart_rate_achieved": max_heart_rate_achieved,
    "family_history_cvd": family_history_cvd, "diabetes": diabetes,
}])

if st.button("Assess risk", type="primary"):
    proba = float(pipe.predict_proba(row)[:, 1][0])
    st.metric("Estimated CVD risk", f"{proba*100:.1f}%")
    if proba >= 0.5:
        st.warning("Model flags ELEVATED cardiovascular risk for this profile.")
    else:
        st.success("Model flags LOWER cardiovascular risk for this profile.")
    st.progress(min(max(proba, 0.0), 1.0))
    st.caption(
        "Explanation: risk rises with age, blood pressure, glucose, BMI, "
        "ST-depression and smoking; it falls with higher HDL, more sleep and "
        "activity. See Stage 05 SHAP plots for the model's global drivers.")
