import streamlit as st
import pandas as pd
import joblib

# ------------------------------------------------------------------
# Page config
# ------------------------------------------------------------------
st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------------------------------------------------------
# Custom CSS
# ------------------------------------------------------------------
st.markdown("""
<style>
    /* Overall page */
    .main {
        background-color: #0e1117;
    }

    /* Header banner */
    .hero {
        background: linear-gradient(135deg, #ff4b4b 0%, #b30000 100%);
        padding: 2.2rem 2rem;
        border-radius: 16px;
        margin-bottom: 1.8rem;
        box-shadow: 0 8px 24px rgba(179, 0, 0, 0.25);
    }
    .hero h1 {
        color: white;
        font-size: 2.3rem;
        margin: 0;
        font-weight: 800;
    }
    .hero p {
        color: rgba(255,255,255,0.9);
        margin-top: 0.4rem;
        font-size: 1.05rem;
    }

    /* Section cards */
    .section-card {
        background: #1a1c24;
        border: 1px solid #2a2d38;
        border-radius: 14px;
        padding: 1.4rem 1.6rem 0.6rem 1.6rem;
        margin-bottom: 1.2rem;
    }
    .section-title {
        font-size: 1.05rem;
        font-weight: 700;
        color: #ff6b6b;
        margin-bottom: 0.8rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    /* Predict button */
    div.stButton > button {
        width: 100%;
        background: linear-gradient(135deg, #ff4b4b 0%, #b30000 100%);
        color: white;
        font-weight: 700;
        font-size: 1.1rem;
        padding: 0.75rem 0;
        border-radius: 12px;
        border: none;
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }
    div.stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 18px rgba(255, 75, 75, 0.4);
        color: white;
    }

    /* Result boxes */
    .result-box {
        padding: 1.6rem;
        border-radius: 14px;
        text-align: center;
        font-size: 1.3rem;
        font-weight: 700;
        margin-top: 1rem;
    }
    .risk-high {
        background: rgba(255, 75, 75, 0.12);
        border: 1.5px solid #ff4b4b;
        color: #ff6b6b;
    }
    .risk-low {
        background: rgba(46, 204, 113, 0.12);
        border: 1.5px solid #2ecc71;
        color: #4ade80;
    }

    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------
# Load model artifacts
# ------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    model = joblib.load("KNN_Heart.pkl")
    scaler = joblib.load("Scaler.pkl")
    expected_columns = joblib.load("columns.pkl")
    return model, scaler, expected_columns

model, scaler, expected_columns = load_artifacts()

# ------------------------------------------------------------------
# Sidebar
# ------------------------------------------------------------------
with st.sidebar:
    st.markdown("### ℹ️ About")
    st.markdown(
        "This tool uses a **K-Nearest Neighbors (KNN)** model trained on "
        "clinical data to estimate a patient's risk of heart disease."
    )
    st.markdown("---")
    st.markdown("### 📋 How to use")
    st.markdown(
        "1. Fill in the patient's details on the right\n"
        "2. Click **Predict**\n"
        "3. Review the estimated risk"
    )
    st.markdown("---")
    st.markdown("### ⚠️ Disclaimer")
    st.caption(
        "This is a machine learning demo and **not** a substitute for "
        "professional medical advice, diagnosis, or treatment."
    )

# ------------------------------------------------------------------
# Hero header
# ------------------------------------------------------------------
st.markdown("""
<div class="hero">
    <h1>🫀 Heart Disease Prediction System</h1>
    <p>Enter the patient's clinical details below to estimate their risk of heart disease.</p>
</div>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------
# Input form
# ------------------------------------------------------------------
with st.form("prediction_form"):

    # --- Demographics ---
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">👤 Demographics</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        age = st.slider("Age", 18, 100, 40)
    with c2:
        sex = st.selectbox("Sex", ["M", "F"])
    st.markdown('</div>', unsafe_allow_html=True)

    # --- Vitals ---
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">💓 Vitals & Labs</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        resting_bp = st.number_input("Resting BP (mm Hg)", 80, 200, 120)
    with c2:
        cholesterol = st.number_input("Cholesterol (mg/dL)", 100, 600, 200)
    with c3:
        fasting_bs = st.selectbox("Fasting Blood Sugar > 120 mg/dL", [0, 1],
                                   format_func=lambda x: "Yes" if x == 1 else "No")
    st.markdown('</div>', unsafe_allow_html=True)

    # --- Cardiac tests ---
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">🩺 Cardiac Test Results</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        chest_pain = st.selectbox("Chest Pain Type", ["ATA", "NAP", "TA", "ASY"])
        resting_ecg = st.selectbox("Resting ECG", ["Normal", "ST", "LVH"])
        st_slope = st.selectbox("ST Slope", ["Up", "Flat", "Down"])
    with c2:
        max_hr = st.slider("Max Heart Rate", 60, 220, 150)
        exercise_angina = st.selectbox("Exercise-Induced Angina", ["Y", "N"])
        oldpeak = st.slider("Oldpeak (ST Depression)", 0.0, 6.0, 1.0, step=0.1)
    st.markdown('</div>', unsafe_allow_html=True)

    submitted = st.form_submit_button("🔍 Predict")

# ------------------------------------------------------------------
# Prediction
# ------------------------------------------------------------------
if submitted:
    raw_input = {
        'Age': age,
        'RestingBP': resting_bp,
        'Cholesterol': cholesterol,
        'FastingBS': fasting_bs,
        'MaxHR': max_hr,
        'Oldpeak': oldpeak,
        'Sex_' + sex: 1,
        'ChestPainType_' + chest_pain: 1,
        'RestingECG_' + resting_ecg: 1,
        'ExerciseAngina_' + exercise_angina: 1,
        'ST_Slope_' + st_slope: 1
    }

    input_df = pd.DataFrame([raw_input])

    for col in expected_columns:
        if col not in input_df.columns:
            input_df[col] = 0

    input_df = input_df[expected_columns]

    scaled_input = scaler.transform(input_df)
    prediction = model.predict(scaled_input)[0]

    # Try to get probability if the model supports it
    proba = None
    if hasattr(model, "predict_proba"):
        try:
            proba = model.predict_proba(scaled_input)[0][1]
        except Exception:
            proba = None

    st.markdown("### 📊 Result")

    if prediction == 1:
        st.markdown(
            '<div class="result-box risk-high">⚠️ High Risk of Heart Disease</div>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            '<div class="result-box risk-low">✅ Low Risk of Heart Disease</div>',
            unsafe_allow_html=True
        )

    if proba is not None:
        st.markdown("#### Estimated Risk Probability")
        st.progress(float(proba))
        st.caption(f"Model-estimated probability of heart disease: **{proba*100:.1f}%**")

    with st.expander("See patient summary"):
        summary_df = pd.DataFrame({
            "Field": ["Age", "Sex", "Chest Pain Type", "Resting BP", "Cholesterol",
                      "Fasting BS > 120", "Resting ECG", "Max HR",
                      "Exercise Angina", "Oldpeak", "ST Slope"],
            "Value": [age, sex, chest_pain, resting_bp, cholesterol,
                      "Yes" if fasting_bs == 1 else "No", resting_ecg, max_hr,
                      exercise_angina, oldpeak, st_slope]
        })
        st.table(summary_df)

    st.caption("⚠️ This prediction is generated by a machine learning model and should not replace professional medical evaluation.")