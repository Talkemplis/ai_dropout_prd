import streamlit as st
import numpy as np
import joblib
from fpdf import FPDF
import plotly.graph_objects as go
import base64
import tempfile

st.set_page_config(
    page_title="AI Dropout Predictor",
    layout="wide",
    page_icon="🤖"
)

# ---------------- CSS Styling ----------------
st.markdown("""
    <style>
        body, .stApp {
            background-color: #f5f7fa;
            font-family: 'Segoe UI', sans-serif;
            color: #212529;
        }
        h1 {
            font-weight: 900;
            font-size: 3rem;
            text-align: center;
            color: #2c3e50;
            margin-bottom: 0;
        }
        p.description {
            text-align: center;
            color: #7f8c8d;
            font-size: 1.1rem;
            margin-bottom: 30px;
        }
    </style>
""", unsafe_allow_html=True)

# ---------------- Title ----------------
st.markdown("""
<h1>AI Dropout Predictor 🤖</h1>
<p class="description">
An AI-powered app to predict student dropout risk. Just fill in the fields below.
</p>
""", unsafe_allow_html=True)

# ---------------- Show Example Help ----------------
with st.expander("📘 Show Me an Example"):
    st.info("""
    **How to Use:**
    - Select the student's **age**, **gender**, and **study details**.
    - Click "Predict" to see the dropout probability.

    **Example:**
    - Age: 22
    - Gender: Female
    - Study Hours: 4.0
    - Video %: 80
    - Assignments: 90
    - Quiz Score: 75
    - Motivation: 6
    """)

# ---------------- Inputs ----------------
col1, col2 = st.columns(2)

with col1:
    age = st.slider("Age", 16, 70, 21)
    hours = st.slider("Study Hours per Week", 0.0, 15.0, 5.0)
    quiz_score = st.slider("Average Quiz Score", 0, 100, 70)
    motivation = st.slider("Motivation Level (1–10)", 1, 10, 5)

with col2:
    gender = st.selectbox("Gender", ["Male", "Female"])
    video_pct = st.slider("Video Watch Percentage", 0, 100, 60)
    assignments = st.slider("Assignment Submission %", 0, 100, 80)

# Convert gender to numeric
gender_val = 0 if gender == "Male" else 1

# Load models (both with joblib)
scaler = joblib.load("scaler.pkl")
model = joblib.load("model.pkl")

# Prepare input
data_input = np.array([[age, gender_val, hours, video_pct, assignments, quiz_score, motivation]])
data_scaled = scaler.transform(data_input)

# Make prediction
pred_prob = model.predict_proba(data_scaled)[0][1]
pred_class = model.predict(data_scaled)[0]

# ---------------- Prediction Output ----------------
st.markdown("---")
st.markdown("## 🎯 Prediction Result")

st.metric("Dropout Risk Probability", f"{pred_prob * 100:.2f}%")

if pred_prob > 0.8:
    st.error("⚠️ High dropout risk. Consider intervention.")
elif pred_prob > 0.5:
    st.warning("📌 Moderate risk. Monitor closely.")
else:
    st.success("✅ Low risk. Student is likely on track.")

# ---------------- Feature Summary Chart ----------------
st.markdown("### 📊 Feature Summary")
features = ["Age", "Study Hours", "Video %", "Assignments", "Quiz Score", "Motivation"]
values = [age, hours, video_pct, assignments, quiz_score, motivation]

fig = go.Figure([go.Bar(x=features, y=values, marker_color='lightskyblue')])
fig.update_layout(title="Input Features Overview", xaxis_title="Feature", yaxis_title="Value")
st.plotly_chart(fig, use_container_width=True)

# ---------------- Generate PDF ----------------
def generate_pdf():
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="AI Dropout Prediction Report", ln=True, align="C")
    pdf.ln(10)

    pdf.set_font("Arial", size=10)
    pdf.cell(200, 10, txt=f"Dropout Probability: {pred_prob * 100:.2f}%", ln=True)
    pdf.cell(200, 10, txt=f"Prediction: {'High Risk' if pred_class else 'Low Risk'}", ln=True)
    pdf.ln(5)
    pdf.cell(200, 10, txt="Input Data:", ln=True)
    for f, v in zip(features, values):
        pdf.cell(200, 10, txt=f"{f}: {v}", ln=True)

    tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    pdf.output(tmp_file.name)
    return tmp_file.name

if st.button("📥 Download Report as PDF"):
    pdf_path = generate_pdf()
    with open(pdf_path, "rb") as f:
        pdf_bytes = f.read()
    b64 = base64.b64encode(pdf_bytes).decode()
    href = f'<a href="data:application/octet-stream;base64,{b64}" download="dropout_report.pdf">Click here to download your report</a>'
    st.markdown(href, unsafe_allow_html=True)
