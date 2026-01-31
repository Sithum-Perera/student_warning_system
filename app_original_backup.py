import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet
import os
from feature_columns import FEATURE_COLUMNS

# ================== LOGIN ==================
st.sidebar.title("🔐 Lecturer Login")

USERNAME = "lecturer"
PASSWORD = "admin123"

username = st.sidebar.text_input("Username")
password = st.sidebar.text_input("Password", type="password")

if username != USERNAME or password != PASSWORD:
    st.sidebar.warning("Please login to continue")
    st.stop()
else:
    st.sidebar.success("Login successful")

# ================== LOAD MODEL ==================
model = joblib.load("student_model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("🧠 Student Early Warning System")
st.write("Academic Risk Prediction & Monitoring Platform")

# ================== INPUTS ==================
department = st.selectbox(
    "Department",
    ["Computer Science", "Engineering", "Business", "Arts"]
)

month = st.selectbox(
    "Month",
    ["January", "February", "March", "April", "May"]
)

gender = st.selectbox("Gender", ["Male", "Female"])
district = st.text_input("District", "Kandy")

attendance = st.slider("Attendance Percentage", 0, 100, 70)
study_hours = st.slider("Study Hours per Week", 0, 60, 15)
gpa = st.slider("Previous GPA", 0.0, 4.0, 3.0)

internet = st.selectbox("Internet Access", ["Yes", "No"])
income = st.number_input("Family Income (LKR)", value=30000)
job = st.selectbox("Part-Time Job", ["Yes", "No"])
sleep = st.slider("Sleep Hours per Night", 0, 12, 7)

# ================== PDF FUNCTION ==================
def generate_pdf(student_data, result, probability):
    file_name = "student_risk_report.pdf"
    doc = SimpleDocTemplate(file_name)
    styles = getSampleStyleSheet()
    content = []

    content.append(Paragraph("Student Academic Risk Report", styles["Title"]))

    for k, v in student_data.items():
        content.append(Paragraph(f"<b>{k}:</b> {v}", styles["Normal"]))

    content.append(Paragraph(f"<b>Prediction:</b> {result}", styles["Normal"]))
    content.append(Paragraph(
        f"<b>At-Risk Probability:</b> {probability*100:.2f}%",
        styles["Normal"]
    ))

    doc.build(content)
    return file_name

# ================== PREDICTION ==================
if st.button("🔍 Predict Student Status"):

    student_data = {
        "Gender": gender,
        "District": district,
        "Attendance_Percentage": attendance,
        "Study_Hours_per_Week": study_hours,
        "Previous_GPA": gpa,
        "Internet_Access": internet,
        "Family_Income_LKR": income,
        "Part_Time_Job": job,
        "Sleep_Hours_per_Night": sleep
    }

    df = pd.DataFrame([student_data])
    df = pd.get_dummies(df)
    df = df.reindex(columns=FEATURE_COLUMNS, fill_value=0)
    df_scaled = scaler.transform(df)

    prediction = model.predict(df_scaled)[0]
    probability = model.predict_proba(df_scaled)[0][0]

    result = "Safe" if prediction == 1 else "At-Risk"

    # ================== OUTPUT ==================
    st.subheader("📊 Prediction Result")
    st.write(f"**At-Risk Probability:** {probability*100:.2f}%")

    if prediction == 1:
        st.success("🟢 Student is SAFE")
    else:
        st.error("🔴 Student is AT RISK")

    # ================== DASHBOARD ==================
    st.subheader("📈 Risk Visualization")
    fig, ax = plt.subplots()
    ax.bar(["At-Risk", "Safe"], [probability*100, (1-probability)*100])
    ax.set_ylabel("Probability (%)")
    st.pyplot(fig)

    # ================== MONTHLY LOG ==================
    log_data = student_data.copy()
    log_data["Department"] = department
    log_data["Month"] = month
    log_data["Risk"] = result

    log_df = pd.DataFrame([log_data])
    log_df.to_csv("risk_log.csv", mode="a", header=not os.path.exists("risk_log.csv"), index=False)

    # ================== PDF ==================
    if st.button("📄 Generate PDF Report"):
        pdf_file = generate_pdf(log_data, result, probability)
        with open(pdf_file, "rb") as f:
            st.download_button(
                "⬇️ Download PDF",
                f,
                file_name=pdf_file
            )

# ================== CSV UPLOAD ==================
st.subheader("📁 Batch Prediction (CSV Upload)")

uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

if uploaded_file:
    batch_df = pd.read_csv(uploaded_file)
    batch_encoded = pd.get_dummies(batch_df)
    batch_encoded = batch_encoded.reindex(columns=FEATURE_COLUMNS, fill_value=0)
    batch_scaled = scaler.transform(batch_encoded)

    preds = model.predict(batch_scaled)
    probs = model.predict_proba(batch_scaled)[:, 0]

    batch_df["Risk_Status"] = ["At-Risk" if p == 0 else "Safe" for p in preds]
    batch_df["Risk_Probability (%)"] = (probs * 100).round(2)

    st.dataframe(batch_df)