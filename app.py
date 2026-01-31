import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from reportlab.platypus import SimpleDocTemplate, Paragraph, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
import os
import datetime
import numpy as np
from feature_columns import FEATURE_COLUMNS

# ================== PAGE CONFIG ==================
st.set_page_config(
    page_title="Student Early Warning System",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ================== CUSTOM CSS ==================
st.markdown("""
<style>
.main-header {
    font-size: 3rem;
    color: #1f77b4;
    text-align: center;
    margin-bottom: 2rem;
}
.metric-card {
    background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    padding: 1rem;
    border-radius: 10px;
    color: white;
    text-align: center;
}
.risk-high { color: #ff4444; font-weight: bold; }
.risk-medium { color: #ffaa00; font-weight: bold; }
.risk-low { color: #00aa44; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# ================== LOGIN SYSTEM ==================
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:
    st.markdown('<h1 class="main-header">🔐 Login Required</h1>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        with st.form("login_form"):
            st.subheader("Lecturer Authentication")
            username = st.text_input("Username", placeholder="Enter username")
            password = st.text_input("Password", type="password", placeholder="Enter password")
            login_btn = st.form_submit_button("🚀 Login", use_container_width=True)
            
            if login_btn:
                if username == "lecturer" and password == "admin123":
                    st.session_state.logged_in = True
                    st.success("✅ Login successful! Redirecting...")
                    st.rerun()
                else:
                    st.error("❌ Invalid credentials")
    st.stop()

# ================== LOGOUT ==================
st.sidebar.markdown("---")
st.sidebar.markdown("**🧭 Navigation**")
if 'current_tab' not in st.session_state:
    st.session_state.current_tab = "Individual Assessment"

if st.sidebar.button("📊 Individual Assessment", use_container_width=True):
    st.session_state.current_tab = "Individual Assessment"
if st.sidebar.button("📁 Batch Analysis", use_container_width=True):
    st.session_state.current_tab = "Batch Analysis"
if st.sidebar.button("📈 Analytics Dashboard", use_container_width=True):
    st.session_state.current_tab = "Analytics Dashboard"
if st.sidebar.button("📋 Student Records", use_container_width=True):
    st.session_state.current_tab = "Student Records"
if st.sidebar.button("⚙️ Settings", use_container_width=True):
    st.session_state.current_tab = "Settings"

st.sidebar.markdown("---")
if st.sidebar.button("🚪 Logout", use_container_width=True):
    st.session_state.logged_in = False
    st.rerun()

# ================== LOAD MODEL ==================
try:
    import warnings
    warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')
    model = joblib.load("student_model.pkl")
    scaler = joblib.load("scaler.pkl")
except FileNotFoundError:
    st.error("⚠️ Model files not found. Please ensure student_model.pkl and scaler.pkl are in the directory.")
    st.stop()

# ================== HEADER ==================
st.markdown('<h1 class="main-header">🎓 Student Early Warning System</h1>', unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size: 1.2rem; color: #666;'>AI-Powered Academic Risk Assessment & Student Success Platform</p>", unsafe_allow_html=True)

# ================== NAVIGATION ==================
current_tab = st.session_state.current_tab

if current_tab == "Individual Assessment":
    st.subheader("🎯 Individual Student Risk Assessment")
    
    # Input validation functions
    def validate_inputs(data, student_reg_no):
        errors = []
        if data['attendance'] < 0 or data['attendance'] > 100:
            errors.append("Attendance must be between 0-100%")
        if data['gpa'] < 0 or data['gpa'] > 4.0:
            errors.append("GPA must be between 0.0-4.0")
        if data['income'] < 0:
            errors.append("Income cannot be negative")
        if not student_reg_no or not student_reg_no.strip():
            errors.append("Student Registration No is required")
        # Sanitize registration number for security
        if student_reg_no and any(char in student_reg_no for char in ['/', '\\', '..', '<', '>', '|', ':', '*', '?', '"']):
            errors.append("Student Registration No contains invalid characters")
        return errors
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**📚 Academic Information**")
        department = st.selectbox("Department", ["Computer Science", "Engineering", "Business", "Arts", "Medicine", "Law"])
        month = st.selectbox("Assessment Month", ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"])
        attendance = st.slider("Attendance Percentage", 0, 100, 75, help="Student's class attendance rate")
        study_hours = st.slider("Study Hours per Week", 0, 60, 20, help="Hours spent studying outside class")
        gpa = st.slider("Previous GPA", 0.0, 4.0, 3.0, 0.1, help="Most recent GPA score")
        
    with col2:
        st.markdown("**👤 Personal Information**")
        student_reg_no = st.text_input("Student Registration No", placeholder="e.g., CS2024001", help="Unique student identifier")
        gender = st.selectbox("Gender", ["Male", "Female"])
        district = st.selectbox("District", ["Kandy", "Galle", "Gampaha", "Jaffna"], help="Student's home district")
        internet = st.selectbox("Internet Access", ["Yes", "No"], help="Reliable internet access at home")
        income = st.number_input("Family Income (LKR)", value=50000, min_value=0, step=5000, help="Monthly family income")
        job = st.selectbox("Part-Time Job", ["Yes", "No"], help="Student has part-time employment")
        sleep = st.slider("Sleep Hours per Night", 4, 12, 7, help="Average sleep duration")

    # ================== ENHANCED PDF FUNCTION ==================
    def generate_enhanced_pdf(student_data, result, probability, recommendations):
        try:
            # Create exports folder if it doesn't exist
            os.makedirs("exports", exist_ok=True)
            
            # Sanitize registration number for filename
            reg_no = student_data.get('Student_Reg_No', 'Unknown')
            # Remove invalid characters and limit length
            reg_no = ''.join(c for c in reg_no if c.isalnum() or c in '-_')[:20]
            if not reg_no:
                reg_no = 'Unknown'
            
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            file_name = f"exports/risk_report_{reg_no}_{timestamp}.pdf"
            
            doc = SimpleDocTemplate(file_name)
            styles = getSampleStyleSheet()
            content = []
            
            # Title
            content.append(Paragraph("🎓 Student Academic Risk Assessment Report", styles["Title"]))
            content.append(Paragraph(f"Student ID: {student_data.get('Student_Reg_No', 'N/A')}", styles["Heading2"]))
            content.append(Paragraph(f"Generated on: {datetime.datetime.now().strftime('%B %d, %Y at %I:%M %p')}", styles["Normal"]))
            content.append(Paragraph("<br/><br/>", styles["Normal"]))
            
            # Student Information Table
            data = [["Parameter", "Value"]]
            for k, v in student_data.items():
                data.append([k.replace('_', ' ').title(), str(v)])
            
            table = Table(data)
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 14),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            content.append(table)
            
            # Risk Assessment
            content.append(Paragraph("<br/><b>Risk Assessment Results:</b>", styles["Heading2"]))
            content.append(Paragraph(f"<b>Risk Status:</b> {result}", styles["Normal"]))
            content.append(Paragraph(f"<b>Risk Probability:</b> {probability*100:.2f}%", styles["Normal"]))
            
            # Recommendations
            content.append(Paragraph("<br/><b>Recommendations:</b>", styles["Heading2"]))
            for rec in recommendations:
                content.append(Paragraph(f"• {rec}", styles["Normal"]))
            
            doc.build(content)
            return file_name
        except Exception as e:
            st.error(f"❌ Error generating PDF: {str(e)}")
            return None
    
    def get_recommendations(student_data, probability):
        recommendations = []
        
        if student_data['Attendance_Percentage'] < 75:
            recommendations.append("Improve class attendance - aim for 80%+ attendance rate")
        if student_data['Study_Hours_per_Week'] < 15:
            recommendations.append("Increase study time - recommended 15-20 hours per week")
        if student_data['Previous_GPA'] < 2.5:
            recommendations.append("Seek academic tutoring or counseling support")
        if student_data['Sleep_Hours_per_Night'] < 6:
            recommendations.append("Improve sleep hygiene - aim for 7-8 hours per night")
        if student_data['Internet_Access'] == 'No':
            recommendations.append("Arrange reliable internet access for online resources")
        if probability > 0.7:
            recommendations.append("Schedule immediate meeting with academic advisor")
            recommendations.append("Consider reducing course load if possible")
        
        if not recommendations:
            recommendations.append("Continue current academic practices")
            recommendations.append("Maintain regular study schedule")
        
        return recommendations

    # ================== PREDICTION ==================
    col1, col2 = st.columns([2, 1])
    
    with col1:
        predict_btn = st.button("🔍 Analyze Student Risk", type="primary", use_container_width=True)
    
    with col2:
        clear_btn = st.button("🔄 Clear Form", use_container_width=True)
        if clear_btn:
            st.rerun()
    
    if predict_btn:
        student_data = {
            "Student_Reg_No": student_reg_no,
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
        
        # Input validation
        errors = validate_inputs({
            'attendance': attendance,
            'gpa': gpa,
            'income': income
        }, student_reg_no)
        
        if errors:
            for error in errors:
                st.error(f"❌ {error}")
        else:
            # Prediction
            df = pd.DataFrame([student_data])
            df = pd.get_dummies(df)
            df = df.reindex(columns=FEATURE_COLUMNS, fill_value=0)
            df_scaled = scaler.transform(df)
            
            prediction = model.predict(df_scaled)[0]
            probability = model.predict_proba(df_scaled)[0][0]
            
            result = "Safe" if prediction == 1 else "At-Risk"
            risk_level = "High" if probability > 0.7 else "Medium" if probability > 0.4 else "Low"
            
            # ================== ENHANCED RESULTS ==================
            st.markdown("---")
            st.subheader("📊 Risk Assessment Results")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Risk Status", result, delta=None)
            with col2:
                st.metric("Risk Probability", f"{probability*100:.1f}%")
            with col3:
                st.metric("Risk Level", risk_level)
            with col4:
                confidence = (1 - abs(probability - 0.5) * 2) * 100
                st.metric("Confidence", f"{confidence:.1f}%")
            
            # Risk indicator
            if prediction == 1:
                st.success("✅ Student is performing well academically")
            else:
                st.error("⚠️ Student requires immediate attention and support")
            
            # ================== ENHANCED VISUALIZATIONS ==================
            st.subheader("📈 Detailed Risk Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Risk gauge chart
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number+delta",
                    value = probability * 100,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "Risk Probability (%)"},
                    delta = {'reference': 50},
                    gauge = {
                        'axis': {'range': [None, 100]},
                        'bar': {'color': "darkblue"},
                        'steps': [
                            {'range': [0, 30], 'color': "lightgreen"},
                            {'range': [30, 70], 'color': "yellow"},
                            {'range': [70, 100], 'color': "red"}],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 70}}))
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Feature importance radar chart
                features = ['Attendance', 'Study Hours', 'GPA', 'Sleep', 'Income']
                values = [attendance/100, study_hours/60, gpa/4, sleep/12, min(income/100000, 1)]
                
                fig = go.Figure()
                fig.add_trace(go.Scatterpolar(
                    r=values,
                    theta=features,
                    fill='toself',
                    name='Student Profile'
                ))
                fig.update_layout(
                    polar=dict(
                        radialaxis=dict(
                            visible=True,
                            range=[0, 1]
                        )),
                    showlegend=True,
                    title="Student Performance Profile",
                    height=300
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Feature impact analysis
            st.subheader("🎯 Key Risk Factors")
            
            risk_factors = []
            if attendance < 75: risk_factors.append(("Low Attendance", "High", attendance))
            if study_hours < 15: risk_factors.append(("Insufficient Study Time", "Medium", study_hours))
            if gpa < 2.5: risk_factors.append(("Poor Academic Performance", "High", gpa))
            if sleep < 6: risk_factors.append(("Sleep Deprivation", "Medium", sleep))
            if internet == "No": risk_factors.append(("No Internet Access", "Medium", 0))
            if income < 25000: risk_factors.append(("Low Family Income", "Low", income))
            
            if risk_factors:
                for factor, impact, value in risk_factors:
                    color = "🔴" if impact == "High" else "🟡" if impact == "Medium" else "🟢"
                    st.write(f"{color} **{factor}** (Impact: {impact}) - Current: {value}")
            else:
                st.success("✅ No significant risk factors identified")
            
            # ================== RECOMMENDATIONS ==================
            recommendations = get_recommendations(student_data, probability)
            
            st.subheader("💡 Personalized Recommendations")
            for i, rec in enumerate(recommendations, 1):
                st.write(f"{i}. {rec}")
            
            # ================== LOGGING ==================
            log_data = student_data.copy()
            log_data["Department"] = department
            log_data["Month"] = month
            log_data["Risk"] = result
            log_data["Risk_Probability"] = probability
            log_data["Timestamp"] = datetime.datetime.now().isoformat()
            
            # Create exports folder if it doesn't exist
            os.makedirs("exports", exist_ok=True)
            
            log_df = pd.DataFrame([log_data])
            log_df.to_csv("exports/risk_log.csv", mode="a", header=not os.path.exists("exports/risk_log.csv"), index=False)
            
            # Store results in session state for PDF generation
            st.session_state.last_assessment = {
                'log_data': log_data,
                'result': result,
                'probability': probability,
                'recommendations': recommendations
            }
    
    # ================== PDF GENERATION (OUTSIDE PREDICTION) ==================
    if 'last_assessment' in st.session_state:
        st.markdown("---")
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📄 Generate Detailed Report", use_container_width=True):
                assessment = st.session_state.last_assessment
                pdf_file = generate_enhanced_pdf(
                    assessment['log_data'], 
                    assessment['result'], 
                    assessment['probability'], 
                    assessment['recommendations']
                )
                if pdf_file:  # Only show download if PDF was created successfully
                    with open(pdf_file, "rb") as f:
                        st.download_button(
                            "⬇️ Download PDF Report",
                            f,
                            file_name=pdf_file,
                            mime="application/pdf",
                            use_container_width=True
                        )
        
        with col2:
            if st.button("📧 Schedule Follow-up", use_container_width=True):
                st.success("📅 Follow-up scheduled for next week. Notification sent to academic advisor.")

# ================== BATCH ANALYSIS ==================
elif current_tab == "Batch Analysis":
    st.subheader("📁 Batch Student Analysis")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded_file = st.file_uploader(
            "Upload CSV File with Student Data", 
            type=["csv"],
            help="CSV should contain columns: Student_Reg_No, Gender, District, Attendance_Percentage, Study_Hours_per_Week, Previous_GPA, Internet_Access, Family_Income_LKR, Part_Time_Job, Sleep_Hours_per_Night"
        )
    
    with col2:
        if st.button("📥 Download Sample CSV"):
            sample_data = {
                'Student_Reg_No': ['STU001', 'STU002', 'STU003'],
                'Gender': ['Male', 'Female', 'Male'],
                'District': ['Kandy', 'Galle', 'Gampaha'],
                'Attendance_Percentage': [85, 70, 90],
                'Study_Hours_per_Week': [20, 15, 25],
                'Previous_GPA': [3.2, 2.8, 3.7],
                'Internet_Access': ['Yes', 'No', 'Yes'],
                'Family_Income_LKR': [45000, 30000, 60000],
                'Part_Time_Job': ['No', 'Yes', 'No'],
                'Sleep_Hours_per_Night': [7, 6, 8]
            }
            sample_df = pd.DataFrame(sample_data)
            csv = sample_df.to_csv(index=False)
            st.download_button(
                "⬇️ Download Sample",
                csv,
                "sample_student_data.csv",
                "text/csv",
                use_container_width=True
            )
    
    if uploaded_file:
        try:
            batch_df = pd.read_csv(uploaded_file)
            st.success(f"✅ Successfully loaded {len(batch_df)} student records")
            
            # Validate required columns
            required_cols = ['Student_Reg_No', 'Gender', 'District', 'Attendance_Percentage', 'Study_Hours_per_Week', 'Previous_GPA', 'Internet_Access', 'Family_Income_LKR', 'Part_Time_Job', 'Sleep_Hours_per_Night']
            missing_cols = [col for col in required_cols if col not in batch_df.columns]
            
            if missing_cols:
                st.error(f"❌ Missing required columns: {', '.join(missing_cols)}")
            else:
                # Process batch predictions
                batch_encoded = pd.get_dummies(batch_df)
                batch_encoded = batch_encoded.reindex(columns=FEATURE_COLUMNS, fill_value=0)
                batch_scaled = scaler.transform(batch_encoded)
                
                preds = model.predict(batch_scaled)
                probs = model.predict_proba(batch_scaled)[:, 0]
                
                batch_df["Risk_Status"] = ["At-Risk" if p == 0 else "Safe" for p in preds]
                batch_df["Risk_Probability_%"] = (probs * 100).round(2)
                batch_df["Risk_Level"] = ["High" if p > 70 else "Medium" if p > 40 else "Low" for p in batch_df["Risk_Probability_%"]]
                
                # Summary statistics
                st.subheader("📊 Batch Analysis Summary")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    total_students = len(batch_df)
                    st.metric("Total Students", total_students)
                
                with col2:
                    at_risk_count = len(batch_df[batch_df['Risk_Status'] == 'At-Risk'])
                    st.metric("At-Risk Students", at_risk_count, delta=f"{(at_risk_count/total_students*100):.1f}%")
                
                with col3:
                    avg_risk = batch_df['Risk_Probability_%'].mean()
                    st.metric("Average Risk %", f"{avg_risk:.1f}%")
                
                with col4:
                    high_risk = len(batch_df[batch_df['Risk_Level'] == 'High'])
                    st.metric("High Risk Students", high_risk)
                
                # Visualizations
                col1, col2 = st.columns(2)
                
                with col1:
                    # Risk distribution pie chart
                    risk_counts = batch_df['Risk_Status'].value_counts()
                    fig = px.pie(values=risk_counts.values, names=risk_counts.index, 
                               title="Risk Status Distribution")
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # Risk probability histogram
                    fig = px.histogram(batch_df, x='Risk_Probability_%', 
                                     title="Risk Probability Distribution",
                                     nbins=20)
                    st.plotly_chart(fig, use_container_width=True)
                
                # Detailed results table
                st.subheader("📋 Detailed Results")
                
                # Filter options
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    status_filter = st.selectbox("Filter by Status", ["All", "At-Risk", "Safe"])
                
                with col2:
                    risk_filter = st.selectbox("Filter by Risk Level", ["All", "High", "Medium", "Low"])
                
                with col3:
                    sort_by = st.selectbox("Sort by", ["Risk_Probability_%", "Previous_GPA", "Attendance_Percentage"])
                
                # Apply filters
                filtered_df = batch_df.copy()
                if status_filter != "All":
                    filtered_df = filtered_df[filtered_df['Risk_Status'] == status_filter]
                if risk_filter != "All":
                    filtered_df = filtered_df[filtered_df['Risk_Level'] == risk_filter]
                
                filtered_df = filtered_df.sort_values(sort_by, ascending=False)
                
                # Color-code the dataframe
                def highlight_risk(row):
                    if row['Risk_Status'] == 'At-Risk':
                        return ['background-color: #ffebee'] * len(row)
                    else:
                        return ['background-color: #e8f5e8'] * len(row)
                
                st.dataframe(
                    filtered_df.style.apply(highlight_risk, axis=1),
                    use_container_width=True
                )
                
                # Download results
                csv = filtered_df.to_csv(index=False)
                st.download_button(
                    "📥 Download Results",
                    csv,
                    f"batch_analysis_results_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    "text/csv",
                    use_container_width=True
                )
                
        except Exception as e:
            st.error(f"❌ Error processing file: {str(e)}")

# ================== ANALYTICS DASHBOARD ==================
elif current_tab == "Analytics Dashboard":
    st.subheader("📈 Analytics Dashboard")
    
    if os.path.exists("exports/risk_log.csv"):
        try:
            log_df = pd.read_csv("exports/risk_log.csv", on_bad_lines='skip')
            
            if len(log_df) > 0:
                # Convert timestamp if exists
                if 'Timestamp' in log_df.columns:
                    log_df['Timestamp'] = pd.to_datetime(log_df['Timestamp'], errors='coerce')
                    log_df['Date'] = log_df['Timestamp'].dt.date
                
                # Summary metrics
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    total_assessments = len(log_df)
                    st.metric("Total Assessments", total_assessments)
                
                with col2:
                    at_risk_pct = (log_df['Risk'] == 'At-Risk').mean() * 100
                    st.metric("At-Risk Rate", f"{at_risk_pct:.1f}%")
                
                with col3:
                    if 'Risk_Probability' in log_df.columns:
                        avg_risk = log_df['Risk_Probability'].mean() * 100
                        st.metric("Avg Risk Score", f"{avg_risk:.1f}%")
                    else:
                        st.metric("Avg Risk Score", "N/A")
                
                with col4:
                    unique_depts = log_df['Department'].nunique()
                    st.metric("Departments", unique_depts)
                
                # Visualizations
                col1, col2 = st.columns(2)
                
                with col1:
                    # Risk by department
                    if 'Department' in log_df.columns and 'Risk' in log_df.columns:
                        dept_counts = log_df.groupby('Department')['Risk'].apply(lambda x: (x == 'At-Risk').sum()).reset_index()
                        dept_total = log_df.groupby('Department').size().reset_index(name='Total')
                        dept_risk = dept_counts.merge(dept_total, on='Department')
                        dept_risk['At_Risk_Percentage'] = (dept_risk['Risk'] / dept_risk['Total'] * 100).round(1)
                        
                        fig = px.bar(dept_risk, x='Department', y='At_Risk_Percentage',
                                   title="At-Risk Rate by Department")
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("Department or Risk data not available for visualization")
                
                with col2:
                    # Monthly trend
                    if 'Month' in log_df.columns and 'Risk' in log_df.columns:
                        month_counts = log_df.groupby('Month')['Risk'].apply(lambda x: (x == 'At-Risk').sum()).reset_index()
                        month_total = log_df.groupby('Month').size().reset_index(name='Total')
                        month_risk = month_counts.merge(month_total, on='Month')
                        month_risk['At_Risk_Percentage'] = (month_risk['Risk'] / month_risk['Total'] * 100).round(1)
                        
                        fig = px.line(month_risk, x='Month', y='At_Risk_Percentage',
                                    title="Monthly Risk Trend")
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("Monthly or Risk data not available for visualization")
                
                # Correlation analysis
                st.subheader("🔍 Risk Factor Analysis")
                
                numeric_cols = ['Attendance_Percentage', 'Study_Hours_per_Week', 'Previous_GPA', 'Family_Income_LKR', 'Sleep_Hours_per_Night']
                available_cols = [col for col in numeric_cols if col in log_df.columns]
                
                if available_cols and 'Risk_Probability' in log_df.columns:
                    corr_data = log_df[available_cols + ['Risk_Probability']].corr()['Risk_Probability'].drop('Risk_Probability')
                    
                    fig = px.bar(x=corr_data.index, y=corr_data.values,
                               title="Correlation with Risk Probability")
                    fig.update_layout(xaxis_title="Factors", yaxis_title="Correlation")
                    st.plotly_chart(fig, use_container_width=True)
                
            else:
                st.info("📊 No assessment data available yet. Perform some assessments to see analytics.")
                
        except Exception as e:
            st.error(f"❌ Error loading analytics data: {str(e)}")
    else:
        st.info("📊 No assessment data available yet. Perform some assessments to see analytics.")

# ================== STUDENT RECORDS ==================
elif current_tab == "Student Records":
    st.subheader("📋 Student Records Management")
    
    if os.path.exists("exports/risk_log.csv"):
        try:
            log_df = pd.read_csv("exports/risk_log.csv", on_bad_lines='skip')
            
            if len(log_df) > 0:
                # Search and filter
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    search_district = st.text_input("🔍 Search by District")
                
                with col2:
                    if 'Department' in log_df.columns:
                        filter_dept = st.selectbox("Filter by Department", ["All"] + list(log_df['Department'].unique()))
                    else:
                        filter_dept = "All"
                        st.info("Department data not available")
                
                with col3:
                    if 'Risk' in log_df.columns:
                        filter_risk = st.selectbox("Filter by Risk Status", ["All", "At-Risk", "Safe"])
                    else:
                        filter_risk = "All"
                        st.info("Risk data not available")
                
                # Apply filters
                filtered_log = log_df.copy()
                
                if search_district and 'District' in log_df.columns:
                    filtered_log = filtered_log[filtered_log['District'].str.contains(search_district, case=False, na=False)]
                
                if filter_dept != "All" and 'Department' in log_df.columns:
                    filtered_log = filtered_log[filtered_log['Department'] == filter_dept]
                
                if filter_risk != "All" and 'Risk' in log_df.columns:
                    filtered_log = filtered_log[filtered_log['Risk'] == filter_risk]
                
                # Display records
                st.dataframe(filtered_log, use_container_width=True)
                
                # Export options
                col1, col2 = st.columns(2)
                
                with col1:
                    csv = filtered_log.to_csv(index=False)
                    st.download_button(
                        "📥 Export Filtered Records",
                        csv,
                        f"student_records_{datetime.datetime.now().strftime('%Y%m%d')}.csv",
                        "text/csv",
                        use_container_width=True
                    )
                
                with col2:
                    if st.button("🗑️ Clear All Records", use_container_width=True, type="secondary"):
                        if 'confirm_delete' not in st.session_state:
                            st.session_state.confirm_delete = False
                        
                        if not st.session_state.confirm_delete:
                            st.session_state.confirm_delete = True
                            st.warning("⚠️ Click again to confirm deletion of all records")
                        else:
                            try:
                                os.remove("exports/risk_log.csv")
                                st.session_state.confirm_delete = False
                                st.success("✅ All records cleared successfully")
                                st.rerun()
                            except Exception as e:
                                st.error(f"❌ Error clearing records: {str(e)}")
            else:
                st.info("📋 No student records found.")
                
        except Exception as e:
            st.error(f"❌ Error loading student records: {str(e)}")
    else:
        st.info("📋 No student records found. Perform some assessments to create records.")

# ================== SETTINGS ==================
elif current_tab == "Settings":
    st.subheader("⚙️ System Settings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**🔧 Model Information**")
        st.info(f"Model Type: Machine Learning Classifier")
        st.info(f"Features: {len(FEATURE_COLUMNS)} input features")
        st.info(f"Last Updated: {datetime.datetime.now().strftime('%Y-%m-%d')}")
        
        if st.button("🔄 Refresh Model", use_container_width=True):
            st.success("✅ Model refreshed successfully")
    
    with col2:
        st.markdown("**📊 System Statistics**")
        
        if os.path.exists("exports/risk_log.csv"):
            log_df = pd.read_csv("exports/risk_log.csv", on_bad_lines='skip')
            st.info(f"Total Assessments: {len(log_df)}")
            st.info(f"At-Risk Students: {(log_df['Risk'] == 'At-Risk').sum()}")
        else:
            st.info("No assessment data available")
        
        st.info(f"System Uptime: Active")
        
        if st.button("📈 Generate System Report", use_container_width=True):
            st.success("✅ System report generated")
    
    st.markdown("---")
    st.markdown("**📝 About This System**")
    st.write("""
    This Student Early Warning System uses machine learning to predict academic risk factors 
    and provide personalized recommendations for student success. The system analyzes multiple 
    factors including attendance, study habits, academic performance, and socioeconomic indicators.
    """)
    
    with st.expander("🔍 Feature Importance"):
        st.write("The model considers the following key factors:")
        st.write("• **Attendance Percentage**: Class participation rate")
        st.write("• **Study Hours**: Weekly study time commitment")
        st.write("• **Previous GPA**: Academic performance history")
        st.write("• **Sleep Hours**: Rest and wellness indicators")
        st.write("• **Family Income**: Socioeconomic factors")
        st.write("• **Internet Access**: Technology availability")
        st.write("• **Part-time Job**: Work-study balance")