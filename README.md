# Student Early Warning System

A comprehensive AI-powered Streamlit application for predicting academic risk factors and providing personalized recommendations for student success.

## 🚀 Features

- **Individual Student Assessment**: Analyze individual student risk factors with interactive visualizations
- **Batch Analysis**: Process multiple students from CSV files with comprehensive reporting
- **Analytics Dashboard**: View trends, statistics, and risk factor correlations
- **Student Records Management**: Track and manage assessment history with search/filter capabilities
- **PDF Report Generation**: Generate detailed assessment reports with personalized recommendations
- **Dark/Light Mode**: Toggle between themes for better user experience
- **Secure Login System**: Lecturer authentication with session management

## 📊 Machine Learning Model

The system uses a Random Forest Classifier trained on student academic data to predict risk factors including:
- Attendance patterns
- Study habits
- Academic performance
- Socioeconomic factors
- Personal wellness indicators

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- pip package manager

### Setup
1. Clone the repository:
```bash
git clone https://github.com/Sithum-Perera/student_warning_system.git
cd student_warning_system
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
streamlit run app.py
```

4. Access the application:
- Open your web browser
- Navigate to: http://localhost:8501
- Login credentials:
  - Username: `lecturer`
  - Password: `admin123`

## 📁 File Structure

```
student_warning_system/
├── app.py                    # Main application file
├── feature_columns.py        # ML model feature definitions
├── student_model.pkl         # Trained machine learning model
├── scaler.pkl               # Data preprocessing scaler
├── requirements.txt         # Python dependencies
└── README.md               # Project documentation
```

## 🎯 Usage

### Individual Assessment
1. Navigate to "Individual Assessment" tab
2. Fill in student information (registration number, demographics, academic data)
3. Click "Analyze Student Risk" to get predictions
4. View detailed risk analysis with visualizations
5. Generate PDF reports with personalized recommendations

### Batch Analysis
1. Go to "Batch Analysis" tab
2. Download the sample CSV template
3. Upload your student data CSV file
4. View batch analysis results with filtering options
5. Export results for further analysis

### Analytics Dashboard
1. Access "Analytics Dashboard" to view:
   - Total assessments and at-risk rates
   - Risk trends by department and month
   - Correlation analysis of risk factors

## 📋 CSV Format for Batch Analysis

Required columns:
- `Student_Reg_No`: Unique student identifier
- `Gender`: Male/Female
- `District`: Student's home district
- `Attendance_Percentage`: Class attendance rate (0-100)
- `Study_Hours_per_Week`: Weekly study hours
- `Previous_GPA`: Most recent GPA (0.0-4.0)
- `Internet_Access`: Yes/No
- `Family_Income_LKR`: Monthly family income
- `Part_Time_Job`: Yes/No
- `Sleep_Hours_per_Night`: Average sleep duration

## 🔧 Technical Details

### Dependencies
- streamlit>=1.28.0
- pandas>=1.5.0
- joblib>=1.3.0
- matplotlib>=3.6.0
- plotly>=5.15.0
- reportlab>=4.0.0
- numpy>=1.24.0
- scikit-learn>=1.3.0

### Security Features
- Input validation and sanitization
- Path traversal protection
- Session-based authentication
- Error handling and recovery

## 🎨 Themes

The application supports both light and dark modes:
- **Light Mode**: Default theme with white background
- **Dark Mode**: Dark theme for reduced eye strain
- Toggle between themes using sidebar buttons

## 📈 Model Performance

The machine learning model considers multiple factors:
- **Attendance Percentage**: Class participation rate
- **Study Hours**: Weekly study time commitment
- **Previous GPA**: Academic performance history
- **Sleep Hours**: Rest and wellness indicators
- **Family Income**: Socioeconomic factors
- **Internet Access**: Technology availability
- **Part-time Job**: Work-study balance

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👨‍💻 Author

**Sithum Perera**
- GitHub: [@Sithum-Perera](https://github.com/Sithum-Perera)

## 🙏 Acknowledgments

- Built with Streamlit framework
- Machine learning powered by scikit-learn
- Visualizations created with Plotly
- PDF generation using ReportLab

## Real-World Application: Streamlit Dashboard 
- A fully functional web application has been deployed and is accessible at:
- 🔗 https://studentwarningsystem-c6gswvr6ubev4skcxngb2z.streamlit.app/

## 📞 Support

If you encounter any issues or have questions, please open an issue on GitHub.

---

⭐ **Star this repository if you find it helpful!**
