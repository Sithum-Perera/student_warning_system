@echo off
echo Starting Student Early Warning System...
cd /d "%~dp0"
python -m streamlit run app.py --server.headless false --server.port 8501
pause