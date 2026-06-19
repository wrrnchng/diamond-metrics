@echo off
echo Activating virtual environment...
call venv\Scripts\activate
if errorlevel 1 (
    echo Failed to activate virtual environment. Make sure venv exists.
    pause
    exit /b 1
)
echo Starting Streamlit app...
streamlit run app.py
pause