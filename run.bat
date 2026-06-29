@echo off
echo ====================================================
echo Starting QuantPricer Options Pricing Engine Setup...
echo ====================================================

:: Check Python installation
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Error: Python is not installed or not in your PATH.
    echo Please install Python 3.8+ to continue.
    pause
    exit /b
)

:: Create Virtual Environment if it doesn't exist
if not exist venv (
    echo Creating python virtual environment venv...
    python -m venv venv
)

:: Activate Virtual Environment and Install Dependencies
echo Activating virtual environment...
call venv\Scripts\activate

echo Installing required Python packages...
python -m pip install --upgrade pip
pip install -r backend/requirements.txt

echo ====================================================
echo Starting FastAPI Web Server at http://127.0.0.1:8000
echo ====================================================
echo You can view the dashboard by visiting: http://127.0.0.1:8000
echo Press Ctrl+C in this terminal window to stop the server.
echo ====================================================

python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
