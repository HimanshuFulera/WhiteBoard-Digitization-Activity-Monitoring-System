@echo off
setlocal

:: Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Python is not installed or not in PATH. Please install Python to continue.
    pause
    exit /b 1
)

:: Define the virtual environment directory
set VENV_DIR=.venv

:: Check if the virtual environment exists
if not exist "%VENV_DIR%\Scripts\activate.bat" (
    echo Creating virtual environment...
    python -m venv %VENV_DIR%
    if %errorlevel% neq 0 (
        echo Failed to create virtual environment.
        pause
        exit /b 1
    )
)

:: Activate the virtual environment
call "%VENV_DIR%\Scripts\activate.bat"

:: Upgrade pip quietly
echo Upgrading pip...
python -m pip install --upgrade pip >nul 2>&1

:: Install requirements
echo Installing dependencies from requirements.txt...
pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo Failed to install dependencies.
    pause
    exit /b 1
)

:: Run the project
echo.
echo ============================================================
echo   STARTING WHITEBOARD NOTES EXTRACTION SYSTEM
echo ============================================================
python main.py

pause
