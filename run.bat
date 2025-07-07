@echo off
title AI VTuber

echo Activating Python virtual environment...
if not exist "venv" (
    echo "venv" not found. Please run the initial setup.
    pause
    exit
)
call "venv\Scripts\activate.bat"

echo.
echo Starting AI VTuber...
python main_app.py

echo Application finished.
pause