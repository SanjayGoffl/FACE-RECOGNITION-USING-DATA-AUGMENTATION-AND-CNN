@echo off
echo Installing Waitress (Production Server)...
pip install waitress

echo.
echo Starting Smart Attendance Server...
echo -----------------------------------
python server.py
pause
