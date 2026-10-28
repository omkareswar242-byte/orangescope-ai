@echo off
cd /d "%~dp0"
python app.py
if errorlevel 1 (
  echo.
  echo Please install the packages first:
  echo python -m pip install -r requirements-web.txt
  pause
)
