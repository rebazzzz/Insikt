@echo off
cd /d "%~dp0"
if not exist "venv\Scripts\python.exe" (
  echo Insikt is not installed yet.
  echo Run INSTALL_FOR_TESTER.bat first.
  pause
  exit /b 1
)
"venv\Scripts\python.exe" -m streamlit run insikt_app.py
pause
