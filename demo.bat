@echo off
rem NiveshRL one-click demo. Double-click this file (or run: demo.bat --full).
rem First run on a new machine: creates .venv, installs packages, downloads data,
rem trains the fast models, then opens the dashboard. Later runs skip what's done.
setlocal
cd /d "%~dp0"
title NiveshRL demo

if not exist ".venv\Scripts\python.exe" (
    echo === Creating virtual environment ===
    python -m venv .venv || goto :fail
    ".venv\Scripts\python.exe" -m pip install --upgrade pip || goto :fail
    echo === Installing PyTorch ^(CPU^) ===
    ".venv\Scripts\python.exe" -m pip install torch --index-url https://download.pytorch.org/whl/cpu || goto :fail
)
".venv\Scripts\python.exe" -c "import niveshrl, streamlit, arch, sklearn" 2>nul
if errorlevel 1 (
    echo === Installing project packages ===
    ".venv\Scripts\python.exe" -m pip install -r requirements.txt || goto :fail
    ".venv\Scripts\python.exe" -m pip install -e . || goto :fail
)

".venv\Scripts\python.exe" scripts\demo.py %*
if errorlevel 1 goto :fail
exit /b 0

:fail
echo.
echo Setup failed - see the messages above. Is Python 3.11+ installed and on PATH?
pause
exit /b 1
