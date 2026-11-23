@echo off
setlocal
cd /d "%~dp0"
title US ETF Momentum Lab
set "SRC=%~dp0"
set "VENV=%LOCALAPPDATA%\us_etf_momentum_venv"
set "WORK=%USERPROFILE%\us_etf_momentum_work"
set PYTHONDONTWRITEBYTECODE=1

rem ---- find the app environment (run setup the first time) ----
if not exist "%VENV%\Scripts\python.exe" (
  echo First run on this computer - starting setup...
  call "%SRC%setup_install.bat"
)
if not exist "%VENV%\Scripts\python.exe" (
  echo Setup did not finish. Double-click setup_install.bat and check install_log.txt.
  pause
  exit /b 1
)
set PY="%VENV%\Scripts\python.exe"

rem ---- install / update libraries (quick after the first time) ----
echo Checking libraries...
%PY% -m pip install -q -r "%SRC%requirements.txt"

rem ---- skip Streamlit's first-run email prompt ----
if not exist "%USERPROFILE%\.streamlit" mkdir "%USERPROFILE%\.streamlit"
if not exist "%USERPROFILE%\.streamlit\credentials.toml" (
  > "%USERPROFILE%\.streamlit\credentials.toml" echo [general]
  >> "%USERPROFILE%\.streamlit\credentials.toml" echo email = ""
)

if not exist "%WORK%" mkdir "%WORK%"
pushd "%WORK%"
echo.
echo Starting the dashboard - it opens in your browser at http://localhost:8501
echo Keep this window open while you use it. Close it to stop the app.
echo.
%PY% -m streamlit run "%SRC%app.py" --browser.gatherUsageStats false --server.headless false
popd
pause
