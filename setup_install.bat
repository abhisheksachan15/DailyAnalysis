@echo off
setlocal
cd /d "%~dp0"
title ETF Momentum Lab - setup
set "LOG=%~dp0install_log.txt"
echo ==== ETF Momentum Lab setup %date% %time% ==== > "%LOG%"
echo.
echo  ETF Momentum Lab - one-time setup on this computer
echo  ---------------------------------------------------
echo.

rem ---- 1. find Python (ignores the Microsoft Store shortcut) ----
call :findpy
if not defined PY (
  echo Python is not installed on this computer.
  echo.
  choice /M "Install Python 3.12 now using Windows Package Manager (winget)"
  if errorlevel 2 goto nopy
  winget install -e --id Python.Python.3.12 --scope user --accept-package-agreements --accept-source-agreements
  set "PATH=%LOCALAPPDATA%\Programs\Python\Python312;%LOCALAPPDATA%\Programs\Python\Python312\Scripts;%LOCALAPPDATA%\Programs\Python\Launcher;%PATH%"
  call :findpy
)
if not defined PY goto nopy
echo [1/4] Python found:
%PY% --version
%PY% --version >> "%LOG%" 2>&1

rem ---- 2. private environment (outside OneDrive so syncing never touches it) ----
set "VENV=%LOCALAPPDATA%\us_etf_momentum_venv"
if not exist "%VENV%\Scripts\python.exe" (
  echo [2/4] Creating the app's Python environment...
  %PY% -m venv "%VENV%" >> "%LOG%" 2>&1
) else (
  echo [2/4] App environment already exists.
)
if not exist "%VENV%\Scripts\python.exe" (
  echo Could not create the environment - see install_log.txt
  pause
  exit /b 1
)
set VPY="%VENV%\Scripts\python.exe"

rem ---- 3. libraries ----
echo [3/4] Installing libraries (pandas, numpy, matplotlib, yfinance, streamlit, plotly)...
echo       This takes 2-5 minutes the first time.
%VPY% -m pip install --upgrade pip >> "%LOG%" 2>&1
%VPY% -m pip install --upgrade -r "%~dp0requirements.txt" >> "%LOG%" 2>&1
if errorlevel 1 (
  echo Library install failed - see install_log.txt
  pause
  exit /b 1
)

rem ---- 4. check everything imports ----
echo [4/4] Checking the installation...
%VPY% -c "import pandas, numpy, matplotlib, yfinance, streamlit, plotly; print('pandas', pandas.__version__, '| streamlit', streamlit.__version__, '| yfinance', yfinance.__version__, '| plotly', plotly.__version__)" >> "%LOG%" 2>&1
if errorlevel 1 (
  echo Check failed - see install_log.txt
  pause
  exit /b 1
)
set PYTHONDONTWRITEBYTECODE=1
%VPY% -c "import sys; sys.path.insert(0, r'%~dp0.'); import us_etf_momentum as m; print('engine OK - markets:', ', '.join(m.PROFILES))" >> "%LOG%" 2>&1

rem ---- skip Streamlit's first-run email prompt ----
if not exist "%USERPROFILE%\.streamlit" mkdir "%USERPROFILE%\.streamlit"
if not exist "%USERPROFILE%\.streamlit\credentials.toml" (
  > "%USERPROFILE%\.streamlit\credentials.toml" echo [general]
  >> "%USERPROFILE%\.streamlit\credentials.toml" echo email = ""
)
if not exist "%USERPROFILE%\us_etf_momentum_work" mkdir "%USERPROFILE%\us_etf_momentum_work"

echo ==== SETUP OK ==== >> "%LOG%"
echo.
echo  Setup complete. Double-click run_dashboard.bat to start the app.
echo  (First start downloads ~20 years of prices - about a minute.)
echo.
pause
exit /b 0

:findpy
set PY=
py -3 --version >nul 2>&1 && set "PY=py -3" && exit /b 0
"%LOCALAPPDATA%\Programs\Python\Python312\python.exe" --version >nul 2>&1 && set PY="%LOCALAPPDATA%\Programs\Python\Python312\python.exe" && exit /b 0
python --version >nul 2>&1 && set PY=python
exit /b 0

:nopy
echo PYTHON_NOT_FOUND >> "%LOG%"
echo.
echo Python is still not available. Install it from https://www.python.org/downloads/
echo (tick "Add python.exe to PATH"), then double-click setup_install.bat again.
pause
exit /b 1
