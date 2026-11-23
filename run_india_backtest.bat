@echo off
setlocal
cd /d "%~dp0"
echo ==== Indian NSE ETF Momentum backtest ==== > run_log_india.txt

call :findpy
if not defined PY (
  echo.
  echo Python is not installed on this computer.
  echo.
  choice /M "Install Python 3.12 now using Windows Package Manager (winget)"
  if errorlevel 2 goto nopy
  winget install -e --id Python.Python.3.12 --scope user --accept-package-agreements --accept-source-agreements
  set "PATH=%LOCALAPPDATA%\Programs\Python\Python312;%LOCALAPPDATA%\Programs\Python\Python312\Scripts;%LOCALAPPDATA%\Programs\Python\Launcher;%PATH%"
  call :findpy
)
if not defined PY goto nopy

echo Using: & %PY% --version
%PY% --version >> run_log_india.txt 2>&1
set "VENV=%LOCALAPPDATA%\us_etf_momentum_venv"
if not exist "%VENV%\Scripts\python.exe" (
  echo Creating virtual environment...
  %PY% -m venv "%VENV%" >> run_log_india.txt 2>&1
)
if exist "%VENV%\Scripts\python.exe" (
  set VPY="%VENV%\Scripts\python.exe"
) else (
  echo Virtual environment failed, using system Python instead >> run_log_india.txt
  set VPY=%PY%
)
echo Installing libraries (first run takes a few minutes)...
%VPY% -m pip install --upgrade pip >> run_log_india.txt 2>&1
%VPY% -m pip install -r requirements.txt >> run_log_india.txt 2>&1
set "SRC=%~dp0"
set "WORK=%USERPROFILE%\us_etf_momentum_work"
if not exist "%WORK%" mkdir "%WORK%"
pushd "%WORK%"
echo Running base backtest (downloads prices)...
%VPY% "%SRC%us_etf_momentum.py" --market india --capital 1000000 --etf-group 1 --refresh >> "%SRC%run_log_india.txt" 2>&1
echo Running comparison sweep...
%VPY% "%SRC%us_etf_momentum.py" --market india --capital 1000000 --etf-group 1 --sweep >> "%SRC%run_log_india.txt" 2>&1
echo Today's signal...
%VPY% "%SRC%us_etf_momentum.py" --market india --capital 1000000 --etf-group 1 --signal >> "%SRC%run_log_india.txt" 2>&1
popd
echo Copying results back to this folder...
robocopy "%WORK%\results" "%SRC%results" /E /NFL /NDL /NJH /NJS >nul
echo ==== DONE ==== >> run_log_india.txt
echo.
echo Done. Results are in the results folder. Log: run_log_india.txt
pause
exit /b 0

:findpy
set PY=
py -3 --version >nul 2>&1 && set "PY=py -3" && exit /b 0
"%LOCALAPPDATA%\Programs\Python\Python312\python.exe" --version >nul 2>&1 && set "PY="%LOCALAPPDATA%\Programs\Python\Python312\python.exe"" && exit /b 0
python --version >nul 2>&1 && set PY=python
exit /b 0

:nopy
echo PYTHON_NOT_FOUND >> run_log_india.txt
echo.
echo Python is still not available. Install it from https://www.python.org/downloads/
echo (tick "Add python.exe to PATH"), then double-click run_backtest.bat again.
pause
exit /b 1
