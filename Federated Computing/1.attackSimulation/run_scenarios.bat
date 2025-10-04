
@echo off
setlocal EnableExtensions EnableDelayedExpansion

REM ===== CONFIG =====
set "ROOT=C:\Users\vfgtr554\Documents\fedComp"
if exist "%ROOT%\1.attackSimulation" set "ROOT=%ROOT%\1.attackSimulation"
set "ENV_NAME=fedcomp"
set "ROUNDS=5"

echo [SETUP] ROOT=%ROOT%
echo [SETUP] ENV =%ENV_NAME%
echo.

REM -- Activate conda in this controller window
call conda activate %ENV_NAME% || goto :conda_err
cd /d "%ROOT%"

REM ===== Launch Flask dashboard (persistent window) =====
echo [FLASK] Starting dashboard...
start "FLASK Dashboard" cmd /k "call conda activate %ENV_NAME% && pushd ""%ROOT%"" && python -m server.app"

REM ===== Give Flask a second to start =====
timeout /t 2 >nul

REM ===== Orchestrate scenarios via Python controller =====
echo [CTRL] Running scenarios A/B/C (attackdefend), %ROUNDS% rounds each phase...
python -m scripts.run_scenarios

echo.
echo [DONE] All scenarios finished.
exit /b 0

:conda_err
echo [ERR] Failed to activate conda env "%ENV_NAME%".
exit /b 1
