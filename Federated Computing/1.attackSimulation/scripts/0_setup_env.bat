@echo off
REM scripts/0_setup_env.bat
REM Create/prepare conda env and install dependencies
call conda activate
conda env list | findstr /C:"fedcomp" >nul
IF %ERRORLEVEL% NEQ 0 (
    conda create -y -n fedcomp python=3.10
)
call conda activate fedcomp

pip install --upgrade pip
pip install flwr flask requests numpy psutil

echo Env ready: fedcomp
