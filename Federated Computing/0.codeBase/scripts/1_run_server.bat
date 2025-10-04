@echo off
REM scripts/1_run_server.bat
set ROOT=C:\Users\vfgtr554\Documents\fedComp
call conda activate fedcomp

cd /d %ROOT%\server

REM Flask 먼저
start "Flask(5000)" cmd /k "call conda activate fedcomp && python app.py"

REM 포트 준비 대기
timeout /t 2 >nul

REM Flower 서버
start "Flower(8080)" cmd /k "call conda activate fedcomp && python flower_server.py"
