@echo off
REM scripts/run_all.bat
set ROOT=C:\Users\vfgtr554\Documents\fedComp
cd /d %ROOT%

call scripts\0_setup_env.bat

REM 서버 기동
call scripts\1_run_server.bat

REM 서버 안정화 대기
timeout /t 4 >nul

REM 클라이언트 순차 실행 (라운드 종료까지 기다리며 순차)
call scripts\2_run_clients_seq.bat

echo Done. Open http://127.0.0.1:5000 in your browser.
