@echo off
REM scripts/2_run_clients_seq.bat
set ROOT=C:\Users\vfgtr554\Documents\fedComp
call conda activate fedcomp
cd /d %ROOT%

echo Running client 1...
python -m client.client1
echo Client 1 done.

echo Running client 2...
python -m client.client2
echo Client 2 done.

echo Running client 3...
python -m client.client3
echo Client 3 done.

echo All clients finished.
