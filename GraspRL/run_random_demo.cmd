@echo off
cd /d "%~dp0"
if exist "..\.conda\grasp_rl\python.exe" (
  "..\.conda\grasp_rl\python.exe" demo.py --policy random --realtime
) else (
  call conda run --no-capture-output -n grasp_rl python demo.py --policy random --realtime
)
pause
