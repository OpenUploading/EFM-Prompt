@echo off
setlocal
cd /d "%~dp0"
"D:\miniconda\envs\csbrain-bcic2a\python.exe" -u run_hygrip_cbramod.py --diagnose-only --device cuda --output-dir "D:\data\HYGRIP-Baselines\CBraMod\diagnostic_frozen_head"
if errorlevel 1 exit /b %errorlevel%
endlocal
