@echo off
setlocal
"D:\bin\matlab.exe" -batch "addpath('%~dp0'); prepare_hygrip_trials('D:\DataSets\HYGRIP\hygrip.h5','D:\data\HYGRIP-Baselines\prepared',14)"
if errorlevel 1 exit /b %errorlevel%
endlocal
