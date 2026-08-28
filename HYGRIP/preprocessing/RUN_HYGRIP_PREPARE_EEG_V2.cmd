@echo off
setlocal
cd /d "%~dp0"
"D:\bin\matlab.exe" -batch "prepare_hygrip_trials_v2('D:\DataSets\HYGRIP\hygrip.h5','D:\data\HYGRIP-Baselines\prepared','D:\data\HYGRIP-Baselines\prepared_eeg_v2',14)"
set "RUN_EXIT=%ERRORLEVEL%"
echo.
if "%RUN_EXIT%"=="0" (echo HYGRIP EEG v2 preprocessing completed.) else (echo HYGRIP EEG v2 preprocessing failed with exit code %RUN_EXIT%.)
echo Output: D:\data\HYGRIP-Baselines\prepared_eeg_v2
pause
endlocal & exit /b %RUN_EXIT%
