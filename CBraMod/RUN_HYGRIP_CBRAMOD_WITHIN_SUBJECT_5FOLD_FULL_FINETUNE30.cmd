@echo off
setlocal
cd /d "%~dp0"
set "HYGRIP_PYTHON=D:\miniconda\envs\csbrain-bcic2a\python.exe"

if not exist "%HYGRIP_PYTHON%" (
  echo Python not found: %HYGRIP_PYTHON%
  set "RUN_EXIT=2"
  goto :finished
)

"%HYGRIP_PYTHON%" -u run_hygrip_cbramod.py --split-protocol within_subject_5fold --all-subjects A-N --fine-tune-backbone --epochs 30 --batch-size 4 --head-lr 0.0001 --backbone-lr 0.00001 --weight-decay 0.0001 --dropout 0.1 --seed 1 --device cuda --output-dir D:\data\HYGRIP-Baselines\CBraMod\run_hygrip_cbramod_within5_3-1-1_fullft30_head1e-4_backbone1e-5_seed1 --cache-dir D:\data\HYGRIP-Baselines\cache\eeg_cbramod_within5
set "RUN_EXIT=%ERRORLEVEL%"

:finished
echo.
if "%RUN_EXIT%"=="0" (
  echo HYGRIP CBraMod full fine-tuning completed successfully.
) else (
  echo HYGRIP CBraMod full fine-tuning failed with exit code %RUN_EXIT%.
)
echo Output: D:\data\HYGRIP-Baselines\CBraMod\run_hygrip_cbramod_within5_3-1-1_fullft30_head1e-4_backbone1e-5_seed1
echo.
pause
endlocal & exit /b %RUN_EXIT%
