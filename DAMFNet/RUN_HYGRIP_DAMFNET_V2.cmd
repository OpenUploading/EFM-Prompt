@echo off
setlocal
cd /d "%~dp0"
set "PYTHON=D:\miniconda\envs\csbrain-bcic2a\python.exe"
"%PYTHON%" -u run_hygrip_damfnet.py --model damfnet --prepared-root D:\data\HYGRIP-Baselines\prepared_eeg_v2 --eeg-normalization channel_zscore --epochs 40 --batch-size 40 --lr 0.0001 --weight-decay 0 --dropout 0.4 --seed 1 --device cuda --output-dir D:\data\HYGRIP-Baselines\DAMFNet\run_hygrip_damfnet_v2_40ep_seed1
set "RUN_EXIT=%ERRORLEVEL%"
echo.
echo Exit code: %RUN_EXIT%
pause
endlocal & exit /b %RUN_EXIT%
