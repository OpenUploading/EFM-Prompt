@echo off
setlocal
cd /d "%~dp0"
set "PYTHON=D:\miniconda\envs\csbrain-bcic2a\python.exe"
"%PYTHON%" -u run_hygrip_eegnet.py --model eegnet --prepared-root D:\data\HYGRIP-Baselines\prepared_eeg_v2 --eeg-normalization channel_zscore --epochs 100 --batch-size 8 --lr 0.001 --weight-decay 0 --dropout 0.5 --seed 1 --device cuda --output-dir D:\data\HYGRIP-Baselines\EEGNet\run_hygrip_eegnet_v2_100ep_seed1
set "RUN_EXIT=%ERRORLEVEL%"
echo.
echo Exit code: %RUN_EXIT%
pause
endlocal & exit /b %RUN_EXIT%
