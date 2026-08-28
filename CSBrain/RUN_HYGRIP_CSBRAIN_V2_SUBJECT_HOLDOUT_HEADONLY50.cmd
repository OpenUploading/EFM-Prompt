@echo off
setlocal
cd /d "%~dp0"
set "PYTHON=D:\miniconda\envs\csbrain-bcic2a\python.exe"
"%PYTHON%" -u run_hygrip_csbrain.py --model csbrain --prepared-root D:\data\HYGRIP-Baselines\prepared_eeg_v2 --train-subjects A-J --val-subjects K-L --test-subjects M-N --epochs 50 --batch-size 64 --feature-batch-size 4 --head-lr 0.0001 --weight-decay 0.01 --dropout 0.1 --seed 1 --device cuda --output-dir D:\data\HYGRIP-Baselines\CSBrain\run_hygrip_csbrain_v2_subject_holdout_headonly50_seed1
set "RUN_EXIT=%ERRORLEVEL%"
echo.
echo Exit code: %RUN_EXIT%
pause
endlocal & exit /b %RUN_EXIT%
