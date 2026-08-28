@echo off
setlocal
cd /d "%~dp0"
set "PYTHON=D:\miniconda\envs\csbrain-bcic2a\python.exe"
"%PYTHON%" -u run_hygrip_cbramod.py --prepared-root D:\data\HYGRIP-Baselines\prepared_eeg_v2 --eeg-preprocessing v2_channel_zscore --split-protocol subject_holdout --train-subjects A-J --val-subjects K-L --test-subjects M-N --fine-tune-backbone --epochs 30 --batch-size 4 --head-lr 0.0001 --backbone-lr 0.00001 --weight-decay 0.0001 --dropout 0.1 --seed 1 --device cuda --output-dir D:\data\HYGRIP-Baselines\CBraMod\run_hygrip_cbramod_v2_subject_holdout_fullft30_seed1 --cache-dir D:\data\HYGRIP-Baselines\cache\eeg_cbramod_v2
set "RUN_EXIT=%ERRORLEVEL%"
echo.
echo Exit code: %RUN_EXIT%
pause
endlocal & exit /b %RUN_EXIT%
