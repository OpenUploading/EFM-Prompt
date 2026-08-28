@echo off
setlocal
cd /d "%~dp0"
"D:\miniconda\envs\csbrain-bcic2a\python.exe" -u run_hygrip_eegnet.py --model eegnet --epochs 100 --batch-size 8 --lr 0.001 --weight-decay 0 --dropout 0.5 --device cuda --output-dir "D:\data\HYGRIP-Baselines\EEGNet\run_hygrip_eegnet_100ep_seed1"
if errorlevel 1 exit /b %errorlevel%
endlocal
