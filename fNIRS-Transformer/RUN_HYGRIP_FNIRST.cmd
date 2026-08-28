@echo off
setlocal
cd /d "%~dp0"
"D:\miniconda\envs\csbrain-bcic2a\python.exe" -u run_hygrip_fnirst.py --model fnirst --epochs 50 --batch-size 32 --head-lr 0.0001 --backbone-lr 0.001 --weight-decay 0.01 --dropout 0.5 --device cuda --output-dir "D:\data\HYGRIP-Baselines\fNIRS-T\run_hygrip_fnirst_50ep_seed1"
if errorlevel 1 exit /b %errorlevel%
endlocal
