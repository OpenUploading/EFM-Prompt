@echo off
setlocal
cd /d "%~dp0"
"D:\miniconda\envs\csbrain-bcic2a\python.exe" -u run_hygrip_cbramod.py --epochs 50 --batch-size 4 --head-lr 0.0001 --weight-decay 0.0001 --dropout 0.1 --seed 1 --device cuda --output-dir "D:\data\HYGRIP-Baselines\CBraMod\run_hygrip_cbramod_frozen50_seed1"
if errorlevel 1 exit /b %errorlevel%
endlocal
