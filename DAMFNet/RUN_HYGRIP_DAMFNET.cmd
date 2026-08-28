@echo off
setlocal
cd /d "%~dp0"
"D:\miniconda\envs\csbrain-bcic2a\python.exe" -u run_hygrip_damfnet.py --model damfnet --epochs 40 --patience 30 --batch-size 16 --lr 0.0001 --weight-decay 0 --dropout 0.5 --device cuda --output-dir "D:\data\HYGRIP-Baselines\DAMFNet\run_hygrip_damfnet_40ep_seed1"
if errorlevel 1 exit /b %errorlevel%
endlocal
