@echo off
chcp 65001 >nul
set "PYTHON=D:\miniconda\envs\csbrain-bcic2a\python.exe"
set "SCRIPT=C:\Users\liumy\Documents\Codex\2026-07-21\d-datasets-shin\work\DAMFNet-official\run_shin_damfnet.py"

"%PYTHON%" "%SCRIPT%" --task ma --output-root "D:\data\DAMFNet-SHIN-CoopHP" --epochs 40 --patience 10 --batch-size 16 --lr 1e-4 --weight-decay 0 --dropout 0.5 --loss-w-eeg 1 --loss-w-hbr 1 --loss-w-fuse 1 --seed 0 --device cuda
if errorlevel 1 (
  echo DAMFNet MA cooperative-hyperparameter training failed.
  pause
  exit /b 1
)
echo DAMFNet MA cooperative-hyperparameter training completed.
pause
