@echo off
chcp 65001 >nul
set "PYTHON=D:\miniconda\envs\csbrain-bcic2a\python.exe"
set "SCRIPT=C:\Users\liumy\Documents\Codex\2026-07-21\d-datasets-shin\work\EEGNet-official\run_shin_eegnet.py"

"%PYTHON%" "%SCRIPT%" --task ma --epochs 100 --batch-size 8 --lr 1e-3 --weight-decay 0 --dropout 0.5 --kernel-length 100 --seed 1 --device cuda
if errorlevel 1 (
  echo EEGNet MA training failed.
  pause
  exit /b 1
)
echo EEGNet MA training completed.
pause
