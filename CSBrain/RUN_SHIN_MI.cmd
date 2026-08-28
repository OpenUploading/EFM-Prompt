@echo off
chcp 65001 >nul
set "PYTHON=D:\miniconda\envs\csbrain-bcic2a\python.exe"
set "SCRIPT=C:\Users\liumy\Documents\Codex\2026-07-21\d-datasets-shin\work\CSBrain-official\run_shin_finetune.py"

"%PYTHON%" "%SCRIPT%" --task mi --epochs 100 --batch-size 8 --head-lr 1e-4 --backbone-lr 1e-5 --weight-decay 1e-4 --unfreeze-epoch 91 --seed 1 --device cuda
if errorlevel 1 (
  echo.
  echo MI training failed with exit code %errorlevel%.
  pause
  exit /b %errorlevel%
)
echo.
echo MI training completed.
pause
