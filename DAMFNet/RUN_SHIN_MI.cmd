@echo off
chcp 65001 >nul
set "PYTHON=D:\miniconda\envs\csbrain-bcic2a\python.exe"
set "SCRIPT=C:\Users\liumy\Documents\Codex\2026-07-21\d-datasets-shin\work\DAMFNet-official\run_shin_damfnet.py"

"%PYTHON%" "%SCRIPT%" --task mi --epochs 40 --batch-size 40 --lr 1e-4 --weight-decay 0 --dropout 0.4 --seed 1 --device cuda
if errorlevel 1 (
  echo DAMFNet MI training failed.
  pause
  exit /b 1
)
echo DAMFNet MI training completed.
pause
