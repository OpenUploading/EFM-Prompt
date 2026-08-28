@echo off
chcp 65001 >nul
set "PYTHON=D:\miniconda\envs\csbrain-bcic2a\python.exe"
set "SCRIPT=C:\Users\liumy\Documents\Codex\2026-07-21\d-datasets-shin\work\DAMFNet-official\run_shin_damfnet.py"

"%PYTHON%" "%SCRIPT%" --task mi --sensor-layout damf_fixed --epoch-start-s -2 --epoch-stop-s 10 --window-seconds 3 --window-stride-seconds 1 --output-root "D:\data\DAMFNet-SHIN-Fixed8-24-Neg2To10-Pat30" --epochs 40 --patience 30 --batch-size 16 --lr 1e-4 --weight-decay 0 --dropout 0.5 --loss-w-eeg 1 --loss-w-hbr 1 --loss-w-fuse 1 --seed 0 --device cuda
if errorlevel 1 (
  echo DAMFNet MI fixed-8/24 -2..10 s patience-30 training failed.
  pause
  exit /b 1
)
echo DAMFNet MI fixed-8/24 -2..10 s patience-30 training completed.
pause
