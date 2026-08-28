@echo off
chcp 65001 >nul
call conda activate bci4models
cd /d "%~dp0"
python run_ds004022_eegnet.py --data-root "D:\0senior student creation\datasets\ds004022_orthopedic_mi_eeg_fnirs" --task mi --train-subjects 1-5 --val-subjects 6 --test-subjects 7 --epochs 100 --batch-size 32 --lr 1e-3 --weight-decay 0 --dropout 0.25 --kernel-length 64 --seed 1 --device cuda
if errorlevel 1 (
  echo EEGNet ds004022 training failed.
  pause
  exit /b 1
)
echo EEGNet ds004022 training completed.
pause
