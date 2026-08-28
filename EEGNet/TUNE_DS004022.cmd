@echo off
chcp 65001 >nul
call conda activate bci4models
cd /d "%~dp0"
python tune_ds004022_eegnet.py --train-subjects 1-5 --val-subjects 6 --test-subjects 7 --epochs 80 --patience 12 --batch-size 32 --seeds 1 --device cuda
if errorlevel 1 (
  echo EEGNet tuning failed.
  pause
  exit /b 1
)
echo EEGNet tuning completed. Review tuning_summary.json and ranking.csv in the printed output directory.
pause
