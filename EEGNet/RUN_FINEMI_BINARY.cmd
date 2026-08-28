@echo off
call D:\Anaconda\Scripts\activate.bat bci4models
cd /d "%~dp0"
python run_finemi_binary_eegnet.py ^
  --task mi ^
  --data-root "D:\0senior student creation\datasets\FineMI_Yi2025_raw\processed_200hz_car_binary_1v6" ^
  --train-subjects 1-12 ^
  --val-subjects 13-15 ^
  --test-subjects 16-18 ^
  --epochs 100 ^
  --batch-size 32 ^
  --lr 0.001 ^
  --dropout 0.5 ^
  --kernel-length 100 ^
  --sampling-rate 200 ^
  --device cuda
