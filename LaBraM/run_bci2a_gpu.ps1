$ErrorActionPreference = "Stop"

$Workspace = "C:\Users\liumy\Documents\Codex\2026-06-28\c-users-liumy-onedrive-eeg-fnirs-2"
$EnvRoot = "D:\miniconda\envs\LaBraM"
$DllFix = Join-Path $Workspace "work\dllfix"
$Output = Join-Path $Workspace "outputs\bci2a_labram_gpu_ep50_lr5e4"
$CacheSource = Join-Path $Workspace "outputs\bci2a_labram_gpu_1epoch\cache"

$env:PYTHONPATH = "$DllFix;$env:PYTHONPATH"
$env:PATH = "$DllFix;$EnvRoot;$EnvRoot\Library\bin;$EnvRoot\Scripts;$env:PATH"
$env:MPLCONFIGDIR = Join-Path $Workspace "work\mplconfig"

New-Item -ItemType Directory -Force -Path (Join-Path $Output "cache") | Out-Null
if (Test-Path $CacheSource) {
    Copy-Item -LiteralPath (Join-Path $CacheSource "train_A01-A02-A03-A04-A05-A06-A07-A08-A09_22ch_4s_200hz.npz") -Destination (Join-Path $Output "cache\train_A01-A02-A03-A04-A05-A06-A07-A08-A09_22ch_4s_200hz.npz") -Force
    Copy-Item -LiteralPath (Join-Path $CacheSource "test_A01-A02-A03-A04-A05-A06-A07-A08-A09_22ch_4s_200hz.npz") -Destination (Join-Path $Output "cache\test_A01-A02-A03-A04-A05-A06-A07-A08-A09_22ch_4s_200hz.npz") -Force
}

& "$EnvRoot\python.exe" run_bci2a_finetune.py `
    --output_dir $Output `
    --epochs 50 `
    --batch_size 16 `
    --lr 5e-4 `
    --device cuda
