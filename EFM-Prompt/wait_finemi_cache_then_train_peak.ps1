$ErrorActionPreference = "Stop"
$Python = "D:\Anaconda\envs\bci4models\python.exe"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$Cache = "D:\0senior student creation\datasets\FineMI_Yi2025_raw\processed_prompt_fnirs_binary_1v6_0_12s"
$Manifest = Join-Path $Cache "manifest.json"

while (-not (Test-Path -LiteralPath $Manifest)) {
    Start-Sleep -Seconds 30
}

$SubjectFiles = Get-ChildItem -LiteralPath $Cache -Filter "subject*_fnirs_prompt.npz" -File
if ($SubjectFiles.Count -ne 18) {
    throw "FineMI fNIRS cache incomplete: expected 18 subject files, found $($SubjectFiles.Count)"
}

$Output = Join-Path $Root "results_finemi_cbramod\three_component_peak_seed1"
& $Python (Join-Path $Root "run_finemi_cbramod_prompt.py") `
    --method three_component --seed 1 --epochs 50 --batch-size 8 `
    --head-lr 1e-4 --prompt-lr 3e-4 --output-dir $Output
