param(
    [string]$DataRoot = "D:\DataSets\SHIN\v1.0.1",
    [string]$OutputRoot = "D:\data\LaBraM-SHIN",
    [string]$CondaEnv = "LaBraM",
    [int]$Epochs = 100,
    [int]$BatchSize = 8,
    [double]$HeadLr = 1e-4,
    [double]$BackboneLr = 1e-5,
    [int]$UnfreezeEpoch = 91,
    [int]$Seed = 1,
    [string]$ExperimentNote = ""
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$Python = "D:\miniconda\envs\$CondaEnv\python.exe"
if (-not (Test-Path -LiteralPath $Python)) {
    throw "Conda environment Python not found: $Python"
}

$Timestamp = Get-Date -Format "yyyyMMdd-HHmmss"
$RunName = "${Timestamp}_ep${Epochs}_headlr1e-4_backbonelr1e-5_unfreeze${UnfreezeEpoch}_seed${Seed}"
$OutputDir = Join-Path $OutputRoot $RunName
$Arguments = @(
    "-B", "run_shin_finetune.py",
    "--data-root", $DataRoot,
    "--output-dir", $OutputDir,
    "--cache-dir", (Join-Path $OutputRoot "cache"),
    "--checkpoint", (Join-Path $RepoRoot "checkpoints\labram-base.pth"),
    "--epochs", "$Epochs",
    "--batch-size", "$BatchSize",
    "--head-lr", "$HeadLr",
    "--backbone-lr", "$BackboneLr",
    "--unfreeze-epoch", "$UnfreezeEpoch",
    "--seed", "$Seed",
    "--device", "cuda"
)
if ($ExperimentNote) {
    $Arguments += @("--experiment-note", $ExperimentNote)
}

Write-Host "LaBraM environment: $CondaEnv"
Write-Host "Output directory: $OutputDir"
Push-Location $RepoRoot
try {
    & $Python @Arguments
    if ($LASTEXITCODE -ne 0) { throw "Training failed with exit code $LASTEXITCODE" }
}
finally {
    Pop-Location
}
