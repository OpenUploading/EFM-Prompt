param(
    [string]$DataRoot = "D:\DataSets\SHIN\v1.0.1",
    [string]$OutputRoot = "D:\data\EFM-Prompt-SHIN",
    [string]$CondaEnv = "cbramod_env",
    [ValidateSet("none", "static", "context")]
    [string]$PromptMode = "static",
    [int]$Epochs = 50,
    [int]$BatchSize = 8,
    [double]$PromptLr = 3e-4,
    [double]$HeadLr = 1e-4,
    [double]$BackboneLr = 1e-5,
    [double]$PromptScale = 0.05,
    [int]$UnfreezeEpoch = 9999,
    [int]$Seed = 1,
    [string]$ExperimentNote = ""
)

$ErrorActionPreference = "Stop"
$ScriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$PortableRoot = Split-Path -Parent $ScriptRoot
$Python = "D:\miniconda\envs\$CondaEnv\python.exe"
if (-not (Test-Path -LiteralPath $Python)) { throw "Conda environment not found: $Python" }

$Timestamp = Get-Date -Format "yyyyMMdd-HHmmss"
$RunName = "cbramod_prompt_${PromptMode}_ep${Epochs}_promptlr${PromptLr}_headlr${HeadLr}_scale${PromptScale}_seed${Seed}_${Timestamp}"
$OutputDir = Join-Path $OutputRoot $RunName
$Checkpoint = Join-Path $PortableRoot "CBraMod\pretrained_weights\pretrained_weights.pth"
$CacheDir = Join-Path $OutputRoot "cache"

$Arguments = @(
    "-B", "-u", (Join-Path $ScriptRoot "run_shin_cbramod_prompt.py"),
    "--portable-root", $PortableRoot,
    "--data-root", $DataRoot,
    "--output-dir", $OutputDir,
    "--cache-dir", $CacheDir,
    "--checkpoint", $Checkpoint,
    "--prompt-mode", $PromptMode,
    "--epochs", "$Epochs",
    "--batch-size", "$BatchSize",
    "--prompt-lr", "$PromptLr",
    "--head-lr", "$HeadLr",
    "--backbone-lr", "$BackboneLr",
    "--prompt-scale", "$PromptScale",
    "--unfreeze-epoch", "$UnfreezeEpoch",
    "--seed", "$Seed",
    "--device", "cuda"
)
if ($ExperimentNote) { $Arguments += @("--experiment-note", $ExperimentNote) }

Write-Host "EFM prompt backbone: CBraMod"
Write-Host "Prompt mode: $PromptMode"
Write-Host "Output directory: $OutputDir"
Push-Location $ScriptRoot
try {
    & $Python @Arguments
    if ($LASTEXITCODE -ne 0) { throw "Training failed with exit code $LASTEXITCODE" }
}
finally { Pop-Location }

