param(
    [string]$CondaEnv = "codebrain-bcic2a",
    [string]$DataRoot = "D:\DataSets\SHIN\v1.0.1",
    [string]$OutDir = "",
    [string]$TrainSubjects = "1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23",
    [string]$ValSubjects = "24,25,26",
    [string]$TestSubjects = "27,28,29",
    [double]$TMin = 0.0,
    [double]$TMax = 10.0,
    [int]$Epochs = 20,
    [int]$BatchSize = 8,
    [double]$HeadLr = 1e-3,
    [double]$BackboneLr = 1e-5,
    [int]$Seed = 1,
    [string]$ExperimentNote = "",
    [int]$UnfreezeBackboneEpoch = 0,
    [switch]$FineTuneBackbone,
    [switch]$DiagnoseOnly
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ProjectRoot

$python = $null
if ($env:CONDA_PREFIX) {
    $activePython = Join-Path $env:CONDA_PREFIX "python.exe"
    if ((Split-Path -Leaf $env:CONDA_PREFIX) -eq $CondaEnv -and (Test-Path -LiteralPath $activePython)) {
        $python = $activePython
    }
}
if (-not $python) {
    $knownPython = "D:\miniconda\envs\$CondaEnv\python.exe"
    if (Test-Path -LiteralPath $knownPython) { $python = $knownPython }
}
if (-not $python) { throw "Python for conda environment '$CondaEnv' was not found." }
if (-not (Test-Path -LiteralPath $DataRoot)) { throw "SHIN BIDS root not found: $DataRoot" }
if ($OutDir -eq "") {
    $stamp = Get-Date -Format "yyyyMMdd-HHmmss"
    $OutDir = Join-Path "D:\data\CodeBrain-SHIN" "${stamp}_ep${Epochs}_headlr${HeadLr}_backbonelr${BackboneLr}_seed${Seed}"
}

$script = Join-Path $ProjectRoot "scripts\shin_train.py"
$checkpoint = Join-Path $ProjectRoot "external\CodeBrain\Checkpoints\CodeBrain.pth"
$argsList = @(
    $script,
    "--data-root", $DataRoot,
    "--out-dir", $OutDir,
    "--pretrained-backbone", $checkpoint,
    "--train-subjects", $TrainSubjects,
    "--val-subjects", $ValSubjects,
    "--test-subjects", $TestSubjects,
    "--tmin", "$TMin",
    "--tmax", "$TMax",
    "--epochs", "$Epochs",
    "--batch-size", "$BatchSize",
    "--head-lr", "$HeadLr",
    "--backbone-lr", "$BackboneLr",
    "--seed", "$Seed",
    "--unfreeze-backbone-epoch", "$UnfreezeBackboneEpoch"
)
if ($FineTuneBackbone) { $argsList += "--finetune-backbone" }
if ($DiagnoseOnly) { $argsList += "--diagnose-only" }
if ($ExperimentNote -ne "") { $argsList += @("--experiment-note", $ExperimentNote) }

Write-Host "Python: $python"
Write-Host "Data:   $DataRoot"
Write-Host "Output: $OutDir"
Write-Host "Seed:   $Seed"
& $python @argsList
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
