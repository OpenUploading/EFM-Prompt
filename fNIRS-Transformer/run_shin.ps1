param(
    [string]$CondaEnv = "codebrain-bcic2a",
    [string]$DataRoot = "D:\DataSets\SHIN\NIRS_01-29",
    [string]$OutDir = "",
    [string]$TrainSubjects = "1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19",
    [string]$ValSubjects = "20,21,22,23,24",
    [string]$TestSubjects = "25,26,27,28,29",
    [int]$Epochs = 50,
    [int]$BatchSize = 128,
    [double]$HeadLr = 1e-4,
    [double]$BackboneLr = 1e-3,
    [int]$Seed = 1,
    [string]$ExperimentNote = "",
    [switch]$DiagnoseOnly
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$python = "D:\miniconda\envs\$CondaEnv\python.exe"
if (-not (Test-Path -LiteralPath $python)) { throw "Conda Python not found: $python" }
if (-not (Test-Path -LiteralPath $DataRoot)) { throw "SHIN fNIRS root not found: $DataRoot" }
if ($OutDir -eq "") {
    $stamp = Get-Date -Format "yyyyMMdd-HHmmss"
    $OutDir = "D:\data\fNIRS-Transformer-SHIN\${stamp}_ep${Epochs}_headlr${HeadLr}_backbonelr${BackboneLr}_seed${Seed}"
}

$argsList = @(
    "-B", "-m", "shin_pipeline.train",
    "--data-root", $DataRoot,
    "--out-dir", $OutDir,
    "--train-subjects", $TrainSubjects,
    "--val-subjects", $ValSubjects,
    "--test-subjects", $TestSubjects,
    "--epochs", "$Epochs",
    "--batch-size", "$BatchSize",
    "--head-lr", "$HeadLr",
    "--backbone-lr", "$BackboneLr",
    "--seed", "$Seed"
)
if ($DiagnoseOnly) { $argsList += "--diagnose-only" }
if ($ExperimentNote -ne "") { $argsList += @("--experiment-note", $ExperimentNote) }

Write-Host "Python: $python"
Write-Host "Data:   $DataRoot"
Write-Host "Output: $OutDir"
Write-Host "Seed:   $Seed"
Set-Location $ProjectRoot
& $python @argsList
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
