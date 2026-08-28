param(
    [string]$PrepRoot = "D:\0senior student creation\braindecode_codebrain_prep",
    [string]$ShinRoot = "",
    [string]$CachePath = "",
    [string]$OutputRoot = "",
    [string]$HeadRoot = "",
    [string]$CondaEnv = "bci4models",
    [string]$CondaRoot = "D:\Anaconda",
    [ValidateSet("mi", "ma")]
    [string]$Task = "mi",
    [int]$Epochs = 50,
    [int]$BatchSize = 8,
    [double]$EegScale = 1.0,
    [ValidateSet("stats", "temporal")]
    [string]$FnirsConditioner = "temporal",
    [double]$FeatureLr = 3e-4,
    [double]$HeadLr = 1e-4,
    [double]$BackboneLr = 1e-5,
    [ValidateSet("conditional", "static")]
    [string]$PromptSource = "conditional",
    [ValidateSet("eeg_only", "pre", "post", "pre_post")]
    [string]$Mode = "pre_post",
    [switch]$ShuffleFnirs,
    [int]$PromptCount = 4,
    [int]$PromptRank = 8,
    [int]$UnfreezeEpoch = 9999,
    [ValidateSet("joint", "prompt_only")]
    [string]$TrainingStrategy = "joint",
    [string]$HeadCheckpoint = "",
    [int]$Seed = 1,
    [string]$ExperimentNote = "",
    [switch]$ValidateOnly
)

$ErrorActionPreference = "Stop"
$ScriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$PortableRoot = Split-Path -Parent $ScriptRoot
if (-not $OutputRoot) { $OutputRoot = Join-Path $ScriptRoot "runs" }
if (-not $HeadRoot) { $HeadRoot = Join-Path $ScriptRoot "runs" }
$Python = Join-Path $CondaRoot "envs\$CondaEnv\python.exe"
if (-not (Test-Path -LiteralPath $Python)) {
    $Python = "D:\miniconda\envs\$CondaEnv\python.exe"
}
if (-not (Test-Path -LiteralPath $Python)) { throw "Conda environment not found: $Python" }
if (-not (Test-Path -LiteralPath $PrepRoot)) { throw "Prep root not found: $PrepRoot" }

if ($TrainingStrategy -eq "prompt_only" -and -not $HeadCheckpoint) {
    $headCandidates = Get-ChildItem -LiteralPath $HeadRoot -Directory -ErrorAction SilentlyContinue |
        Where-Object {
            $_.Name -match "^cbramod_boundary_prompt_${Task}_.*joint_eeg_only_.*_seed${Seed}_" -and
            (Test-Path -LiteralPath (Join-Path $_.FullName "best_model.pth"))
        } |
        Sort-Object LastWriteTime -Descending
    if (-not $headCandidates) {
        throw "No EEG-only head found for task=$Task seed=$Seed under $HeadRoot. Run EEG-only first, or pass -HeadCheckpoint explicitly."
    }
    $HeadCheckpoint = Join-Path $headCandidates[0].FullName "best_model.pth"
    Write-Host "Matched EEG-only head: $HeadCheckpoint"
}

$Timestamp = Get-Date -Format "yyyyMMdd-HHmmss"
$Pairing = if ($ShuffleFnirs) { "shuffled" } else { "aligned" }
$RunName = "cbramod_boundary_prompt_${Task}_${PromptSource}_${TrainingStrategy}_${Mode}_${Pairing}_ep${Epochs}_featurelr${FeatureLr}_headlr${HeadLr}_m${PromptCount}_r${PromptRank}_seed${Seed}_${Timestamp}"
$OutputDir = Join-Path $OutputRoot $RunName
$Checkpoint = Join-Path $PortableRoot "CBraMod\pretrained_weights\pretrained_weights.pth"

$Arguments = @(
    "-B", "-u", (Join-Path $ScriptRoot "run_shin2017_cbramod_fnirs_feature_stage1.py"),
    "--portable-root", $PortableRoot,
    "--prep-root", $PrepRoot,
    "--output-dir", $OutputDir,
    "--checkpoint", $Checkpoint,
    "--task", $Task,
    "--epochs", "$Epochs",
    "--batch-size", "$BatchSize",
    "--eeg-scale", "$EegScale",
    "--fnirs-conditioner", $FnirsConditioner,
    "--feature-lr", "$FeatureLr",
    "--head-lr", "$HeadLr",
    "--backbone-lr", "$BackboneLr",
    "--prompt-source", $PromptSource,
    "--mode", "$Mode",
    "--prompt-count", "$PromptCount",
    "--prompt-rank", "$PromptRank",
    "--unfreeze-epoch", "$UnfreezeEpoch",
    "--training-strategy", $TrainingStrategy,
    "--seed", "$Seed",
    "--device", "cuda"
)
if ($ShuffleFnirs) { $Arguments += "--shuffle-fnirs" }
if ($ShinRoot) { $Arguments += @("--shin-root", $ShinRoot) }
if ($CachePath) { $Arguments += @("--cache-path", $CachePath) }
if ($HeadCheckpoint) { $Arguments += @("--head-checkpoint", $HeadCheckpoint) }
if ($ExperimentNote) { $Arguments += @("--experiment-note", $ExperimentNote) }

Write-Host "EFM prompt: boundary conditional prompt / mode=$Mode / pairing=$Pairing"
Write-Host "Backbone: CBraMod"
Write-Host "Output directory: $OutputDir"
if ($ValidateOnly) {
    Write-Host "Parameter validation only: Epochs=$Epochs BatchSize=$BatchSize FeatureLr=$FeatureLr HeadLr=$HeadLr"
    return
}
Push-Location $ScriptRoot
try {
    & $Python @Arguments
    if ($LASTEXITCODE -ne 0) { throw "Training failed with exit code $LASTEXITCODE" }
}
finally { Pop-Location }
