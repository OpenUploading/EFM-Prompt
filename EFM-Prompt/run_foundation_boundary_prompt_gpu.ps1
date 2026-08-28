param(
    [string]$PrepRoot = "D:\0senior student creation\braindecode_codebrain_prep",
    [string]$ShinRoot = "",
    [string]$CachePath = "",
    [string]$OutputRoot = "",
    [string]$HeadRoot = "",
    [string]$CondaEnv = "bci4models",
    [string]$CondaRoot = "D:\Anaconda",
    [ValidateSet("codebrain", "csbrain")]
    [string]$Backbone = "codebrain",
    [string]$BackboneCheckpoint = "",
    [ValidateSet("mi", "ma")]
    [string]$Task = "mi",
    [ValidateSet("eeg_only", "pre", "post", "pre_post")]
    [string]$Mode = "pre_post",
    [ValidateSet("joint", "prompt_only")]
    [string]$TrainingStrategy = "prompt_only",
    [string]$HeadCheckpoint = "",
    [switch]$ShuffleFnirs,
    [ValidateSet("stats", "temporal")]
    [string]$FnirsConditioner = "temporal",
    [ValidateSet("conditional", "static")]
    [string]$PromptSource = "conditional",
    [int]$Epochs = 50,
    [int]$BatchSize = 8,
    [double]$EegScale = 100.0,
    [double]$FeatureLr = 3e-4,
    [double]$HeadLr = 1e-4,
    [double]$BackboneLr = 1e-5,
    [int]$PromptCount = 6,
    [int]$PromptRank = 8,
    [int]$ExpertCount = 16,
    [double]$RouterTemperature = 0.1,
    [double]$RouterNoiseStd = 0.00390625,
    [double]$ImportanceThreshold = 0.05,
    [double]$ImportanceWeight = 0.01,
    [ValidateSet("flat", "tap4x4")]
    [string]$DynamicExpertMode = "flat",
    [double]$TAPAttributeWeight = 0.1,
    [ValidateSet("none", "static", "dynamic", "mapped")]
    [string]$DropComponent = "none",
    [ValidateSet("mlp", "sgformer")]
    [string]$MappedMode = "mlp",
    [string]$SGFormerCachePath = "",
    [int]$SGFormerGraphDimension = 128,
    [double]$SGFormerAttentionResidualWeight = 0.5,
    [double]$SGFormerGraphWeight = 0.8,
    [int]$Seed = 1,
    [switch]$DiagnoseOnly,
    [switch]$ValidateOnly
)

$ErrorActionPreference = "Stop"
$ScriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$PortableRoot = Split-Path -Parent $ScriptRoot
if (-not $OutputRoot) { $OutputRoot = Join-Path $ScriptRoot "runs_foundation" }
if (-not $HeadRoot) { $HeadRoot = $OutputRoot }
if (-not $BackboneCheckpoint) {
    if ($Backbone -eq "codebrain") {
        $BackboneCheckpoint = Join-Path $PortableRoot "CodeBrain\pretrained_weights\CodeBrain.pth"
    } else {
        $BackboneCheckpoint = "D:\0senior student creation\2026-06-27_MI_BCI_IV_2a_4models_experiment_log\repos\CSBrain\pth_downloaded\pth\CSBrain.pth"
    }
}
if (-not (Test-Path -LiteralPath $BackboneCheckpoint)) {
    throw "Backbone checkpoint not found: $BackboneCheckpoint"
}
if (-not $HeadCheckpoint -and $TrainingStrategy -eq "prompt_only") {
    $headCandidates = Get-ChildItem -LiteralPath $HeadRoot -Directory -ErrorAction SilentlyContinue |
        Where-Object {
            $_.Name -match "^foundation_${Backbone}_${Task}_.*joint_eeg_only_.*_seed${Seed}_" -and
            (Test-Path -LiteralPath (Join-Path $_.FullName "best_model.pth"))
        } |
        Sort-Object LastWriteTime -Descending
    if (-not $headCandidates) {
        throw "No $Backbone EEG-only head found for task=$Task seed=$Seed under $HeadRoot. Run EEG-only first."
    }
    $HeadCheckpoint = Join-Path $headCandidates[0].FullName "best_model.pth"
    Write-Host "Matched EEG-only head: $HeadCheckpoint"
}

$Python = Join-Path $CondaRoot "envs\$CondaEnv\python.exe"
if (-not (Test-Path -LiteralPath $Python)) { $Python = "D:\miniconda\envs\$CondaEnv\python.exe" }
if (-not (Test-Path -LiteralPath $Python)) { throw "Conda environment not found: $Python" }
$Timestamp = Get-Date -Format "yyyyMMdd-HHmmss"
$Pairing = if ($ShuffleFnirs) { "shuffled" } else { "aligned" }
$RunName = "foundation_${Backbone}_${Task}_mope_${TrainingStrategy}_${Mode}_dynamic-${DynamicExpertMode}_mapped-${MappedMode}_${Pairing}_ep${Epochs}_l${PromptCount}_k${ExpertCount}_seed${Seed}_${Timestamp}"
$OutputDir = Join-Path $OutputRoot $RunName

$Arguments = @(
    "-B", "-u", (Join-Path $ScriptRoot "run_shin2017_foundation_boundary_prompt.py"),
    "--portable-root", $PortableRoot,
    "--prep-root", $PrepRoot,
    "--output-dir", $OutputDir,
    "--backbone", $Backbone,
    "--backbone-checkpoint", $BackboneCheckpoint,
    "--task", $Task,
    "--mode", $Mode,
    "--training-strategy", $TrainingStrategy,
    "--epochs", "$Epochs",
    "--batch-size", "$BatchSize",
    "--eeg-scale", "$EegScale",
    "--fnirs-conditioner", $FnirsConditioner,
    "--prompt-source", $PromptSource,
    "--feature-lr", "$FeatureLr",
    "--head-lr", "$HeadLr",
    "--backbone-lr", "$BackboneLr",
    "--prompt-count", "$PromptCount",
    "--prompt-rank", "$PromptRank",
    "--expert-count", "$ExpertCount",
    "--router-temperature", "$RouterTemperature",
    "--router-noise-std", "$RouterNoiseStd",
    "--importance-threshold", "$ImportanceThreshold",
    "--importance-weight", "$ImportanceWeight",
    "--dynamic-expert-mode", $DynamicExpertMode,
    "--tap-attribute-weight", "$TAPAttributeWeight",
    "--mope-drop-component", $DropComponent,
    "--mapped-mode", $MappedMode,
    "--sgformer-graph-dimension", "$SGFormerGraphDimension",
    "--sgformer-attention-residual-weight", "$SGFormerAttentionResidualWeight",
    "--sgformer-graph-weight", "$SGFormerGraphWeight",
    "--seed", "$Seed",
    "--device", "cuda"
)
if ($HeadCheckpoint) { $Arguments += @("--head-checkpoint", $HeadCheckpoint) }
if ($ShinRoot) { $Arguments += @("--shin-root", $ShinRoot) }
if ($CachePath) { $Arguments += @("--cache-path", $CachePath) }
if ($SGFormerCachePath) { $Arguments += @("--sgformer-cache-path", $SGFormerCachePath) }
if ($ShuffleFnirs) { $Arguments += "--shuffle-fnirs" }
if ($DiagnoseOnly) { $Arguments += "--diagnose-only" }

Write-Host "Foundation boundary prompt: $Backbone / task=$Task / mode=$Mode / dynamic=$DynamicExpertMode / mapped=$MappedMode / pairing=$Pairing"
Write-Host "Prompt location: after native patch embedding and after backbone output"
Write-Host "Output directory: $OutputDir"
if ($ValidateOnly) { return }

Push-Location $ScriptRoot
try {
    & $Python @Arguments
    if ($LASTEXITCODE -ne 0) { throw "Training failed with exit code $LASTEXITCODE" }
}
finally { Pop-Location }
