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
    [ValidateSet("eeg_only", "pre", "post", "pre_post")]
    [string]$Mode = "pre_post",
    [ValidateSet("conditional", "static")]
    [string]$PromptSource = "conditional",
    [ValidateSet("joint", "prompt_only")]
    [string]$TrainingStrategy = "prompt_only",
    [string]$HeadCheckpoint = "",
    [switch]$ShuffleFnirs,
    [ValidateSet("stats", "temporal")]
    [string]$FnirsConditioner = "temporal",
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
    [ValidateSet("none", "dynamic_mapped_class_ot")]
    [string]$MoPEContrastMode = "none",
    [double]$OTTemperature = 0.1,
    [double]$SinkhornEpsilon = 0.1,
    [int]$SinkhornIterations = 20,
    [double]$OTPairWeight = 0.1,
    [double]$OTClassWeight = 0.02,
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
    [int]$UnfreezeEpoch = 9999,
    [int]$Seed = 1,
    [string]$ExperimentNote = "",
    [switch]$DiagnoseOnly,
    [switch]$ValidateOnly
)

$ErrorActionPreference = "Stop"
$ScriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$PortableRoot = Split-Path -Parent $ScriptRoot
if (-not $OutputRoot) { $OutputRoot = Join-Path $ScriptRoot "runs_mope" }
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
$ContrastTag = if ($MoPEContrastMode -eq "none") { "contrast-none" } else { "contrast-${MoPEContrastMode}" }
$RunName = "cbramod_mope_boundary_${Task}_${PromptSource}_${TrainingStrategy}_${Mode}_dynamic-${DynamicExpertMode}_mapped-${MappedMode}_${ContrastTag}_${Pairing}_drop${DropComponent}_ep${Epochs}_l${PromptCount}_k${ExpertCount}_tau${RouterTemperature}_wimp${ImportanceWeight}_seed${Seed}_${Timestamp}"
$OutputDir = Join-Path $OutputRoot $RunName
$Checkpoint = Join-Path $PortableRoot "CBraMod\pretrained_weights\pretrained_weights.pth"

$Arguments = @(
    "-B", "-u", (Join-Path $ScriptRoot "run_shin2017_cbramod_fnirs_feature_stage1.py"),
    "--portable-root", $PortableRoot,
    "--prep-root", $PrepRoot,
    "--output-dir", $OutputDir,
    "--checkpoint", $Checkpoint,
    "--task", $Task,
    "--mode", $Mode,
    "--prompt-family", "mope",
    "--prompt-source", $PromptSource,
    "--training-strategy", $TrainingStrategy,
    "--epochs", "$Epochs",
    "--batch-size", "$BatchSize",
    "--eeg-scale", "$EegScale",
    "--fnirs-conditioner", $FnirsConditioner,
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
    "--mope-contrast-mode", $MoPEContrastMode,
    "--ot-temperature", "$OTTemperature",
    "--sinkhorn-epsilon", "$SinkhornEpsilon",
    "--sinkhorn-iterations", "$SinkhornIterations",
    "--ot-pair-weight", "$OTPairWeight",
    "--ot-class-weight", "$OTClassWeight",
    "--dynamic-expert-mode", $DynamicExpertMode,
    "--tap-attribute-weight", "$TAPAttributeWeight",
    "--mope-drop-component", $DropComponent,
    "--mapped-mode", $MappedMode,
    "--sgformer-graph-dimension", "$SGFormerGraphDimension",
    "--sgformer-attention-residual-weight", "$SGFormerAttentionResidualWeight",
    "--sgformer-graph-weight", "$SGFormerGraphWeight",
    "--unfreeze-epoch", "$UnfreezeEpoch",
    "--seed", "$Seed",
    "--device", "cuda"
)
if ($ShuffleFnirs) { $Arguments += "--shuffle-fnirs" }
if ($ShinRoot) { $Arguments += @("--shin-root", $ShinRoot) }
if ($CachePath) { $Arguments += @("--cache-path", $CachePath) }
if ($SGFormerCachePath) { $Arguments += @("--sgformer-cache-path", $SGFormerCachePath) }
if ($HeadCheckpoint) { $Arguments += @("--head-checkpoint", $HeadCheckpoint) }
if ($ExperimentNote) { $Arguments += @("--experiment-note", $ExperimentNote) }
if ($DiagnoseOnly) { $Arguments += "--diagnose-only" }

Write-Host "EFM prompt: CBraMod boundary MoPE / task=$Task / mode=$Mode / dynamic=$DynamicExpertMode / mapped=$MappedMode / pairing=$Pairing"
Write-Host "MoPE: prompts=$PromptCount experts=$ExpertCount tau=$RouterTemperature w_imp=$ImportanceWeight"
Write-Host "MoPE contrast: mode=$MoPEContrastMode pair_weight=$OTPairWeight class_weight=$OTClassWeight"
Write-Host "Output directory: $OutputDir"
if ($ValidateOnly) {
    Write-Host "Parameter validation only."
    return
}

Push-Location $ScriptRoot
try {
    & $Python @Arguments
    if ($LASTEXITCODE -ne 0) { throw "Training failed with exit code $LASTEXITCODE" }
}
finally {
    Pop-Location
}
