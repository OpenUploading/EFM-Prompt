param(
    [string]$OutputRoot = "",
    [int]$Epochs = 50,
    [int]$BatchSize = 8,
    [double]$FeatureLr = 3e-4,
    [double]$HeadLr = 1e-4,
    [int]$PromptCount = 4,
    [int]$PromptRank = 8,
    [int]$Seed = 1,
    [switch]$ValidateOnly
)

$ErrorActionPreference = "Stop"
$Runner = Join-Path $PSScriptRoot "run_cbramod_fnirs_feature_stage1_gpu.ps1"
if (-not $OutputRoot) { $OutputRoot = Join-Path $PSScriptRoot "runs" }

function Invoke-Ablation {
    param([string]$Mode, [bool]$ShuffleFnirs = $false)

    if ($ShuffleFnirs) {
        & $Runner -OutputRoot $script:OutputRoot -Epochs $script:Epochs -BatchSize $script:BatchSize `
            -FeatureLr $script:FeatureLr -HeadLr $script:HeadLr -PromptCount $script:PromptCount `
            -PromptRank $script:PromptRank -Mode $Mode -Seed $script:Seed -ShuffleFnirs:$ShuffleFnirs `
            -ValidateOnly:$script:ValidateOnly
    }
    else {
        & $Runner -OutputRoot $script:OutputRoot -Epochs $script:Epochs -BatchSize $script:BatchSize `
            -FeatureLr $script:FeatureLr -HeadLr $script:HeadLr -PromptCount $script:PromptCount `
            -PromptRank $script:PromptRank -Mode $Mode -Seed $script:Seed -ValidateOnly:$script:ValidateOnly
    }
    if ($null -ne $LASTEXITCODE -and $LASTEXITCODE -ne 0) {
        throw "Ablation $Mode failed with exit code $LASTEXITCODE"
    }
}

# All runs share the same 10-patch EEG window, frozen CBraMod encoder,
# official all_patch_reps classifier, split, seed, and optimization settings.
Invoke-Ablation -Mode "eeg_only"
Invoke-Ablation -Mode "pre"
Invoke-Ablation -Mode "post"
Invoke-Ablation -Mode "pre_post"
Invoke-Ablation -Mode "pre_post" -ShuffleFnirs $true
