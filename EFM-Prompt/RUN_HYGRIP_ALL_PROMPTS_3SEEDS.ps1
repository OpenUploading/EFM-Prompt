param(
    [string]$Python = "D:\Anaconda\envs\bci4models\python.exe",
    [string]$PreparedRoot = "D:\data\HYGRIP-Baselines\prepared_eeg_v2",
    [string]$Device = "cuda"
)

$ErrorActionPreference = "Stop"
$PromptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$PortableRoot = Split-Path -Parent $PromptRoot
$BaselineRunner = Join-Path $PortableRoot "CBraMod\run_hygrip_cbramod.py"
$PromptRunner = Join-Path $PromptRoot "run_hygrip_cbramod_prompt.py"
$Cache = "D:\data\HYGRIP-Baselines\cache\eeg_cbramod_v2"
$OutputRoot = Join-Path $PromptRoot "results_prompt_transfer\hygrip"
$HeadRoot = Join-Path $OutputRoot "eegonly"
$Methods = @(
    "mope",
    "deep_conditional",
    "deep_three_component_shared",
    "tmpa_final",
    "hierarchical_cross_attention",
    "bidirectional_contrast"
)

if (-not (Test-Path -LiteralPath $Python)) { throw "Python not found: $Python" }
if (-not (Test-Path -LiteralPath $PreparedRoot)) { throw "HYGRIP prepared_eeg_v2 not found: $PreparedRoot" }

# Create the three seed-matched frozen-backbone EEG-only heads when absent.
foreach ($Seed in 1, 2, 3) {
    $HeadOutput = Join-Path $HeadRoot "seed$Seed"
    $Head = Join-Path $HeadOutput "best_head.pt"
    if (-not (Test-Path -LiteralPath $Head)) {
        if ((Test-Path -LiteralPath $HeadOutput) -and (Get-ChildItem -LiteralPath $HeadOutput -Force)) {
            throw "Incomplete non-empty EEG-only output directory: $HeadOutput"
        }
        & $Python $BaselineRunner `
            --prepared-root $PreparedRoot `
            --eeg-preprocessing v2_channel_zscore `
            --split-protocol subject_holdout `
            --train-subjects A-J `
            --val-subjects K-L `
            --test-subjects M-N `
            --epochs 50 `
            --batch-size 4 `
            --head-lr 1e-4 `
            --weight-decay 1e-4 `
            --dropout 0.1 `
            --seed $Seed `
            --cache-dir $Cache `
            --output-dir $HeadOutput `
            --device $Device
        if ($LASTEXITCODE -ne 0) { throw "HYGRIP EEG-only seed=$Seed failed with $LASTEXITCODE" }
    } else {
        Write-Host "SKIP existing HYGRIP EEG-only head: seed=$Seed"
    }
}

foreach ($Method in $Methods) {
    foreach ($Seed in 1, 2, 3) {
        $Head = Join-Path $HeadRoot "seed$Seed\best_head.pt"
        $Output = Join-Path $OutputRoot "${Method}_headfrozen_seed$Seed"
        $Summary = Join-Path $Output "summary.json"
        if (Test-Path -LiteralPath $Summary) {
            Write-Host "SKIP complete: HYGRIP $Method seed=$Seed"
            continue
        }
        if ((Test-Path -LiteralPath $Output) -and (Get-ChildItem -LiteralPath $Output -Force)) {
            throw "Incomplete non-empty output directory: $Output"
        }
        & $Python $PromptRunner `
            --method $Method `
            --prepared-root $PreparedRoot `
            --train-subjects A-J `
            --val-subjects K-L `
            --test-subjects M-N `
            --fnirs-window 3 13 `
            --eegonly-head-checkpoint $Head `
            --epochs 50 `
            --batch-size 4 `
            --prompt-lr 3e-4 `
            --weight-decay 1e-4 `
            --dropout 0.1 `
            --lambda-pair 0.1 `
            --lambda-class 0.02 `
            --importance-weight 0.01 `
            --prompt-boundary pre `
            --seed $Seed `
            --output-dir $Output `
            --device $Device
        if ($LASTEXITCODE -ne 0) { throw "HYGRIP $Method seed=$Seed failed with $LASTEXITCODE" }
    }
}

Write-Host "HYGRIP: all six prompt methods, seeds 1/2/3 complete."
