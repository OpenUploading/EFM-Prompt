param(
    [string]$Python = "D:\Anaconda\envs\bci4models\python.exe",
    [string]$Device = "cuda"
)

$ErrorActionPreference = "Stop"
$PromptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$Runner = Join-Path $PromptRoot "run_finemi_cbramod_prompt.py"
$HeadRoot = Join-Path $PromptRoot "results_finemi_cbramod"
$OutputRoot = Join-Path $PromptRoot "results_prompt_transfer\finemi"
$EEGCache = "D:\0senior student creation\datasets\FineMI_Yi2025_raw\processed_200hz_rawuv_binary_1v6"
$Methods = @(
    "mope",
    "deep_conditional",
    "deep_three_component_shared",
    "tmpa_final",
    "hierarchical_cross_attention",
    "bidirectional_contrast"
)

if (-not (Test-Path -LiteralPath $Python)) { throw "Python not found: $Python" }
if (-not (Test-Path -LiteralPath $EEGCache)) { throw "FineMI matching raw-uV cache not found: $EEGCache" }

foreach ($Method in $Methods) {
    foreach ($Seed in 1, 2, 3) {
        $Head = Join-Path $HeadRoot "eegonly_seed$Seed\best_prompt_and_head.pth"
        $Output = Join-Path $OutputRoot "${Method}_headfrozen_seed$Seed"
        $Summary = Join-Path $Output "summary.json"
        if (Test-Path -LiteralPath $Summary) {
            Write-Host "SKIP complete: FineMI $Method seed=$Seed"
            continue
        }
        if (-not (Test-Path -LiteralPath $Head)) { throw "FineMI seed-$Seed EEG-only head missing: $Head" }
        if ((Test-Path -LiteralPath $Output) -and (Get-ChildItem -LiteralPath $Output -Force)) {
            throw "Incomplete non-empty output directory: $Output"
        }
        & $Python $Runner `
            --method $Method `
            --eeg-cache-root $EEGCache `
            --seed $Seed `
            --epochs 50 `
            --batch-size 8 `
            --prompt-lr 3e-4 `
            --head-lr 1e-4 `
            --weight-decay 1e-4 `
            --dropout 0.1 `
            --lambda-pair 0.1 `
            --lambda-class 0.02 `
            --importance-weight 0.01 `
            --prompt-boundary pre `
            --fnirs-window 3 7 `
            --freeze-classifier `
            --eegonly-head-checkpoint $Head `
            --output-dir $Output `
            --device $Device
        if ($LASTEXITCODE -ne 0) { throw "FineMI $Method seed=$Seed failed with $LASTEXITCODE" }
    }
}

Write-Host "FineMI: all six prompt methods, seeds 1/2/3 complete."
