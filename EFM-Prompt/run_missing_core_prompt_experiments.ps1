param(
    [ValidateSet("all", "tmpa", "cross", "deep")]
    [string]$ExperimentSet = "all",
    [string]$Python = "D:\Anaconda\envs\bci4models\python.exe",
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
$PromptRoot = $PSScriptRoot
$PortableRoot = Split-Path -Parent $PromptRoot
$OutputRoot = Join-Path $PromptRoot "runs_core_supplement"

$Scripts = @{
    tmpa  = Join-Path $PromptRoot "run_shin2017_foundation_tmpa_token_alignment.py"
    cross = Join-Path $PromptRoot "run_shin2017_foundation_hierarchical_cross_attention.py"
    deep  = Join-Path $PromptRoot "run_shin2017_foundation_deep_prompt.py"
}

$Heads = @{
    "cbramod-mi-1" = Join-Path $PromptRoot "runs\cbramod_boundary_prompt_mi_conditional_joint_eeg_only_aligned_ep50_featurelr0.0003_headlr0.0001_m4_r8_seed1_20260801-110046\best_model.pth"
    "cbramod-mi-2" = Join-Path $PromptRoot "runs\cbramod_boundary_prompt_mi_conditional_joint_eeg_only_aligned_ep50_featurelr0.0003_headlr0.0001_m4_r8_seed2_20260804-154547\best_model.pth"
    "cbramod-mi-3" = Join-Path $PromptRoot "runs\cbramod_boundary_prompt_mi_conditional_joint_eeg_only_aligned_ep50_featurelr0.0003_headlr0.0001_m4_r8_seed3_20260804-180131\best_model.pth"
    "cbramod-ma-1" = Join-Path $PromptRoot "runs\cbramod_boundary_prompt_ma_conditional_joint_eeg_only_aligned_ep50_featurelr0.0003_headlr0.0001_m4_r8_seed1_20260801-145417\best_model.pth"
    "cbramod-ma-2" = Join-Path $PromptRoot "runs\cbramod_boundary_prompt_ma_conditional_joint_eeg_only_aligned_ep50_featurelr0.0003_headlr0.0001_m4_r8_seed2_20260804-165543\best_model.pth"
    "cbramod-ma-3" = Join-Path $PromptRoot "runs\cbramod_boundary_prompt_ma_conditional_joint_eeg_only_aligned_ep50_featurelr0.0003_headlr0.0001_m4_r8_seed3_20260804-190406\best_model.pth"
    "codebrain-mi-1" = Join-Path $PromptRoot "runs_foundation\foundation_codebrain_mi_mope_joint_eeg_only_aligned_ep50_l6_k16_seed1_20260805-091803\best_model.pth"
    "codebrain-ma-1" = Join-Path $PromptRoot "runs_foundation\foundation_codebrain_ma_mope_joint_eeg_only_aligned_ep50_l6_k16_seed1_20260805-113828\best_model.pth"
    "csbrain-mi-1" = Join-Path $PromptRoot "runs_foundation\foundation_csbrain_mi_mope_joint_eeg_only_aligned_ep50_l6_k16_seed1_20260805-135132\best_model.pth"
    "csbrain-ma-1" = Join-Path $PromptRoot "runs_foundation\foundation_csbrain_ma_mope_joint_eeg_only_mapped-mlp_aligned_ep50_l6_k16_seed1_20260805-183836\best_model.pth"
}

function Add-Job {
    param(
        [System.Collections.Generic.List[object]]$List,
        [string]$Method,
        [string]$Backbone,
        [string]$Task,
        [int]$Seed,
        [ValidateSet("frozen", "trainable")]
        [string]$HeadMode
    )
    $List.Add([pscustomobject]@{
        Method = $Method
        Backbone = $Backbone
        Task = $Task
        Seed = $Seed
        HeadMode = $HeadMode
    })
}

$Jobs = [System.Collections.Generic.List[object]]::new()

if ($ExperimentSet -in @("all", "tmpa")) {
    # Existing CBraMod seed 1-3 runs used a trainable head; supplement frozen-head runs.
    foreach ($task in @("mi", "ma")) {
        foreach ($seed in 1..3) {
            Add-Job $Jobs "tmpa" "cbramod" $task $seed "frozen"
        }
    }
    # CodeBrain and CSBrain require both classifier protocols for the main comparison.
    foreach ($backbone in @("codebrain", "csbrain")) {
        foreach ($task in @("mi", "ma")) {
            Add-Job $Jobs "tmpa" $backbone $task 1 "trainable"
            Add-Job $Jobs "tmpa" $backbone $task 1 "frozen"
        }
    }
}

if ($ExperimentSet -in @("all", "cross")) {
    # Existing CBraMod seed 1-3 runs used a trainable head.
    foreach ($task in @("mi", "ma")) {
        foreach ($seed in 1..3) {
            Add-Job $Jobs "cross" "cbramod" $task $seed "frozen"
        }
    }
    foreach ($backbone in @("codebrain", "csbrain")) {
        foreach ($task in @("mi", "ma")) {
            Add-Job $Jobs "cross" $backbone $task 1 "trainable"
            Add-Job $Jobs "cross" $backbone $task 1 "frozen"
        }
    }
}

if ($ExperimentSet -in @("all", "deep")) {
    # Deep prompt is defined as strict prompt-only: both EFM and classifier stay frozen.
    foreach ($task in @("mi", "ma")) {
        foreach ($seed in 2..3) {
            Add-Job $Jobs "deep" "cbramod" $task $seed "frozen"
        }
    }
    foreach ($backbone in @("codebrain", "csbrain")) {
        foreach ($task in @("mi", "ma")) {
            Add-Job $Jobs "deep" $backbone $task 1 "frozen"
        }
    }
}

if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
    throw "Python not found: $Python"
}
foreach ($script in $Scripts.Values) {
    if (-not (Test-Path -LiteralPath $script -PathType Leaf)) {
        throw "Experiment script not found: $script"
    }
}

$index = 0
foreach ($job in $Jobs) {
    $index += 1
    $name = "$($job.Method)_$($job.Backbone)_$($job.Task)_seed$($job.Seed)_head-$($job.HeadMode)"
    $outputDir = Join-Path $OutputRoot $name
    $summary = Join-Path $outputDir "summary.json"
    if (Test-Path -LiteralPath $summary -PathType Leaf) {
        Write-Host "[$index/$($Jobs.Count)] SKIP completed: $name"
        continue
    }
    if (Test-Path -LiteralPath $outputDir) {
        throw "Incomplete output directory already exists: $outputDir. Rename or remove it before resuming."
    }

    $arguments = @(
        $Scripts[$job.Method],
        "--portable-root", $PortableRoot,
        "--backbone", $job.Backbone,
        "--task", $job.Task,
        "--seed", $job.Seed,
        "--epochs", 50,
        "--batch-size", 8,
        "--weight-decay", 1e-4,
        "--dropout", 0.1,
        "--output-dir", $outputDir
    )

    if ($job.Method -eq "deep") {
        $arguments += @(
            "--prompt-lr", 3e-4,
            "--prompt-dim", 128,
            "--prompt-tokens", 4,
            "--fnirs-temporal-tokens", 10,
            "--attention-heads", 8,
            "--prompt-scale", 0.05
        )
    } else {
        $arguments += @(
            "--feature-lr", 3e-4,
            "--alignment-dim", 128,
            "--fnirs-temporal-tokens", 10,
            "--mode-count", 4,
            "--prompt-tokens-per-mode", 2,
            "--attention-heads", 8,
            "--prompt-scale", 0.05,
            "--contrast-temperature", 0.1,
            "--lambda-pair", 0.1,
            "--lambda-class", 0.02
        )
    }

    if ($job.HeadMode -eq "frozen") {
        $headKey = "$($job.Backbone)-$($job.Task)-$($job.Seed)"
        $headCheckpoint = $Heads[$headKey]
        if (-not $headCheckpoint -or -not (Test-Path -LiteralPath $headCheckpoint -PathType Leaf)) {
            throw "Matching EEG-only head not found for $headKey"
        }
        if ($job.Method -ne "deep") {
            $arguments += "--freeze-classifier"
        }
        $arguments += @("--head-checkpoint", $headCheckpoint)
    }

    Write-Host "[$index/$($Jobs.Count)] RUN $name"
    if ($DryRun) {
        Write-Host ($Python + " " + ($arguments -join " "))
        continue
    }
    & $Python @arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Experiment failed: $name (exit code $LASTEXITCODE)"
    }
}

Write-Host "Core prompt supplement complete. Results: $OutputRoot"
