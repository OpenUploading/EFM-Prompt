$ErrorActionPreference = 'Stop'

$python = 'D:\Anaconda\envs\bci4models\python.exe'
$runner = Join-Path $PSScriptRoot 'run_finemi_eegonly_5fold.py'
$cache = 'D:\0senior student creation\datasets\FineMI_Yi2025_raw\processed_200hz_paper_car_uv100_binary_1v6'
$resultsRoot = 'D:\0senior student creation\results'

function Run-Experiment {
    param(
        [string]$Name,
        [string]$LearningRate,
        [string]$WeightDecay,
        [int]$BatchSize,
        [string]$Dropout
    )
    $output = Join-Path $resultsRoot $Name
    $summary = Join-Path $output 'summary.json'
    if (Test-Path -LiteralPath $summary) { return $output }
    if (Test-Path -LiteralPath $output) {
        if ((Get-ChildItem -LiteralPath $output -Force | Measure-Object).Count -eq 0) {
            Remove-Item -LiteralPath $output
        } else {
            throw "Refusing to overwrite incomplete non-empty run directory: $output"
        }
    }
    New-Item -ItemType Directory -Path $output | Out-Null
    $log = Join-Path $output 'run.log'
    & $python $runner --model cbramod --protocol single --cache-root $cache --output-dir $output --seed 1 --epochs 50 --patience 50 --selection-metric accuracy --batch-size $BatchSize --lr $LearningRate --weight-decay $WeightDecay --dropout $Dropout --optimizer adamw --finetune-mode frozen --scheduler none --binary-loss ce --device cuda *> $log
    if ($LASTEXITCODE -ne 0) { throw "Run failed: $Name (see its run.log)" }
    if (-not (Test-Path -LiteralPath $summary)) { throw "Run completed without summary: $Name" }
    return $output
}

function Read-Score {
    param([string]$Output, [double]$Distance)
    $s = Get-Content -LiteralPath (Join-Path $Output 'summary.json') -Raw | ConvertFrom-Json
    [PSCustomObject]@{
        Output = $Output
        BestVal = [double]$s.folds[0].best_val_selection_score
        Distance = $Distance
    }
}

# Stage 1: lr x weight decay, fixed SHIN-style batch 8 / dropout .1.
$stage1 = @()
foreach ($lr in @('3e-5', '1e-4', '3e-4')) {
    foreach ($wd in @('1e-4', '1e-3', '1e-2')) {
        $name = ('finemi_cbramod_frozen_shin_phase1_lr{0}_wd{1}_seed1' -f $lr, $wd).Replace('-', 'm')
        $out = Run-Experiment -Name $name -LearningRate $lr -WeightDecay $wd -BatchSize 8 -Dropout '0.1'
        $distance = [math]::Pow([math]::Log10(([double]$lr) / 1e-4), 2) + [math]::Pow([math]::Log10(([double]$wd) / 1e-4), 2)
        $stage1 += Read-Score -Output $out -Distance $distance | Add-Member -PassThru NoteProperty LR $lr | Add-Member -PassThru NoteProperty WD $wd
    }
}
$best1 = $stage1 | Sort-Object @{Expression='BestVal';Descending=$true}, @{Expression='Distance';Ascending=$true} | Select-Object -First 1

# Stage 3: batch size, fixed to the stage-1 winning lr/wd.
$stage3 = @()
foreach ($bs in @(8, 16, 32)) {
    $name = "finemi_cbramod_frozen_shin_phase3_bs${bs}_seed1"
    $out = Run-Experiment -Name $name -LearningRate $best1.LR -WeightDecay $best1.WD -BatchSize $bs -Dropout '0.1'
    $stage3 += Read-Score -Output $out -Distance ([math]::Abs($bs - 8)) | Add-Member -PassThru NoteProperty BatchSize $bs
}
$best3 = $stage3 | Sort-Object @{Expression='BestVal';Descending=$true}, @{Expression='Distance';Ascending=$true} | Select-Object -First 1

# Stage 4: dropout, fixed to the stage-1 and stage-3 winners.
$stage4 = @()
foreach ($dropout in @('0.0', '0.1', '0.3', '0.5')) {
    $name = ('finemi_cbramod_frozen_shin_phase4_dropout{0}_seed1' -f $dropout).Replace('.', 'p')
    $out = Run-Experiment -Name $name -LearningRate $best1.LR -WeightDecay $best1.WD -BatchSize $best3.BatchSize -Dropout $dropout
    $stage4 += Read-Score -Output $out -Distance ([math]::Abs(([double]$dropout) - 0.1)) | Add-Member -PassThru NoteProperty Dropout $dropout
}
$best4 = $stage4 | Sort-Object @{Expression='BestVal';Descending=$true}, @{Expression='Distance';Ascending=$true} | Select-Object -First 1

[PSCustomObject]@{
    stage1 = $stage1
    stage3 = $stage3
    stage4 = $stage4
    selected = [PSCustomObject]@{
        lr = $best1.LR
        weight_decay = $best1.WD
        batch_size = $best3.BatchSize
        dropout = $best4.Dropout
        output = $best4.Output
    }
} | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath (Join-Path $resultsRoot 'finemi_cbramod_frozen_shin_single_seed_tuning_summary.json') -Encoding utf8
