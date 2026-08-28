$ErrorActionPreference = "Stop"
$Python = "D:\Anaconda\envs\bci4models\python.exe"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$Runner = Join-Path $Root "run_finemi_cbramod_prompt.py"
$Results = Join-Path $Root "results_finemi_cbramod"

foreach ($Seed in 1, 2, 3) {
    & $Python $Runner --method eegonly --seed $Seed --epochs 50 --batch-size 64 `
        --head-lr 1e-4 --output-dir (Join-Path $Results "eegonly_seed$Seed")
}

foreach ($Method in "three_component", "tmpa") {
    foreach ($Seed in 1, 2, 3) {
        & $Python $Runner --method $Method --seed $Seed --epochs 50 --batch-size 8 `
            --head-lr 1e-4 --prompt-lr 3e-4 `
            --output-dir (Join-Path $Results "${Method}_seed$Seed")
    }
}
