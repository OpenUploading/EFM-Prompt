param(
    [string]$OutDir = "external\CodeBrain\Checkpoints"
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ProjectRoot

$TargetDir = Join-Path $ProjectRoot $OutDir
New-Item -ItemType Directory -Force -Path $TargetDir | Out-Null

$files = @(
    @{
        Name = "CodeBrain.pth"
        Url = "https://huggingface.co/YjMajy/CodeBrain/resolve/main/CodeBrain.pth"
    },
    @{
        Name = "CodeBrain_Tokenizer.pth"
        Url = "https://huggingface.co/YjMajy/CodeBrain/resolve/main/CodeBrain_Tokenizer.pth"
    }
)

foreach ($file in $files) {
    $dest = Join-Path $TargetDir $file.Name
    if (Test-Path -LiteralPath $dest) {
        $sizeMb = [math]::Round((Get-Item -LiteralPath $dest).Length / 1MB, 2)
        Write-Host "Exists: $dest ($sizeMb MB)"
        continue
    }

    Write-Host "Downloading $($file.Name) ..."
    Invoke-WebRequest -Uri $file.Url -OutFile $dest
    $sizeMb = [math]::Round((Get-Item -LiteralPath $dest).Length / 1MB, 2)
    Write-Host "Saved: $dest ($sizeMb MB)"
}

Write-Host ""
Write-Host "Backbone checkpoint path:"
Write-Host (Join-Path $TargetDir "CodeBrain.pth")
