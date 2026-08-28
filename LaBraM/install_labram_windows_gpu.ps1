$ErrorActionPreference = "Stop"

$EnvPython = "D:\miniconda\envs\LaBraM\python.exe"
$RepoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path

if (-not (Test-Path $EnvPython)) {
    throw "Cannot find LaBraM environment Python: $EnvPython"
}

Push-Location $RepoRoot
try {
    & $EnvPython -m pip install --upgrade pip setuptools wheel

    # Install PyTorch before DeepSpeed. DeepSpeed imports torch during its build step.
    & $EnvPython -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

    & $EnvPython -m pip install -r requirements-windows-gpu.txt

    @'
import importlib
mods = ["torch", "numpy", "scipy", "sklearn", "mne", "timm", "einops", "h5py", "pandas", "tensorboardX"]
for name in mods:
    mod = importlib.import_module(name)
    print(f"{name}: {getattr(mod, '__version__', 'OK')}")
import torch
print("cuda_available:", torch.cuda.is_available())
print("cuda_version:", torch.version.cuda)
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
    print("capability:", torch.cuda.get_device_capability(0))
'@ | & $EnvPython -

    Write-Host ""
    Write-Host "Core LaBraM dependencies installed."
    Write-Host "DeepSpeed is optional for single-GPU fine-tuning. If you still need it, run after torch is installed:"
    Write-Host "  $EnvPython -m pip install deepspeed"
}
finally {
    Pop-Location
}
