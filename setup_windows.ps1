# One-time setup on Windows. Run in the VS Code terminal (PowerShell) from the project folder:
#   powershell -ExecutionPolicy Bypass -File setup_windows.ps1
# Creates .venv, installs PyTorch with CUDA (NVIDIA GPU) and the other requirements, then checks the GPU.

$ErrorActionPreference = "Stop"

Write-Host "1/4 Checking for an NVIDIA GPU..."
$hasGpu = $false
try { nvidia-smi -L; $hasGpu = $true } catch { Write-Host "   No NVIDIA GPU / driver found - will install the CPU version." }

Write-Host "2/4 Creating virtual environment .venv ..."
if (-not (Test-Path ".venv")) { python -m venv .venv }
& .\.venv\Scripts\python.exe -m pip install --upgrade pip

Write-Host "3/4 Installing PyTorch ..."
if ($hasGpu) {
    # CUDA 12.6 build. If this fails, copy the current command from https://pytorch.org/get-started/locally/
    & .\.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cu126
} else {
    & .\.venv\Scripts\python.exe -m pip install torch
}
& .\.venv\Scripts\python.exe -m pip install -r requirements.txt

Write-Host "4/4 Checking PyTorch ..."
& .\.venv\Scripts\python.exe -c "import torch; print('PyTorch', torch.__version__, '| CUDA available:', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')"
Write-Host "Done. In VS Code: Ctrl+Shift+P -> 'Python: Select Interpreter' -> choose .venv"
