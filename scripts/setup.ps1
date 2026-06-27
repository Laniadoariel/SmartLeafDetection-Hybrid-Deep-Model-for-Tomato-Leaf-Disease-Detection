# Cross-platform setup for Windows (PowerShell).
# Creates a Python virtual environment, installs Python deps, and installs the
# frontend node modules.
#
# If you get "running scripts is disabled on this system", run PowerShell once as:
#   Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

$Python = if ($env:PYTHON) { $env:PYTHON } else { "python" }

Write-Host "==> Repo: $RepoRoot"
Write-Host "==> Creating virtual environment (venv\) with $Python"
& $Python -m venv venv

# Activate for this session
& "$RepoRoot\venv\Scripts\Activate.ps1"

Write-Host "==> Upgrading pip"
python -m pip install --upgrade pip

Write-Host "==> Installing core pipeline + training deps (requirements.txt)"
pip install -r requirements.txt

Write-Host "==> Installing web backend deps (webapp\backend\requirements.txt)"
pip install -r webapp\backend\requirements.txt

if (Get-Command npm -ErrorAction SilentlyContinue) {
  Write-Host "==> Installing frontend deps (npm install)"
  Push-Location webapp\frontend
  npm install
  Pop-Location
} else {
  Write-Host "!! npm not found - skipping frontend install. Install Node.js 18+ then run:"
  Write-Host "     cd webapp\frontend ; npm install"
}

Write-Host "==> Done. Activate the env with:  .\venv\Scripts\Activate.ps1"
