# Start the FastAPI backend on Windows (PowerShell).
# Usage:  .\scripts\start_backend.ps1 [-Port 8000]
#   $env:LEAF_CONF=0.3 ; .\scripts\start_backend.ps1
param([int]$Port = 8000)
$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

# Activate venv if present and not already inside one.
if (-not $env:VIRTUAL_ENV -and (Test-Path "$RepoRoot\venv\Scripts\Activate.ps1")) {
  & "$RepoRoot\venv\Scripts\Activate.ps1"
}

# Make the project root importable so `smart_leaf_detection` resolves.
if ($env:PYTHONPATH) {
  $env:PYTHONPATH = "$RepoRoot;$env:PYTHONPATH"
} else {
  $env:PYTHONPATH = "$RepoRoot"
}
if (-not $env:LEAF_CONF) { $env:LEAF_CONF = "0.3" }

Set-Location webapp\backend
Write-Host "==> Backend on http://localhost:$Port  (LEAF_CONF=$env:LEAF_CONF)"
python -m uvicorn app.main:app --port $Port --reload
