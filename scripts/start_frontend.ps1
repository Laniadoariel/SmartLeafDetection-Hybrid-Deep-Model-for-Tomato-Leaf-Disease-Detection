# Start the React/Vite frontend dev server on Windows (PowerShell).
$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location "$RepoRoot\webapp\frontend"

if (-not (Test-Path node_modules)) {
  Write-Host "==> node_modules missing - running npm install"
  npm install
}

Write-Host "==> Frontend dev server starting (Vite)"
npm run dev
