# Launch the NeuroForge live research dashboard (read-only; zero training overhead).
#   powershell -ExecutionPolicy Bypass -File scripts\dashboard.ps1
# Opens http://127.0.0.1:8765/ in your browser and watches full_run.log.
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot          # repo root (parent of scripts\)
$py   = Join-Path $root ".venv\Scripts\python.exe"
if (-not (Test-Path $py)) { $py = "python" }      # fall back to PATH python
& $py (Join-Path $PSScriptRoot "dashboard\server.py") --open @args
