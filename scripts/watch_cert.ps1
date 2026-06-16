# Lightweight live monitor for any single-run log (e.g. the H4/H5 certificate run).
# Tails the log + shows GPU — no stage/progress model, so nothing looks "stuck".
# Launch:  powershell -ExecutionPolicy Bypass -File scripts\watch_cert.ps1 -Log cert_run.log
param([string]$Log = "cert_run.log")
$ErrorActionPreference = "SilentlyContinue"
if (-not [System.IO.Path]::IsPathRooted($Log)) { $Log = Join-Path (Join-Path $PSScriptRoot "..") $Log }
$start = Get-Date
$Host.UI.RawUI.WindowTitle = "NeuroForge — certificate run monitor"

while ($true) {
    Clear-Host
    $el = (Get-Date) - $start
    Write-Host "==========================================================" -ForegroundColor Cyan
    Write-Host "  NeuroForge — H4/H5 certificate run (live)" -ForegroundColor Cyan
    Write-Host "==========================================================" -ForegroundColor Cyan
    Write-Host ("  watching : {0}" -f $Log) -ForegroundColor DarkGray
    Write-Host ("  elapsed  : {0:hh\:mm\:ss}" -f $el)
    Write-Host ""
    Write-Host "  LATEST OUTPUT" -ForegroundColor Yellow
    if (Test-Path $Log) {
        $raw = Get-Content $Log -Raw
        $lines = $raw -split "[`r`n]+" | Where-Object { $_ -and $_.Trim() -ne "" }
        $tail = if ($lines.Count -ge 6) { $lines[($lines.Count-6)..($lines.Count-1)] } else { $lines }
        foreach ($l in $tail) { $s = $l.Trim(); if ($s.Length -gt 96) { $s = $s.Substring(0,96) + "..." }; Write-Host ("    " + $s) }
    } else { Write-Host "    (waiting for log...)" }
    Write-Host ""
    Write-Host "  GPU (RTX 4070 Ti)" -ForegroundColor Yellow
    $g = nvidia-smi --query-gpu=temperature.gpu,utilization.gpu,memory.used,memory.total,power.draw --format=csv,noheader,nounits
    if ($g) {
        $p = $g -split ",\s*"
        $tc = if ([int]$p[0] -ge 84) { "Red" } elseif ([int]$p[0] -ge 75) { "Yellow" } else { "Green" }
        Write-Host ("    temp  : {0} C" -f $p[0]) -ForegroundColor $tc
        Write-Host ("    util  : {0} %   power : {1} W" -f $p[1], $p[4])
        Write-Host ("    vram  : {0} / {1} MiB" -f $p[2], $p[3])
    } else { Write-Host "    (nvidia-smi unavailable)" }
    if ((Test-Path $Log) -and ((Get-Content $Log -Raw) -match 'H4|coverage|contraction|\[done\]|Traceback|Error')) {
        Write-Host ""
        Write-Host "  >>> certificate stage reached (coverage/contraction) — near done" -ForegroundColor Green
    }
    Write-Host ""
    Write-Host "  refresh 4s — close any time (Ctrl+C)" -ForegroundColor DarkGray
    Start-Sleep -Seconds 4
}
