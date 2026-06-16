# Live dashboard for the NeuroForge full research run.
# Shows: current stage / latest progress line, GPU temp+usage, elapsed time.
# Launch:  powershell -ExecutionPolicy Bypass -File scripts\watch_run.ps1
$ErrorActionPreference = "SilentlyContinue"
$log   = Join-Path $PSScriptRoot "..\full_run.log"
$start = Get-Date
$Host.UI.RawUI.WindowTitle = "NeuroForge — full run monitor"

function Get-LastStatus {
    if (-not (Test-Path $log)) { return "(waiting for log…)" }
    # Read raw, split on CR and LF (tqdm uses \r), keep non-empty lines.
    $raw = Get-Content $log -Raw
    if (-not $raw) { return "(log empty…)" }
    $lines = $raw -split "[`r`n]+" | Where-Object { $_ -and $_.Trim() -ne "" }
    if (-not $lines) { return "(no output yet…)" }
    return ($lines[-1]).Trim()
}

function Get-StageLine {
    if (-not (Test-Path $log)) { return "(starting…)" }
    $raw = Get-Content $log -Raw
    $st  = ($raw -split "[`r`n]+" | Where-Object { $_ -match '\[stage|\[ablation\]|\[done\]|seed \d' })
    if ($st) { return ($st[-1]).Trim() } else { return "(stage 1 — preparing data)" }
}

while ($true) {
    Clear-Host
    $el = (Get-Date) - $start
    Write-Host "==========================================================" -ForegroundColor Cyan
    Write-Host "  NeuroForge CFD — full research run (live monitor)" -ForegroundColor Cyan
    Write-Host "==========================================================" -ForegroundColor Cyan
    Write-Host ("  monitor elapsed : {0:hh\:mm\:ss}" -f $el)
    Write-Host ""

    Write-Host "  STAGE / ARM" -ForegroundColor Yellow
    Write-Host ("    " + (Get-StageLine))
    Write-Host ""
    Write-Host "  LATEST PROGRESS" -ForegroundColor Yellow
    $status = Get-LastStatus
    # wrap long lines
    if ($status.Length -gt 100) { $status = $status.Substring(0,100) + "…" }
    Write-Host ("    " + $status)
    Write-Host ""

    Write-Host "  GPU (RTX 4070 Ti)" -ForegroundColor Yellow
    $g = nvidia-smi --query-gpu=temperature.gpu,utilization.gpu,memory.used,memory.total,power.draw --format=csv,noheader,nounits
    if ($g) {
        $p = $g -split ",\s*"
        $tempColor = if ([int]$p[0] -ge 84) { "Red" } elseif ([int]$p[0] -ge 75) { "Yellow" } else { "Green" }
        Write-Host ("    temp  : {0} C" -f $p[0]) -ForegroundColor $tempColor
        Write-Host ("    util  : {0} %" -f $p[1])
        Write-Host ("    vram  : {0} / {1} MiB" -f $p[2], $p[3])
        Write-Host ("    power : {0} W" -f $p[4])
    } else {
        Write-Host "    (nvidia-smi unavailable)"
    }
    Write-Host ""

    if ((Test-Path $log) -and ((Get-Content $log -Raw) -match '\[done\]')) {
        Write-Host "  >>> RUN COMPLETE. See results\full_research\SUMMARY.md" -ForegroundColor Green
        Write-Host "  (this window will keep refreshing; close it any time)" -ForegroundColor DarkGray
    }
    Write-Host "  refreshing every 4s — close this window any time (Ctrl+C)" -ForegroundColor DarkGray
    Start-Sleep -Seconds 4
}
