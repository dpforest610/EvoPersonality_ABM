# Helper called by start_run.bat. Launches the actual sweep as an
# independent, hidden process that keeps running even after this window
# (and the start_run.bat window that called it) is closed.
#
# Design choice: "survives the launching window being closed" was chosen
# over "runs in a minimized window you must not close".
#
# The command to run is first written to a one-off .cmd file, then that
# file is launched -- rather than handing Start-Process a command string
# with nested quotes directly. Letting cmd.exe re-interpret a quoted
# PowerShell string is a common, hard-to-notice failure mode; writing it
# to a file sidesteps that entirely.
#
# This file is plain ASCII on purpose. Windows PowerShell 5.1 (the
# version that ships with Windows, as opposed to PowerShell 7) can
# misdetect the encoding of a BOM-less UTF-8 file and garble non-ASCII
# text, which previously caused a parse error here. ASCII bytes are
# unambiguous under every encoding Windows might guess, so this avoids
# the whole problem rather than working around it.

param(
    [int]$Workers = 3
)

$RepoRoot = Split-Path -Parent $PSCommandPath
Set-Location $RepoRoot

$PythonExe = Join-Path $RepoRoot "venv/Scripts/python.exe"
$ScriptPath = Join-Path $RepoRoot "scripts/run_opponent_count_precision_sweep_chunked.py"
$LogDir = Join-Path $RepoRoot "logs"
$LogPath = Join-Path $LogDir "sweep_run.log"
$LauncherPath = Join-Path $LogDir "_launcher.cmd"

if (-not (Test-Path $PythonExe)) {
    Write-Host "[ERROR] venv not found. Double-click setup_windows.bat first."
    exit 1
}

# Created here AND in start_run.bat before this script is even invoked --
# belt and suspenders, since nothing later in this script (including the
# log redirect inside the generated launcher file) works without it.
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

$LauncherLines = @(
    "@echo off",
    "`"$PythonExe`" `"$ScriptPath`" --workers $Workers --status-interval-seconds 3600 > `"$LogPath`" 2>&1"
)
Set-Content -Path $LauncherPath -Value $LauncherLines -Encoding ASCII

Start-Process -FilePath $LauncherPath -WorkingDirectory $RepoRoot -WindowStyle Hidden

Write-Host "Launch command sent. Waiting a few seconds to check status..."
Start-Sleep -Seconds 3

$Tail = ""
if (Test-Path $LogPath) {
    $Tail = (Get-Content $LogPath -Tail 10 -ErrorAction SilentlyContinue) -join "`n"
}

if ($Tail -match "already holds the lock") {
    Write-Host ""
    Write-Host "=============================================="
    Write-Host " Already running. This did NOT start a second copy."
    Write-Host "=============================================="
    Write-Host $Tail
} else {
    Write-Host ""
    Write-Host "=============================================="
    Write-Host " Started. You can close this window now."
    Write-Host " The run continues as a hidden background process."
    Write-Host " Check progress: check_progress.bat"
    Write-Host " Stop:           stop_run.bat"
    Write-Host "=============================================="
    if ($Tail) {
        Write-Host ""
        Write-Host "--- Recent log output ---"
        Write-Host $Tail
    }
}
