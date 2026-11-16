@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0"

echo ============================================
echo  Progress check
echo ============================================
echo.

if not exist venv\Scripts\python.exe (
    echo [ERROR] venv not found. Double-click setup_windows.bat first.
    pause
    exit /b 1
)

venv\Scripts\python.exe scripts\check_progress.py --k-values 99,64,80

echo.
echo --- Process status ---
set LOCKFILE=scripts\output\opponent_count_precision_sweep\.run.lock
if exist "%LOCKFILE%" (
    set LOCKPID=
    for /f "delims=" %%P in (%LOCKFILE%) do set LOCKPID=%%P
    tasklist /FI "PID eq !LOCKPID!" 2>nul | find "!LOCKPID!" >nul
    if errorlevel 1 (
        echo Lock file exists, but process PID !LOCKPID! is not running.
        echo It may have crashed, but checkpoints should still be on disk.
        echo Double-click start_run.bat to resume from where it left off.
    ) else (
        echo Running (PID !LOCKPID!).
    )
) else (
    echo No run is currently in progress (no lock file).
)

echo.
echo --- Recent log output (last 20 lines) ---
if exist logs\sweep_run.log (
    powershell -NoProfile -Command "Get-Content 'logs\sweep_run.log' -Tail 20"
) else (
    echo No log file yet (nothing has been started, or the log is elsewhere).
)

echo.
echo Press any key to close this window.
pause >nul
