@echo off
cd /d "%~dp0"

echo ============================================
echo  Stop
echo ============================================
echo.

set LOCKFILE=scripts\output\opponent_count_precision_sweep\.run.lock
if not exist "%LOCKFILE%" (
    echo No run is currently in progress (no lock file).
    pause
    exit /b 0
)

set LOCKPID=
for /f "delims=" %%P in (%LOCKFILE%) do set LOCKPID=%%P

if "%LOCKPID%"=="" (
    echo [ERROR] Could not read the lock file's contents.
    pause
    exit /b 1
)

tasklist /FI "PID eq %LOCKPID%" 2>nul | find "%LOCKPID%" >nul
if errorlevel 1 (
    echo Process PID %LOCKPID% is not running (it may have already stopped).
    echo Nothing to do -- start_run.bat will resume from the last checkpoint.
    pause
    exit /b 0
)

echo Stopping process PID %LOCKPID%...
taskkill /PID %LOCKPID% /T /F
if errorlevel 1 (
    echo [ERROR] Failed to stop it. Please end python.exe manually in Task Manager.
    pause
    exit /b 1
)

echo.
echo Stopped. Only the most recent chunk (up to 500 generations, up to about
echo 12 minutes for k=99) of progress may be lost. Everything before that is
echo already on disk, so start_run.bat will resume from there.
echo.
pause
