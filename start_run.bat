@echo off
cd /d "%~dp0"

if not exist venv\Scripts\python.exe (
    echo [ERROR] venv not found. Double-click setup_windows.bat first.
    pause
    exit /b 1
)

if not exist logs mkdir logs

for /f "delims=" %%C in ('venv\Scripts\python.exe -c "import os; print(max(1, (os.cpu_count() or 2) - 1))"') do set WORKERS=%%C

echo Starting with %WORKERS% worker(s) (CPU core count minus 1).
echo.

powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0run_background.ps1" -Workers %WORKERS%

echo.
pause
