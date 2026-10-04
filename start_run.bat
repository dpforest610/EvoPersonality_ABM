@echo off
chcp 65001 >nul
cd /d "%~dp0"

if not exist venv\Scripts\python.exe (
    echo [ERROR] venv が見つかりません。先に setup_windows.bat をダブルクリックしてください。
    pause
    exit /b 1
)

for /f "delims=" %%C in ('venv\Scripts\python.exe -c "import os; print(max(1, (os.cpu_count() or 2) - 1))"') do set WORKERS=%%C

echo このPCのコア数から、ワーカー数 %WORKERS% で起動します。
echo.

powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0run_background.ps1" -Workers %WORKERS%

echo.
pause
