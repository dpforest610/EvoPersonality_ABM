@echo off
setlocal enabledelayedexpansion
chcp 65001 >nul
cd /d "%~dp0"

echo ============================================
echo  進捗確認
echo ============================================
echo.

if not exist venv\Scripts\python.exe (
    echo [ERROR] venv が見つかりません。先に setup_windows.bat をダブルクリックしてください。
    pause
    exit /b 1
)

venv\Scripts\python.exe scripts\check_progress.py --k-values 99,64,80

echo.
echo --- プロセスの状態 ---
set LOCKFILE=scripts\output\opponent_count_precision_sweep\.run.lock
if exist "%LOCKFILE%" (
    set LOCKPID=
    for /f "delims=" %%P in (%LOCKFILE%) do set LOCKPID=%%P
    tasklist /FI "PID eq !LOCKPID!" 2>nul | find "!LOCKPID!" >nul
    if errorlevel 1 (
        echo ロックファイルはありますが、そのプロセス(PID !LOCKPID!)は現在動いていません。
        echo 異常終了した可能性がありますが、チェックポイントは残っているはずです。
        echo start_run.bat を実行すれば、続きから自動的に再開されます。
    ) else (
        echo 実行中です(PID !LOCKPID!)。
    )
) else (
    echo 現在、実行中のランはありません(ロックファイルなし)。
)

echo.
echo --- ログの末尾(直近20行) ---
if exist logs\sweep_run.log (
    powershell -NoProfile -Command "Get-Content 'logs\sweep_run.log' -Tail 20"
) else (
    echo ログファイルがまだありません(まだ一度も起動していないか、ログの場所が違います)。
)

echo.
echo 何かキーを押すとこのウィンドウを閉じます。
pause >nul
