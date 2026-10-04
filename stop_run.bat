@echo off
chcp 65001 >nul
cd /d "%~dp0"

echo ============================================
echo  停止
echo ============================================
echo.

set LOCKFILE=scripts\output\opponent_count_precision_sweep\.run.lock
if not exist "%LOCKFILE%" (
    echo 現在、実行中のランはありません(ロックファイルなし)。
    pause
    exit /b 0
)

set LOCKPID=
for /f "delims=" %%P in (%LOCKFILE%) do set LOCKPID=%%P

if "%LOCKPID%"=="" (
    echo [ERROR] ロックファイルの内容を読み取れませんでした。
    pause
    exit /b 1
)

tasklist /FI "PID eq %LOCKPID%" 2>nul | find "%LOCKPID%" >nul
if errorlevel 1 (
    echo プロセス(PID %LOCKPID%)は既に動いていません(前回異常終了した可能性があります)。
    echo 特に何もする必要はありません。次回 start_run.bat で続きから再開されます。
    pause
    exit /b 0
)

echo プロセス(PID %LOCKPID%)を停止します...
taskkill /PID %LOCKPID% /T /F
if errorlevel 1 (
    echo [ERROR] 停止に失敗しました。タスクマネージャーから python.exe を手動で終了してください。
    pause
    exit /b 1
)

echo.
echo 停止しました。直前のチャンク(最大500世代、k=99で最大約12分)分の進捗だけ
echo 失われる可能性がありますが、それより前はチェックポイントに保存されているので、
echo 次回 start_run.bat を実行すれば続きから再開されます。
echo.
pause
