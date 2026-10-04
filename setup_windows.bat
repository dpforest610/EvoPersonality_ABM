@echo off
chcp 65001 >nul
cd /d "%~dp0"

echo ============================================
echo  初回セットアップ: 精密kスイープ延長ラン
echo ============================================
echo.

where py >nul 2>nul
if %errorlevel%==0 (
    set PY_LAUNCHER=py -3
) else (
    where python >nul 2>nul
    if %errorlevel%==0 (
        set PY_LAUNCHER=python
    ) else (
        echo [ERROR] Python が見つかりません。
        echo.
        echo   https://www.python.org/downloads/ からインストーラーをダウンロードし、
        echo   インストール開始画面の下の方にある
        echo   「Add python.exe to PATH」に必ずチェックを入れてからインストールしてください。
        echo   インストール後、このウィンドウを閉じて setup_windows.bat をもう一度
        echo   ダブルクリックしてください。
        echo.
        pause
        exit /b 1
    )
)

echo [1/6] Python のバージョンを確認しています...
%PY_LAUNCHER% --version
if errorlevel 1 (
    echo [ERROR] Python の起動に失敗しました。上のメッセージを確認してください。
    pause
    exit /b 1
)
echo.

echo [2/6] 仮想環境(venv)を作成しています...
if exist venv (
    echo すでに venv フォルダがあるので、作成はスキップします。
) else (
    %PY_LAUNCHER% -m venv venv
    if errorlevel 1 (
        echo [ERROR] venv の作成に失敗しました。上のエラーメッセージを確認してください。
        pause
        exit /b 1
    )
)
echo OK
echo.

echo [3/6] 依存パッケージをインストールしています(数分かかることがあります)...
venv\Scripts\python.exe -m pip install --upgrade pip
if errorlevel 1 (
    echo [ERROR] pip の更新に失敗しました。インターネット接続を確認してください。
    pause
    exit /b 1
)
venv\Scripts\python.exe -m pip install -r requirements.txt
if errorlevel 1 (
    echo [ERROR] 依存パッケージのインストールに失敗しました。インターネット接続を確認してください。
    pause
    exit /b 1
)
echo OK
echo.

echo [4/6] クラウド側で進行中だったチェックポイントを取得しています...
git fetch origin checkpoint-scratch
if errorlevel 1 (
    echo [情報] checkpoint-scratch の取得に失敗しました(初回・既に取り込み済みなら問題ありません)。続行します。
) else (
    git checkout origin/checkpoint-scratch -- scripts/output/opponent_count_precision_sweep/
    git restore --staged scripts/output/opponent_count_precision_sweep/
    echo OK
)
echo.

echo [5/6] テストを実行しています...
venv\Scripts\python.exe -m pytest -q
if errorlevel 1 (
    echo [ERROR] テストが失敗しました。上の出力をそのまま報告してください。
    pause
    exit /b 1
)
echo OK: 全テスト成功
echo.

echo [6/6] 現在の進捗:
venv\Scripts\python.exe scripts\check_progress.py --k-values 99,64,80
echo.

echo ============================================
echo  セットアップ完了。次は start_run.bat をダブルクリックしてください。
echo ============================================
pause
