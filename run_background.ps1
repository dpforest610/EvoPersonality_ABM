# start_run.bat から呼ばれるヘルパー。実際のランを、このウィンドウ(と、
# これを呼んだ start_run.bat のウィンドウ)が閉じても生き続ける、独立した
# 非表示プロセスとして起動する。
#
# 設計上の選択: 「このウィンドウを閉じても続く」方式(ユーザーが誤って
# ウィンドウを閉じても処理が止まらない)を選び、「閉じないでください」と
# 表示する最小化ウィンドウ方式は採用しなかった。
#
# 実行したいコマンドは、PowerShellの文字列として直接 Start-Process に渡すの
# ではなく、一度 .cmd ファイルに書き出してから、そのファイルを起動する。
# PowerShell の引用符を cmd.exe 側でもう一度解釈させる形は、入れ子の引用符が
# 崩れやすい典型的な失敗パターンのため、避けている。

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
    Write-Host "[ERROR] venv が見つかりません。先に setup_windows.bat をダブルクリックしてください。"
    exit 1
}

New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

$LauncherLines = @(
    "@echo off",
    "`"$PythonExe`" `"$ScriptPath`" --workers $Workers --status-interval-seconds 3600 > `"$LogPath`" 2>&1"
)
Set-Content -Path $LauncherPath -Value $LauncherLines -Encoding ASCII

Start-Process -FilePath $LauncherPath -WorkingDirectory $RepoRoot -WindowStyle Hidden

Write-Host "起動コマンドを送信しました。数秒待って状態を確認します..."
Start-Sleep -Seconds 3

$Tail = ""
if (Test-Path $LogPath) {
    $Tail = (Get-Content $LogPath -Tail 10 -ErrorAction SilentlyContinue) -join "`n"
}

if ($Tail -match "already holds the lock") {
    Write-Host ""
    Write-Host "=============================================="
    Write-Host " すでに実行中です。多重には起動していません。"
    Write-Host "=============================================="
    Write-Host $Tail
} else {
    Write-Host ""
    Write-Host "=============================================="
    Write-Host " 起動しました。このウィンドウは閉じて構いません。"
    Write-Host " ランは裏側(非表示のプロセス)で動き続けます。"
    Write-Host " 進捗確認: check_progress.bat"
    Write-Host " 停止:     stop_run.bat"
    Write-Host "=============================================="
    if ($Tail) {
        Write-Host ""
        Write-Host "--- ログ(直近) ---"
        Write-Host $Tail
    }
}
