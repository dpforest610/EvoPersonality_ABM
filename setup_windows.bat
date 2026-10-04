@echo off
cd /d "%~dp0"

echo ============================================
echo  Setup: k-sweep extension run (first time only)
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
        echo [ERROR] Python was not found.
        echo.
        echo   Go to https://www.python.org/downloads/ and download the installer.
        echo   On the FIRST screen of the installer, make sure you check the box
        echo   "Add python.exe to PATH" before clicking Install.
        echo   After installing, close this window and double-click
        echo   setup_windows.bat again.
        echo.
        pause
        exit /b 1
    )
)

echo [1/6] Checking Python version...
%PY_LAUNCHER% --version
if errorlevel 1 (
    echo [ERROR] Failed to run Python. See the message above.
    pause
    exit /b 1
)
echo.

echo [2/6] Creating virtual environment (venv)...
if exist venv (
    echo venv already exists, skipping.
) else (
    %PY_LAUNCHER% -m venv venv
    if errorlevel 1 (
        echo [ERROR] Failed to create venv. See the error message above.
        pause
        exit /b 1
    )
)
echo OK
echo.

echo [3/6] Installing dependencies (this can take a few minutes)...
venv\Scripts\python.exe -m pip install --upgrade pip
if errorlevel 1 (
    echo [ERROR] Failed to upgrade pip. Check your internet connection.
    pause
    exit /b 1
)
venv\Scripts\python.exe -m pip install -r requirements.txt
if errorlevel 1 (
    echo [ERROR] Failed to install dependencies. Check your internet connection.
    pause
    exit /b 1
)
echo OK
echo.

echo [4/6] Fetching in-progress checkpoints from the cloud run...
git fetch origin checkpoint-scratch
if errorlevel 1 (
    echo [INFO] Could not fetch checkpoint-scratch. This is fine if it does not
    echo exist yet, or you already have these checkpoints. Continuing.
) else (
    git checkout origin/checkpoint-scratch -- scripts/output/opponent_count_precision_sweep/
    git restore --staged scripts/output/opponent_count_precision_sweep/
    echo OK
)
echo.

echo [5/6] Running tests...
venv\Scripts\python.exe -m pytest -q
if errorlevel 1 (
    echo [ERROR] Tests failed. Please report the output above exactly as shown.
    pause
    exit /b 1
)
echo OK: all tests passed
echo.

echo [6/6] Current progress:
venv\Scripts\python.exe scripts\check_progress.py --k-values 99,64,80
echo.

echo ============================================
echo  Setup complete. Next, double-click start_run.bat
echo ============================================
pause
