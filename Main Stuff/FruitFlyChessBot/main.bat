@echo off
setlocal EnableExtensions
title Fruit Fly Chess
cd /d "%~dp0"

set "VENV_DIR=%~dp0.venv"
set "VENV_PY=%VENV_DIR%\Scripts\python.exe"
if not exist "%VENV_PY%" goto :find_python
"%VENV_PY%" -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 9) else 1)" >nul 2>nul
if not errorlevel 1 goto :check_packages

echo The existing .venv uses Python older than 3.9; leaving it untouched.
echo Trying a separate environment with a newer Python...
set "VENV_DIR=%~dp0.venv-py312"
set "VENV_PY=%VENV_DIR%\Scripts\python.exe"
if exist "%VENV_PY%" goto :check_existing_fallback

:find_python
set "PY="

rem Prefer Python 3.12 through the Windows Python Launcher.
where py >nul 2>nul
if not errorlevel 1 (
    py -3.12 -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 9) else 1)" >nul 2>nul
    if not errorlevel 1 set "PY=py -3.12"
)

rem If the launcher is unavailable, check the python command on PATH.
if not defined PY (
    where python >nul 2>nul
    if not errorlevel 1 (
        python -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 9) else 1)" >nul 2>nul
        if not errorlevel 1 set "PY=python"
    )
)

rem Last fallback: any registered Python 3.9+.
if not defined PY (
    where py >nul 2>nul
    if not errorlevel 1 (
        py -3 -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 9) else 1)" >nul 2>nul
        if not errorlevel 1 set "PY=py -3"
    )
)

if not defined PY goto :python_not_found
if exist "%VENV_PY%" goto :check_existing_fallback

echo Creating local environment with:
%PY% -c "import sys; print(sys.executable); print(sys.version)"
%PY% -m venv "%VENV_DIR%"
if errorlevel 1 goto :venv_failed

goto :check_packages

:check_existing_fallback
"%VENV_PY%" -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 9) else 1)" >nul 2>nul
if not errorlevel 1 goto :check_packages
echo The existing .venv-py312 is also older than Python 3.9.
echo Rename that environment folder before trying again.
goto :failed

:check_packages
"%VENV_PY%" -c "import chess, pygame, numpy" >nul 2>nul
if errorlevel 1 (
    echo Installing the core chess and graphics packages...
    "%VENV_PY%" -m pip install python-chess "pygame>=2.5" "numpy>=1.24"
    if errorlevel 1 goto :install_failed
)

echo Starting Fruit Fly Chess...
"%VENV_PY%" fly_connectome_chess.py %*
set "APP_EXIT=%ERRORLEVEL%"
if not "%APP_EXIT%"=="0" echo Fruit Fly Chess exited with code %APP_EXIT%.
echo.
if not defined FRUIT_FLY_NO_PAUSE pause
exit /b %APP_EXIT%

:python_not_found
echo Could not find Python 3.9 or newer.
echo Checking Python 3.12 explicitly may help; from Command Prompt run:
echo   py -3.12 --version
echo   python --version
echo If Python 3.12 is installed but neither command works, add Python to PATH.
set "APP_EXIT=1"
goto :failed

:venv_failed
echo Could not create the local Python environment.
set "APP_EXIT=1"
goto :failed

:install_failed
echo Dependency installation failed. Check internet access and Python pip.
set "APP_EXIT=1"
goto :failed

:failed
echo.
if not defined FRUIT_FLY_NO_PAUSE pause
exit /b %APP_EXIT%
