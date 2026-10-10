@echo off
setlocal EnableExtensions
title Fruit Fly Chess Training
cd /d "%~dp0"

echo Starting fast offline self-play training.
echo Defaults: 25 games, up to 160 plies each, no search wait or GUI animation.
echo Existing connectome_weights.npz will be loaded and updated.
echo To change the run, pass options such as: train.bat --games 50 --ply-limit 200

echo.
set "FRUIT_FLY_NO_PAUSE=1"
call "%~dp0main.bat" --offline --train --games 25 --ply-limit 160 --think 0 --speed 0 --weights "%~dp0connectome_weights.npz" --save-weights "%~dp0connectome_weights.npz" %*
set "TRAIN_EXIT=%ERRORLEVEL%"
if not "%TRAIN_EXIT%"=="0" (
    echo.
    echo Training failed with exit code %TRAIN_EXIT%.
    pause
) else (
    echo.
    echo Training complete. Learned weights saved to connectome_weights.npz.
    pause
)
exit /b %TRAIN_EXIT%
