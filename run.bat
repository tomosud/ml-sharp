@echo off
echo Activating virtual environment...
call .venv\Scripts\activate

echo.
echo Running SHARP prediction...

sharp predict -i input\images -o output\gaussians -c models\sharp_2572gikvuh.pt

rem sharp render -i output\gaussians -o output\renderings

echo.
echo Prediction complete!
pause
