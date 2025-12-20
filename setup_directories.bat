@echo off
echo Creating directories...

if not exist "input\images" (
    mkdir input\images
    echo Created input\images
) else (
    echo input\images already exists
)

if not exist "output\gaussians" (
    mkdir output\gaussians
    echo Created output\gaussians
) else (
    echo output\gaussians already exists
)

if not exist "models" (
    mkdir models
    echo Created models
) else (
    echo models already exists
)

echo.
echo Downloading model file...
powershell -Command "Invoke-WebRequest -Uri 'https://ml-site.cdn-apple.com/models/sharp/sharp_2572gikvuh.pt' -OutFile 'models\sharp_2572gikvuh.pt'"

if exist "models\sharp_2572gikvuh.pt" (
    echo Model downloaded successfully!
) else (
    echo Failed to download model file.
)

echo.
echo Setup complete!
pause
