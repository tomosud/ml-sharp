@echo off
echo Activating virtual environment...
call .venv\Scripts\activate

echo.
echo Upgrading pip, setuptools, and wheel...
.venv\Scripts\python.exe -m pip install --upgrade pip setuptools wheel

echo.
echo Installing requirements...
.venv\Scripts\python.exe -m pip install -r requirements.txt

echo.
echo Uninstalling existing PyTorch...
.venv\Scripts\python.exe -m pip uninstall -y torch torchvision torchaudio

echo.
echo Installing PyTorch with CUDA 12.8 support...
.venv\Scripts\python.exe -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

echo.
echo Setup complete!
pause
