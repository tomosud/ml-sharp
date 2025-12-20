@echo off
echo ========================================
echo Visual Studio Build Tools Installation
echo ========================================
echo.
echo This will open the download page for Visual Studio Build Tools.
echo.
echo Please follow these steps:
echo.
echo 1. Download "Build Tools for Visual Studio 2022"
echo 2. Run the installer
echo 3. Select "Desktop development with C++"
echo 4. Make sure these are checked:
echo    - MSVC v143 - VS 2022 C++ x64/x86 build tools
echo    - Windows SDK (latest version)
echo 5. Install (this may take 15-30 minutes)
echo 6. Restart your command prompt after installation
echo.
pause
echo.
echo Opening download page...
start https://visualstudio.microsoft.com/downloads/
echo.
echo After installation, run setup_env.bat again.
pause
