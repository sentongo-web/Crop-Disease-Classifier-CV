@echo off
REM ============================================================
REM  Crop Disease Classifier — Windows Setup Script
REM  Creates virtual environment and installs CPU-only deps
REM ============================================================

echo.
echo ============================================================
echo  Crop Disease Classifier Setup
echo ============================================================
echo.

REM Check Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found. Install Python 3.10+ from https://python.org
    pause
    exit /b 1
)

echo [1/4] Creating virtual environment...
python -m venv venv
if errorlevel 1 (
    echo [ERROR] Failed to create virtual environment.
    pause
    exit /b 1
)

echo [2/4] Activating virtual environment...
call venv\Scripts\activate.bat

echo [3/4] Upgrading pip...
python -m pip install --upgrade pip --quiet

echo [4/4] Installing dependencies (CPU-only PyTorch)...
pip install -r requirements.txt
if errorlevel 1 (
    echo [ERROR] Dependency installation failed. Check requirements.txt.
    pause
    exit /b 1
)

echo.
echo ============================================================
echo  Setup complete!
echo ============================================================
echo.
echo  Next steps:
echo.
echo  1. Activate the environment:
echo        venv\Scripts\activate
echo.
echo  2. Create a sample dataset (quick test, no download needed):
echo        python data\download_dataset.py --source sample --output data\sample
echo.
echo  3. Train the model on sample data:
echo        python src\train.py --data data\sample --arch mobilenet_v3_small --epochs 5 --batch-size 8
echo.
echo  4. (Optional) Download full PlantVillage dataset from HuggingFace:
echo        python data\download_dataset.py --source huggingface --output data\plantvillage
echo        python src\train.py --data data\plantvillage --arch mobilenet_v3_small --epochs 15
echo.
echo  5. Start the web app:
echo        python scripts\run_local.py
echo        Open http://localhost:8000 in your browser
echo.
pause
