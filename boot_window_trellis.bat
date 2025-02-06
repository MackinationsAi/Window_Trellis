git pull
@echo off
setlocal EnableDelayedExpansion

:: Suppress Python warnings
set PYTHONWARNINGS=ignore
set PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32

:: Activate conda environment
call conda activate window_trellis 2>nul
if !ERRORLEVEL! NEQ 0 (
    echo Error: Failed to activate conda environment 'window_trellis'
    echo Please ensure the environment is created and conda is initialized
    pause
    exit /b 1
)

:: Set required environment variables
set ATTN_BACKEND=flash-attn
set SPCONV_ALGO=native

:: Create a temporary Python script to modify warning filters
echo import warnings > suppress_warnings.py
echo import os >> suppress_warnings.py
echo warnings.filterwarnings('ignore') >> suppress_warnings.py
echo os.environ['PYTHONWARNINGS'] = 'ignore' >> suppress_warnings.py
echo exec(open('./app.py').read()) >> suppress_warnings.py

:: Launch the application with suppressed warnings
echo Booting up Window_Trellis...
echo This may take several minutes on first launch...
python suppress_warnings.py

:: Clean up
del suppress_warnings.py

:: Keep window open if there's an error
if !ERRORLEVEL! NEQ 0 (
    echo.
    echo An error occurred while running Window_Trellis
    pause
)

endlocal
