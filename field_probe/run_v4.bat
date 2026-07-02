@echo off
rem One-click run of the locked V4-FINAL field probe (GPT-2 small, local CPU).
rem Takes ~2 minutes. Prints the results table, then opens the plot.
cd /d "%~dp0"
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
"C:\Users\amptk\AppData\Local\Programs\Python\Python312\python.exe" v4final.py
if exist v4final_field.png start "" v4final_field.png
echo.
pause
