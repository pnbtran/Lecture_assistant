@echo off
chcp 65001 >nul
if not defined RA_PYTHON set "RA_PYTHON=%~dp0..\runtime\python_env\Scripts\python.exe"
if not defined RA_OLLAMA set "RA_OLLAMA=%~dp0..\runtime\ollama\ollama.exe"
title Tro ly bai giang (dong cua so nay de tat)
cd /d "%~dp0app"
rem Bat may chu Ollama neu chua chay (flash attention + KV cache 8-bit)
set "OLLAMA_FLASH_ATTENTION=1"
set "OLLAMA_KV_CACHE_TYPE=q8_0"
tasklist /FI "IMAGENAME eq ollama.exe" 2>nul | find /I "ollama.exe" >nul
if errorlevel 1 (
  if exist "%RA_OLLAMA%" (start "" /min "%RA_OLLAMA%" serve) else (start "" /min ollama serve)
  timeout /t 3 >nul
)
"%RA_PYTHON%" app.py
echo.
echo App da tat. Nhan phim bat ky de dong cua so.
pause >nul
