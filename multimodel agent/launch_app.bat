@echo off
title Alzheimer's Multi-Modal Diagnosis Launcher
echo =======================================================
echo Starting Alzheimer's Multi-Modal Diagnosis App...
echo =======================================================

:: Navigate to the backend directory containing main.py and venv
cd /d "%~dp0backend"

:: Activate the local python virtual environment
echo Activating Python virtual environment...
call .\venv\Scripts\activate.bat

:: Open the default web browser to the server root URL
echo Opening default web browser to http://127.0.0.1:8000...
start http://127.0.0.1:8000

:: Start the Uvicorn FastAPI server on port 8000
echo Starting FastAPI Uvicorn server on port 8000...
uvicorn main:app --host 127.0.0.1 --port 8000

pause
