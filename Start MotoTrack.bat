@echo off
REM Double-click launcher for the MotoTrack desktop window.
REM Keeps the console open so a startup error is readable instead of vanishing.
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
  echo Virtual environment not found.
  echo Run the setup steps in README.md first.
  pause
  exit /b 1
)

".venv\Scripts\python.exe" -m mototrack.cli desktop
if errorlevel 1 (
  echo.
  echo MotoTrack exited with an error code.
  pause
)
endlocal