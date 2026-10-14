@echo off
REM Rakhsh one-click start (Windows)
cd /d "%~dp0"
if not exist .venv (
  py -3 -m venv .venv
  .venv\Scripts\pip install -q -e ".[formats]"
)
.venv\Scripts\rakhsh serve %*
