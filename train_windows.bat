@echo off
call .venv\Scripts\activate.bat
python -m local_llm.train
pause
