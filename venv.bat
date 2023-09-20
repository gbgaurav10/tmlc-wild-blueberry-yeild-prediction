@echo off

:: Define your virtual environment name (change it if needed)
set "venv_name=blueberry"

:: Create a virtual environment
python -m venv "%venv_name%"

:: Activate the virtual environment
call "%venv_name%\Scripts\activate"

:: Install dependencies from requirements.txt
pip install -r requirements.txt

:: Deactivate the virtual environment
deactivate

echo Virtual environment created and requirements installed.
