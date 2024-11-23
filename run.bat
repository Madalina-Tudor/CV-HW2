@echo off
echo Setting up virtual environment...

REM Check if virtual environment folder exists
IF NOT EXIST venv (
    python -m venv venv
    echo Virtual environment created.
) ELSE (
    echo Virtual environment already exists.
)

echo Activating virtual environment...
call venv\Scripts\activate

echo Installing requirements...
pip install -r requirements.txt

echo Running the project...
python code\color_filters.py
