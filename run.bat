@echo off
echo Activating Conda environment...
call conda activate smartagro

echo Starting FastAPI backend...
start cmd /k "python src\ML_Server.py"

echo Starting Streamlit UI...
start cmd /k "streamlit run src\Streamlitapp.py"

echo Both services started!
pause
