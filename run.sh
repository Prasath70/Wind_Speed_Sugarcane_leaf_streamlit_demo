#!/bin/bash

echo "Activating Conda environment..."
source ~/miniconda3/etc/profile.d/conda.sh
conda activate smartagro

echo "Starting FastAPI backend..."
python src/ML_Server.py &

echo "Starting Streamlit UI..."
streamlit run src/Streamlitapp.py &

echo "Both services started!"
wait
