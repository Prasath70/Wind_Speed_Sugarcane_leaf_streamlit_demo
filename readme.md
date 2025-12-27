🌱 Smart Agro AI

Wind speed prediction & sugarcane leaf disease detection using machine learning

<p align="center"> <img src="https://img.shields.io/badge/Python-3.10-blue?logo=python" /> <img src="https://img.shields.io/badge/FastAPI-Backend-009688?logo=fastapi" /> <img src="https://img.shields.io/badge/Streamlit-Frontend-FF4B4B?logo=streamlit" /> <img src="https://img.shields.io/badge/TensorFlow-Model-FF6F00?logo=tensorflow" /> <img src="https://img.shields.io/badge/License-MIT-green" /> </p>

Project Structure
```
project/
├─ data/                     # sample leaf images
├─ model/                    # ML models (wind + disease)
├─ src/
│   ├─ __init__.py
│   ├─ ml_server.py          # FastAPI backend
│   └─ Streamlitapp.py       # UI
└─ run.py                    # starts backend + frontend
```
Setup
``conda env create -f dependencies/environment.yml``
``conda activate backend``


(Optional — pip install instead of conda)

``pip install -r dependencies/requirements.txt``

Start both services

Run from the project root:

python run.py


API →`` http://127.0.0.1:8000``

UI → ``http://127.0.0.1:8501``

Stop with ENTER.

Start only the API
``python -m src.ml_server``


Test:

curl http://127.0.0.1:8000

API Endpoints
Task	Method	URL
wind speed prediction	POST	/predict/wind
leaf disease prediction	POST (file)	/predict/image

Example wind request:

{
  "features": [10.2, 42.5, 7.1, 88.0]
}

Expected Output
{"predicted_wind_speed": 12.87}