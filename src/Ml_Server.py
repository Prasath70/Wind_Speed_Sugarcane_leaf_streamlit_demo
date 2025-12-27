from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
import numpy as np
from keras.models import model_from_json, load_model
from keras.utils import load_img, img_to_array
import uvicorn

app = FastAPI()

def load_model_image(model_path: str):
    model = load_model(model_path,compile=False)
    return model

def predict_wind_speed(model, features: list[float]):
    x = np.array(features).reshape(1, -1)
    x = x / 2015
    pred = model.predict(x)
    return float(pred[0][0])


def predict_disease(model, img_file, class_names: list[str]):
    img = load_img(img_file, target_size=(64, 64))
    img_array = img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    pred = model.predict(img_array)
    class_index = int(np.argmax(pred[0]))
    return class_names[class_index], float(pred[0][class_index])


# ------------------------------
# Load models once at startup
# ------------------------------
wind_model = load_model_image("model/wind_prediction.keras")
image_model = load_model_image("model/sugarcaneleafprediction.h5")
class_labels = ["Healthy", "Mosaic", "RedRot", "Rust", "Yellow"]


# ------------------------------
# Request model for wind-speed
# ------------------------------
class WindRequest(BaseModel):
    features: list[float]


# ------------------------------
# API Routes
# ------------------------------
@app.post("/predict/wind")
def predict_wind(data: WindRequest):
    result = predict_wind_speed(wind_model, data.features)
    return {"predicted_wind_speed": result}


@app.post("/predict/image")
async def predict_image_api(file: UploadFile = File(...)):
    contents = await file.read()

    temp_path = "uploaded_image.jpg"
    with open(temp_path, "wb") as f:
        f.write(contents)

    label, confidence = predict_disease(image_model, temp_path, class_labels)
    return {
        "predicted_label": label,
        "confidence": confidence
    }


# ------------------------------
# root for test
# ------------------------------
@app.get("/")
def root():
    return {"status": "API running"}

if __name__ == "__main__":
    uvicorn.run(
        "src.Ml_Server:app",
        host="127.0.0.1",
        port=8000,
        reload=True
    )
