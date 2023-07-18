import numpy as np
from cv2 import imdecode, IMREAD_COLOR
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel

from ml.model import load_model

model = None
app = FastAPI()


class FacePredictionResponse(BaseModel):
    x1: int
    y1: int
    width: int
    height: int
    age: int


# create a route
@app.get("/")
def index():
    return {"text": "Face Detection and Age Estimation"}


# Register the function to run during startup
@app.on_event("startup")
def startup_event():
    global model
    model = load_model()


@app.get("/predict")
async def predict_faces(image: UploadFile = File(...)):
    contents = await image.read()
    npimg = np.frombuffer(contents, np.uint8)
    image_decoded = imdecode(npimg, IMREAD_COLOR)

    face_predictions = model(image_decoded)
    response = []
    for face_prediction in face_predictions:
        response.append(
            FacePredictionResponse(x1=face_prediction.x1,
                                   y1=face_prediction.y1,
                                   width=face_prediction.width,
                                   height=face_prediction.height,
                                   age=face_prediction.age_prediction))
    return response
