import numpy as np
from cv2 import imdecode, IMREAD_COLOR
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel

from ml.model import load_model

model = None
app = FastAPI()


# Data model for the face prediction response
class FacePredictionResponse(BaseModel):
    x1: int
    y1: int
    width: int
    height: int
    age: int


@app.get("/")
def index():
    """
    Root route that returns a welcome message.
    """
    return {"text": "Face Detection and Age Estimation"}


@app.on_event("startup")
def startup_event():
    """
    Function to run during server startup, loads the model for face detection age estimation.
    """
    global model
    model = load_model()


@app.post("/predict")
async def predict_faces(image: UploadFile = File(...)):
    """
    Route for detecting faces and estimating their ages from an image.

    Parameters:
    image: UploadFile - The image uploaded in the request body.

    Returns:
    List of predicted face parameters: (x1, y1) coordinates, width, height, and estimated age.
    """
    contents = await image.read()
    npimg = np.frombuffer(contents, np.uint8)
    image_decoded = imdecode(npimg, IMREAD_COLOR)

    face_predictions = model(image_decoded)
    response = []
    for face_prediction in face_predictions:
        response.append(
            FacePredictionResponse(
                x1=face_prediction.x1,
                y1=face_prediction.y1,
                width=face_prediction.width,
                height=face_prediction.height,
                age=face_prediction.age_prediction,
            )
        )
    return response
