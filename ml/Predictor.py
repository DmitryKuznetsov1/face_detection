from pathlib import Path

import torch

from ml.face_detection import CascadeClassifier
from ml.age_estimation import AgeRegressor
import numpy as np
from dataclasses import dataclass


@dataclass
class FacePrediction:
    """Class representing a per face predictor result"""

    x1: int
    y1: int
    width: int
    height: int
    age_prediction: int


class Predictor:
    def __init__(self):
        self.detector = None
        self.age_estimator = None

    def setup(self) -> None:
        """
        Set up the predictor by initializing the face detector and age estimator.
        """
        self.detector = CascadeClassifier()

        self.age_estimator = AgeRegressor()
        current_module_path = Path(__file__).resolve()
        weights_folder = current_module_path.parent / "age_estimation/weights"
        weights_file = "2.EfficientNetB5-60.5.pth"
        weights_path = weights_folder / weights_file
        self.age_estimator.load_state_dict(torch.load(weights_path, map_location=torch.device('cpu')))
        self.age_estimator.eval()

    def predict(self, image: np.ndarray) -> list[FacePrediction]:
        """
        Perform face detection and age estimation on the given image.

        Args:
            image (np.ndarray): The input image as a NumPy array.

        Returns:
            list[FacePrediction]: A list of FacePrediction objects containing the predicted face coordinates and ages.
        """
        faces: np.ndarray = self.detector.detect_faces(image)
        output: list[FacePrediction] = []

        for face_coordinates in faces:
            x1, y1, width, height = face_coordinates
            face = image[y1:y1 + height, x1:x1 + width, :]
            age_prediction = self.age_estimator(face)
            output.append(
                FacePrediction(
                    x1=x1,
                    y1=y1,
                    width=width,
                    height=height,
                    age_prediction=age_prediction
                )
            )

        return output
