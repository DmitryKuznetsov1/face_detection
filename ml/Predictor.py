import configparser
from pathlib import Path

import torch

from ml.face_detection import CascadeClassifier
from ml.age_estimation import EfficientNetRegressor
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
        self.current_module_dir = Path(__file__).resolve().parent

    def setup(self) -> None:
        """
        Set up the predictor by initializing the face detector and age estimator.
        """
        config = configparser.ConfigParser()
        config.read(self.current_module_dir / "model_config.ini")

        self.detector = CascadeClassifier()

        self.age_estimator = EfficientNetRegressor()
        age_estimation_weights_folder = self.current_module_dir / "age_estimation/weights"
        age_estimation_weights_file = config.get("AGE ESTIMATION", 'weights')
        weights_path = age_estimation_weights_folder / age_estimation_weights_file
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
