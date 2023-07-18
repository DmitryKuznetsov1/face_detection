import numpy as np
from ml.Predictor import Predictor, FacePrediction


def load_model():
    """
    Returns:
        function: A function that takes an image and returns a list of FacePrediction objects.
    """
    model_inst = Predictor()
    model_inst.setup()

    def model(image: np.array) -> list[FacePrediction]:
        return model_inst.predict(image)

    return model
