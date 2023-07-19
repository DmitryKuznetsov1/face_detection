import cv2
import numpy as np


class CascadeClassifier:
    def __init__(self):
        """
        Initializes the FaceDetector.

        """
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

    def detect_faces(
        self,
        image: np.ndarray,
        scale_factor: float = 1.1,
        min_neighbors: int = 5,
        min_size: tuple[int, int] = (30, 30),
        visualize: bool = False,
    ) -> np.ndarray:
        """
        Detects faces in the given image.

        Args:
            image (np.ndarray): Input image in BGR format.
            scale_factor (float): Parameter specifying how much the image size is reduced at each image scale.
            min_neighbors (int): Parameter specifying how many neighbors each candidate rectangle should have
                                 to retain it.
            min_size (Tuple[int, int]): Minimum possible object size.
            visualize (bool): Flag indicating whether to draw bounding boxes on the image.

        Returns:
            np.ndarray: Array representing the coordinates of the detected faces
                                in the format (x, y, width, height).
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=scale_factor, minNeighbors=min_neighbors, minSize=min_size
        )
        if visualize:
            for x, y, w, h in faces:
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 3)

        return faces
