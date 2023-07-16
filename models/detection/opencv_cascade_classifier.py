import cv2


class CascadeClassifier:
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    def predict_bboxes(self, image, scale_factor=1.1, min_neighbors=5, min_size=(30, 30), visualize=False):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=scale_factor,
                                                   minNeighbors=min_neighbors, minSize=min_size)
        if visualize:
            for (x, y, w, h) in faces:
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 3)

        return image, faces