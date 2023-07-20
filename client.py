import requests
import cv2
import os

url = "http://localhost:80/predict"
img_path = "data/img3.jpg"

with open(img_path, "rb") as image_file:
    files = {"image": image_file}
    response = requests.post(url, files=files)

    if response.status_code == 200:
        result_data = response.json()

        image = cv2.imread(img_path)
        image_copy = image.copy()

        for obj in result_data:
            x1, y1, width, height, age = obj["x1"], obj["y1"], obj["width"], obj["height"], obj["age"]

            cv2.rectangle(image_copy, (x1, y1), (x1 + width, y1 + height), (0, 255, 0), 4)

            text = f"Age: {age}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = min(width, height) / 200.0
            thickness = max(int(round(font_scale * 2.3)), 2)

            text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
            cv2.putText(image_copy, text, (x1, y1 + height + text_size[1]), font, font_scale, (255, 255, 255), thickness)

        folder, filename = os.path.split(img_path)
        filename_without_extension, extension = os.path.splitext(filename)
        new_filename = f"{filename_without_extension}-pred{extension}"
        new_img_path = os.path.join(folder, new_filename)
        cv2.imwrite(new_img_path, image_copy)

    else:
        print("Ошибка:", response.status_code)
