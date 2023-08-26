import cv2, os
from PIL import Image
import numpy as np

path = 'samples'
recognizer = cv2.face.LBPHFaceRecognizer_create()
dectector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")


def Images_And_lable(path):
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
    faceSamples = []
    ids = []
    for imagePath in imagePaths:
        grey_img = Image.open(imagePath).convert('L')
        img_arr = np.array(grey_img, 'uint8')

        id = int(os.path.split(imagePath)[-1].split(".")[1])
        faces = dectector.detectMultiScale(img_arr)
        for (x, y, w, h) in faces:
            faceSamples.append(img_arr[y:y + h, x:x + w])
            ids.append(id)

    return faceSamples, ids


print("Training faces it will take few seconds please wait")
faces, ids = Images_And_lable(path)
recognizer.train(faces, np.array(ids))
recognizer.write('trainer/trainer.yml')
print("Model Train now we can recognize your face.")