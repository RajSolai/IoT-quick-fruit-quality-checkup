import subprocess
from PIL import Image, ImageOps
from keras.models import load_model
from keras.preprocessing import image
import cv2
import numpy as np

FRAME_SOURCE = "http://raspberrypi.local:5000/live"

def make_prediction():
    # save image to show later

    vid = cv2.VideoCapture(FRAME_SOURCE)
    ret, img = vid.read()
    if not ret:
        return "Camera Unable to read"
    # Load the model
    model = load_model('model_new_3.h5')
    data = np.ndarray(shape=(1, 150, 150, 3), dtype=np.float32)
    size = (150, 150)
    img = cv2.resize(img, size)
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = img/255

    images = np.vstack([img])
    classes = model.predict(images, batch_size=10)
    label = np.where(classes[0] > 0.5, 1,0)

    print("OUTPUT IS ",label)
    if label == 0 or label == 0.5:
        return "Fresh Fruit"
    
    return "Rotton Fruit"

