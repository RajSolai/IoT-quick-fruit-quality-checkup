import subprocess
from PIL import Image, ImageOps
from keras.models import load_model
import cv2
import numpy as np

FRAME_SOURCE = "http://raspberrypi.local:5000/live"

CLASSES = [
    'rottenapples', 'freshbanana', 'freshoranges', 'rottenbanana',
    'freshapples', 'rottenoranges'
]


def make_prediction():
    vid = cv2.VideoCapture(FRAME_SOURCE)
    ret, img = vid.read()
    if not ret:
        return "Camera Unable to read"
    # Load the model
    model = load_model('model_densenet.h5')

    # Create the array of the right shape to feed into the keras model
    # The 'length' or number of images you can put into the array is
    # determined by the first position in the shape tuple, in this case 1.
    data = np.ndarray(shape=(1, 128, 128, 3), dtype=np.float32)
    # Replace this with the path to your image
    # image = Image.open("./input.png")
    # resize the image to a 224x224 with the same strategy as in TM2:
    # resizing the image to be at least 224x224 and then cropping from the center
    size = (128, 128)
    image = cv2.resize(img, size)

    # turn the image into a numpy array
    image_array = np.asarray(image)
    # Normalize the image
    normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
    # Load the image into the array
    data[0] = normalized_image_array

    # run the inference
    prediction = model.predict(data)
    print(prediction)
    # out = []
    # out += prediction[0]
    # out.append(np.argmax(prediction))
    # print(prediction)
    out=np.append(prediction[0],np.argmax(prediction[0]))
    print(out)
    return prediction[0]
