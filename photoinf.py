import subprocess
from keras.models import load_model
from PIL import Image, ImageOps
import cv2
import numpy as np


# This function is used to capture a picture using opencv on system

def take_photo():
    cap = cv2.VideoCapture(0)
    _,frame = cap.read() 
    cv2.imwrite('./input.png',frame)
    cap.release()


def make_prediction():
    # Load the model
    model = load_model('keras_model.h5')

    # Create the array of the right shape to feed into the keras model
    # The 'length' or number of images you can put into the array is
    # determined by the first position in the shape tuple, in this case 1.
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    # Replace this with the path to your image
    image = Image.open("./input.png")
    # resize the image to a 224x224 with the same strategy as in TM2:
    # resizing the image to be at least 224x224 and then cropping from the center
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.ANTIALIAS)

    # turn the image into a numpy array
    image_array = np.asarray(image)
    # Normalize the image
    normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
    # Load the image into the array
    data[0] = normalized_image_array

    # run the inference
    prediction = model.predict(data)
    prediction = prediction[0]
    # print(prediction[0])
    if (prediction[0] > 0.5 and prediction[1] < 0.5):
        return "good"
    if (prediction[1] > 0.5 and prediction[0] < 0.5):
        return "bad"
    return "Please Retake Image"