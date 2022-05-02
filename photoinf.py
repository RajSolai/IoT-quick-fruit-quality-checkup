import subprocess
import tflite_runtime.interpreter as tflite
import cv2
import numpy as np

FRAME_SOURCE = "http://raspberrypi.local/live"

def make_prediction():
    _c = cv2.VideoCapture(FRAME_SOURCE)
    # Load the model
    interpreter = tflite.Interpreter(model_path="model_densenet.tflite")
    # interpreter = tflite.Interpreter(model_path="model_unquant.tflite")
    interpreter.allocate_tensors()
    _,img = _c.read()
    img = cv2.resize(img, (128, 128))
    input_tensor = np.array(np.expand_dims(img, 0), dtype=np.float32)
    input_index = interpreter.get_input_details()[0]["index"]
    interpreter.set_tensor(input_index, input_tensor)
    interpreter.invoke()
    output_details = interpreter.get_output_details()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    pred = np.squeeze(output_data)
    print(pred)
    return "bad"
