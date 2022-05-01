from flask import Flask, request
from flask import render_template
from sensorinf import predict_model, train_model
from photoinf import make_prediction, take_photo

app = Flask(__name__)

sensor_data = 0
cam = cv2.VideoCapture(0)

def gen_frames():
    while True:
      succ, frame = cam.read()
      if not succ:
        break
      else:
        ret, buff = cv2.imencode('.jpg',frame)
        frame = buff.tobytes()
        yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + bytearray(frame) + b'\r\n')

@app.route('/live')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route("/detectsensor")
def detect_sensor():
    input_data = [[sensor_data]]
    return predict_model(input_data)


@app.route("/detect")
def detect_image():
    return make_prediction()


@app.route("/init")
def initialize():
    train_model()
    return "Done"

# web hook for sensor data

@app.route("/sensor", methods=['POST'])
def get_sensor_data():
    global sensor_data
    print("Got Data From ESP8266")
    sensor_data = float(request.get_json()['data'])
    return str(sensor_data)


@app.route("/")
def render_main():
    return render_template('main.html')


app.run(host='0.0.0.0', port=5000)
