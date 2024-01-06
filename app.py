from flask import Flask, render_template, Response
from flask_sqlalchemy import SQLAlchemy
import cv2

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
db = SQLAlchemy(app)

face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Access the Webcam
video_capture = cv2.VideoCapture(0)

recording = False

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(50), nullable=False)
    image = db.Column(db.LargeBinary, nullable=True)

def detect_bounding_box(frame):
    gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray_image, 1.1, 5, minSize=(40, 40))
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 4)
    return frame

def generate_frames():
    global recording
    while True:
        success, frame = video_capture.read()
        if not success:
            break
        if recording:
            frame = detect_bounding_box(frame)
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/start_recording')
def start_recording():
    global recording
    recording = True
    return "Recording started"

@app.route('/stop_recording/<name>', methods=['POST'])
def stop_recording(name):
    global recording
    recording = False

    with app.app_context():
        db.create_all()  # Move db.create_all() inside the app context
        user = User(name=name, image=cv2.imencode('.jpg', video_capture.read())[1].tobytes())
        db.session.add(user)
        db.session.commit()

    return f"Recording stopped for {name}. Image saved."

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True)
