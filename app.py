from flask import Flask, render_template, Response, request, redirect, url_for, session, flash
import cv2
import pytesseract
import os
import re
import threading
import time
from werkzeug.utils import secure_filename
import numpy as np
from collections import Counter
import mysql.connector
from mysql.connector import Error
from werkzeug.security import generate_password_hash, check_password_hash
import Levenshtein
from config import Config

app = Flask(__name__)
app.config.from_object(Config)
output_dir = 'static/output_frames'
uploaded_videos_dir = 'static/uploads'

# Ensure the output directories exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
if not os.path.exists(uploaded_videos_dir):
    os.makedirs(uploaded_videos_dir)
def connect_to_mysql():
    try:
        conn = mysql.connector.connect(
            host=app.config['MYSQL_HOST'],
            user=app.config['MYSQL_USER'],
            password=app.config['MYSQL_PASSWORD'],
            database=app.config['MYSQL_DB']
        )
        if conn.is_connected():
            return conn
    except Error as e:
        print(e)
        return None

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        connection = connect_to_mysql()
        if connection:
            cursor = connection.cursor(dictionary=True)
            cursor.execute("SELECT * FROM users WHERE username = %s", (username,))
            user = cursor.fetchone()
            cursor.close()
            connection.close()

            if user and check_password_hash(user['password'], password):
                session['username'] = username
                return render_template('Upload.html')
            else:
                flash('Invalid username or password')
    
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        location = request.form['location']

        connection = connect_to_mysql()
        if connection:
            cursor = connection.cursor(dictionary=True)
            cursor.execute("SELECT * FROM users WHERE username = %s", (username,))
            existing_user = cursor.fetchone()

            if existing_user:
                flash('Username already exists')
            else:
                hashed_password = generate_password_hash(password)
                cursor.execute("INSERT INTO users (username, email, password, location) VALUES (%s, %s, %s, %s)",
                               (username, email, hashed_password, location))
                connection.commit()
                flash('Registration successful')
                return redirect(url_for('login'))

            cursor.close()
            connection.close()
    
    return render_template('register.html')


@app.route('/logout')
def logout():
    session.pop('username', None)
    return redirect(url_for('login'))





# Global variable to keep track of the running thread
process_thread = None
process_lock = threading.Lock()
stop_event = threading.Event()

app.config['UPLOAD_FOLDER'] = 'static/uploads/'

# Set the Tesseract OCR path
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Update with your Tesseract executable path

# Paths to YOLO configuration, weights, and COCO class names
YOLO_CONFIG_PATH = 'yolov4.cfg'
YOLO_WEIGHTS_PATH = 'yolov4.weights'
YOLO_CLASSES_PATH = 'coco.names'
NUMBER_PLATE_CASCADE_PATH = 'haarcascade_russian_plate_number.xml'

# Load YOLO model
net = cv2.dnn.readNet(YOLO_WEIGHTS_PATH, YOLO_CONFIG_PATH)
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Load the COCO class labels YOLO was trained on
with open(YOLO_CLASSES_PATH, "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Load the number plate detection cascade
plate_cascade = cv2.CascadeClassifier(NUMBER_PLATE_CASCADE_PATH)

# Global variables for threading
output_frame = None
output_lock = threading.Lock()

def detect_cars(frame):
    height, width = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    class_ids = []
    confidences = []
    boxes = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5 and classes[class_id] == "car":
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    return [boxes[i] for i in indexes]

def detect_number_plate(car_roi):
    gray = cv2.cvtColor(car_roi, cv2.COLOR_BGR2GRAY)
    plates = plate_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(25, 25))
    return plates

def is_valid_number_plate(text):
    return bool(re.match(r'^[A-Z][A-Z\d\s-]*[A-Z\d\s-]*$', text)) and len(text) > 5

def process_video(file_path):
    global output_frame, output_lock, stop_event
    cap = cv2.VideoCapture(file_path)
    frames_with_strings = {}

    while cap.isOpened() and not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            break

        cars = detect_cars(frame)
        for (x, y, w, h) in cars:
            car_roi = frame[y:y+h, x:x+w]
            plates = detect_number_plate(car_roi)
            for (px, py, pw, ph) in plates:
                number_plate_roi = car_roi[py:py+ph, px:px+pw]
                number_plate_text = pytesseract.image_to_string(number_plate_roi, config='--psm 8')
                number_plate_text = re.sub(r'^\d+', '', number_plate_text)
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                if is_valid_number_plate(number_plate_text):
                    cv2.rectangle(car_roi, (px, py), (px+pw, py+ph), (255, 0, 0), 2)
                    cv2.putText(frame, number_plate_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
                    filename = re.sub(r'\W+', '', number_plate_text.upper())
                    frame_name = os.path.join('static/output_frames', f'{filename}.jpg')
                    cv2.imwrite(frame_name, frame)
                    frames_with_strings[frame_name] = number_plate_text.upper()

        with output_lock:
            output_frame = frame.copy()

    cap.release()

@app.route('/Upload')
def index2():
    return render_template('Upload.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    global process_thread, process_lock, stop_event

    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        with process_lock:
            if process_thread and process_thread.is_alive():
                stop_event.set()
                process_thread.join()
            stop_event.clear()
            process_thread = threading.Thread(target=process_video, args=(file_path,))
            process_thread.daemon = True
            process_thread.start()

        return redirect(url_for('process'))
    return redirect(request.url)

@app.route('/process')
def process():
    return render_template('process.html')

def generate():
    global output_frame, output_lock
    while True:
        with output_lock:
            if output_frame is None:
                continue
            (flag, encoded_image) = cv2.imencode(".jpg", output_frame)
            if not flag:
                continue
        yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encoded_image) + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/find_numberplate', methods=['GET', 'POST'])
def find_numberplate():
    image = None
    if request.method == 'POST':
        numberplate = request.form['numberplate']
        image = find_image_by_numberplate(numberplate)
    return render_template('find_numberplate.html', image=image)

def calculate_similarity(a, b):
    return Levenshtein.ratio(a, b)

def find_image_by_numberplate(numberplate):
    folder = os.path.join(app.root_path, 'static', 'output_frames')
    images = os.listdir(folder)
    best_match = None
    best_similarity = 0.8  # Set threshold to 80%

    for image in images:
        filename_without_extension = os.path.splitext(image)[0]
        similarity = calculate_similarity(numberplate.lower(), filename_without_extension.lower())
        if similarity > best_similarity:
            best_similarity = similarity
            best_match = image

    return best_match

if __name__ == '__main__':
    app.run(debug=True)
