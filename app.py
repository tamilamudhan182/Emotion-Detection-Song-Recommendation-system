from flask import Flask, render_template, request
import numpy as np
import cv2
from keras.models import load_model
import webbrowser
import logging
import time

app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 1

# Setup logging
logging.basicConfig(level=logging.INFO)

# Global info dictionary
info = {}

# Load model and cascade
haarcascade = "haarcascade_frontalface_default.xml"
label_map = ['ANGER', 'NEUTRAL', 'FEAR', 'HAPPY', 'SAD', 'SURPRISE']
logging.info("Loading emotion detection model...")
model = load_model('model.h5')
cascade = cv2.CascadeClassifier(haarcascade)

# Emotion-to-genre mapping (optional enhancement)
emotion_genre_map = {
    'HAPPY': 'pop',
    'SAD': 'acoustic',
    'ANGER': 'rock',
    'FEAR': 'ambient',
    'SURPRISE': 'electronic',
    'NEUTRAL': 'chill'
}

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/choose_singer', methods=["POST"])
def choose_singer():
    info['language'] = request.form['language']
    logging.info(f"Selected language: {info['language']}")
    return render_template('choose_singer.html', data=info['language'])

@app.route('/emotion_detect', methods=["POST"])
def emotion_detect():
    info['singer'] = request.form['singer']
    logging.info(f"Selected singer: {info['singer']}")

    cap = cv2.VideoCapture(0)
    found = False
    roi = None

    while not found:
        ret, frm = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frm, cv2.COLOR_BGR2GRAY)
        faces = cascade.detectMultiScale(gray, 1.4, 1)

        for x, y, w, h in faces:
            found = True
            roi = gray[y:y+h, x:x+w]
            filename = f"static/face_{int(time.time())}.jpg"
            cv2.imwrite(filename, roi)
            break

    cap.release()

    if not found or roi is None:
        logging.warning("No face detected.")
        return render_template("emotion_detect.html", data="No face detected", youtube="#", spotify="#")

    # Predict emotion
    roi = cv2.resize(roi, (48, 48))
    roi = roi / 255.0
    roi = np.reshape(roi, (1, 48, 48, 1))
    prediction = model.predict(roi)
    emotion = label_map[np.argmax(prediction)]
    logging.info(f"Detected emotion: {emotion}")

    # Generate links
    genre = emotion_genre_map.get(emotion, '')
    you_link = f"https://www.youtube.com/results?search_query={info['singer']}+{emotion}+{info['language']}+song"
    spo_link = f"https://open.spotify.com/search/{info['singer']}%20{genre}%20{info['language']}%20song"

    # Open links
    webbrowser.open(you_link)
    webbrowser.open(spo_link)

    return render_template("emotion_detect.html", data=emotion, youtube=you_link, spotify=spo_link)

if __name__ == "__main__":
    app.run(debug=True)