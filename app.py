from flask import Flask, render_template, Response, jsonify
import cv2
import numpy as np
from keras.models import load_model
import pygame
import os
import time
from collections import deque

app = Flask(__name__)

# ===================== Load Sub-Emotion Model =====================
model = load_model(r"C:\Users\prava\OneDrive\Desktop\3rd sem\FAI\subemotions\C__Users_prava_OneDrive_Desktop_3rd sem_FAI_with sleep_best_emotion.multi_model.keras")
face_cascade = cv2.CascadeClassifier(
    r"C:\Users\prava\OneDrive\Desktop\3rd sem\FAI\subemotions\haarcascade_frontalface_default.xml"
)

# ===================== Hardcoded Main Emotion Folders =====================
# Each main folder contains sub-emotion folders
emotion_folders = {
    'angry': r"C:\Users\prava\OneDrive\Desktop\3rd sem\FAI\mp3 music with sleep\Angry",
    'fear': r"C:\Users\prava\OneDrive\Desktop\3rd sem\FAI\mp3 music with sleep\Fear",
    'happy': r"C:\Users\prava\OneDrive\Desktop\3rd sem\FAI\mp3 music with sleep\Happy",
    'neutral': r"C:\Users\prava\OneDrive\Desktop\3rd sem\FAI\mp3 music with sleep\Neutral",
    'sad': r"C:\Users\prava\OneDrive\Desktop\3rd sem\FAI\mp3 music with sleep\Sad",
    'sleep': r"C:\Users\prava\OneDrive\Desktop\3rd sem\FAI\mp3 music with sleep\Sleep",
    'surprise': r"C:\Users\prava\OneDrive\Desktop\3rd sem\FAI\mp3 music with sleep\Suprise"
}

# Detect sub-emotions automatically
sub_to_main = {}
sub_emotions = []

for main, path in emotion_folders.items():
    if os.path.exists(path):
        sub_folders = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
        for sub in sub_folders:
            sub_emotions.append(sub)
            sub_to_main[sub] = main
    else:
        print(f"Main folder not found: {path}")

print("Detected sub-emotions:", sub_emotions)

# ===================== Music =====================
song_index = {}
current_song_name = None
prev_songs = deque(maxlen=5)
emotion_queue = deque()
current_sub = None
last_play_time = 0
play_duration = 30

pygame.mixer.init()

def get_music_file(sub_emotion):
    main_emotion = sub_to_main[sub_emotion]
    dir_path = os.path.join(emotion_folders[main_emotion], sub_emotion)
    if os.path.exists(dir_path):
        files = [f for f in os.listdir(dir_path) if f.lower().endswith(".mp3")]
        files.sort()
        if files:
            if sub_emotion not in song_index:
                song_index[sub_emotion] = 0
            file_path = os.path.join(dir_path, files[song_index[sub_emotion] % len(files)])
            song_index[sub_emotion] += 1
            return file_path, files[song_index[sub_emotion]-1]
    return None, None

def play_music(sub_emotion):
    global current_song_name
    music_file, file_name = get_music_file(sub_emotion)
    if music_file:
        pygame.mixer.music.stop()
        pygame.mixer.music.load(music_file)
        pygame.mixer.music.play()
        current_song_name = file_name
        prev_songs.appendleft({"song": current_song_name, "emotion": sub_emotion})
        print(f"> Playing {current_song_name} for sub-emotion: {sub_emotion}")
    else:
        print(f"No songs for {sub_emotion}")

# ===================== Video Feed =====================
camera = cv2.VideoCapture(0)

def generate_frames():
    global current_sub, last_play_time

    while True:
        success, frame = camera.read()
        if not success:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 3)

        for x, y, w, h in faces:
            face_img = gray[y:y+h, x:x+w]
            resized = cv2.resize(face_img, (48, 48))
            normalized = resized / 255.0
            reshaped = np.reshape(normalized, (1, 48, 48, 1))

            # Predict sub-emotion
            result = model.predict(reshaped, verbose=0)
            label_idx = np.argmax(result)
            sub_emotion = sub_emotions[label_idx]
            main_emotion = sub_to_main[sub_emotion]

            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, f"{main_emotion} - {sub_emotion}", (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            current_time = time.time()
            if current_sub is None or (current_time - last_play_time >= play_duration):
                if emotion_queue:
                    current_sub = emotion_queue.popleft()
                else:
                    current_sub = sub_emotion
                play_music(current_sub)
                last_play_time = time.time()
            else:
                if sub_emotion != current_sub and (not emotion_queue or emotion_queue[-1] != sub_emotion):
                    emotion_queue.append(sub_emotion)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# ===================== Routes =====================
@app.route('/')
def index():
    return render_template('index.html', prev_songs=list(prev_songs), queue=list(emotion_queue))

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/skip_song', methods=['POST'])
def skip_song():
    global current_sub, last_play_time
    pygame.mixer.music.stop()
    if emotion_queue:
        current_sub = emotion_queue.popleft()
        play_music(current_sub)
        last_play_time = time.time()
    else:
        current_sub = None
    return jsonify({"status": "skipped", "prev_songs": list(prev_songs), "queue": list(emotion_queue)})

@app.route('/next_song', methods=['POST'])
def next_song():
    global current_sub, last_play_time
    if current_sub:
        play_music(current_sub)
        last_play_time = time.time()
    return jsonify({"status": "next song", "prev_songs": list(prev_songs), "queue": list(emotion_queue)})

@app.route('/stop_song', methods=['POST'])
def stop_song():
    global current_sub
    pygame.mixer.music.stop()
    current_sub = None
    return jsonify({"status": "stopped", "prev_songs": list(prev_songs), "queue": list(emotion_queue)})

if __name__ == "__main__":
    app.run(debug=True)
