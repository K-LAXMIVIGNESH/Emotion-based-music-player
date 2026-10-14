import cv2
import numpy as np
from tensorflow.keras.models import load_model
import pygame
import os
import time
from collections import deque

# ===================== Load Trained Sub-Emotion Model =====================
model_path = r"C:\Users\prava\OneDrive\Desktop\3rd sem\FAI\subemotions\cnn subemotions\C__Users_prava_OneDrive_Desktop_3rd sem_FAI_with sleep_best_emotion.multi_model.keras"
model = load_model(model_path)
print("Loaded model:", model_path)

# Haar cascade for face detection
face_cascade = cv2.CascadeClassifier(
    r"C:\Users\prava\OneDrive\Desktop\3rd sem\FAI\subemotions\haarcascade_frontalface_default.xml"
)

# ===================== Hardcoded main emotion folders =====================
# Each main emotion folder contains sub-emotion folders with songs
emotion_folders = {
    'angry': r"C:\Users\prava\OneDrive\Desktop\3rd sem\FAI\mp3 music with sleep\Angry",
    'fear': r"C:\Users\prava\OneDrive\Desktop\3rd sem\FAI\mp3 music with sleep\Fear",
    'happy': r"C:\Users\prava\OneDrive\Desktop\3rd sem\FAI\mp3 music with sleep\Happy",
    'neutral': r"C:\Users\prava\OneDrive\Desktop\3rd sem\FAI\mp3 music with sleep\Neutral",
    'sad': r"C:\Users\prava\OneDrive\Desktop\3rd sem\FAI\mp3 music with sleep\Sad",
    'sleep': r"C:\Users\prava\OneDrive\Desktop\3rd sem\FAI\mp3 music with sleep\Sleep",
    'surprise': r"C:\Users\prava\OneDrive\Desktop\3rd sem\FAI\mp3 music with sleep\Suprise"
}

# Automatically detect all sub-emotion folders
sub_to_main = {}
sub_emotions = []

for main_emotion, main_path in emotion_folders.items():
    if os.path.exists(main_path):
        sub_folders = [d for d in os.listdir(main_path) if os.path.isdir(os.path.join(main_path, d))]
        for sub in sub_folders:
            sub_emotions.append(sub)
            sub_to_main[sub] = main_emotion
    else:
        print(f"Main emotion folder not found: {main_path}")

print("Detected sub-emotions:", sub_emotions)

# ===================== Music Setup =====================
song_index = {}

def get_music_file(sub_emotion):
    """Get next MP3 file for sub-emotion in main/sub folder structure"""
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
            return file_path
    else:
        print(f"Directory not found: {dir_path}")
    return None

# Initialize pygame mixer
pygame.mixer.init()

def play_music(sub_emotion):
    """Play music for detected sub-emotion"""
    music_file = get_music_file(sub_emotion)
    if music_file:
        pygame.mixer.music.stop()
        try:
            pygame.mixer.music.load(music_file)
            pygame.mixer.music.play()
            print(f"> Playing {music_file} for sub-emotion: {sub_emotion} (main: {sub_to_main[sub_emotion]})")
        except Exception as e:
            print(f"Error playing {music_file}: {e}")
    else:
        print(f"No music file to play for {sub_emotion}")

# ===================== Webcam Setup =====================
video = cv2.VideoCapture(0)

user_input = input("Enter song duration in seconds (default=30): ").strip()
play_duration = int(user_input) if user_input.isdigit() else 30
print(f"Song duration set to {play_duration} seconds")

current_sub = None
last_play_time = 0
sub_queue = deque()

while True:
    ret, frame = video.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 3)

    for x, y, w, h in faces:
        face_img = gray[y:y+h, x:x+w]
        resized = cv2.resize(face_img, (48, 48))
        normalized = resized / 255.0
        reshaped = np.reshape(normalized, (1, 48, 48, 1))

        # Predict Sub-Emotion
        result = model.predict(reshaped, verbose=0)
        label_idx = np.argmax(result)
        sub_emotion = sub_emotions[label_idx]
        main_emotion = sub_to_main[sub_emotion]

        # Draw bounding box & text
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)
        cv2.putText(frame, f"{main_emotion} - {sub_emotion}", (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

        current_time = time.time()

        # Play music if none playing or duration expired
        if current_sub is None or (current_time - last_play_time >= play_duration):
            if sub_queue:
                current_sub = sub_queue.popleft()
            else:
                current_sub = sub_emotion
            play_music(current_sub)
            last_play_time = time.time()
        else:
            # Queue next sub-emotion if different
            if sub_emotion != current_sub and (not sub_queue or sub_queue[-1] != sub_emotion):
                sub_queue.append(sub_emotion)
                print(f"Queued sub-emotion: {sub_emotion}")

    cv2.imshow("Live Sub-Emotion Player", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):  # Quit
        break
    elif key == ord('n'):  # Skip to next in queue
        pygame.mixer.music.stop()
        if sub_queue:
            current_sub = sub_queue.popleft()
            play_music(current_sub)
            last_play_time = time.time()
        else:
            current_sub = None

video.release()
cv2.destroyAllWindows()
pygame.mixer.quit()
