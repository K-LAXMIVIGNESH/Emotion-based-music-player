# 🎵 Emotion-Based Music Recommendation System

An AI-powered music recommendation system that detects facial emotions in real time using a webcam and recommends music based on the detected emotion. The project combines deep learning, computer vision, and automated music playback to create an emotion-aware listening experience.

## ✨ Features

- Real-time facial emotion detection using a webcam.
- Sub-emotion classification using deep learning.
- Two model implementations: **CNN and FastNet**.
- Automatic music recommendations based on detected sub-emotions.
- Emotion-based song queue management.
- Live webcam feed through a Flask web interface.
- Music controls to skip, replay, and stop songs.
- Standalone webcam-based music player.

## 🛠️ Technologies Used

- Python
- TensorFlow and Keras
- OpenCV
- NumPy
- Flask
- Pygame
- HTML, CSS, and JavaScript

## 🧠 Emotion Categories

The system organizes music into seven main emotion categories:

- Angry
- Fear
- Happy
- Neutral
- Sad
- Sleep
- Surprise

Each category can contain multiple sub-emotions, with corresponding songs organized into separate folders.

## ⚙️ System Workflow

1. Capture live video from the webcam.
2. Detect faces using OpenCV and a Haar Cascade classifier.
3. Convert the detected face to grayscale.
4. Resize the face image to `48 × 48` pixels and normalize the pixel values.
5. Predict the sub-emotion using a trained deep learning model.
6. Map the predicted sub-emotion to its main emotion category.
7. Select an MP3 file from the corresponding music folder.
8. Play the selected song using Pygame.
9. Display the live webcam feed and music controls through Flask.

## 📊 Model Performance

The CNN and FastNet models were trained and evaluated on the project's seven-class emotion dataset.

| Model | Final Validation Accuracy | Evaluation Accuracy |
|---|---:|---:|
| CNN | 62.97% | 63% |
| FastNet | 56.45% | 56% |

**Results:** In the recorded experiments, CNN achieved higher classification accuracy than FastNet.

*Note: These values are from the recorded notebook outputs. Performance may vary depending on the dataset split, training configuration, and evaluation procedure.*

## 📂 Project Structure

```text
Emotion-based-music-player/
│
├── app.py
├── Final_code.py
├── README.md
├── Emotion_Based_Music_Recommendation_System (FINAL).pdf
│
├── cnn subemotions/
│   └── CNN training and evaluation notebook
│
├── fastnet subemotions/
│   └── FastNet training and evaluation notebook
│
└── templates/
    └── index.html
```

## 🚀 Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/K-LAXMIVIGNESH/Emotion-based-music-player.git
cd Emotion-based-music-player
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

Activate the environment on Windows:

```bash
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install flask opencv-python numpy tensorflow pygame
```

## 🔧 Configuration

Before running the application:

1. Ensure the trained `.keras` model is available.
2. Ensure `haarcascade_frontalface_default.xml` is available.
3. Update the model and Haar Cascade paths in the Python files.
4. Configure the paths to your local music folders.
5. Organize MP3 files into the appropriate emotion and sub-emotion directories.

Example music folder structure:

```text
music/
├── Happy/
│   └── Meeting Friends/
│       ├── song1.mp3
│       └── song2.mp3
├── Sad/
│   └── Missing Someone/
│       └── song3.mp3
└── Neutral/
    └── Sitting in Class/
        └── song4.mp3
```

**Note:** The current Python code uses local Windows paths. Update these paths to match your computer before running the application.

## ▶️ Running the Application

### Flask Web Application

Run:

```bash
python app.py
```

Open your browser and visit:

```text
http://127.0.0.1:5000
```

### Standalone Webcam Application

Run:

```bash
python Final_code.py
```

Enter the desired song duration when prompted. Press `q` to close the application.

## 🧪 Deep Learning Models

### CNN (Convolutional Neural Network)

The CNN model uses convolutional layers, batch normalization, max pooling, and dropout to extract facial features and classify the input image into one of seven emotion classes.

### FastNet

FastNet explores a lightweight convolutional architecture using depthwise convolutions and related operations to perform facial emotion classification.

## 🎯 Applications

- Emotion-aware music recommendation.
- Personalized music-listening experiences.
- Human-computer interaction.
- Emotion-responsive multimedia applications.

## 🔮 Future Enhancements

- Improve sub-emotion classification accuracy.
- Add personalized playlists and user preferences.
- Optimize inference for real-time performance.
- Improve the user interface and accessibility.
- Evaluate models using precision, recall, F1-score, and inference latency.

## ⚠️ Limitations

- Prediction quality depends on lighting, webcam quality, and model performance.
- Facial expressions may not always reflect a person's actual emotional state.
- A compatible trained model, webcam, and configured music library are required.
- Model files and music tracks may need to be downloaded or configured separately.

## 👨‍💻 Author

**Konkimalla Laxmi Vignesh**

GitHub: [K-LAXMIVIGNESH](https://github.com/K-LAXMIVIGNESH)

## 🔗 Project Repository

[Emotion-Based Music Recommendation System](https://github.com/K-LAXMIVIGNESH/Emotion-based-music-player)

---

*Developed using deep learning and computer vision for emotion-based music recommendation.*
