# Gesture-Based System Controller (ML Project)

## 📌 Overview

This project implements a real-time gesture-based system controller using computer vision and machine learning.
It allows users to control the mouse (move, click, drag) using hand gestures captured via a webcam.

---

## 🚀 Features

* Real-time hand tracking using MediaPipe
* Custom gesture dataset collection
* Machine Learning-based gesture recognition (MLP Classifier)
* Cursor control using index finger
* Pinch gesture for mouse click
* Fist gesture for drag and drop
* Gesture smoothing using sliding window (history buffer)
* Stable and responsive interaction

---

## 🧠 How It Works

### 1. Hand Detection

* MediaPipe detects 21 hand landmarks from webcam input

### 2. Data Collection

* Custom dataset created using recorded hand gestures
* Each sample stores normalized landmark coordinates

### 3. Model Training

* Trained using Scikit-learn MLPClassifier
* Input: 42 features (x, y for 21 landmarks)
* Output: Gesture class (open, fist, point, pinch)

### 4. Real-Time Prediction

* Landmarks → ML model → predicted gesture
* Uses sliding window (history buffer) for stability

### 5. System Control

* Index finger → cursor movement
* Pinch → mouse click
* Fist → drag and drop

---

## 🛠️ Tech Stack

* Python
* OpenCV
* MediaPipe
* Scikit-learn
* PyAutoGUI

---

## 📂 Project Structure

```
gesture-system-controller/
│
├── main.py                # Real-time gesture control
├── train_model.py        # Model training script
├── detection/            # Hand detection module
├── utils/                # Helper functions
├── data/                 # Dataset (ignored in repo)
├── gesture_model.pkl     # Trained model (optional)
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation

```bash
git clone https://github.com/utkarsh-rau/gesture-system-controller.git
cd gesture-system-controller
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

## ▶️ Run the Project

```bash
python main.py
```

---

## 🎮 Gesture Mapping

| Gesture              | Action      |
| -------------------- | ----------- |
| Open Hand            | Idle        |
| Point (Index Finger) | Move Cursor |
| Pinch                | Mouse Click |
| Fist                 | Drag        |

---

## 📊 Model Training

```bash
python train_model.py
```

---

## ⚠️ Limitations

* Performance depends on lighting conditions
* Requires camera stability
* Gesture accuracy depends on dataset quality

---

## 🚀 Future Improvements

* Add scroll gesture
* Volume control using pinch distance
* Deep learning-based model (CNN / LSTM)
* Multi-hand support
* UI feedback system

---

## 👨‍💻 Author

Utkarsh Raut

---
