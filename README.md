# 🚧 Pothole & Speed Breaker Detection System

## 📌 Project Overview

This project is an intelligent road safety system that detects **potholes and speed breakers** using **deep learning and image processing**.

The system analyzes road images (live camera or stored images) and classifies them into:

* Normal Road
* Pothole
* Speed Breaker

It uses a **CNN model based on MobileNetV2** for fast and accurate detection.

---

## 🎯 Objectives

* Detect road hazards automatically
* Reduce accidents caused by potholes
* Provide real-time alerts to users
* Replace manual road inspection

---

## ⚙️ Features

* 📷 Real-time detection using webcam
* 🖼️ Image-based detection
* 🤖 Deep learning model (MobileNetV2)
* 🚨 Alert system for potholes & speed breakers
* 🖥️ Simple GUI using Tkinter
* ⚡ Fast and efficient processing

---

## 🛠️ Technologies Used

* Python
* OpenCV
* TensorFlow / Keras
* NumPy
* Tkinter

---

## 🧠 How It Works

1. Capture image from camera or dataset
2. Preprocess image (resize, normalize)
3. Pass image to trained CNN model
4. Model predicts road condition
5. Display result with alert

---

## 📂 Project Structure

project_root/
│
├── dataset/
│   ├── train/
│   │   ├── normal/
│   │   ├── pothole/
│   │   └── speedbreaker/
│   └── validation/
│
├── models/
│   └── road_detection_model.h5
│
├── debug_saves/
│
├── detect_image.py
├── live_predict_image.py
├── train_model.py
├── main.py
│
├── requirements.txt
└── README.md

---

## 💻 Installation (Step-by-Step)

### 1️⃣ Clone the repository



### 2️⃣ Open project folder

cd your-repo-name

### 3️⃣ Create virtual environment

python -m venv venv

### 4️⃣ Activate environment

venv\Scripts\activate

### 5️⃣ Install dependencies

pip install -r requirements.txt

---

## ▶️ How to Run

### 🔹 Run main application

python main.py

### 🔹 Run live detection (camera)

python live_predict_image.py

### 🔹 Run image detection

python detect_image.py

### 🔹 Train model

python train_model.py

---

## 📊 Output

* Detects potholes and speed breakers
* Shows prediction with confidence
* Displays alerts on screen

---

## 💡 System Requirements

* OS: Windows / Linux
* RAM: 8GB (recommended)
* Processor: i3 or above
* Webcam required

---

## 🚀 Future Improvements

* GPS integration
* Mobile app support
* Cloud-based road monitoring
* Automatic reporting system

---

## 👨‍💻 Author

**Sumanth H**
MCA Student – CMR Institute of Technology

---

## 📌 Notes

* Do NOT upload `venv/` or `.venv/`
* Use `requirements.txt` to install dependencies
