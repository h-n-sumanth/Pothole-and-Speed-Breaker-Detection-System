1.Install: Python, TensorFlow, Keras, OpenCV, MobileNetV2, NumPy, Tkinter.

2. Using the Pre-trained Model
This repository includes a pre-trained MobileNetV2 model located in the models/ directory.

File: models/road_detection_model.h5 
Usage: The live detection and image testing scripts load this file by default. You do not need to download a dataset to run the application.

3. Training Your Own Model
If you wish to retrain the model or improve its accuracy with fresh data, follow these steps:

A. Download the Dataset
Since the full dataset is not hosted in this repository to save space, you can download the required images from Kaggle:

Source: Search for "Pothole and Speed Breaker Dataset" on Kaggle.
Categories Needed: Ensure your dataset has images for Normal Road, Pothole, and Speed Breaker.
    
4.Project Folder Structure
pothole_speedbreaker/
│
├── models/
│   └── road_detection_model.h5        # Trained CNN model
│
├── dataset/                           # (Only needed for training)
│   ├── train/
│   │   ├── pothole/
│   │   ├── speedbreaker/
│   │   └── normal/
│   │
│   └── validation/
│       ├── pothole/
│       ├── speedbreaker/
│       └── normal/
│
│
├── train_model.py                     # Train the CNN model
├── main.py                            # Simple camera detection (optional)
├── live_predict_camera.py             # Final UI + live detection (MAIN FILE)
│
├── requirements.txt                   # Python libraries list
│
└── README.txt                         # (Optional) How to run project

5.Run main UI detection:
python live_predict_camera.py

