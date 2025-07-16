# 🚦 Street Sense – Real-Time Traffic Monitoring System

**Street Sense** is a real-time traffic analysis system designed to detect and track vehicles and estimate their speeds using only a single webcam. This project aims to provide a cost-effective, intelligent solution for monitoring traffic flow and identifying speed violations.

> ⚠️ This project is under active development — new features and improvements are being added continuously.


## 📌 Features

- 🚗 **Vehicle Detection** using the YOLO object detection algorithm
- 🎯 **Speed Estimation** based on real-time tracking data
- 📹 Works with any webcam or video input
- 🧠 **Single-camera solution** — no radar, lidar, or multiple sensors required
- 📊 Logs vehicle data including:
  - Detected speed
  - Timestamp
  - Snapshot filename
- 📁 Saves detection results to a CSV file



## 🧪 Technologies Used

- **YOLO (You Only Look Once)** – Object detection
- **OpenCV** – Video processing and object tracking
- **Python** – Core implementation
- **CSV** – Data logging


## 🎬 Demo Video Source

Video used for development/testing:
[![Watch the video](https://github.com/user-attachments/assets/57cd225d-13f4-4eab-8f62-487a2251674e)](https://malakagunawardana.pages.dev/projects/images/street-sense/animation_1.mp4)

[Motor Vehicles Traveling on a Highway – Pexels](https://www.pexels.com/video/motor-vehicles-traveling-on-a-highway-5473757)


## 🛠️ Setup Instructions

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/street-sense.git
   cd street-sense```

2. **Install required dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the main script**:
   ```bash
   python main.py
   ```



## 📂 Output

- `snapshots/`: Contains images of detected vehicles.
- `log.csv`: CSV file containing timestamp, speed, and snapshot filename for each detection.

---

##  🚧 Under Development

I'm working on:
- Enhancing object tracking for better accuracy
- Calibrated speed estimation using camera parameters
- UI dashboard for visualizing real-time stats

Stay tuned for updates! 😊

