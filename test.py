
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator
import cv2, sys
from PyQt5.QtCore import QThread
from PyQt5.QtWidgets import QApplication, QMainWindow
import threading




def track():
    model = YOLO("YOLOv8/yolov8m.pt")
    capture = cv2.VideoCapture("example.mp4")
    while True:
        success, frame = capture.read()
        results = model.predict(frame, show=True)

        cv2.waitKey(1)

a = threading.Thread(target=track)
a.start()