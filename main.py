import cv2, sys, datetime
import numpy as np
from PyQt5 import uic
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QListWidgetItem, QCheckBox
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import QThread
from utlis import *
from YOLO_v8 import *
from tracker import *
from shapely.geometry import Point, Polygon

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        uic.loadUi('ui/main.ui', self)
        self.setWindowIcon(QIcon("./ui/icon.png"))
        self.show()
        self.btn_start_feed.clicked.connect(lambda: self.start_camera_live_feed(id=self.input_cam_id.text()))
        self.btn_start_service.clicked.connect(self.start_service)
        self.btn_load_config.clicked.connect(self.load_config)

        self.frame = None
        self.prev_frame = None
        self.frame_resized = None
        self.prev_frame_resized = None
        self.frame_resize_factor = None
        self.vehicles = []
        self.workers = []
        self.startTime = datetime.datetime.now()

    def load_config(self):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(self, "Select config file", "", "All Files (*);;Config Files (*.config)", options=options)
        if not file_name:return
        else:
            with open(file_name, "r") as f:
                data = f.read().split("\n")[:-1]

        settings = eval(data[0])

        self.input_cam_id.setText(str(settings["cam_id"]))
        self.input_fps.setText(str(settings["fps"]))
        self.input_w.setText(str(settings["width"]))
        self.input_h.setText(str(settings["height"]))
    
        data = data[1:]
        self.regions = []
        for id, line in enumerate(data):
            self.regions.append(eval(line))
            self.add_checkbox_item(f'Preview | {id} : {self.regions[id]["name"]}', id=id)
        self.list_lanes.setStyleSheet("""color: rgb(200, 200, 200); background: rgb(30, 30, 30);""")
        self.btn_start_feed.setEnabled(True)


    # Camera live feed
    def start_camera_live_feed(self, id=0):
        try:
            id = int(id)
            print(f"Camera ID : {id}")
        except:
            print(f"Input Video : {id}")
        
        # Read a frame to test input
        self.capture = cv2.VideoCapture(id)
        success, frame = self.capture.read()
        if success:
            try:
                # Fix resolution errors based on user inputs
                self.frame_real_dims = (int(self.input_w.text()), int(self.input_h.text()))
                self.frame_ratio = self.frame_real_dims[0]/self.frame_real_dims[1]

                if self.frame_ratio > 1: self.frame_resized_dims = (640, int(640/self.frame_ratio))
                else: self.frame_resized_dims = (int(360*self.frame_ratio), 360)

                #####edit######
                self.frame_width, self.frame_height = self.frame_real_dims

                self.fps = int(self.input_fps.text())
                self.cv_wait_time = int((1/self.fps) * 1000)
                self.cam_id = id

                # Start live feed worker thread
                self.thread_camera_live = worker_camera_live_feed()
                self.thread_camera_live.start()

                # UI disable inputs and buttons
                self.btn_start_feed.setEnabled(False)
                self.input_cam_id.setEnabled(False)
                self.input_fps.setEnabled(False)
                self.input_w.setEnabled(False)
                self.input_h.setEnabled(False)
                self.btn_start_service.setEnabled(True)

            except Exception as e:
                print(e)
        else:
            print("Failed to read input.")

    def start_service(self):       
        for id in range(len(self.regions)):
            region = self.regions[id]
            self.workers.append(worker_service( id=id,
                                                name=region["name"],
                                                area=region["area"], 
                                                tracker_dimensions=region["tracker_dimensions"],
                                                vehicle_direction = region["vehicle_direction"],
                                                abs_pos=region["abs_pos"], 
                                                abs_distance=region["abs_distance"],
                                                distorted_factor=region["distorted_factor"],
                                                buffer_dimensions=region["buffer_dimensions"],
                                                speed_limits=region["speed_limits"]))
            self.workers[id].start()
        self.list_lanes.setEnabled(True)
        self.btn_start_service.setEnabled(False)

    def add_checkbox_item(self, text, id):
        item = QListWidgetItem(self.list_lanes)
        checkbox = QCheckBox(text)
        checkbox.setChecked(False)
        checkbox.stateChanged.connect(lambda state: self.on_checkbox_state_changed(id, state))
        self.list_lanes.setItemWidget(item, checkbox)

    def on_checkbox_state_changed(self, id, state):
        if state == 2:
            self.workers[id].preview = True
        else:
            self.workers[id].preview = False


class worker_camera_live_feed(QThread):
    def __init__(self):
        super().__init__()
        print(f"Input : {_main.cam_id}\ndims : {_main.frame_real_dims}\nratio : {_main.frame_ratio}\nresized_dims : {_main.frame_resized_dims}")
        self.areas = []
        _main.vehicles_in_regions = []
        for i in range(len(_main.regions)):
            area = _main.regions[i]["area"]
            self.areas.append(Polygon([area[0],area[1],area[3],area[2]]))
            _main.vehicles_in_regions.append([])

    def run(self):
        model = load_model("YOLOv8/yolov8s.pt")
        while True:
            success, _main.frame = _main.capture.read()

            # Resize based on user inputs
            _main.frame = cv2.resize(_main.frame, _main.frame_real_dims, interpolation=cv2.INTER_AREA)

            # Resize frame for yolo
            _main.frame_resized = cv2.resize(_main.frame, _main.frame_resized_dims, interpolation=cv2.INTER_AREA)

            # Predict resized frame using YOLO v8 - in(640px,360px) - ret(type, or_pos, or_image_of_vehicle)
            _main.vehicles = YOLOv8(or_frame=_main.frame, yolo_frame=_main.frame_resized,model=model,_main=_main)

            # Lanes
            for i in range(len(_main.regions)):
                _main.vehicles_in_regions[i] = []

            for vehicle in _main.vehicles:
                for i in range(len(_main.regions)):
                    if self.areas[i].contains(Point(vehicle[1])):
                        _main.vehicles_in_regions[i].append(vehicle)
                        break

            #cv2.imshow("Original Image", _main.frame_resized)
            _main.lbl_video_live.setPixmap(cvtQtimg(_main.frame_resized, w=_main.frame_resized_dims[0], h=_main.frame_resized_dims[1]))
            _main.prev_frame = _main.frame
            _main.prev_frame_resized = _main.frame_resized
            cv2.waitKey(_main.cv_wait_time)


class worker_service(QThread):
    def __init__(self, id, area, name, tracker_dimensions, vehicle_direction, abs_pos, abs_distance, distorted_factor, buffer_dimensions, speed_limits):
        super().__init__()
        self.id = id
        self.name = name
        self.area = np.array(area, np.int32)
        self.tracker_dimensions = tracker_dimensions
        self.tracker_width = tracker_dimensions[0]
        self.tracker_height = tracker_dimensions[1]
        self.distorted_factor = distorted_factor
        self.vehicle_direction = vehicle_direction
        self.buffer_dimensions = buffer_dimensions
        self.speed_limits =speed_limits

        self.log = {}

        self.perspective_matrix = cv2.getPerspectiveTransform(self.area.astype(np.float32), np.array([[0, 0], [self.tracker_width, 0], [0, self.tracker_height], [self.tracker_width, self.tracker_height]], np.float32))

        self.abs_pos = abs_pos
        self.abs_distance = abs_distance

        self.preview = False

        print(f"Starting worker for lane ID : {self.id}.\nArea : {self.area}\nDimentions : {self.tracker_dimensions}\nMapping pos(s) : {self.abs_pos}\nABS_distance : {self.abs_distance}\nD_factor : {self.distorted_factor}\nBuffer dims : {self.buffer_dimensions}")

    def run(self):
        print(f"Running worker for lane ID : {self.id}")
        self.tracker = Tracker(_main=_main, _worker=self.id, tracker_dimensions=self.tracker_dimensions, vehicle_direction=self.vehicle_direction, buffer_dimensions=self.buffer_dimensions, log=self.log, distorted_factor=self.distorted_factor, solver_data=[self.abs_pos, self.abs_distance])

        while True:
            vehicles = []
            if self.preview : self.bird_eye_image = self.bird_eye_view(_main.frame, self.perspective_matrix, self.tracker_width, self.tracker_height, self.distorted_factor)

            for vehicle in _main.vehicles_in_regions[self.id]:
                solved_point = self.solve_perspective(_main.frame, self.perspective_matrix, self.tracker_width, self.tracker_height, self.distorted_factor, vehicle[1])
                vehicles.append([vehicle[0], solved_point, vehicle[1], vehicle[2]])

            results = self.tracker.track(vehicles=vehicles,speed_limits=self.speed_limits)

            # Display the result
            if self.preview : 
                try:
                    for item in results:
                        cv2.putText(self.bird_eye_image, str(item[0]), (int(item[1][0]), int(item[1][1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    cv2.imshow(f"Lane {self.id}", self.bird_eye_image)
                except:pass
            cv2.waitKey(_main.cv_wait_time)

    def bird_eye_view(self, frame, perspective_matrix, width, height, d_factor):
        # Apply perspective transformation
        transformed_frame = cv2.warpPerspective(frame, perspective_matrix, (width, height))
        # Create an output image with the same size as the input
        bird_eye_image = np.zeros_like(transformed_frame, dtype=np.float32)  # Use float32 for intermediate values
        # Stretch the image from top to bottom
        for y in range(height):
            stretch_factor = 1 + ((height - y) / height) * d_factor
            bird_eye_row_position = int(y * stretch_factor)
            stretched_row = cv2.resize(transformed_frame[y:y+1, :, :], (width, 1), interpolation=cv2.INTER_AREA)
            bird_eye_image[bird_eye_row_position-1:bird_eye_row_position + 1, :, :] = stretched_row
        # Normalize the result to uint8 range
        bird_eye_image = np.clip(bird_eye_image, 0, 255).astype(np.uint8)
        return bird_eye_image

    def solve_perspective(self, frame, perspective_matrix, width, height, d_factor, pos):
        frame = cv2.resize(frame, _main.frame_real_dims, interpolation=cv2.INTER_AREA)
        original_point = np.array([pos], dtype=np.float32)
        # Apply perspective transformation
        transformed_point = cv2.perspectiveTransform(original_point.reshape(1, 1, 2), perspective_matrix)
        transformed_point_coordinates = tuple(transformed_point[0][0].astype(int))
        # Stretch the image from top to bottom
        stretch_factor = 1 + ((height - transformed_point_coordinates[1]) / height) * d_factor
        stretched_row_position = int(transformed_point_coordinates[1] * stretch_factor)
        stretched_point_coordinates = (transformed_point_coordinates[0], stretched_row_position)
        return stretched_point_coordinates


if __name__ == '__main__':
    _app = QApplication(sys.argv)
    _main = MainWindow()
    sys.exit(_app.exec_())