import cv2, sys
import numpy as np
from PyQt5 import uic
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog
from PyQt5.QtGui import QPixmap, QPainter, QPen
from PyQt5.QtGui import QPixmap, QIcon
from PyQt5.QtCore import Qt, QThread, QPoint
from utlis import *

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        uic.loadUi('ui/new_config.ui', self)
        self.setWindowIcon(QIcon("./ui/icon.png"))
        self.show()
        self.btn_start_feed.clicked.connect(lambda: self.start_camera_live_feed(id=self.input_cam_id.text()))
        self.btn_add_lane.clicked.connect(self.add_lane)
        self.btn_save.clicked.connect(self.save_config)

        self.frame = None
        self.frame_resized = None
   
        # 
        self.regions = [None,None,None,None]
        self.lane_id = 0

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

                # Start live feed worker thread
                self.thread_camera_live = worker_camera_live_feed()
                self.thread_camera_live.start()

                # UI disable inputs and buttons
                self.btn_start_feed.setEnabled(False)
                self.input_cam_id.setEnabled(False)
                self.input_fps.setEnabled(False)
                self.input_w.setEnabled(False)
                self.input_h.setEnabled(False)
                self.btn_add_lane.setEnabled(True)

            except Exception as e:
                print(e)
        else:
            print("Failed to read input.")

    def add_lane(self):
        # Get lane points
        self.getpoints = window_get_point(image = self.frame, dims=self.frame_real_dims, count=4, guide="ui/guide_region.png")
        result = self.getpoints.exec_()
        if result == QDialog.Accepted:
            if len(self.getpoints.results) !=0:
                region_area = self.getpoints.results
            else:return
        else:return

        # Default values
        self.regions[self.lane_id] = {  "name" : "lane",
                                        "area" : region_area,
                                        "tracker_dimensions":(400, 900),
                                        "vehicle_direction" : 0 ,
                                        "abs_pos":[(0, 0),(0, 0)], 
                                        "abs_distance":1,
                                        "distorted_factor": 0,
                                        "buffer_dimensions":[200,200],
                                        "speed_limits":{ "car":2.5,
                                                         "motorbike":2.5,
                                                         "bus":2.0,
                                                         "truck":2.0,}}
        self.region = self.regions[self.lane_id]

        self.worker = worker_service(id=self.lane_id, region=self.region)
        self.worker.start()

        # 
        lane = window_lane_settings()
        result = lane.exec_()
        if result == QDialog.Accepted:
            name = SyS_InputDialog()
            result = name.exec_()
            if result == QDialog.Accepted:
                print("Success")
                name = name.input.text()
                self.list_lanes.addItem(f"{self.lane_id} : {name}")
                self.regions[self.lane_id]["name"] = name
                self.btn_save.setEnabled(True)
                self.lane_id += 1
            else:
                print("Please enter a name")
        else:
            print("Aborted")

        self.worker.stop()
        del lane, result
        return
        
    def save_config(self):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getSaveFileName(self, "Save File", "", "Configuration Files (*.config)", options=options)
        if file_name:
            with open(file_name, "w") as f:
                cam_id = self.input_cam_id.text()
                fps = int(self.input_fps.text())
                width = int(self.input_w.text())
                height = int(self.input_h.text())
                settings = {"cam_id" : cam_id,
                            "fps" : fps,
                            "width" : width,
                            "height" : height,}

                f.write(f"{settings}\n")

                for id, lane in enumerate(self.regions):
                    if lane != None:
                        f.write(f"{lane}\n")
        else:
            print("Please select an output file.")


class window_lane_settings(QDialog):   
    def __init__(self, parent=None):
        super(window_lane_settings, self).__init__(parent)
        # Display the about window
        self = uic.loadUi('ui/window_lane_settings.ui', self)
        self.setWindowModality(Qt.ApplicationModal)
        self.setWindowTitle("Settings")
        self.setWindowIcon(QIcon("./ui/icon.png"))

        self.input_width.valueChanged.connect(self.update_values)
        self.input_height.valueChanged.connect(self.update_values)
        self.input_search_width.valueChanged.connect(self.update_values)
        self.input_search_height.valueChanged.connect(self.update_values)
        self.input_d_factor.valueChanged.connect(self.update_values)
        self.input_abs_distance.valueChanged.connect(self.update_values)

        self.input_speed_car.valueChanged.connect(self.update_values)
        self.input_speed_motorbike.valueChanged.connect(self.update_values)
        self.input_speed_bus.valueChanged.connect(self.update_values)
        self.input_speed_truck.valueChanged.connect(self.update_values)
        self.cBox_direction.currentIndexChanged.connect(self.update_values)


        self.btn_pos.clicked.connect(self.get_points)

        self.btn_accept.clicked.connect(self.accept)
        self.btn_accept.setDefault(False)
        self.btn_cancel.setDefault(False)
        self.btn_cancel.clicked.connect(self.reject)

        self.show()

    def update_values(self):

        _main.region["tracker_dimensions"] = (int(self.input_width.value()), int(self.input_height.value()))
        _main.region["buffer_dimensions"] = (int(self.input_search_width.value()), int(self.input_search_height.value()))
        _main.region["vehicle_direction"] = int(self.cBox_direction.currentIndex())
        _main.region["distorted_factor"] = float(self.input_d_factor.value())
        _main.region["abs_distance"] = float(self.input_abs_distance.value())

        _main.region["speed_limits"]["car"] = float(self.input_speed_car.value())
        _main.region["speed_limits"]["motorbike"] = float(self.input_speed_motorbike.value())
        _main.region["speed_limits"]["bus"] = float(self.input_speed_bus.value())
        _main.region["speed_limits"]["truck"] = float(self.input_speed_truck.value())

    def get_points(self):
        self.getpoints = window_get_point(image = _main.worker.bird_eye_image, dims=_main.worker.tracker_dimensions, count=2, guide="ui/guide_points.png")
        result = self.getpoints.exec_()
        if result == QDialog.Accepted:
            if len(self.getpoints.results) !=0:
                _main.region["abs_pos"] = self.getpoints.results
        return


class SyS_InputDialog(QDialog):   
    def __init__(self, parent=None, title="Input a new name", msg="Please enter a name", msg2="Confirm password :", ispassword=False, password_confirm=False):
        super(SyS_InputDialog, self).__init__(parent)
        self = uic.loadUi('ui/dlg_input.ui', self)
        self.setWindowModality(Qt.ApplicationModal)
        self.setWindowTitle(title)
        self.setWindowIcon(QIcon("./ui/icon.png"))
        self.text.setText(msg)
        self.btn_ok.clicked.connect(self.accept)
        self.btn_cancel.clicked.connect(self.reject)
        self.btn_ok.setDefault(True)
        self.show()


class worker_camera_live_feed(QThread):
    def __init__(self):
        super().__init__()
        self.is_running = True

    def run(self):
        while self.is_running:
            success, _main.frame = _main.capture.read()

            # Resize based on user inputs
            _main.frame = cv2.resize(_main.frame, _main.frame_real_dims, interpolation=cv2.INTER_AREA)

            # Resize frame for yolo
            _main.frame_resized = cv2.resize(_main.frame, _main.frame_resized_dims, interpolation=cv2.INTER_AREA)

            #cv2.imshow("Original Image", _main.frame_resized)
            _main.lbl_video_live.setPixmap(cvtQtimg(_main.frame_resized, w=_main.frame_resized_dims[0], h=_main.frame_resized_dims[1]))
            
            self.msleep(_main.cv_wait_time)


class worker_service(QThread):
    def __init__(self, id, region):
        super().__init__()
        self.id = id
        self.region = region
        self.area = np.array(region["area"], np.int32)
        self.is_running = True

    def run(self):
        while self.is_running:
            self.tracker_dimensions = self.region["tracker_dimensions"]
            self.tracker_width = self.tracker_dimensions[0]
            self.tracker_height = self.tracker_dimensions[1]
            self.distorted_factor = self.region["distorted_factor"]
            self.buffer_dimensions = self.region["buffer_dimensions"]
            self.speed_limits = self.region["speed_limits"]

            self.perspective_matrix = cv2.getPerspectiveTransform(self.area.astype(np.float32), np.array([[0, 0], [self.tracker_width, 0], [0, self.tracker_height], [self.tracker_width, self.tracker_height]], np.float32))

            self.abs_pos = self.region["abs_pos"]
            self.abs_distance = self.region["abs_distance"]
       
            self.bird_eye_image = bird_eye_view(_main.frame, self.perspective_matrix, self.tracker_width, self.tracker_height, self.distorted_factor)
            cv2.circle(self.bird_eye_image, self.abs_pos[0], 5, (255, 0, 0), -1)
            cv2.circle(self.bird_eye_image, self.abs_pos[1], 5, (255, 0, 0), -1)

            x = 20
            y = 20
            w = self.buffer_dimensions[0]
            h = self.buffer_dimensions[1]
            
            cv2.rectangle(self.bird_eye_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Display the result
            cv2.imshow(f"Lane {self.id}", self.bird_eye_image)
            cv2.waitKey(_main.cv_wait_time)

    def stop(self):
        self.is_running = False


class window_get_point(QDialog):
    def __init__(self, image, dims, count, guide="ui/guide_region.png"):
        super(window_get_point, self).__init__(None)
        self = uic.loadUi('ui/dlg_pick.ui', self)
        self.setWindowModality(Qt.ApplicationModal)
        self.setWindowTitle(f'Select {count} points')
        self.setWindowIcon(QIcon("./ui/icon.png"))
    
        self.image_label = self.lbl_img
        self.count = count

        self.selected_points = []
        self.point_radius = 5

        pixmap = QPixmap(cvtQtimg(image, w=dims[0], h=dims[1]))
        self.image_label.setPixmap(pixmap)

        self.results = []
        self.btn_ok.clicked.connect(self.accept)
        self.image_label.setGeometry(0, 0, dims[0], dims[1])
        self.line.setGeometry(dims[0]+10, 10,20, dims[1])
        
        self.lbl_guide.setGeometry(dims[0]+40, 10, 320, 180)
        self.lbl_guide.setPixmap(QPixmap(guide))

        self.btn_ok.setGeometry(dims[0]+370-85, dims[1]+10, 75, 23)
        self.setGeometry(100, 100, dims[0]+370, dims[1]+40)
        self.show()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton and len(self.selected_points) < self.count:
            self.selected_points.append(event.pos())
            self.updateLabel()

    def updateLabel(self):
        if self.selected_points:
            pixmap = self.image_label.pixmap().copy()
            painter = QPainter(pixmap)

            pen = QPen(Qt.red)
            pen.setWidth(2)
            painter.setPen(pen)

            for id, point in enumerate(self.selected_points):
                painter.drawText(QPoint(point.x()+10,point.y()), str(id+1))
                painter.drawEllipse(point, self.point_radius, self.point_radius)

            painter.end()
            self.image_label.setPixmap(pixmap)

            if len(self.selected_points) == self.count:
                for i in self.selected_points:
                    self.results.append((i.x(),i.y()))

def bird_eye_view(frame, perspective_matrix, width, height, d_factor):
    # Apply perspective transformation
    transformed_frame = cv2.warpPerspective(frame, perspective_matrix, (width, height))
    # Create an output image with the same size as the input
    bird_eye_image = np.zeros_like(transformed_frame, dtype=np.float32)  # Use float32 for intermediate values
    # Stretch the image from top to bottom
    for y in range(height):
        stretch_factor = 1 + ((height - y) / height) * d_factor # Adjust the factor as needed
        bird_eye_row_position = int(y * stretch_factor)
        stretched_row = cv2.resize(transformed_frame[y:y+1, :, :], (width, 1), interpolation=cv2.INTER_AREA)
        bird_eye_image[bird_eye_row_position-1:bird_eye_row_position + 1, :, :] = stretched_row
    # Normalize the result to uint8 range
    bird_eye_image = np.clip(bird_eye_image, 0, 255).astype(np.uint8)
    return bird_eye_image

if __name__ == '__main__':
    _app = QApplication(sys.argv)
    _main = MainWindow()
    sys.exit(_app.exec_())