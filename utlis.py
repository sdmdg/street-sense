import numpy as np
import cv2
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QDialog
from PyQt5 import uic
from PyQt5.QtGui import QPixmap, QIcon
from PyQt5.QtCore import Qt


# 1. Reoder points
def reoder(myPoints):
    myPoints = myPoints.reshape((4, 2))
    myPointsNew = np.zeros((4, 1, 2), dtype=np.int32)
    add = myPoints.sum(1)
    myPointsNew[0] = myPoints[np.argmin(add)]
    myPointsNew[3] = myPoints[np.argmax(add)]
    diff = np.diff(myPoints, axis=1)
    myPointsNew[1] = myPoints[np.argmin(diff)]
    myPointsNew[2] = myPoints[np.argmax(diff)]
    return myPointsNew

# 2. Convert cv2 img to QtPixMap
def cvtQtimg(img, w=384, h=216):
    """get cv2 img and return pixmap for Qt"""
    frame = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = QImage(frame, w, h, 3 * w, QImage.Format_RGB888)
    return QPixmap.fromImage(img)

# 3. About
class dlg_about(QDialog):   
    def __init__(self, parent=None):
        super(dlg_about, self).__init__(parent)
        # Display the about window
        self = uic.loadUi('ui/dlg_about.ui', self)
        self.setWindowModality(Qt.ApplicationModal)
        self.setWindowTitle("About")
        self.setWindowIcon(QIcon("./ui/icon.png"))
        self.dummy_3.setText(''.join(chr(ord(char) - 1) for char in "Efwfmpqfe!cz;!Nbmblb!E/Hvobxbsebob/"))
        pixmap = QPixmap("./ui/icon.png")
        pixmap = pixmap.scaledToWidth(200, Qt.SmoothTransformation)
        self.icon.setPixmap(pixmap)
        self.btn_ok.clicked.connect(self.accept)
        self.btn_ok.setDefault(True)
        self.show()