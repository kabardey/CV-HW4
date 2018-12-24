import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QHBoxLayout, QVBoxLayout, QWidget, QGroupBox, QAction, QFileDialog, qApp, QPushButton, QMessageBox
from PyQt5.QtWidgets import QLabel
from PyQt5.QtGui import QPixmap, QImage, QIcon
from PyQt5.QtCore import Qt
import numpy as np
import cv2
import random


class App(QMainWindow):
    def __init__(self):
        super(App, self).__init__()
        self.title = 'Corner Detection and Tumor Segmentation'
        self.left = 10
        self.top = 10
        self.width = 1000
        self.height = 600

        self.input_opened = False

        self.corner_points = []

        self.initUI()

    def placeImgToLabel1(self, image):

        pixmap_label = self.qlabel1
        height, width, channel = image.shape

        bytesPerLine = 3 * width
        qImg = QImage(image.data, width, height, bytesPerLine, QImage.Format_RGB888).rgbSwapped()
        pixmap = QPixmap(qImg)
        pixmap_label.setPixmap(pixmap)

    def placeImgToLabel2(self, image):

        pixmap_label = self.qlabel1
        height, width = image.shape

        bytesPerLine = width
        qImg = QImage(image.data, width, height, bytesPerLine, QImage.Format_Grayscale8)
        pixmap = QPixmap(qImg)
        pixmap_label.setPixmap(pixmap)

    def openInputImage(self):
        # ******** place image into qlabel object *********************
        self.input_opened = True

        imagePath, _ = QFileDialog.getOpenFileName()
        self.inputImg = cv2.imread(imagePath)

        pixmap_label = self.qlabel1
        height, width, channel = self.inputImg.shape

        bytesPerLine = 3 * width
        qImg = QImage(self.inputImg.data, width, height, bytesPerLine, QImage.Format_RGB888).rgbSwapped()
        pixmap = QPixmap(qImg)
        pixmap_label.setPixmap(pixmap)

    def compute_matrix(self, sobelx_pad, sobely_pad, height, width, padding):

        temp_image = np.zeros((height, width, 4))  # temp_image with G matrix in the z axis

        for i in range(padding, height+padding):
            for j in range(padding, width+padding):

                window1 = sobelx_pad[i - padding:i + padding + 1, j - padding:j + padding + 1]
                window2 = sobely_pad[i - padding:i + padding + 1, j - padding:j + padding + 1]

                IxSquare = np.sum(np.multiply(window1, window1))
                temp_image[i - padding, j - padding, 0] = IxSquare

                IxMultiplyIy = np.sum(np.multiply(window1, window2))
                temp_image[i - padding, j - padding, 1] = IxMultiplyIy
                temp_image[i - padding, j - padding, 2] = IxMultiplyIy

                IySquare = np.sum(np.multiply(window2, window2))
                temp_image[i - padding, j - padding, 3] = IySquare

        self.findCorners(temp_image)

    #  Draw a point
    def draw_point1(self, p, color):

        tmp_image = self.inputImg
        cv2.circle(tmp_image, p, 1, color, 2, 6, 0)
        self.placeImgToLabel1(tmp_image)


    def findCorners(self, temp_image):

        height, width, g_vector = temp_image.shape
        threshold_value = 10000000

        for i in range(height):
            for j in range(width):
                g_vector = temp_image[i, j, :]
                g_vector = g_vector.reshape((2, 2))

                s = np.linalg.svd(g_vector, compute_uv=False)

                if s[1] > threshold_value:
                    self.corner_points.append([j, i])

        for k in range(len(self.corner_points)):
            x = self.corner_points[k][0]
            y = self.corner_points[k][1]

            self.draw_point1((x, y), (0, 0, 0))


    def detectCorners(self):
        if (self.input_opened == False):
            return QMessageBox.question(self, 'Error Message', "Please, load input image", QMessageBox.Ok, QMessageBox.Ok)

        height, width, _ = self.inputImg.shape

        gray = cv2.cvtColor(self.inputImg, cv2.COLOR_RGB2GRAY)
        kernel_size = 5
        blur = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)

        sobelx = cv2.Sobel(blur, cv2.CV_64F, 1, 0, ksize=kernel_size)
        sobely = cv2.Sobel(blur, cv2.CV_64F, 0, 1, ksize=kernel_size)

        sobelx_pad = np.pad(sobelx, ((2, 2), (2, 2)), 'constant', constant_values=(0, 0))
        sobely_pad = np.pad(sobely, ((2, 2), (2, 2)), 'constant', constant_values=(0, 0))

        padding = int((kernel_size-1)/2)

        self.compute_matrix(sobelx_pad, sobely_pad, height, width, padding)


    def findTumor(self):
        if (self.input_opened == False):
            return QMessageBox.question(self, 'Error Message', "Please, load input image", QMessageBox.Ok, QMessageBox.Ok)

        gray = cv2.cvtColor(self.inputImg, cv2.COLOR_RGB2GRAY)
        height, width = gray.shape

        temp_gray = gray.copy()
        threshold = 56
        for i in range(height):
            for j in range(width):
                if temp_gray[i, j] > threshold:
                    temp_gray[i, j] = 255
                else:
                    temp_gray[i, j] = 0

        erode_img = cv2.erode(temp_gray, np.ones((17, 17)))

        mask = cv2.bitwise_and(gray, erode_img)

        k = 4

        random.seed(None)
        centers = np.array(random.sample(range(1, 256), k))

        tmp_centers = np.zeros((k,), dtype=np.uint8)

        while np.sum(tmp_centers-centers) != 0:

            listofsamples = [np.array([]) for _ in range(k)]

            for i in range(height):
                for j in range(width):
                    if mask[i, j] != 0:
                        index = np.argmin(np.abs(centers - mask[i, j]))
                        listofsamples[index] = np.append(listofsamples[index], mask[i, j])

            tmp_centers = centers.copy()

            for m in range(len(listofsamples)):
                if listofsamples[m].shape[0] != 0:
                    centers[m] = np.average(listofsamples[m])

        centers[::-1].sort()

        mask[np.abs(np.float32(mask) - centers[0]) <= np.abs(np.float32(mask) - centers[1])] = 255
        mask[np.abs(np.float32(mask) - centers[0]) > np.abs(np.float32(mask) - centers[1])] = 0

        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, (9, 9))

        small_mask = cv2.erode(mask, np.ones((5, 5)))

        borders = cv2.bitwise_xor(mask, small_mask)

        self.inputImg[borders[:, :].astype(bool)] = 0
        np.copyto(self.inputImg[:, :, 2], borders, where=borders[:, :].astype(bool))

        self.placeImgToLabel1(self.inputImg)


    def initUI(self):
        # Write GUI initialization code

        self.setGeometry(self.left, self.top, self.width, self.height)
        self.setWindowTitle(self.title)

        #****************add the labels for images*********************
        wid = QWidget(self)
        self.setCentralWidget(wid)

        b1 = QPushButton("Detect Corners")
        b1.clicked.connect(self.detectCorners)

        b2 = QPushButton("Find Tumor")
        b2.clicked.connect(self.findTumor)

        self.groupBox = QGroupBox()
        self.hBoxlayout = QHBoxLayout()

        self.qlabel1 = QLabel('Input', self)
        self.qlabel1.setStyleSheet("border: 1px inset grey; min-height: 200px; ")
        self.qlabel1.setAlignment(Qt.AlignCenter)
        self.hBoxlayout.addWidget(self.qlabel1)

        self.groupBox.setLayout(self.hBoxlayout)

        vBox = QVBoxLayout()
        vBox.addWidget(b1)
        vBox.addWidget(b2)
        vBox.addWidget(self.groupBox)

        wid.setLayout(vBox)

        #****************menu bar***********
        menubar = self.menuBar()
        fileMenu = menubar.addMenu('File')

        openAction = QAction('Open Input', self)
        openAction.triggered.connect(self.openInputImage)
        fileMenu.addAction(openAction)

        exitAct = QAction(QIcon('exit.png'), '&Exit', self)
        exitAct.setShortcut('Ctrl+Q')
        exitAct.setStatusTip('Exit application')
        exitAct.triggered.connect(qApp.quit)
        fileMenu.addAction(exitAct)

        #------------------------------------

        self.show()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = App()
    sys.exit(app.exec_())


