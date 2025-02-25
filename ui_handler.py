from PyQt5.QtCore import Qt
from PyQt5 import QtWidgets, uic
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtWidgets import QFileDialog, QGraphicsScene
import cv2
import numpy as np
from add_noise import add_gaussian_noise, add_salt_pepper_noise, add_uniform_noise
from apply_filter import apply_average_filter, apply_gaussian_filter, apply_median_filter

class UIHandler:
    def __init__(self, main_window):
        self.main_window = main_window
        uic.loadUi(r'ui_2.ui', self.main_window)
        self.image_label = self.main_window.findChild(QtWidgets.QLabel, 'image1')
        self.output_image_view = self.main_window.findChild(QtWidgets.QGraphicsView, 'output_image')
        self.image_label.setAlignment(Qt.AlignCenter)
        
        self.image_label.mouseDoubleClickEvent = self.open_image
        self.main_window.apply_noise_btn.clicked.connect(self.apply_noise)
        self.main_window.apply_filter_btn.clicked.connect(self.apply_filter)
        self.main_window.kernel_size_slider.valueChanged.connect(self.update_kernel_size)
        self.main_window.kernel_size_slider.setValue(3)
        self.image = None
        self.kernel_size = 3

    def open_image(self, event):
        options = QtWidgets.QFileDialog.Options()
        file_name, _ = QtWidgets.QFileDialog.getOpenFileName(self.main_window, "Select Image", "", "Image Files (*.png *.jpg *.bmp)", options=options)
        if file_name:
            pixmap = QPixmap(file_name)
            # Preserve aspect ratio in QLabel
            self.image_label.setPixmap(pixmap.scaled(self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
            self.image_label.setScaledContents(False)
            
            self.image = cv2.imread(file_name, cv2.IMREAD_COLOR)

    def update_kernel_size(self):
        self.kernel_size = self.main_window.kernel_size_slider.value()
        self.main_window.kernel_size_label.setText(f"{self.kernel_size}")

    def apply_noise(self):
        # print('Applying noise')
        if self.image is None:
            return

        noise_type = self.main_window.noise_type_combo.currentText()
        if noise_type == 'guassian noise':
            noisy_image = add_gaussian_noise(self.image)
        elif noise_type == 'salt and pepper noise':
            noisy_image = add_salt_pepper_noise(self.image)
        elif noise_type == 'uniform noise':
            noisy_image = add_uniform_noise(self.image)
        else:
            return

        self.display_image(noisy_image)

    def apply_filter(self):
        if self.image is None:
            return
        filter_type = self.main_window.filter_combo.currentText()
        if filter_type == 'average filter':
            filtered_image = apply_average_filter(self.image , self.kernel_size)
        elif filter_type == 'gaussian filter':
            filtered_image = apply_gaussian_filter(self.image, self.kernel_size)
        elif filter_type == 'median filter':
            filtered_image = apply_median_filter(self.image, self.kernel_size)
        else:
            return
        self.display_image(filtered_image)

    def display_image(self, image):
        height, width, channel = image.shape
        bytes_per_line = 3 * width
        q_image = QImage(image.data, width, height, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
        pixmap = QPixmap.fromImage(q_image)
        scene = QGraphicsScene()
        scene.addPixmap(pixmap)
        self.output_image_view.setScene(scene)
        self.output_image_view.fitInView(scene.itemsBoundingRect(), Qt.KeepAspectRatio)
