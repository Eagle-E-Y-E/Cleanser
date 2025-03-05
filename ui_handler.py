from PyQt5.QtCore import Qt
from PyQt5 import QtWidgets, uic
from PyQt5.QtGui import QPixmap, QImage, QPainter
from PyQt5.QtWidgets import QApplication, QGraphicsView, QGraphicsScene, QGraphicsPixmapItem, QLabel, QVBoxLayout, QWidget, QFileDialog
import cv2
import numpy as np
from add_noise import add_gaussian_noise, add_salt_pepper_noise, add_uniform_noise
from EdgeDetection import EdgeDetection
from apply_filter import apply_average_filter, apply_gaussian_filter, apply_median_filter
from Histogram_Equalization import HistogramEqualization
from Thresholding import Thresholding
from RGB2GRAY import RGB2GRAY
from RGB_Hist import RGB_Hist
from Freq_filters import Freq_filters
from Hybrid import Hybrid
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import QDialog, QLabel, QVBoxLayout
from PyQt5.QtCore import QTimer

# ui_names:
# - image1
# - output_image
# - apply_noise_btn
# - apply_filter_btn
# - noise_type_combo
# - filter_combo
# - kernel_size_slider
# - kernel_size_label
# - convert_to_grayscale_btn
# - equalize_image_btn
# - edge_detection_method_combo
# - detect_edges_btn
# - histogram_1
# - histogram_2
# - image1_mix
# - image2_mix
# - output_img_mix
# - mix_btn


class UIHandler:
    def __init__(self, main_window):
        self.main_window = main_window
        self.edge_detector = EdgeDetection()
        uic.loadUi(r'ui_2.ui', self.main_window)
        self.image_label = self.main_window.findChild(
            QtWidgets.QLabel, 'image1')
        self.output_image_view = self.main_window.findChild(
            QtWidgets.QGraphicsView, 'output_image')
        self.original_histogram = self.main_window.findChild(
            QtWidgets.QGraphicsView, 'histogram_1')
        self.equalized_histogram = self.main_window.findChild(
            QtWidgets.QGraphicsView, 'histogram_2')
        self.image_label.setAlignment(Qt.AlignCenter)

        self.image_label.mouseDoubleClickEvent = self.open_image
        self.main_window.apply_noise_btn.clicked.connect(self.apply_noise)
        self.main_window.apply_filter_btn.clicked.connect(self.apply_filter)
        self.main_window.kernel_size_slider.valueChanged.connect(
            self.update_kernel_size)
        self.main_window.kernel_size_slider.setValue(3)
        self.main_window.apply_thresholding_btn.clicked.connect(
            self.thresholding_control)
        self.main_window.equalize_image_btn.clicked.connect(
            self.equalize_btn_ctrl)
        self.main_window.detect_edges_btn.clicked.connect(self.detect_edges)
        self.main_window.export_btn.clicked.connect(self.export)
        self.main_window.normalize_btn.clicked.connect(self.normalize)

        
        self.main_window.noise_intensity_slider.valueChanged.connect(lambda: self.main_window.noise_intensity_label.setText(
            f"{self.main_window.noise_intensity_slider.value()}"))
        self.main_window.window_size_slider.valueChanged.connect(lambda: self.main_window.window_size_label.setText(
            f"{self.main_window.window_size_slider.value()}"))
        self.main_window.sensitivity_slider.valueChanged.connect(lambda: self.main_window.sensitivity_label.setText(
            f"{self.main_window.sensitivity_slider.value()}"))
        self.main_window.threshold_slider.valueChanged.connect(lambda: self.main_window.threshold_label.setText(
            f"{self.main_window.threshold_slider.value()}"))
        self.main_window.mixing_kenel_size_slider.valueChanged.connect(lambda: self.main_window.mixing_kenel_size_label.setText(
            f"{self.main_window.mixing_kenel_size_slider.value()}"))
        self.main_window.mixing_sigma_slider.valueChanged.connect(lambda: self.main_window.mixing_sensitivity_label.setText(
            f"{self.main_window.mixing_sigma_slider.value()}"))
        self.main_window.cuttoff_freq_slider.valueChanged.connect(lambda: self.main_window.cuttofffreq_label.setText(
            f"{self.main_window.cuttoff_freq_slider.value()}"))

        # double click histograms to maximize
        self.main_window.histogram_1.mouseDoubleClickEvent = lambda event: self.show_large_image(
            self.main_window.histogram_1, event)
        self.main_window.histogram_2.mouseDoubleClickEvent = lambda event: self.show_large_image(
            self.main_window.histogram_2, event)
        self.main_window.histogram_3.mouseDoubleClickEvent = lambda event: self.show_large_image(
            self.main_window.histogram_3, event)
        self.main_window.histogram_4.mouseDoubleClickEvent = lambda event: self.show_large_image(
            self.main_window.histogram_4, event)

        self.main_window.thresholding_combo.currentIndexChanged.connect(
            self.thresholding_control)
        self.main_window.Local_widget.hide()
        self.main_window.convert_to_grayscale_btn.clicked.connect(
            self.apply_grayscale)
        self.main_window.cuttofffreq_widgwt.hide()
        self.main_window.filter_combo.currentIndexChanged.connect(
            self.freq_control)

        self.main_window.mix_btn.clicked.connect(self.mix_images)
        
        self.image1_mix = self.main_window.findChild(
            QtWidgets.QLabel, 'image1_mix')
        self.image2_mix = self.main_window.findChild(
            QtWidgets.QLabel, 'image2_mix')
        self.output_img_mix = self.main_window.findChild(
            QtWidgets.QLabel, 'output_img_mix')
        self.image1_for_mixing = None
        self.image2_for_mixing = None

        
        self.image1_mix.mouseDoubleClickEvent = self.open_image1_for_mixing
        self.image2_mix.mouseDoubleClickEvent = self.open_image2_for_mixing

        self.image = None
        self.output_image = None
        self.gray_image = None
        self.kernel_size = 3

        self.R_cdf = None
        self.G_cdf = None
        self.B_cdf = None

        self.main_window.output_radio.toggled.connect(self.histogram_control)
        self.main_window.input_radio.toggled.connect(self.histogram_control)

        self.is_input_gray = False
        self.is_output_gray = False

    def histogram_control(self):
        if self.main_window.output_radio.isChecked():
            if self.output_image is not None:
                if self.is_output_gray:
                    self.grayscale_hist(self.output_image)
                    self.main_window.histogram_1.hide()
                    self.main_window.histogram_2.show()
                    self.main_window.histogram_3.hide()
                    self.main_window.histogram_4.hide()
                else:
                    self.rgb_hist(self.output_image)
                    self.main_window.histogram_1.show()
                    self.main_window.histogram_2.show()
                    self.main_window.histogram_3.show()
                    self.main_window.histogram_4.show()
            else:
                QtWidgets.QMessageBox.warning(self.main_window, "Warning",
                                              "Please apply a filter or noise to display the output histogram")
                
                
                # Use QTimer to delay the execution of the radio button state change
                QTimer.singleShot(0, lambda: self.main_window.output_radio.setChecked(False))
                QTimer.singleShot(0, lambda: self.main_window.input_radio.setChecked(True))

        elif self.main_window.input_radio.isChecked():
            if self.is_input_gray:
                self.grayscale_hist(self.image)
                self.main_window.histogram_1.show()
                self.main_window.histogram_2.hide()
                self.main_window.histogram_3.hide()
                self.main_window.histogram_4.hide()
            else:
                self.rgb_hist(self.image)
                self.main_window.histogram_1.show()
                self.main_window.histogram_2.show()
                self.main_window.histogram_3.show()
                self.main_window.histogram_4.show()

    def equalize_btn_ctrl(self):
        if self.is_input_gray:
            self.equalize_grayscale()
        else:
            self.RGB_historgram_equalization(self.image, self.R_cdf,
                                             self.G_cdf, self.B_cdf)

    def open_image1_for_mixing(self, event):
        options = QtWidgets.QFileDialog.Options()
        file_name, _ = QtWidgets.QFileDialog.getOpenFileName(
            self.main_window, "Select First Image for Mixing", "", "Image Files (*.png *.jpg *.bmp)", options=options)
        if file_name:
            pixmap = QPixmap(file_name)
            self.image1_mix.setPixmap(pixmap.scaled(
                self.image1_mix.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
            self.image1_mix.setScaledContents(False)
            self.image1_for_mixing = cv2.imread(file_name, cv2.IMREAD_COLOR)
            
            

    def open_image2_for_mixing(self, event):
        options = QtWidgets.QFileDialog.Options()
        file_name, _ = QtWidgets.QFileDialog.getOpenFileName(
            self.main_window, "Select Second Image for Mixing", "", "Image Files (*.png *.jpg *.bmp)", options=options)
        if file_name:
            pixmap = QPixmap(file_name)
            self.image2_mix.setPixmap(pixmap.scaled(
                self.image2_mix.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
            self.image2_mix.setScaledContents(False)
            self.image2_for_mixing = cv2.imread(file_name, cv2.IMREAD_COLOR)

    def mix_images(self):
        if self.image1_for_mixing is None or self.image2_for_mixing is None:
            
            QtWidgets.QMessageBox.warning(self.main_window, "Warning",
                                          "Please load both images for mixing by double-clicking on the image areas.")
            return

        
        
        
        kernel_size = self.main_window.mixing_kenel_size_slider.value()
        print(kernel_size)
        sigma = self.main_window.mixing_sigma_slider.value()     
        print(sigma)

        
        self.hybrid_image(self.image1_for_mixing,
                          self.image2_for_mixing, kernel_size, sigma)

    def open_image(self, event):
        options = QtWidgets.QFileDialog.Options()
        file_name, _ = QtWidgets.QFileDialog.getOpenFileName(
            self.main_window, "Select Image", "", "Image Files (*.png *.jpg *.bmp)", options=options)
        if file_name:
            pixmap = QPixmap(file_name)
            
            self.image_label.setPixmap(pixmap.scaled(
                self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
            self.image_label.setScaledContents(False)

            self.image = cv2.imread(file_name, cv2.IMREAD_UNCHANGED)
            self.gray_image = cv2.imread(file_name, cv2.IMREAD_GRAYSCALE)
            print(self.image.shape)
            if len(self.image.shape) == 3:
                self.is_input_gray = False
            elif len(self.image.shape) == 2:
                self.is_input_gray = True
            self.output_image = None
            self.histogram_control()
            if self.output_image_view.scene():
                self.output_image_view.scene().clear()
            

    def show_large_image(self, histogram, event):
        scene = histogram.scene()  # Get the scene of the clicked histogram
        if scene is not None:
            
            for item in scene.items():
                if isinstance(item, QGraphicsPixmapItem):  
                    print('Showing large image')

                    
                    dialog = QDialog(self.main_window)
                    dialog.setWindowTitle("Large Image")
                    layout = QVBoxLayout()

                    
                    label = QLabel()
                    label.setPixmap(item.pixmap().scaled(
                        600, 600, Qt.KeepAspectRatio))
                    label.setAlignment(Qt.AlignCenter)

                    layout.addWidget(label)
                    dialog.setLayout(layout)
                    dialog.resize(650, 650)

                    
                    dialog.exec_()
                    return  

        print('No QGraphicsPixmapItem found in the scene')

    def thresholding_control(self):
        if self.main_window.thresholding_combo.currentText() == 'Global':
            self.main_window.Global_widget.show()
            self.main_window.Local_widget.hide()
            self.apply_thresholding(
                self.image, type='global', threshold_value=self.main_window.threshold_slider.value())

        else:
            self.main_window.Local_widget.show()
            self.main_window.Global_widget.hide()
            self.apply_thresholding(self.image, type='local', window_size=self.main_window.window_size_slider.value(
            ), sensitivity=self.main_window.sensitivity_slider.value())

    def freq_control(self):
        if self.main_window.filter_combo.currentText() == 'high pass filter' or self.main_window.filter_combo.currentText() == 'low pass filter':
            self.main_window.cuttofffreq_widgwt.show()
            self.main_window.kernel_size_widget.hide()

        else:
            self.main_window.cuttofffreq_widgwt.hide()
            self.main_window.kernel_size_widget.show()

    def update_kernel_size(self, event):
        self.kernel_size = self.main_window.kernel_size_slider.value()
        self.main_window.kernel_size_label.setText(f"{self.kernel_size}")

    def apply_noise(self):
        
        if self.image is None:
            return

        noise_type = self.main_window.noise_type_combo.currentText()
        noise_intensity = self.main_window.noise_intensity_slider.value()
        if noise_type == 'guassian noise':
            noisy_image = add_gaussian_noise(self.image, sigma=noise_intensity)
        elif noise_type == 'salt and pepper noise':
            noisy_image = add_salt_pepper_noise(
                self.image, salt_prob=noise_intensity/100, pepper_prob=noise_intensity/100)
        elif noise_type == 'uniform noise':
            noisy_image = add_uniform_noise(
                self.image, intensity=noise_intensity)
        else:
            return
        
        self.output_image = noisy_image
        self.display_image(self.output_image_view, noisy_image)
        

    def apply_filter(self):
        if self.image is None:
            return
        filter_type = self.main_window.filter_combo.currentText()
        if filter_type == 'average filter':
            filtered_image = apply_average_filter(self.image, self.kernel_size)
        elif filter_type == 'gausssian filter':
            filtered_image = apply_gaussian_filter(
                self.image, self.kernel_size)
        elif filter_type == 'median filter':
            filtered_image = apply_median_filter(self.image, self.kernel_size)
        elif filter_type == 'high pass filter':
            filtered_image = self.apply_frequency_filters(
                self.image, filter_type='high', D0=self.main_window.cuttoff_freq_slider.value())
        elif filter_type == 'low pass filter':
            filtered_image = self.apply_frequency_filters(
                self.image, filter_type='low', D0=self.main_window.cuttoff_freq_slider.value())
        else:
            return
        self.output_image = filtered_image
        self.display_image(self.output_image_view, filtered_image)
        

    def display_image(self, view, image):
        pixmap = self.convert_cv_to_pixmap(image)
        scene = QGraphicsScene()
        scene.addPixmap(pixmap)
        view.setScene(scene)
        view.fitInView(
            scene.itemsBoundingRect(), Qt.KeepAspectRatio)

        # update output is gray?
        if view == self.output_image_view:
            if len(image.shape) == 3:
                self.is_output_gray = False
            elif len(image.shape) == 2:
                self.is_output_gray = True

            self.histogram_control() # update histograms

    def plot_histogram(self, data):
        fig, ax = plt.subplots(figsize=(5, 4), dpi=150)
        ax.plot(data, color='blue')  
        ax.set_title('Histogram')
        ax.set_xlabel('Pixel Intensity')
        ax.set_ylabel('Frequency')
        fig.canvas.draw()
        width, height = fig.canvas.get_width_height()
        img = np.frombuffer(fig.canvas.tostring_rgb(),
                            dtype=np.uint8).reshape(height, width, 3)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  

        plt.close(fig)  
        return img

    def convert_cv_to_pixmap(self, cv_img):
        """ Converts an OpenCV image to QPixmap (automatically detects grayscale or RGB) """
        if len(cv_img.shape) == 2:  
            height, width = cv_img.shape
            bytes_per_line = width
            q_img = QImage(cv_img.data, width, height,
                           bytes_per_line, QImage.Format_Grayscale8)
        else:  
            height, width, channel = cv_img.shape
            bytes_per_line = 3 * width
            q_img = QImage(cv_img.data, width, height,
                           bytes_per_line, QImage.Format_RGB888).rgbSwapped()

        return QPixmap.fromImage(q_img)

    def grayscale_hist(self, image):
        histogram, equalized_hist, new_image = HistogramEqualization.equalize(
            image)
        # Plot histograms as images
        hist_pixmap = self.plot_histogram(histogram)
        equalized_hist_pixmap = self.plot_histogram(equalized_hist)

        # Display in respective QGraphicsView widgets
        self.display_image(self.equalized_histogram, equalized_hist_pixmap)
        self.display_image(self.original_histogram, hist_pixmap)

        # self.output_image = new_image

    def equalize_grayscale(self):
        histogram, equalized_hist, new_image = HistogramEqualization.equalize(
            self.image)
        self.output_image = new_image
        self.display_image(self.output_image_view, new_image)
        

    def detect_edges(self):
        self.edge_detector.mask_selection = self.main_window.edge_detection_method_combo.currentText()
        self.edge_detector.image = self.gray_image
        gradient_magnitude = [[]]

        if self.edge_detector.mask_selection == "Sobel":
            Gx, Gy = self.edge_detector.sobel_kernel()
            gradient_magnitude = np.sqrt(Gx ** 2 + Gy ** 2)

        elif self.edge_detector.mask_selection == "Prewitt":
            Gx, Gy = self.edge_detector.prewitt_kernel()
            gradient_magnitude = np.sqrt(Gx ** 2 + Gy ** 2)

        elif self.edge_detector.mask_selection == "Roberts":
            Gx, Gy = self.edge_detector.roberts_kernel()
            gradient_magnitude = np.sqrt(Gx ** 2 + Gy ** 2)

        elif self.edge_detector.mask_selection == "Canny":
            gradient_magnitude = self.edge_detector.canny_kernel(
                self.edge_detector.image)

        
        gradient_magnitude = (gradient_magnitude / gradient_magnitude.max()) * 255
        gradient_magnitude = gradient_magnitude.astype(np.uint8)
        self.output_image = gradient_magnitude
        self.display_image(self.output_image_view, gradient_magnitude)
        

    def convert_to_grayscale(self, image):
        if len(image.shape) == 3:
            gray_image = RGB2GRAY.convert_to_grayscale(image)
            return gray_image
        return image

    def apply_grayscale(self):
        if self.image is None:
            return

        gray_image = self.convert_to_grayscale(self.image)
        self.output_image = gray_image
        self.display_image(self.output_image_view, gray_image)
        

    def apply_thresholding(self, image, threshold_value=128, window_size=15, sensitivity=2, type='global'):
        gray_image = self.convert_to_grayscale(image)

        if type == 'global':
            image = Thresholding.global_thresholding(
                gray_image, threshold_value)

        elif type == 'local':
            image = Thresholding.local_thresholding(
                gray_image, window_size, sensitivity)

        else:
            return

        self.output_image = image
        self.display_image(self.output_image_view, image)
        

    def apply_frequency_filters(self, image, filter_type='high', D0=30):
        gray_image = self.convert_to_grayscale(image)

        image = cv2.resize(gray_image, (256, 256))
 
        image = np.float32(image)

        rows, cols = image.shape
        x = np.arange(rows)
        y = np.arange(cols)
        X, Y = np.meshgrid(y, x)
        centered_data = image * ((-1) ** (X + Y))

        F = Freq_filters.DFT_2D(centered_data)
        H = Freq_filters.create_filter(rows, cols, D0, filter_type)
        G = Freq_filters.apply_filter(F, H)
        g = Freq_filters.IDFT_2D(G)
        recentered_data = g * ((-1) ** (X + Y))

        normalized_image = Freq_filters.normalize_image(recentered_data)
 
        image = np.array(normalized_image, dtype=np.uint8)
        return image

    def hybrid_image(self, image1, image2, kernel_size, sigma):
        
        image1 = cv2.resize(image1, (256, 256))
        image2 = cv2.resize(image2, (256, 256))

        image1 = np.float32(image1)
        image2 = np.float32(image2)

        
        low_frequencies = Hybrid.extract_low_frequencies(
            image1, kernel_size=kernel_size, sigma=sigma)
        high_frequencies = Hybrid.extract_high_frequencies(image2, kernel_size=kernel_size, sigma=sigma)

        # Combine frequencies
        hybrid_image = Hybrid.combine_frequencies(
            low_frequencies, high_frequencies)

        hybrid_image = np.array(hybrid_image, dtype=np.uint8)

        
        height, width, channel = hybrid_image.shape
        bytes_per_line = 3 * width
        q_image = QImage(hybrid_image.data, width, height,
                         bytes_per_line, QImage.Format_RGB888).rgbSwapped()

        
        pixmap = QPixmap.fromImage(q_image)

        
        self.output_img_mix.setPixmap(pixmap.scaled(
            self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def rgb_hist(self, image):

        
        height, width, channels = image.shape
        R_values, G_values, B_values = RGB_Hist.extract_rgb_channels(
            image, width, height)

        
        R_histogram = RGB_Hist.compute_histogram(R_values)
        G_histogram = RGB_Hist.compute_histogram(G_values)
        B_histogram = RGB_Hist.compute_histogram(B_values)

        
        self.R_cdf = RGB_Hist.compute_cdf(R_histogram)
        self.G_cdf = RGB_Hist.compute_cdf(G_histogram)
        self.B_cdf = RGB_Hist.compute_cdf(B_histogram)

        
        self.display_image(self.main_window.histogram_1,
                           RGB_Hist.plot_histogram(R_histogram, 'Red'))
        self.display_image(self.main_window.histogram_2,
                           RGB_Hist.plot_histogram(G_histogram, 'Green'))
        self.display_image(self.main_window.histogram_3,
                           RGB_Hist.plot_histogram(B_histogram, 'Blue'))
        self.display_image(self.main_window.histogram_4,
                           RGB_Hist.plot_combined_cdf(self.R_cdf, self.G_cdf, self.B_cdf))


        
        # return R_cdf, G_cdf, B_cdf

    def RGB_historgram_equalization(self, image, R_cdf, G_cdf, B_cdf):
        # Get image dimensions
        height, width, channels = image.shape
        R_values, G_values, B_values = RGB_Hist.extract_rgb_channels(
            image, width, height)

        
        R_equalized = RGB_Hist.histogram_equalization(R_values, R_cdf)
        G_equalized = RGB_Hist.histogram_equalization(G_values, G_cdf)
        B_equalized = RGB_Hist.histogram_equalization(B_values, B_cdf)

        
        equalized_image = cv2.merge([
            np.array(B_equalized, dtype=np.uint8).reshape(height, width),
            np.array(G_equalized, dtype=np.uint8).reshape(height, width),
            np.array(R_equalized, dtype=np.uint8).reshape(height, width)
        ])

        # self.rgb_hist(equalized_image)
        self.output_image = equalized_image
        self.display_image(self.output_image_view, equalized_image)
        

        return equalized_image

    def export(self):
        if self.output_image is None:
            return
        file_dialog = QFileDialog(self.main_window)
        file_dialog.setAcceptMode(QFileDialog.AcceptSave)
        file_dialog.setNameFilter('Images (*.png *.jpg *.bmp *.tiff)')
        file_dialog.setDefaultSuffix('png')

        if file_dialog.exec_():
            file_path = file_dialog.selectedFiles()[0]
            print(type(self.output_image), 555, self.output_image.ndim)
            if self.output_image.ndim == 3 and self.output_image.shape[2] == 3:
                height, width, channels = self.output_image.shape
                qimg = QImage(self.output_image.data, width, height, 3 * width, QImage.Format_RGB888)                
            elif self.output_image.ndim == 2:
                    height, width = self.output_image.shape
                    qimg = QImage(self.output_image.data, width, height, width, QImage.Format_Grayscale8)                    
            qimg.save(file_path)

    def normalize(self):
        if self.image is None:
            return
        self.output_image = Freq_filters.normalize_image(self.image)
        self.display_image(self.output_image_view, self.output_image)
