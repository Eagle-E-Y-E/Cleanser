from PyQt5.QtCore import Qt
from PyQt5 import QtWidgets, uic
from PyQt5.QtGui import QPixmap, QImage, QPainter
from PyQt5.QtWidgets import QFileDialog, QGraphicsScene, QGraphicsPixmapItem
from PyQt5.QtWidgets import QApplication, QGraphicsView, QGraphicsScene, QGraphicsPixmapItem, QLabel, QVBoxLayout, QWidget
import cv2
import numpy as np
from add_noise import add_gaussian_noise, add_salt_pepper_noise, add_uniform_noise
from apply_filter import apply_average_filter, apply_gaussian_filter, apply_median_filter
from Histogram_Equalization import HistogramEqualization
from Thresholding import Thresholding
from RGB2GRAY import RGB2GRAY
from RGB_Hist import RGB_Hist
from Freq_filters import Freq_filters
from Hybrid import Hybrid
import matplotlib.pyplot as plt

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
        # sliders
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

        self.main_window.histogram_1.mouseDoubleClickEvent = self.show_large_image
        self.large_image_dialog = None
        self.main_window.thresholding_combo.currentIndexChanged.connect(
            self.thresholding_control)
        self.main_window.Local_widget.hide()
        self.main_window.convert_to_grayscale_btn.clicked.connect(
            self.apply_grayscale)
        self.main_window.cuttofffreq_widgwt.hide()
        self.main_window.filter_combo.currentIndexChanged.connect(
            self.freq_control)

        self.main_window.mix_btn.clicked.connect(self.mix_images)
        # Add references to image mix labels if not already initialized
        self.image1_mix = self.main_window.findChild(
            QtWidgets.QLabel, 'image1_mix')
        self.image2_mix = self.main_window.findChild(
            QtWidgets.QLabel, 'image2_mix')
        self.output_img_mix = self.main_window.findChild(
            QtWidgets.QLabel, 'output_img_mix')

        # Initialize image storage variables
        self.image1_for_mixing = None
        self.image2_for_mixing = None

        # Let users double-click to load images into the mixing panes
        self.image1_mix.mouseDoubleClickEvent = self.open_image1_for_mixing
        self.image2_mix.mouseDoubleClickEvent = self.open_image2_for_mixing

        self.image = None
        self.gray_image = None
        self.main_window.equalize_image_btn.clicked.connect(
            self.equalize_image)
        self.kernel_size = 3
        self.window_size = self.main_window.window_size_slider.value()
        self.sensitivity = self.main_window.sensitivity_slider.value()
        self.threshold = self.main_window.threshold_slider.value()

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
            # print(self.image1_for_mixing.shape)
            # self.display_image(self.image2_mix, self.image1_for_mixing)

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
            # Show error message or return silently
            QtWidgets.QMessageBox.warning(self.main_window, "Warning",
                                          "Please load both images for mixing by double-clicking on the image areas.")
            return

        # Get kernel size and sigma from UI or use default values
        # You might want to add sliders for these parameters in your UI
        # Default or from a slider
        kernel_size = self.main_window.mixing_kenel_size_slider.value()
        print(kernel_size)
        sigma = self.main_window.mixing_sigma_slider.value()     # Default or from a slider
        print(sigma)

        # Call the hybrid image function
        self.hybrid_image(self.image1_for_mixing,
                          self.image2_for_mixing, kernel_size, sigma)

    def open_image(self, event):
        options = QtWidgets.QFileDialog.Options()
        file_name, _ = QtWidgets.QFileDialog.getOpenFileName(
            self.main_window, "Select Image", "", "Image Files (*.png *.jpg *.bmp)", options=options)
        if file_name:
            pixmap = QPixmap(file_name)
            # Preserve aspect ratio in QLabel
            self.image_label.setPixmap(pixmap.scaled(
                self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
            self.image_label.setScaledContents(False)

            self.image = cv2.imread(file_name, cv2.IMREAD_COLOR)
            self.gray_image = cv2.imread(file_name, cv2.IMREAD_GRAYSCALE)
            self.rgb_hist(self.image)

    def show_large_image(self, event):
        # Get the scene from the QGraphicsView
        scene = self.main_window.histogram_1.scene()
        if scene is not None:
            # Iterate through the items in the scene to find the QGraphicsPixmapItem
            for item in scene.items():
                if isinstance(item, QGraphicsPixmapItem):
                    print('Showing large image')
                    dialog = QWidget()
                    dialog.setWindowTitle("Large Image")
                    layout = QVBoxLayout()

                    label = QLabel()
                    label.setPixmap(item.pixmap().scaled(
                        600, 600, Qt.KeepAspectRatio))  # Scale it larger
                    label.setAlignment(Qt.AlignCenter)

                    layout.addWidget(label)
                    dialog.setLayout(layout)
                    dialog.resize(650, 650)
                    dialog.show()
                    return  # Exit after showing the first found pixmap item
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
        # print('Applying noise')
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
            filtered_image = self.apply_frequency_filters(self.image, filter_type='high', D0=self.main_window.cuttoff_freq_slider.value())
        elif filter_type == 'low pass filter':
            filtered_image = self.apply_frequency_filters(self.image, filter_type='low', D0=self.main_window.cuttoff_freq_slider.value())
        else:
            return
        self.display_image(self.output_image_view, filtered_image)

    def display_image(self, view, image):
        pixmap = self.convert_cv_to_pixmap(image)
        scene = QGraphicsScene()
        scene.addPixmap(pixmap)
        view.setScene(scene)
        view.fitInView(
            scene.itemsBoundingRect(), Qt.KeepAspectRatio)

    def plot_histogram(self, data):
        fig, ax = plt.subplots(figsize=(5, 4), dpi=150)
        ax.plot(data, color='blue')  # Black color for grayscale consistency
        ax.set_title('Histogram')
        ax.set_xlabel('Pixel Intensity')
        ax.set_ylabel('Frequency')
        fig.canvas.draw()
        width, height = fig.canvas.get_width_height()
        img = np.frombuffer(fig.canvas.tostring_rgb(),
                            dtype=np.uint8).reshape(height, width, 3)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  # Convert to grayscale

        plt.close(fig)  # Close the figure to prevent memory leaks
        return img

    def convert_cv_to_pixmap(self, cv_img):
        """ Converts an OpenCV image to QPixmap (automatically detects grayscale or RGB) """
        if len(cv_img.shape) == 2:  # Grayscale
            height, width = cv_img.shape
            bytes_per_line = width
            q_img = QImage(cv_img.data, width, height,
                           bytes_per_line, QImage.Format_Grayscale8)
        else:  # RGB
            height, width, channel = cv_img.shape
            bytes_per_line = 3 * width
            q_img = QImage(cv_img.data, width, height,
                           bytes_per_line, QImage.Format_RGB888).rgbSwapped()

        return QPixmap.fromImage(q_img)

    def equalize_image(self):
        histogram, equalized_hist, new_image = HistogramEqualization.equalize(
            self.gray_image)
        # Plot histograms as images
        hist_pixmap = self.plot_histogram(histogram)
        equalized_hist_pixmap = self.plot_histogram(equalized_hist)

        # Display in respective QGraphicsView widgets
        self.display_image(self.original_histogram, hist_pixmap)
        self.display_image(self.equalized_histogram, equalized_hist_pixmap)
        self.display_image(self.output_image_view, new_image)

    def convert_to_grayscale(self, image):
        if len(image.shape) == 3:
            gray_image = RGB2GRAY.convert_to_grayscale(image)

        return gray_image

    def apply_grayscale(self):
        if self.image is None:
            return

        gray_image = self.convert_to_grayscale(self.image)
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

        # diplay the thresholded image
        self.display_image(self.output_image_view, image)

    def apply_frequency_filters(self, image, filter_type='high', D0=30):
        gray_image = self.convert_to_grayscale(image)

        image = cv2.resize(gray_image, (256, 256))

        # Convert image to float32 for FFT processing
        image = np.float32(image)

        # Center the image by multiplying with (-1)^(x+y)
        rows, cols = image.shape
        x = np.arange(rows)
        y = np.arange(cols)
        X, Y = np.meshgrid(y, x)
        centered_data = image * ((-1) ** (X + Y))

        # Apply 2D DFT
        F = Freq_filters.DFT_2D(centered_data)

        # Create a filter
        H = Freq_filters.create_filter(rows, cols, D0, filter_type)

        # Apply the filter
        G = Freq_filters.apply_filter(F, H)

        # Apply inverse DFT
        g = Freq_filters.IDFT_2D(G)

        # Re-center the image
        recentered_data = g * ((-1) ** (X + Y))

        # Normalize the image data
        normalized_image = Freq_filters.normalize_image(recentered_data)

        # Convert the normalized image to uint8
        image = np.array(normalized_image, dtype=np.uint8)
        return image
    
        # display the image
        # self.display_image(self.output_image_view, image)

    def hybrid_image(self, image1, image2, kernel_size, sigma):
        # Resize images to the same dimensions
        image1 = cv2.resize(image1, (256, 256))
        image2 = cv2.resize(image2, (256, 256))

        image1 = np.float32(image1)
        image2 = np.float32(image2)

        # Extract frequency components
        low_frequencies = Hybrid.extract_low_frequencies(
            image1, kernel_size=kernel_size, sigma=sigma)
        high_frequencies = Hybrid.extract_high_frequencies(image2)

        # Combine frequencies
        hybrid_image = Hybrid.combine_frequencies(
            low_frequencies, high_frequencies)

        hybrid_image = np.array(hybrid_image, dtype=np.uint8)

        # Convert numpy array to QImage
        height, width, channel = hybrid_image.shape
        bytes_per_line = 3 * width
        q_image = QImage(hybrid_image.data, width, height,
                         bytes_per_line, QImage.Format_RGB888).rgbSwapped()

        # Convert QImage to QPixmap
        pixmap = QPixmap.fromImage(q_image)

        # Display the pixmap
        self.output_img_mix.setPixmap(pixmap.scaled(
            self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def rgb_hist(self, image):

        # Get image dimensions
        height, width, channels = image.shape
        R_values, G_values, B_values = RGB_Hist.extract_rgb_channels(
            image, width, height)

        # Compute histograms
        R_histogram = RGB_Hist.compute_histogram(R_values)
        G_histogram = RGB_Hist.compute_histogram(G_values)
        B_histogram = RGB_Hist.compute_histogram(B_values)

        # Compute CDFs
        R_cdf = RGB_Hist.compute_cdf(R_histogram)
        G_cdf = RGB_Hist.compute_cdf(G_histogram)
        B_cdf = RGB_Hist.compute_cdf(B_histogram)

        # Plot histograms and CDFs
        self.display_image(self.main_window.histogram_1,
                           RGB_Hist.plot_histogram(R_histogram, 'Red'))
        self.display_image(self.main_window.histogram_2,
                           RGB_Hist.plot_histogram(G_histogram, 'Green'))
        self.display_image(self.main_window.histogram_3,
                           RGB_Hist.plot_histogram(B_histogram, 'Blue'))
        self.display_image(self.main_window.histogram_4,
                           RGB_Hist.plot_combined_cdf(R_cdf, G_cdf, B_cdf))

        # RGB_Hist.plot_histogram(G_histogram, 'Green')
        # RGB_Hist.plot_histogram(B_histogram, 'Blue')

        # # Plot combined CDFs
        # RGB_Hist.plot_combined_cdf(R_cdf, G_cdf, B_cdf)

        # # display the image
        # self.display_image(self.output_image_view, image)

        return R_cdf, G_cdf, B_cdf

    def historgram_equalization(self, image, R_cdf, G_cdf, B_cdf):
        # Get image dimensions
        height, width, channels = image.shape
        R_values, G_values, B_values = RGB_Hist.extract_rgb_channels(
            image, width, height)

        # Equalize each channel
        R_equalized = RGB_Hist.histogram_equalization(R_values, R_cdf)
        G_equalized = RGB_Hist.histogram_equalization(G_values, G_cdf)
        B_equalized = RGB_Hist.histogram_equalization(B_values, B_cdf)

        # Reconstruct the equalized image
        equalized_image = cv2.merge([
            np.array(B_equalized, dtype=np.uint8).reshape(height, width),
            np.array(G_equalized, dtype=np.uint8).reshape(height, width),
            np.array(R_equalized, dtype=np.uint8).reshape(height, width)
        ])

        RGB_Hist(equalized_image)

        return equalized_image
