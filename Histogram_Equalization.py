import cv2
import numpy as np
import matplotlib.pyplot as plt

class Hisogram_Equalization:
    @staticmethod
    def compute_histogram(image):
        histogram = np.zeros(256, dtype=int)
        for pixel_value in image.flatten():
            histogram[pixel_value] += 1
        return histogram

    @staticmethod
    def compute_CDF(histogram):
        cdf_norm = np.cumsum(histogram) / np.sum(histogram) 
        return cdf_norm

    @staticmethod
    def equalize(image, grey_levels=255):
        image_shape = image.shape
        flatten_image = image.flatten()
        histogram = Hisogram_Equalization.compute_histogram(flatten_image)
        cdf_norm = Hisogram_Equalization.compute_CDF(histogram)
        equalized_values = np.round(cdf_norm * (grey_levels - 1)).astype(int)
        equalized_hist = np.zeros(grey_levels, dtype=int)
        for i in range(grey_levels):
            equalized_hist[equalized_values[i]] += histogram[i]
        new_flattened_image = equalized_values[flatten_image]
        new_image = new_flattened_image.reshape(image_shape)
        return equalized_hist, new_image

image_path = 'ED-image1.png'
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
if image is None:
    raise ValueError("Image not found or could not be loaded.")

_, new_image = Hisogram_Equalization.equalize(image, 8)
plt.imshow(new_image, cmap='gray')
plt.title('Equalized Image')
plt.axis('off')  # Hide axis
plt.show()