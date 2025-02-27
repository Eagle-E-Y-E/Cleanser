import cv2
import numpy as np
import matplotlib.pyplot as plt

class HistogramEqualization:
    @staticmethod
    def compute_histogram(image):
        histogram = np.zeros(256, dtype=int)
        for pixel_value in image.flatten():
            histogram[pixel_value] += 1
        return histogram

    @staticmethod
    def compute_CDF(histogram, grey_levels=256):
        cdf = np.cumsum(histogram)
        cdf_norm = np.round(cdf / cdf[-1] * (grey_levels - 1)).astype(int)
        return cdf_norm

    @staticmethod
    def equalize(image, grey_levels=256):
        if image is None: return
        image_shape = image.shape
        histogram = HistogramEqualization.compute_histogram(image)
        cdf_norm = HistogramEqualization.compute_CDF(histogram, grey_levels)
        equalized_hist = np.zeros(grey_levels, dtype=int)
        for i in range(grey_levels):
            equalized_hist[cdf_norm[i]] += histogram[i]
        new_flattened_image = cdf_norm[image]
        new_image = new_flattened_image.reshape(image_shape)
        return histogram, equalized_hist, new_image.astype(np.uint8)
