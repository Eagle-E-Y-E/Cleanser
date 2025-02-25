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
        image_shape = image.shape
        histogram = HistogramEqualization.compute_histogram(image)
        cdf_norm = HistogramEqualization.compute_CDF(histogram, grey_levels)
        equalized_hist = np.zeros(grey_levels, dtype=int)
        for i in range(grey_levels):
            equalized_hist[cdf_norm[i]] += histogram[i]
        new_flattened_image = cdf_norm[image]
        new_image = new_flattened_image.reshape(image_shape)
        return histogram, equalized_hist, new_image

image_path = 'dark.jpg'
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
if image is None:
    raise ValueError("Image not found or could not be loaded.")

histogram, equalized_hist, new_image = HistogramEqualization.equalize(image)

# Create a figure
plt.figure(figsize=(10, 10))

# Plot image1
plt.subplot(2, 2, 1)
plt.imshow(image, cmap='gray')
plt.title('Equalized Image')
plt.axis('off')  # Hide axis

# Plot image2
plt.subplot(2, 2, 2)
plt.plot(histogram)
plt.title('Histogram of Original Image')
plt.xlabel('Pixel Intensity')
plt.ylabel('Frequency')
plt.axis('off')

# Plot image3
plt.subplot(2, 2, 3)
plt.imshow(new_image, cmap='gray')
plt.title('Equalized Image')
plt.axis('off')  # Hide axis

# Plot image4
plt.subplot(2, 2, 4)
plt.plot(equalized_hist)
plt.title('Histogram of Equalized Image')
plt.xlabel('Pixel Intensity')
plt.ylabel('Frequency')
plt.axis('off')

# Show the plot
plt.show()