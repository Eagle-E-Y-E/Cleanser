import cv2
import numpy as np


class EdgeDetection:
    def __init__(self):
        self.mask_selection = "Prewitt"

        self.roberts_x = None
        self.roberts_y = None

        self.prewitt_x = None
        self.prewitt_y = None

        self.sobel_x = None
        self.sobel_y = None

        self.image = None
        # if self.image is None:
        #     raise ValueError("Error: Unable to load image. Check the file path.")

        self.height, self.width = 0, 0

    def apply_kernel(self, kernel):
        self.height, self.width = self.image.shape[0], self.image.shape[1]
        pad = 1
        # pad the edges with zeroes to perserve the boundary pixels
        padded_image = np.pad(self.image, pad, mode='constant')
        # Make a zeros matrix of the same size as the image to store values later
        output_image = np.zeros_like(self.image)

        n = 1
        k = 1
        if self.mask_selection == "Roberts":
            n = 2
            k = 2
        elif self.mask_selection == "Sobel":  # Sobel
            n = 3
            k = 8
        elif self.mask_selection == "Prewitt":
            n = 3
            k = 6

        for i in range(self.height):
            for j in range(self.width):
                # from i to i+n because the End in slicing is excluded
                region = padded_image[i:i + n, j:j + n]
                output_image[i, j] = np.sum(region * kernel) / k

        return output_image

    def sobel_kernel(self):
        self.sobel_x = np.array([[-1, 0, 1],
                                 [-2, 0, 2],
                                 [-1, 0, 1]])

        self.sobel_y = np.array([[-1, -2, -1],
                                 [0, 0, 0],
                                 [1, 2, 1]])
        output_sobel_x = self.apply_kernel(self.sobel_x)
        output_sobel_y = self.apply_kernel(self.sobel_y)
        return output_sobel_x, output_sobel_y

    def prewitt_kernel(self):
        self.prewitt_x = np.array([[-1, 0, 1],
                                   [-1, 0, 1],
                                   [-1, 0, 1]])

        self.prewitt_y = np.array([[-1, -1, -1],
                                   [0, 0, 0],
                                   [1, 1, 1]])
        output_prewitt_x = self.apply_kernel(self.prewitt_x)
        output_prewitt_y = self.apply_kernel(self.prewitt_y)
        return output_prewitt_x, output_prewitt_y

    def roberts_kernel(self):
        self.roberts_x = np.array([[1, 0],
                                   [0, -1]])

        self.roberts_y = np.array([[0, 1],
                                   [-1, 0]])

        output_roberts_x = self.apply_kernel(self.roberts_x)
        output_roberts_y = self.apply_kernel(self.roberts_y)
        return output_roberts_x, output_roberts_y

    def canny_kernel(self, image):
        #Apply Gaussian blur
        # Compute gradients using Sobel filters.
        # Apply Non-Maximum Suppression.
        # Implement Double Thresholding.
        # Apply Edge Tracking by Hysteresis.
        blurred = cv2.GaussianBlur(image, (5, 5), 1.4)
        output_image = cv2.Canny(blurred, 20, 200)
        return output_image
