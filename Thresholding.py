import numpy as np


class Thresholding:
    @staticmethod
    def global_thresholding(image, threshold):
        """
        Apply global thresholding to a grayscale image.

        Parameters:
        - image: 2D list or array representing the grayscale image.
        - threshold: Integer value for thresholding.

        Returns:
        - A new image array after applying global thresholding.
        """
        # Apply thresholding using NumPy vectorization
        thresholded_image = (image > threshold) * 255
        # ensures the image has the correct data type for saving and displaying
        thresholded_image = thresholded_image.astype('uint8')
        return thresholded_image

    @staticmethod
    def local_thresholding(image, window_size, C):
        """
        Apply local adaptive thresholding to a grayscale image.

        Parameters:
        - image: 2D NumPy array representing the grayscale image.
        - window_size: Size of the local window (must be an odd integer).
        - C: Constant subtracted from the mean to fine-tune the threshold.

        Returns:
        - A new image array after applying local thresholding.
        """
        # Ensure window size is odd
        if window_size % 2 == 0:
            raise ValueError("window_size must be an odd integer.")

        # Pad the image to handle borders
        pad_size = window_size // 2
        padded_image = np.pad(image, pad_size, mode='reflect')

        # Create an empty array for the thresholded image
        thresholded_image = np.zeros_like(image, dtype=np.uint8)

        # Iterate over each pixel in the original image
        for y in range(image.shape[0]):
            for x in range(image.shape[1]):
                # Extract the local window
                y0, y1 = y, y + window_size
                x0, x1 = x, x + window_size
                window = padded_image[y0:y1, x0:x1]

                # Calculate the local mean
                local_mean = np.mean(window)

                # Calculate local threshold and apply it
                T_local = local_mean - C
                thresholded_image[y, x] = 255 if image[y, x] > T_local else 0

        return thresholded_image

    @staticmethod
    def bradley_localThreshold(image, window_size=15, p=0.15):
        """
        Applies Bradley's thresholding to a grayscale image without using built-in functions.

        Parameters:
        - image: 2D NumPy array of the grayscale image.
        - window_size: Size of the local window (must be an odd integer).
        - p: Percentage parameter (positive value between 0 and 1).

        Returns:
        - binary_image: Thresholded binary image.
        """
        height, width = image.shape
        half_size = window_size // 2

        # Pad the image
        padded_image = np.pad(image, half_size, mode='reflect')

        # Initialize output image
        binary_image = np.zeros((height, width), dtype=np.uint8)

        # Iterate over each pixel
        for i in range(height):
            for j in range(width):
                x_start = j
                x_end = j + window_size
                y_start = i
                y_end = i + window_size

                window = padded_image[y_start:y_end, x_start:x_end]

                # Local mean
                mean = np.sum(window) / (window_size * window_size)

                # Threshold
                T = mean * (1 - p)

                # Apply threshold
                if image[i, j] > T:
                    binary_image[i, j] = 255
                else:
                    binary_image[i, j] = 0

        return binary_image
