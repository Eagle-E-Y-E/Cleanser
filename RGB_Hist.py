import matplotlib.pyplot as plt
import cv2
import numpy as np

class RGB_Hist:
    @staticmethod
    def extract_rgb_channels(image, width, height):
        """
        Extracts R, G, and B channel values from the image.

        Returns:
        - R_values: List of all red channel values.
        - G_values: List of all green channel values.
        - B_values: List of all blue channel values.
        """
        # Check if the image is RGB
        if len(image.shape) == 3 and image.shape[2] == 3:
            # The image is RGB (or BGR in OpenCV terms)
            height, width, channels = image.shape
            R_values = []
            G_values = []
            B_values = []

            for y in range(height):
                for x in range(width):
                    B, G, R , A = image[y, x]  # OpenCV uses BGR format
                    R_values.append(R)
                    G_values.append(G)
                    B_values.append(B)

            return R_values, G_values, B_values
        else:
            raise ValueError("The input image is not an RGB image.")

    @staticmethod
    def compute_histogram(channel_values):
        """
        Computes histogram of a channel without built-in functions.

        Parameters:
        - channel_values: List of intensity values for a channel.

        Returns:
        - histogram: List where index represents intensity value and value at index is the count.
        """
        # Initialize histogram with zeros for all possible intensity values (0-255)
        histogram = [0] * 256

        # Count the frequency of each intensity value
        for value in channel_values:
            histogram[value] += 1

        return histogram

    @staticmethod
    def compute_cdf(histogram):
        """
        Computes the cumulative distribution function of a histogram.

        Parameters:
        - histogram: The histogram list.

        Returns:
        - cdf: Normalized cumulative distribution function as a list.
        """
        cdf = []
        cumulative = 0
        total_pixels = sum(histogram)
        for count in histogram:
            cumulative += count
            cdf_value = cumulative / total_pixels  # Normalize to [0,1]
            cdf.append(cdf_value)
        return cdf

    @staticmethod
    def plot_histogram(histogram, color):
        fig, ax = plt.subplots(figsize=(10, 8), dpi=150)
        ax.bar(range(256), histogram, color=color.lower(), alpha=0.7)
        ax.set_title(f'{color} Histogram')
        ax.set_xlabel('Pixel Intensity')
        ax.set_ylabel('Frequency')
        ax.set_xlim(0, 255)
        ax.grid(True)
        fig.canvas.draw()
        width, height = fig.canvas.get_width_height()
        img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(height, width, 3)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        plt.close(fig)
        return img

    @staticmethod
    def plot_cdf(cdf, channel_name):
        """
        Plots the cumulative distribution function for the given channel using Matplotlib.

        Parameters:
        - cdf: The cumulative distribution function list.
        - channel_name: Name of the color channel.
        """
        plt.figure(figsize=(10, 4))
        plt.plot(range(256), cdf, color=channel_name.lower())
        plt.title(f'Cumulative Distribution Function for {channel_name} Channel')
        plt.xlabel('Intensity Value')
        plt.ylabel('Cumulative Probability')
        plt.xlim(0, 255)
        plt.ylim(0, 1)
        plt.grid(True)
        plt.show()

    @staticmethod
    def plot_combined_histogram(R_histogram, G_histogram, B_histogram):
        """
        Plots the combined histograms of the R, G, and B channels.

        Parameters:
        - R_histogram: Histogram list for the Red channel.
        - G_histogram: Histogram list for the Green channel.
        - B_histogram: Histogram list for the Blue channel.
        """
        plt.figure(figsize=(10, 4))
        plt.plot(range(256), R_histogram, color='red', label='Red')
        plt.plot(range(256), G_histogram, color='green', label='Green')
        plt.plot(range(256), B_histogram, color='blue', label='Blue')
        plt.title('Combined RGB Histograms')
        plt.xlabel('Intensity Value')
        plt.ylabel('Frequency')
        plt.xlim(0, 255)
        plt.legend()
        plt.grid(True)
        plt.show()

    @staticmethod
    def plot_combined_cdf(R_cdf, G_cdf, B_cdf):
        fig, ax = plt.subplots(figsize=(5, 4), dpi=150)
        ax.plot(R_cdf, color='red', label='Red CDF')
        ax.plot(G_cdf, color='green', label='Green CDF')
        ax.plot(B_cdf, color='blue', label='Blue CDF')
        ax.set_title('Combined CDF')
        ax.set_xlabel('Pixel Intensity')
        ax.set_ylabel('Cumulative Frequency')
        ax.legend()
        fig.canvas.draw()
        width, height = fig.canvas.get_width_height()
        img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(height, width, 3)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        plt.close(fig)
        return img

    @staticmethod
    def histogram_equalization(channel_values, cdf):
        """
        Applies histogram equalization to the channel values.

        Parameters:
        - channel_values: List of intensity values for a channel.
        - cdf: Cumulative distribution function for the channel.

        Returns:
        - equalized_values: List of equalized intensity values.
        """
        cdf_min = min(filter(lambda x: x > 0, cdf))  # Get the minimum non-zero value
        equalized_values = []
        for value in channel_values:
            equalized_value = round((cdf[value] - cdf_min) / (1 - cdf_min) * 255)
            equalized_values.append(equalized_value)
        return equalized_values
