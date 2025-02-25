import numpy as np

class Hybrid:
    @staticmethod
    def generate_gaussian_kernel(size, sigma):
        """
        Generate a square Gaussian kernel.

        Parameters:
        - size: Kernel size (must be an odd integer).
        - sigma: Standard deviation of the Gaussian.

        Returns:
        - kernel: 2D list representing the Gaussian kernel.
        """
        kernel = []
        center = size // 2
        total = 0

        for x in range(size):
            row = []
            for y in range(size):
                exponent = -((x - center) ** 2 + (y - center) ** 2) / (2 * sigma ** 2)
                value = (1 / (2 * np.pi * sigma ** 2)) * np.exp(exponent)
                row.append(value)
                total += value
            kernel.append(row)

        # Normalize the kernel
        for x in range(size):
            for y in range(size):
                kernel[x][y] /= total

        return kernel

    @staticmethod
    def convolve_image(image, kernel):
        """
        Apply convolution between an image and a kernel.

        Parameters:
        - image: 2D or 3D numpy array of the image.
        - kernel: 2D list representing the kernel.

        Returns:
        - convolved_image: numpy array of the filtered image.
        """
        if len(image.shape) == 2:
            # Grayscale image
            height, width = image.shape
            channels = 1
        else:
            # Color image
            height, width, channels = image.shape

        k_size = len(kernel)
        pad = k_size // 2
        # Pad the image for each channel
        if channels == 1:
            padded_image = np.pad(image, pad, mode='edge')
        else:
            padded_image = np.pad(image, ((pad, pad), (pad, pad), (0, 0)), mode='edge')

        convolved_image = np.zeros_like(image)

        for c in range(channels):
            for i in range(height):
                for j in range(width):
                    acc = 0
                    for ki in range(k_size):
                        for kj in range(k_size):
                            pi = i + ki
                            pj = j + kj
                            if channels == 1:
                                acc += padded_image[pi, pj] * kernel[ki][kj]
                            else:
                                acc += padded_image[pi, pj, c] * kernel[ki][kj]
                    if channels == 1:
                        convolved_image[i, j] = acc
                    else:
                        convolved_image[i, j, c] = acc
        return convolved_image

    @staticmethod
    def extract_low_frequencies(image):
        """
        Extract low frequencies using Gaussian blur.

        Parameters:
        - image: 2D or 3D numpy array of the image.

        Returns:
        - low_frequencies: numpy array of the low-frequency components.
        """
        kernel_size = 15  # Adjust based on desired blurriness
        sigma = 5
        gaussian_kernel = Hybrid.generate_gaussian_kernel(kernel_size, sigma)
        low_frequencies = Hybrid.convolve_image(image, gaussian_kernel)
        return low_frequencies

    @staticmethod
    def extract_high_frequencies(image):
        """
        Extract high frequencies by subtracting the blurred image from the original.

        Parameters:
        - image: 2D or 3D numpy array of the image.

        Returns:
        - high_frequencies: numpy array of the high-frequency components.
        """
        low_frequencies = Hybrid.extract_low_frequencies(image)
        high_frequencies = image - low_frequencies
        return high_frequencies

    
    @staticmethod
    def combine_frequencies(low_freq_image, high_freq_image):
        """
        Combine low and high frequency components.

        Parameters:
        - low_freq_image: 2D numpy array of low-frequency image.
        - high_freq_image: 2D numpy array of high-frequency image.

        Returns:
        - hybrid_image: 2D numpy array of the hybrid image.
        """
        hybrid_image = low_freq_image + high_freq_image

        # Clip values to valid range [0, 255]
        hybrid_image = np.clip(hybrid_image, 0, 255)
        return hybrid_image

