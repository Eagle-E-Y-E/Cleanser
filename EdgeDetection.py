import cv2
import numpy as np


class EdgeDetection:
    def __init__(self, image_path):
        self.mask_selection = "Roberts"

        self.roberts_x = None
        self.roberts_y = None

        self.prewitt_x = None
        self.prewitt_y = None

        self.sobel_x = None
        self.sobel_y = None

        self.image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if self.image is None:
            raise ValueError("Error: Unable to load image. Check the file path.")

        self.height, self.width = self.image.shape

    def apply_kernel(self, kernel):
        pad = 1
        # pad the edges with zeroes to perserve the boundary pixels
        padded_image = np.pad(self.image, pad,mode='constant')
        # Make a zeros matrix of the same size as the image to store values later
        output_image = np.zeros_like(self.image)
        if self.mask_selection == "Roberts":
            n = 2
        else:
            n = 3
        for i in range(self.height):
            for j in range(self.width):
                # from i to i+n because the End in slicing is excluded
                region = padded_image[i:i + n, j:j + n]
                output_image[i, j] = np.sum(region * kernel)

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

    def detect_edges(self, save_path="output.jpg"):
        if self.mask_selection == "Sobel":
            Gx, Gy = self.sobel_kernel()
        elif self.mask_selection == "Prewitt":
            Gx, Gy = self.prewitt_kernel()
        elif self.mask_selection == "Roberts":
            Gx, Gy = self.roberts_kernel()
        elif self.mask_selection == "Canny":
            Gx, Gy = 0, 0

        gradient_magnitude = np.sqrt(Gx ** 2 + Gy ** 2)
        gradient_magnitude = (gradient_magnitude / gradient_magnitude.max()) * 255
        gradient_magnitude = gradient_magnitude.astype(np.uint8)

        # Save the image
        cv2.imwrite(save_path, gradient_magnitude)
        print(f"Edge-detected image saved as: {save_path}")


# Example Usage
if __name__ == "__main__":
    edge_detector = EdgeDetection("ED-image1_gray.png")
    edge_detector.detect_edges("output.jpg")
