import math
import cmath
import cv2
import numpy as np

class Freq_filters:
    @staticmethod
    def DFT_2D(image):
        # Apply 2D FFT using NumPy
        F = np.fft.fft2(image)
        return F

    @staticmethod
    def create_filter(M, N, D0, filter_type='low'):
        # Create a meshgrid for frequency domain coordinates
        u = np.arange(M)
        v = np.arange(N)
        U, V = np.meshgrid(u, v, indexing='ij')
        D = np.sqrt((U - M//2)**2 + (V - N//2)**2)
        
        if filter_type == 'low':
            H = np.where(D <= D0, 1, 0)
        elif filter_type == 'high':
            H = np.where(D > D0, 1, 0)
        return H
    
    @staticmethod
    def apply_filter(F, H):
        # Element-wise multiplication
        G = F * H
        return G

    @staticmethod
    def IDFT_2D(F):
        # Apply 2D inverse FFT using NumPy
        image = np.fft.ifft2(F)
        return image.real  # Return the real part
    
    @staticmethod
    def normalize_image(image):
        M = len(image)
        N = len(image[0])
        min_val = min([min(row) for row in image])
        max_val = max([max(row) for row in image])
        normalized = [[0 for _ in range(N)] for _ in range(M)]
        for y in range(M):
            for x in range(N):
                normalized[y][x] = ((image[y][x] - min_val) / (max_val - min_val)) * 255
        return normalized
    
    # @staticmethod
    # def normalize_image(image):
    #     # Normalize the image to the range 0-255
    #     normalized = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
    #     return normalized
