import math
import cmath

class Freq_filters:
    @staticmethod
    def DFT_2D(image):
        M = len(image)        # Number of rows (height)
        N = len(image[0])     # Number of columns (width)
        F = [[0 for _ in range(N)] for _ in range(M)]
        
        for u in range(M):
            for v in range(N):
                sum_value = 0
                for y in range(M):
                    for x in range(N):
                        angle = -2 * math.pi * ((u * y) / M + (v * x) / N)
                        exponent = cmath.exp(complex(0, angle))
                        sum_value += image[y][x] * exponent
                F[u][v] = sum_value
        return F

    @staticmethod
    def create_filter(M, N, D0, filter_type='low'):
        H = [[0 for _ in range(N)] for _ in range(M)]
        center_u, center_v = M // 2, N // 2
        for u in range(M):
            for v in range(N):
                D = math.sqrt((u - center_u) ** 2 + (v - center_v) ** 2)
                if filter_type == 'low':
                    H[u][v] = 1 if D <= D0 else 0
                elif filter_type == 'high':
                    H[u][v] = 0 if D <= D0 else 1
        return H
    
    @staticmethod
    def apply_filter(F, H):
        M = len(F)
        N = len(F[0])
        G = [[0 for _ in range(N)] for _ in range(M)]
        for u in range(M):
            for v in range(N):
                G[u][v] = F[u][v] * H[u][v]
        return G

    @staticmethod
    def IDFT_2D(F):
        M = len(F)
        N = len(F[0])
        image = [[0 for _ in range(N)] for _ in range(M)]
        
        for y in range(M):
            for x in range(N):
                sum_value = 0
                for u in range(M):
                    for v in range(N):
                        angle = 2 * math.pi * ((u * y) / M + (v * x) / N)
                        exponent = cmath.exp(complex(0, angle))
                        sum_value += F[u][v] * exponent
                image[y][x] = (1 / (M * N)) * sum_value.real  # Take the real part
        return image  # Return the reconstructed image
    
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
