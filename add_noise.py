import cv2
import numpy as np
import matplotlib.pyplot as plt


def add_gaussian_noise(image, mean=0, sigma=25):
    sigma = sigma * 2.55  # Convert to [0, 255] range
    noisy = image + np.random.normal(mean, sigma, image.shape) # guassian = normal distribution
    return np.clip(noisy, 0, 255).astype(np.uint8)


def add_salt_pepper_noise(image, salt_prob=0.02, pepper_prob=0.02):
    noisy = np.copy(image)
    total_pixels = image.size
    num_salt = int(total_pixels * salt_prob)
    num_pepper = int(total_pixels * pepper_prob)

    # Add salt (white) noise
    salt_coords = [np.random.randint(0, i, num_salt) for i in image.shape]
    noisy[salt_coords[0], salt_coords[1]] = 255

    # Add pepper (black) noise
    pepper_coords = [np.random.randint(0, i, num_pepper) for i in image.shape]
    noisy[pepper_coords[0], pepper_coords[1]] = 0

    return noisy


def add_uniform_noise(image, intensity=50):
    intensity = intensity * 2.55  # Convert to [0, 255] range
    noisy = image + np.random.uniform(-intensity, intensity, image.shape)
    return np.clip(noisy, 0, 255).astype(np.uint8)
