import numpy as np

def apply_average_filter(image, kernel_size=30):
    kernel = np.ones((kernel_size, kernel_size)) / (kernel_size ** 2)
    # pad image to ensure that filter is applied to edges
    padded_image = np.pad(image, ((kernel_size//2, kernel_size//2), (kernel_size //
                          2, kernel_size//2), (0, 0)), mode='constant', constant_values=0)
    filtered_image = np.zeros_like(image)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            for k in range(image.shape[2]):
                window = padded_image[i:i+kernel_size, j:j+kernel_size, k]
                filtered_image[i, j, k] = np.sum(window * kernel)

    return np.clip(filtered_image, 0, 255).astype(np.uint8)


def apply_gaussian_filter(image, kernel_size=3, sigma=100):

    ax = np.linspace(-(kernel_size // 2), kernel_size // 2,
                     kernel_size)  # Generate a Gaussian kernel
    gauss = np.exp(-0.5 * np.square(ax) / np.square(sigma))
    kernel = np.outer(gauss, gauss)
    kernel /= np.sum(kernel)

    # Apply convolution
    padded_image = np.pad(image, ((kernel_size//2, kernel_size//2), (kernel_size //
                          2, kernel_size//2), (0, 0)), mode='constant', constant_values=0)
    filtered_image = np.zeros_like(image)

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            for k in range(image.shape[2]):
                region = padded_image[i:i+kernel_size, j:j+kernel_size, k]
                filtered_image[i, j, k] = np.sum(region * kernel)

    return np.clip(filtered_image, 0, 255).astype(np.uint8)


def apply_median_filter(image, kernel_size=3):
    padded_image = np.pad(image, ((kernel_size//2, kernel_size//2), (kernel_size //
                          2, kernel_size//2), (0, 0)), mode='constant', constant_values=0)
    filtered_image = np.zeros_like(image)

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            for k in range(image.shape[2]):
                region = padded_image[i:i+kernel_size, j:j+kernel_size, k]
                filtered_image[i, j, k] = np.median(region)

    return filtered_image
