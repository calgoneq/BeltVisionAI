import numpy as np
import cv2

def smooth_data(data: np.ndarray, kernel_size: int) -> np.ndarray:
    profile = data.astype(np.float32).reshape(1, -1)
    kernel = cv2.getGaussianKernel(ksize=kernel_size, sigma=kernel_size/6.0)
    smooth = cv2.filter2D(profile, -1, kernel).reshape(-1)

    return smooth

def calculate_gradient(smooth_data: np.ndarray) -> np.ndarray:
    gradient = np.gradient(smooth_data)
    
    return gradient


if __name__ == "__main__":
    zeros = np.zeros(25)
    numbers = np.arange(100,256)
    data = np.concatenate([zeros, numbers])
    smooth_data = smooth_data(data=data, kernel_size=13)
    gradient_data = calculate_gradient(smooth_data)

    print(gradient_data)