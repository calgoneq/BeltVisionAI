import numpy as np
import cv2

from core.logger_config import logger

def smooth_data(data: np.ndarray, kernel_size: int) -> np.ndarray:
    profile = data.astype(np.float32).reshape(1, -1)
    kernel = cv2.getGaussianKernel(ksize=kernel_size, sigma=kernel_size/6.0)
    smooth = cv2.filter2D(profile, -1, kernel).reshape(-1)

    return smooth

def calculate_gradient(smooth_data: np.ndarray) -> np.ndarray:
    gradient = np.gradient(smooth_data)
    
    return gradient

def find_edge(gradient: np.ndarray, x_min: int, x_max: int, edge_type: str) -> tuple[int, float]: 
    x_min = max(0, int(x_min))
    x_max = min(len(gradient) - 1, int(x_max))
    segment = gradient[x_min:x_max + 1]

    if edge_type == "left":
        idx_local = np.argmax(segment)
    else:
        idx_local = np.argmin(segment)
    
    x = x_min + idx_local
    strength = abs(segment[idx_local])

    return x, strength

def measure_width(data: np.ndarray, kernel_size: int, l_min: int, l_max: int, r_min: int, r_max: int) -> tuple[float, int, int]:
    smooth_profile = smooth_data(data=data, kernel_size=kernel_size)
    gradient_data = calculate_gradient(smooth_profile)
    right_x, right_str = find_edge(gradient=gradient_data, x_min=r_min, x_max=r_max, edge_type="right")
    left_x, left_str= find_edge(gradient=gradient_data, x_min=l_min, x_max=l_max, edge_type="left")

    return float(right_x - left_x), int(left_x), int(right_x)

if __name__ == "__main__":
    zeros = np.zeros(25)
    numbers_up = np.arange(100,256)
    numbers_down = np.arange(255, 99, -1)
    data = np.concatenate([zeros, numbers_up, numbers_down, zeros])
    logger.info(measure_width(data=data, kernel_size=13, l_min=0, l_max=180, r_min=181, r_max=362))