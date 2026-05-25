import numpy as np
import cv2
from scipy.ndimage import gaussian_filter, convolve


def normalize_image(image):
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = image.astype(np.float64)
    m = np.mean(image)
    v = np.std(image)
    normalized = (image - m) / (v + 1e-8)
    return normalized


def calculate_gradient(image):
    sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    return sobel_x, sobel_y


def compute_orientation_field(image, block_size=16, gradient_sigma=1.0, orientation_sigma=3.0):
    normalized = normalize_image(image)
    
    dx, dy = calculate_gradient(normalized)
    
    dx2 = dx * dx
    dy2 = dy * dy
    dxy = dx * dy
    
    if gradient_sigma > 0:
        dx2 = gaussian_filter(dx2, gradient_sigma)
        dy2 = gaussian_filter(dy2, gradient_sigma)
        dxy = gaussian_filter(dxy, gradient_sigma)
    
    kernel = np.ones((block_size, block_size), dtype=np.float64)
    vx = convolve(dx2 - dy2, kernel)
    vy = 2 * convolve(dxy, kernel)
    
    theta = 0.5 * np.arctan2(vy, vx)
    
    if orientation_sigma > 0:
        sin_2theta = np.sin(2 * theta)
        cos_2theta = np.cos(2 * theta)
        
        sin_2theta = gaussian_filter(sin_2theta, orientation_sigma)
        cos_2theta = gaussian_filter(cos_2theta, orientation_sigma)
        
        theta = 0.5 * np.arctan2(sin_2theta, cos_2theta)
    
    coherence = np.sqrt(vx**2 + vy**2) / (dx2 + dy2 + 1e-8).sum()
    coherence = np.clip(coherence, 0, 1)
    
    return theta, coherence


def get_orientation_at(theta_field, x, y):
    h, w = theta_field.shape
    x = np.clip(x, 0, w - 1)
    y = np.clip(y, 0, h - 1)
    return theta_field[int(y), int(x)]


def visualize_orientation_field(image, theta_field, step=8, scale=0.7):
    if len(image.shape) == 2:
        vis = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        vis = image.copy()
    
    h, w = theta_field.shape
    color = (0, 255, 0)
    
    for y in range(0, h, step):
        for x in range(0, w, step):
            angle = theta_field[y, x]
            length = step * scale
            dx = length * np.cos(angle)
            dy = length * np.sin(angle)
            
            x1 = int(x - dx / 2)
            y1 = int(y - dy / 2)
            x2 = int(x + dx / 2)
            y2 = int(y + dy / 2)
            
            cv2.line(vis, (x1, y1), (x2, y2), color, 1)
    
    return vis
