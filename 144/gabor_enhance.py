import numpy as np
import cv2
from scipy.ndimage import gaussian_filter


def create_gabor_kernel(ksize, sigma, theta, lambd, gamma, psi):
    kernel = cv2.getGaborKernel(
        (ksize, ksize), sigma, theta, lambd, gamma, psi, ktype=cv2.CV_64F
    )
    return kernel


def estimate_ridge_frequency(image, block_size=32):
    h, w = image.shape
    freq_map = np.zeros((h, w), dtype=np.float64)
    
    for y in range(0, h, block_size):
        for x in range(0, w, block_size):
            block = image[y:y+block_size, x:x+block_size]
            if block.size == 0:
                continue
            
            mean = np.mean(block)
            std = np.std(block)
            if std < 1e-3:
                freq_map[y:y+block_size, x:x+block_size] = 1.0 / 8.0
                continue
            
            fft = np.fft.fft2(block - mean)
            fft_mag = np.abs(np.fft.fftshift(fft))
            
            center_y, center_x = block_size // 2, block_size // 2
            max_r = block_size // 2
            radii = np.zeros(max_r)
            
            for r in range(1, max_r):
                mask = np.zeros((block_size, block_size), dtype=bool)
                for by in range(block_size):
                    for bx in range(block_size):
                        dist = np.sqrt((by - center_y)**2 + (bx - center_x)**2)
                        if r - 0.5 <= dist < r + 0.5:
                            mask[by, bx] = True
                radii[r] = np.mean(fft_mag[mask])
            
            peak_r = np.argmax(radii[1:]) + 1
            if peak_r > 0:
                freq = peak_r / block_size
            else:
                freq = 1.0 / 8.0
            
            freq_map[y:y+block_size, x:x+block_size] = np.clip(freq, 1.0/16.0, 1.0/4.0)
    
    freq_map = cv2.GaussianBlur(freq_map, (block_size*2+1, block_size*2+1), block_size/2)
    return freq_map


def enhance_fingerprint(image, theta_field, ridge_frequency=None, ksize=31):
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    image = image.astype(np.float64)
    h, w = image.shape
    
    if ridge_frequency is None:
        ridge_frequency = estimate_ridge_frequency(image)
    
    enhanced = np.zeros_like(image)
    confidence = np.zeros_like(image)
    
    for y in range(h):
        for x in range(w):
            theta = theta_field[y, x] + np.pi / 2
            freq = ridge_frequency[y, x]
            if freq < 1e-3:
                freq = 1.0 / 8.0
            lambd = 1.0 / freq
            
            sigma = lambd * 0.5
            gamma = 0.5
            psi = 0
            
            kernel = create_gabor_kernel(ksize, sigma, theta, lambd, gamma, psi)
            kernel = kernel - np.mean(kernel)
            kernel = kernel / (np.sum(np.abs(kernel)) + 1e-8)
            
            half_k = ksize // 2
            y1 = max(0, y - half_k)
            y2 = min(h, y + half_k + 1)
            x1 = max(0, x - half_k)
            x2 = min(w, x + half_k + 1)
            
            ky1 = half_k - (y - y1)
            ky2 = half_k + (y2 - y)
            kx1 = half_k - (x - x1)
            kx2 = half_k + (x2 - x)
            
            if y2 > y1 and x2 > x1:
                patch = image[y1:y2, x1:x2]
                k_patch = kernel[ky1:ky2, kx1:kx2]
                if patch.shape == k_patch.shape:
                    enhanced[y, x] = np.sum(patch * k_patch)
    
    enhanced = cv2.normalize(enhanced, None, 0, 255, cv2.NORM_MINMAX)
    enhanced = enhanced.astype(np.uint8)
    
    return enhanced


def fast_enhance_fingerprint(image, theta_field, ridge_frequency=None, ksize=25):
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    image = image.astype(np.float64)
    h, w = image.shape
    
    if ridge_frequency is None:
        ridge_frequency = estimate_ridge_frequency(image)
    
    enhanced = np.zeros_like(image)
    
    num_orientations = 16
    orientation_bins = np.linspace(0, np.pi, num_orientations, endpoint=False)
    
    for idx, base_theta in enumerate(orientation_bins):
        freq = np.mean(ridge_frequency)
        if freq < 1e-3:
            freq = 1.0 / 8.0
        lambd = 1.0 / freq
        
        sigma = lambd * 0.5
        gamma = 0.5
        
        kernel_0 = create_gabor_kernel(ksize, sigma, base_theta, lambd, gamma, 0)
        kernel_90 = create_gabor_kernel(ksize, sigma, base_theta, lambd, gamma, np.pi/2)
        
        kernel_0 = kernel_0 - np.mean(kernel_0)
        kernel_0 = kernel_0 / (np.sum(np.abs(kernel_0)) + 1e-8)
        
        kernel_90 = kernel_90 - np.mean(kernel_90)
        kernel_90 = kernel_90 / (np.sum(np.abs(kernel_90)) + 1e-8)
        
        filtered_0 = cv2.filter2D(image, -1, kernel_0)
        filtered_90 = cv2.filter2D(image, -1, kernel_90)
        
        theta_diff = np.abs(theta_field - base_theta)
        theta_diff = np.minimum(theta_diff, np.pi - theta_diff)
        weight = np.exp(-theta_diff**2 / (2 * (np.pi/12)**2))
        
        enhanced += weight * (filtered_0 - filtered_90)
    
    enhanced = cv2.normalize(enhanced, None, 0, 255, cv2.NORM_MINMAX)
    enhanced = enhanced.astype(np.uint8)
    
    return enhanced


def multiscale_enhance(image, theta_field, scales=[1.0, 0.75, 0.5]):
    h, w = image.shape[:2]
    enhanced_pyramid = []
    
    for scale in scales:
        new_h, new_w = int(h * scale), int(w * scale)
        
        if scale < 1.0:
            scaled_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
            scaled_theta = cv2.resize(theta_field, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        else:
            scaled_image = image
            scaled_theta = theta_field
        
        enhanced = fast_enhance_fingerprint(scaled_image, scaled_theta)
        
        if scale < 1.0:
            enhanced = cv2.resize(enhanced, (w, h), interpolation=cv2.INTER_LINEAR)
        
        enhanced_pyramid.append(enhanced.astype(np.float64))
    
    weights = np.array(scales) / np.sum(scales)
    final_enhanced = np.zeros_like(image, dtype=np.float64)
    for i, enhanced in enumerate(enhanced_pyramid):
        final_enhanced += weights[i] * enhanced
    
    final_enhanced = cv2.normalize(final_enhanced, None, 0, 255, cv2.NORM_MINMAX)
    final_enhanced = final_enhanced.astype(np.uint8)
    
    return final_enhanced


def estimate_quality_map(image, theta_field, block_size=16):
    h, w = image.shape
    quality_map = np.zeros((h, w), dtype=np.float64)
    
    if len(image.shape) == 3:
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        image_gray = image
    
    for y in range(0, h, block_size):
        for x in range(0, w, block_size):
            y_end = min(y + block_size, h)
            x_end = min(x + block_size, w)
            
            block = image_gray[y:y_end, x:x_end]
            theta_block = theta_field[y:y_end, x:x_end]
            
            if block.size == 0:
                continue
            
            std_val = np.std(block)
            
            sin_2theta = np.sin(2 * theta_block)
            cos_2theta = np.cos(2 * theta_block)
            coherence = np.sqrt(np.mean(sin_2theta)**2 + np.mean(cos_2theta)**2)
            
            contrast = std_val / (np.mean(block) + 1e-8)
            
            quality = coherence * (1 - np.exp(-std_val / 30.0))
            quality_map[y:y_end, x:x_end] = np.clip(quality, 0, 1)
    
    quality_map = gaussian_filter(quality_map, block_size)
    return quality_map
