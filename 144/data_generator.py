import numpy as np
import cv2
import os
import random
from scipy.ndimage import gaussian_filter, map_coordinates

try:
    import torch
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


def generate_synthetic_fingerprint_with_keypoints(size=(256, 256), fingerprint_type='whorl'):
    h, w = size
    y, x = np.mgrid[0:h, 0:w]
    cx, cy = w // 2, h // 2
    
    cores = []
    deltas = []
    
    if fingerprint_type == 'loop_right':
        dy = y - cy
        dx = x - cx
        angle = np.arctan2(dy, dx)
        r = np.sqrt(dx**2 + dy**2)
        ridge_pattern = np.sin(r / 4 + angle)
        
        cores.append((int(cx - 30), int(cy - 20)))
        deltas.append((int(cx + 40), int(cy + 30)))
        
    elif fingerprint_type == 'loop_left':
        dy = y - cy
        dx = x - cx
        angle = np.arctan2(dy, -dx)
        r = np.sqrt(dx**2 + dy**2)
        ridge_pattern = np.sin(r / 4 + angle)
        
        cores.append((int(cx + 30), int(cy - 20)))
        deltas.append((int(cx - 40), int(cy + 30)))
        
    elif fingerprint_type == 'whorl':
        dy = y - cy
        dx = x - cx
        angle = np.arctan2(dy, dx)
        r = np.sqrt(dx**2 + dy**2)
        ridge_pattern = np.sin(r / 3 + 2 * angle)
        
        cores.append((int(cx - 25), int(cy)))
        cores.append((int(cx + 25), int(cy)))
        deltas.append((int(cx), int(cy - 50)))
        deltas.append((int(cx), int(cy + 50)))
        
    elif fingerprint_type == 'arch':
        ridge_freq = 1 / 8.0
        ridge_pattern = np.sin(2 * np.pi * ridge_freq * (y + 0.3 * x))
        
    else:
        ridge_freq = 1 / 8.0
        ridge_pattern = np.sin(2 * np.pi * ridge_freq * y)
    
    ridge_pattern = (ridge_pattern + 1) / 2
    noise = np.random.normal(0, 0.1, (h, w))
    ridge_pattern = np.clip(ridge_pattern + noise, 0, 1)
    
    pattern = (ridge_pattern * 255).astype(np.uint8)
    
    mask = np.zeros((h, w), dtype=np.float64)
    center = (w // 2, h // 2)
    axes = (int(w * 0.4), int(h * 0.4))
    cv2.ellipse(mask, center, axes, 0, 0, 360, 1, -1)
    mask = cv2.GaussianBlur(mask, (21, 21), 5)
    
    pattern = (pattern * mask + 128 * (1 - mask)).astype(np.uint8)
    
    return pattern, cores, deltas


def apply_wet_effect(image, severity=0.5):
    h, w = image.shape
    
    alpha = 1.0 - 0.3 * severity
    beta = 30 * severity
    darkened = cv2.convertScaleAbs(image, alpha=alpha, beta=-beta)
    
    sigma = 1.5 * severity
    if sigma > 0:
        darkened = gaussian_filter(darkened, sigma=sigma)
    
    water_noise = np.random.normal(0, 10 * severity, (h, w))
    result = darkened.astype(np.float64) + water_noise
    result = np.clip(result, 0, 255).astype(np.uint8)
    
    return result


def apply_dry_effect(image, severity=0.5):
    h, w = image.shape
    
    alpha = 1.0 + 0.5 * severity
    beta = -20 * severity
    contrast = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    
    grain = np.random.randn(h, w) * 20 * severity
    result = contrast.astype(np.float64) + grain
    result = np.clip(result, 0, 255).astype(np.uint8)
    
    kernel_size = int(3 + 2 * severity)
    if kernel_size % 2 == 0:
        kernel_size += 1
    result = cv2.medianBlur(result, kernel_size)
    
    return result


def apply_blur_effect(image, severity=0.5):
    kernel_size = int(3 + 6 * severity)
    if kernel_size % 2 == 0:
        kernel_size += 1
    
    blurred = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
    return blurred


def apply_noise_effect(image, severity=0.5):
    h, w = image.shape
    noise = np.random.normal(0, 25 * severity, (h, w))
    noisy = image.astype(np.float64) + noise
    noisy = np.clip(noisy, 0, 255).astype(np.uint8)
    return noisy


def apply_elastic_deformation(image, sigma=10, alpha=30):
    h, w = image.shape
    
    dx = gaussian_filter((np.random.rand(h, w) * 2 - 1), sigma) * alpha
    dy = gaussian_filter((np.random.rand(h, w) * 2 - 1), sigma) * alpha
    
    x, y = np.meshgrid(np.arange(w), np.arange(h))
    
    indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1))
    
    deformed = map_coordinates(image, indices, order=1).reshape(h, w)
    
    return deformed


def apply_rotation(image, cores, deltas, angle=15):
    h, w = image.shape
    center = (w // 2, h // 2)
    
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=128)
    
    def rotate_point(px, py):
        new_x = M[0, 0] * px + M[0, 1] * py + M[0, 2]
        new_y = M[1, 0] * px + M[1, 1] * py + M[1, 2]
        return int(new_x), int(new_y)
    
    rotated_cores = [rotate_point(x, y) for x, y in cores]
    rotated_deltas = [rotate_point(x, y) for x, y in deltas]
    
    return rotated, rotated_cores, rotated_deltas


def apply_scaling(image, cores, deltas, scale=0.9):
    h, w = image.shape
    
    new_h, new_w = int(h * scale), int(w * scale)
    scaled = cv2.resize(image, (new_w, new_h))
    
    pad_h = (h - new_h) // 2
    pad_w = (w - new_w) // 2
    result = np.full((h, w), 128, dtype=np.uint8)
    result[pad_h:pad_h+new_h, pad_w:pad_w+new_w] = scaled
    
    def scale_point(px, py):
        new_x = int(px * scale + pad_w)
        new_y = int(py * scale + pad_h)
        return new_x, new_y
    
    scaled_cores = [scale_point(x, y) for x, y in cores]
    scaled_deltas = [scale_point(x, y) for x, y in deltas]
    
    return result, scaled_cores, scaled_deltas


def augment_fingerprint(image, cores, deltas, aug_prob=0.5):
    h, w = image.shape
    
    if random.random() < aug_prob:
        effect_type = random.choice(['wet', 'dry', 'blur', 'noise', 'elastic'])
        severity = random.uniform(0.2, 0.8)
        
        if effect_type == 'wet':
            image = apply_wet_effect(image, severity)
        elif effect_type == 'dry':
            image = apply_dry_effect(image, severity)
        elif effect_type == 'blur':
            image = apply_blur_effect(image, severity)
        elif effect_type == 'noise':
            image = apply_noise_effect(image, severity)
        elif effect_type == 'elastic':
            image = apply_elastic_deformation(image, sigma=8, alpha=25)
    
    if random.random() < aug_prob:
        angle = random.uniform(-20, 20)
        image, cores, deltas = apply_rotation(image, cores, deltas, angle)
    
    if random.random() < aug_prob:
        scale = random.uniform(0.8, 1.2)
        image, cores, deltas = apply_scaling(image, cores, deltas, scale)
    
    if random.random() < aug_prob:
        brightness = random.uniform(-30, 30)
        image = cv2.convertScaleAbs(image, alpha=1.0, beta=brightness)
    
    return image, cores, deltas


def create_augmented_variants(image, cores, deltas, num_variants=5):
    variants = []
    
    variants.append(('original', image, cores, deltas))
    
    variants.append(('wet', apply_wet_effect(image, severity=0.6), cores, deltas))
    
    variants.append(('dry', apply_dry_effect(image, severity=0.6), cores, deltas))
    
    variants.append(('blur', apply_blur_effect(image, severity=0.5), cores, deltas))
    
    variants.append(('noisy', apply_noise_effect(image, severity=0.5), cores, deltas))
    
    for i in range(num_variants - 5):
        aug_img, aug_cores, aug_deltas = augment_fingerprint(image.copy(), cores.copy(), deltas.copy(), aug_prob=0.7)
        variants.append((f'random_{i}', aug_img, aug_cores, aug_deltas))
    
    return variants


if TORCH_AVAILABLE:
    class FingerprintDataset(Dataset):
        def __init__(self, num_samples=1000, img_size=256, num_cores=2, num_deltas=2, 
                     augment=True, use_heatmap=False, heatmap_size=64):
            self.num_samples = num_samples
            self.img_size = img_size
            self.num_cores = num_cores
            self.num_deltas = num_deltas
            self.augment = augment
            self.use_heatmap = use_heatmap
            self.heatmap_size = heatmap_size
            self.fingerprint_types = ['loop_right', 'loop_left', 'whorl', 'arch']
        
        def __len__(self):
            return self.num_samples
        
        def __getitem__(self, idx):
            fp_type = random.choice(self.fingerprint_types)
            image, cores, deltas = generate_synthetic_fingerprint_with_keypoints(
                size=(self.img_size, self.img_size),
                fingerprint_type=fp_type
            )
            
            if self.augment:
                image, cores, deltas = augment_fingerprint(image, cores, deltas, aug_prob=0.6)
            
            image_tensor = self._preprocess_image(image)
            
            if self.use_heatmap:
                target = self._generate_heatmap_target(cores, deltas)
            else:
                target = self._generate_regression_target(cores, deltas)
            
            return image_tensor, target
        
        def _preprocess_image(self, image):
            image = image.astype(np.float32) / 255.0
            image = (image - 0.5) / 0.5
            return torch.from_numpy(image).unsqueeze(0)
        
        def _generate_regression_target(self, cores, deltas):
            target = torch.zeros(self.num_cores + self.num_deltas, 3)
            
            for i, (x, y) in enumerate(cores[:self.num_cores]):
                target[i, 0] = x / self.img_size
                target[i, 1] = y / self.img_size
                target[i, 2] = 1.0
            
            for i, (x, y) in enumerate(deltas[:self.num_deltas]):
                idx = self.num_cores + i
                target[idx, 0] = x / self.img_size
                target[idx, 1] = y / self.img_size
                target[idx, 2] = 1.0
            
            return target
        
        def _generate_heatmap_target(self, cores, deltas):
            from deep_learning import generate_heatmap
            
            num_keypoints = self.num_cores + self.num_deltas
            target = torch.zeros(num_keypoints, self.heatmap_size, self.heatmap_size)
            
            for i, (x, y) in enumerate(cores[:self.num_cores]):
                hm = generate_heatmap(x, y, self.img_size, self.heatmap_size, sigma=2)
                target[i] = torch.from_numpy(hm)
            
            for i, (x, y) in enumerate(deltas[:self.num_deltas]):
                idx = self.num_cores + i
                hm = generate_heatmap(x, y, self.img_size, self.heatmap_size, sigma=2)
                target[idx] = torch.from_numpy(hm)
            
            return target


def generate_dataset_for_visualization(num_samples=5, output_dir='dataset_samples'):
    os.makedirs(output_dir, exist_ok=True)
    
    fp_types = ['loop_right', 'loop_left', 'whorl', 'arch']
    
    for i, fp_type in enumerate(fp_types):
        image, cores, deltas = generate_synthetic_fingerprint_with_keypoints(
            size=(256, 256), fingerprint_type=fp_type
        )
        
        variants = create_augmented_variants(image, cores, deltas, num_variants=8)
        
        for j, (name, img, c, d) in enumerate(variants):
            vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            
            for (x, y) in c:
                cv2.circle(vis, (x, y), 6, (0, 0, 255), -1)
            for (x, y) in d:
                cv2.circle(vis, (x, y), 6, (255, 0, 0), -1)
            
            cv2.putText(vis, name, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            output_path = os.path.join(output_dir, f'{fp_type}_{name}.png')
            cv2.imwrite(output_path, vis)
    
    print(f"Generated dataset samples in {output_dir}")


if __name__ == '__main__':
    print("Testing data generation...")
    
    image, cores, deltas = generate_synthetic_fingerprint_with_keypoints(
        size=(256, 256), fingerprint_type='whorl'
    )
    print(f"Generated fingerprint: {len(cores)} cores, {len(deltas)} deltas")
    
    wet = apply_wet_effect(image, severity=0.7)
    dry = apply_dry_effect(image, severity=0.7)
    blurred = apply_blur_effect(image, severity=0.5)
    noisy = apply_noise_effect(image, severity=0.5)
    
    print("Data generation test passed!")
    
    generate_dataset_for_visualization()
