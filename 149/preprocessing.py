import numpy as np
from scipy import ndimage
from skimage import filters, morphology, segmentation


class VesselPreprocessor:
    def __init__(self, spacing=(1.0, 1.0, 1.0)):
        self.spacing = np.array(spacing)

    def normalize(self, image):
        image = image.astype(np.float32)
        p2 = np.percentile(image, 2)
        p98 = np.percentile(image, 98)
        image = np.clip(image, p2, p98)
        image = (image - p2) / (p98 - p2 + 1e-8)
        return image

    def enhance_vessels(self, image, sigma_range=(1, 4)):
        image = image.astype(np.float32)
        sigmas = np.linspace(sigma_range[0], sigma_range[1], 4)
        response = np.zeros_like(image)
        
        for sigma in sigmas:
            hessian = []
            for i in range(3):
                for j in range(i, 3):
                    der = ndimage.gaussian_filter(image, sigma, order=[1 if k == i or k == j else 0 for k in range(3)])
                    hessian.append(der)
            
            Dxx, Dxy, Dxz, Dyy, Dyz, Dzz = hessian
            eigenvalues = np.zeros(image.shape + (3,))
            
            for idx in np.ndindex(image.shape):
                H = np.array([
                    [Dxx[idx], Dxy[idx], Dxz[idx]],
                    [Dxy[idx], Dyy[idx], Dyz[idx]],
                    [Dxz[idx], Dyz[idx], Dzz[idx]]
                ])
                eigvals = np.linalg.eigvalsh(H)
                eigenvalues[idx] = eigvals
            
            lambda1, lambda2, lambda3 = eigenvalues[..., 0], eigenvalues[..., 1], eigenvalues[..., 2]
            blobness = np.zeros_like(image)
            mask = (lambda2 < 0) & (lambda3 < 0)
            blobness[mask] = (lambda2[mask] * lambda3[mask]) / (lambda1[mask] + 1e-8)
            response = np.maximum(response, blobness)
        
        return response

    def threshold_segmentation(self, image, threshold='otsu'):
        if isinstance(threshold, str):
            if threshold == 'otsu':
                thresh_val = filters.threshold_otsu(image)
            elif threshold == 'li':
                thresh_val = filters.threshold_li(image)
            else:
                thresh_val = filters.threshold_mean(image)
        else:
            thresh_val = threshold
        
        binary = image > thresh_val
        return binary

    def post_process(self, binary, min_size=100, closing_radius=2):
        footprint = morphology.ball(closing_radius)
        binary = morphology.closing(binary, footprint)
        binary = morphology.remove_small_objects(binary, min_size=min_size)
        binary = morphology.remove_small_holes(binary, area_threshold=min_size // 2)
        return binary

    def full_pipeline(self, image, vesselness=True, threshold='otsu'):
        normalized = self.normalize(image)
        
        if vesselness:
            enhanced = self.enhance_vessels(normalized)
        else:
            enhanced = normalized
        
        binary = self.threshold_segmentation(enhanced, threshold)
        processed = self.post_process(binary)
        
        return {
            'normalized': normalized,
            'enhanced': enhanced if vesselness else None,
            'binary': binary,
            'processed': processed
        }


def generate_synthetic_vessel_image(shape=(100, 100, 100), num_branches=5):
    image = np.zeros(shape, dtype=np.float32)
    center = np.array(shape) // 2
    
    def draw_vessel(start, end, radius):
        start = np.array(start)
        end = np.array(end)
        vec = end - start
        length = np.linalg.norm(vec)
        vec = vec / (length + 1e-8)
        
        for t in np.linspace(0, 1, int(length * 2)):
            pos = start + vec * t * length
            for dx in [-radius, 0, radius]:
                for dy in [-radius, 0, radius]:
                    for dz in [-radius, 0, radius]:
                        if dx*dx + dy*dy + dz*dz <= radius*radius:
                            x, y, z = int(pos[0] + dx), int(pos[1] + dy), int(pos[2] + dz)
                            if 0 <= x < shape[0] and 0 <= y < shape[1] and 0 <= z < shape[2]:
                                image[x, y, z] = 1.0
    
    current_pos = center
    draw_vessel(current_pos - np.array([20, 0, 0]), current_pos + np.array([20, 0, 0]), 3)
    
    branch_points = [
        (center + np.array([10, 0, 0]), np.array([1, 1, 0]), 2),
        (center + np.array([5, 0, 0]), np.array([1, -1, 1]), 2),
        (center - np.array([10, 0, 0]), np.array([-1, 0, 1]), 2),
        (center - np.array([5, 0, 0]), np.array([-1, 1, -1]), 2),
    ]
    
    for start_pos, direction, radius in branch_points:
        end_pos = start_pos + direction * 25
        draw_vessel(start_pos, end_pos, radius)
        
        sub_branch_start = start_pos + direction * 15
        sub_dir = np.array([-direction[1], direction[0], direction[2]])
        draw_vessel(sub_branch_start, sub_branch_start + sub_dir * 12, radius - 1)
    
    image = ndimage.gaussian_filter(image, sigma=0.8)
    image = image * 0.8 + np.random.randn(*shape) * 0.05
    image = np.clip(image, 0, 1)
    
    return image
