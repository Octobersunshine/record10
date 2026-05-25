import numpy as np
import cv2
from typing import Tuple, Optional


class OpticalFlowCalculator:
    def __init__(self, method: str = 'horn_schunck', **kwargs):
        self.method = method.lower()
        self.params = kwargs
        
        if self.method == 'horn_schunck':
            self.alpha = kwargs.get('alpha', 1.0)
            self.iterations = kwargs.get('iterations', 100)
            self.anisotropic = kwargs.get('anisotropic', True)
            self.edge_threshold = kwargs.get('edge_threshold', 0.05)
            self.edge_sensitivity = kwargs.get('edge_sensitivity', 2.0)
        elif self.method == 'lucas_kanade':
            self.window_size = kwargs.get('window_size', 15)
        else:
            raise ValueError(f"Unknown method: {method}. Use 'horn_schunck' or 'lucas_kanade'")
    
    def calculate(self, frame1: np.ndarray, frame2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if frame1.shape != frame2.shape:
            raise ValueError("Frames must have the same dimensions")
        
        if len(frame1.shape) == 3:
            gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        else:
            gray1 = frame1
            gray2 = frame2
        
        gray1 = gray1.astype(np.float32) / 255.0
        gray2 = gray2.astype(np.float32) / 255.0
        
        if self.method == 'horn_schunck':
            return self._horn_schunck(gray1, gray2)
        else:
            return self._lucas_kanade(gray1, gray2)
    
    def _horn_schunck(self, img1: np.ndarray, img2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        rows, cols = img1.shape
        
        u = np.zeros((rows, cols), dtype=np.float32)
        v = np.zeros((rows, cols), dtype=np.float32)
        
        kernel_x = np.array([[-1, 1], [-1, 1]]) / 4.0
        kernel_y = np.array([[-1, -1], [1, 1]]) / 4.0
        kernel_t = np.array([[1, 1], [1, 1]]) / 4.0
        
        Ix = cv2.filter2D(img1, -1, kernel_x) + cv2.filter2D(img2, -1, kernel_x)
        Iy = cv2.filter2D(img1, -1, kernel_y) + cv2.filter2D(img2, -1, kernel_y)
        It = cv2.filter2D(img2, -1, kernel_t) - cv2.filter2D(img1, -1, kernel_t)
        
        if self.anisotropic:
            edge_weights = self._compute_edge_weights(img1, img2)
        else:
            edge_weights = None
        
        for _ in range(self.iterations):
            if self.anisotropic:
                u_avg = self._anisotropic_smooth(u, edge_weights)
                v_avg = self._anisotropic_smooth(v, edge_weights)
            else:
                avg_kernel = np.array([[1/12, 1/6, 1/12],
                                       [1/6, 0, 1/6],
                                       [1/12, 1/6, 1/12]], dtype=np.float32)
                u_avg = cv2.filter2D(u, -1, avg_kernel)
                v_avg = cv2.filter2D(v, -1, avg_kernel)
            
            numerator = Ix * u_avg + Iy * v_avg + It
            denominator = self.alpha**2 + Ix**2 + Iy**2
            
            u = u_avg - Ix * numerator / denominator
            v = v_avg - Iy * numerator / denominator
        
        return u, v
    
    def _compute_edge_weights(self, img1: np.ndarray, img2: np.ndarray) -> np.ndarray:
        sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
        sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)
        
        avg_img = (img1 + img2) / 2.0
        gx = cv2.filter2D(avg_img, -1, sobel_x)
        gy = cv2.filter2D(avg_img, -1, sobel_y)
        
        grad_mag = np.sqrt(gx**2 + gy**2)
        
        k = self.edge_sensitivity
        edge_weights = np.exp(-(grad_mag / self.edge_threshold)**k)
        
        return edge_weights
    
    def _anisotropic_smooth(self, field: np.ndarray, edge_weights: np.ndarray) -> np.ndarray:
        rows, cols = field.shape
        
        padded = np.pad(field, 1, mode='edge')
        weights_padded = np.pad(edge_weights, 1, mode='edge')
        
        result = np.zeros_like(field)
        
        w_center = 4.0
        w_neighbor = 1.0
        
        for i in range(rows):
            for j in range(cols):
                i_pad = i + 1
                j_pad = j + 1
                
                w_c = w_center
                w_n = w_neighbor * weights_padded[i_pad - 1, j_pad]
                w_s = w_neighbor * weights_padded[i_pad + 1, j_pad]
                w_e = w_neighbor * weights_padded[i_pad, j_pad + 1]
                w_w = w_neighbor * weights_padded[i_pad, j_pad - 1]
                w_ne = w_neighbor * 0.5 * weights_padded[i_pad - 1, j_pad + 1]
                w_nw = w_neighbor * 0.5 * weights_padded[i_pad - 1, j_pad - 1]
                w_se = w_neighbor * 0.5 * weights_padded[i_pad + 1, j_pad + 1]
                w_sw = w_neighbor * 0.5 * weights_padded[i_pad + 1, j_pad - 1]
                
                total_weight = w_c + w_n + w_s + w_e + w_w + w_ne + w_nw + w_se + w_sw
                
                val = (w_c * padded[i_pad, j_pad] +
                       w_n * padded[i_pad - 1, j_pad] +
                       w_s * padded[i_pad + 1, j_pad] +
                       w_e * padded[i_pad, j_pad + 1] +
                       w_w * padded[i_pad, j_pad - 1] +
                       w_ne * padded[i_pad - 1, j_pad + 1] +
                       w_nw * padded[i_pad - 1, j_pad - 1] +
                       w_se * padded[i_pad + 1, j_pad + 1] +
                       w_sw * padded[i_pad + 1, j_pad - 1])
                
                result[i, j] = val / total_weight
        
        return result
    
    def _lucas_kanade(self, img1: np.ndarray, img2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        rows, cols = img1.shape
        half_window = self.window_size // 2
        
        kernel_x = np.array([[-1, 1], [-1, 1]]) / 4.0
        kernel_y = np.array([[-1, -1], [1, 1]]) / 4.0
        kernel_t = np.array([[1, 1], [1, 1]]) / 4.0
        
        Ix = cv2.filter2D(img1, -1, kernel_x) + cv2.filter2D(img2, -1, kernel_x)
        Iy = cv2.filter2D(img1, -1, kernel_y) + cv2.filter2D(img2, -1, kernel_y)
        It = cv2.filter2D(img2, -1, kernel_t) - cv2.filter2D(img1, -1, kernel_t)
        
        u = np.zeros((rows, cols), dtype=np.float32)
        v = np.zeros((rows, cols), dtype=np.float32)
        
        Ix_pad = np.pad(Ix, half_window, mode='edge')
        Iy_pad = np.pad(Iy, half_window, mode='edge')
        It_pad = np.pad(It, half_window, mode='edge')
        
        for i in range(rows):
            for j in range(cols):
                i_start = i
                i_end = i + self.window_size
                j_start = j
                j_end = j + self.window_size
                
                Ix_win = Ix_pad[i_start:i_end, j_start:j_end].flatten()
                Iy_win = Iy_pad[i_start:i_end, j_start:j_end].flatten()
                It_win = It_pad[i_start:i_end, j_start:j_end].flatten()
                
                A = np.vstack((Ix_win, Iy_win)).T
                b = -It_win
                
                ATA = A.T @ A
                ATb = A.T @ b
                
                eigenvalues = np.linalg.eigvalsh(ATA)
                if eigenvalues.min() > 0.001:
                    flow = np.linalg.inv(ATA) @ ATb
                    u[i, j] = flow[0]
                    v[i, j] = flow[1]
        
        return u, v


def flow_to_color(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    magnitude, angle = cv2.cartToPolar(u, v)
    
    hsv = np.zeros((u.shape[0], u.shape[1], 3), dtype=np.uint8)
    hsv[..., 1] = 255
    
    hsv[..., 0] = (angle * 180 / np.pi / 2).astype(np.uint8)
    hsv[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def read_video_frames(video_path: str, max_frames: Optional[int] = None) -> list:
    cap = cv2.VideoCapture(video_path)
    frames = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
        if max_frames is not None and len(frames) >= max_frames:
            break
    
    cap.release()
    return frames


def calculate_optical_flow_video(video_path: str, method: str = 'horn_schunck', **kwargs) -> list:
    frames = read_video_frames(video_path)
    flows = []
    
    calculator = OpticalFlowCalculator(method=method, **kwargs)
    
    for i in range(len(frames) - 1):
        u, v = calculator.calculate(frames[i], frames[i + 1])
        flows.append((u, v))
    
    return flows


if __name__ == "__main__":
    print("Dense Optical Flow Calculator")
    print("=" * 50)
    print("Methods available:")
    print("  1. Horn-Schunck (global method)")
    print("     - With anisotropic smoothing (edge-preserving)")
    print("  2. Lucas-Kanade (local method)")
    print()
    print("Horn-Schunck Parameters:")
    print("  alpha: Smoothness weight (default: 1.0)")
    print("  iterations: Number of iterations (default: 100)")
    print("  anisotropic: Enable edge-preserving smoothing (default: True)")
    print("  edge_threshold: Edge sensitivity threshold (default: 0.05)")
    print("  edge_sensitivity: Edge weight decay rate (default: 2.0)")
    print()
    print("Usage example (with edge-preserving):")
    print("  calculator = OpticalFlowCalculator(")
    print("      method='horn_schunck',")
    print("      alpha=1.0,")
    print("      iterations=100,")
    print("      anisotropic=True,")
    print("      edge_threshold=0.05,")
    print("      edge_sensitivity=2.0")
    print("  )")
    print("  u, v = calculator.calculate(frame1, frame2)")
    print()
    print("  or for original isotropic smoothing:")
    print("  calculator = OpticalFlowCalculator(method='horn_schunck', anisotropic=False)")
    print()
    print("  or for Lucas-Kanade:")
    print("  calculator = OpticalFlowCalculator(method='lucas_kanade', window_size=15)")
    print("  u, v = calculator.calculate(frame1, frame2)")
    print()
    print("Visualization:")
    print("  color_flow = flow_to_color(u, v)")
    print("  cv2.imshow('Optical Flow', color_flow)")
