import numpy as np
import cv2
from typing import Tuple, Optional, Union


class OpticalFlow:
    def __init__(self, method: str = 'raft', **kwargs):
        self.method = method.lower()
        self.params = kwargs
        self.calculator = None
        self._init_calculator()
    
    def _init_calculator(self):
        traditional_methods = ['horn_schunck', 'lucas_kanade', 'farneback']
        deep_methods = ['raft', 'flownet2', 'liteflownet']
        
        if self.method in traditional_methods:
            self._init_traditional()
        elif self.method in deep_methods:
            self._init_deep()
        else:
            raise ValueError(
                f"Unknown method: {self.method}. "
                f"Use one of: {traditional_methods + deep_methods}"
            )
    
    def _init_traditional(self):
        from dense_optical_flow import OpticalFlowCalculator as TraditionalCalculator
        
        if self.method == 'horn_schunck':
            self.calculator = TraditionalCalculator(
                method='horn_schunck',
                alpha=self.params.get('alpha', 1.0),
                iterations=self.params.get('iterations', 100),
                anisotropic=self.params.get('anisotropic', True),
                edge_threshold=self.params.get('edge_threshold', 0.05),
                edge_sensitivity=self.params.get('edge_sensitivity', 2.0)
            )
        elif self.method == 'lucas_kanade':
            self.calculator = TraditionalCalculator(
                method='lucas_kanade',
                window_size=self.params.get('window_size', 15)
            )
        elif self.method == 'farneback':
            self.calculator = _FarnebackWrapper(**self.params)
        
        self._type = 'traditional'
    
    def _init_deep(self):
        from deep_optical_flow import DeepOpticalFlow
        
        self.calculator = DeepOpticalFlow(
            model_name=self.method,
            device=self.params.get('device', 'auto')
        )
        self._type = 'deep'
    
    def calculate(self, frame1: np.ndarray, frame2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        return self.calculator.calculate(frame1, frame2)
    
    def calculate_batch(self, frames_pairs: list) -> list:
        if hasattr(self.calculator, 'calculate_batch'):
            return self.calculator.calculate_batch(frames_pairs)
        else:
            results = []
            for f1, f2 in frames_pairs:
                results.append(self.calculate(f1, f2))
            return results
    
    def refine_subpixel(self, u: np.ndarray, v: np.ndarray, 
                        img1: np.ndarray, img2: np.ndarray,
                        iterations: int = 3, window_size: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        from deep_optical_flow import SubpixelRefinement
        return SubpixelRefinement.refine_flow(u, v, img1, img2, iterations, window_size)
    
    @staticmethod
    def visualize(u: np.ndarray, v: np.ndarray) -> np.ndarray:
        from deep_optical_flow import flow_to_color
        return flow_to_color(u, v)


class _FarnebackWrapper:
    def __init__(self, **kwargs):
        self.params = {
            'pyr_scale': kwargs.get('pyr_scale', 0.5),
            'levels': kwargs.get('levels', 5),
            'winsize': kwargs.get('winsize', 15),
            'iterations': kwargs.get('iterations', 5),
            'poly_n': kwargs.get('poly_n', 5),
            'poly_sigma': kwargs.get('poly_sigma', 1.2),
            'flags': kwargs.get('flags', 0)
        }
    
    def calculate(self, frame1: np.ndarray, frame2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if len(frame1.shape) == 3:
            gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        else:
            gray1 = frame1
            gray2 = frame2
        
        flow = cv2.calcOpticalFlowFarneback(gray1, gray2, None, **self.params)
        return flow[..., 0].astype(np.float32), flow[..., 1].astype(np.float32)


def get_available_methods() -> dict:
    return {
        'traditional': {
            'horn_schunck': 'Global variational method with anisotropic smoothing (edge-preserving)',
            'lucas_kanade': 'Local window-based least squares method',
            'farneback': 'Dense optical flow based on polynomial expansion (OpenCV)'
        },
        'deep_learning': {
            'raft': 'Recurrent All-Pairs Field Transforms (high accuracy, subpixel)',
            'flownet2': 'Deep network for optical flow estimation',
            'liteflownet': 'Lightweight deep optical flow network'
        }
    }


if __name__ == "__main__":
    print("Unified Optical Flow Interface")
    print("=" * 60)
    print()
    
    methods = get_available_methods()
    
    print("Traditional Methods:")
    for name, desc in methods['traditional'].items():
        print(f"  - {name}: {desc}")
    
    print()
    print("Deep Learning Methods:")
    for name, desc in methods['deep_learning'].items():
        print(f"  - {name}: {desc}")
    
    print()
    print("Usage Example:")
    print("  from optical_flow_unified import OpticalFlow")
    print()
    print("  # Traditional method with edge preservation")
    print("  flow = OpticalFlow('horn_schunck', anisotropic=True)")
    print("  u, v = flow.calculate(frame1, frame2)")
    print()
    print("  # Deep learning method")
    print("  flow = OpticalFlow('raft', device='cuda')")
    print("  u, v = flow.calculate(frame1, frame2)")
    print()
    print("  # Subpixel refinement")
    print("  u_ref, v_ref = flow.refine_subpixel(u, v, frame1, frame2)")
    print()
    print("  # Visualize")
    print("  color_flow = OpticalFlow.visualize(u, v)")
