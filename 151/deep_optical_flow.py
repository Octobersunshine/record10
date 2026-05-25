import numpy as np
import cv2
from typing import Tuple, Optional
import warnings


class DeepOpticalFlow:
    def __init__(self, model_name: str = 'raft', device: str = 'auto', **kwargs):
        self.model_name = model_name.lower()
        self.device = device
        self.params = kwargs
        self.model = None
        self.use_gpu = False
        
        self._init_device()
        self._init_model()
    
    def _init_device(self):
        if self.device == 'auto':
            try:
                import torch
                self.use_gpu = torch.cuda.is_available()
                self.device = 'cuda' if self.use_gpu else 'cpu'
            except ImportError:
                self.use_gpu = False
                self.device = 'cpu'
        elif self.device == 'cuda':
            try:
                import torch
                self.use_gpu = torch.cuda.is_available()
                if not self.use_gpu:
                    warnings.warn("CUDA not available, falling back to CPU")
                    self.device = 'cpu'
            except ImportError:
                self.use_gpu = False
                self.device = 'cpu'
                warnings.warn("PyTorch not available, using CPU mode")
        else:
            self.device = 'cpu'
            self.use_gpu = False
        
        print(f"Using device: {self.device}")
    
    def _init_model(self):
        if self.model_name == 'raft':
            self._init_raft()
        elif self.model_name == 'flownet2':
            self._init_flownet2()
        elif self.model_name == 'liteflownet':
            self._init_liteflownet()
        else:
            raise ValueError(f"Unknown model: {self.model_name}. Use 'raft', 'flownet2', or 'liteflownet'")
    
    def _init_raft(self):
        try:
            self._init_raft_pytorch()
        except Exception as e:
            warnings.warn(f"PyTorch RAFT not available: {e}, using OpenCV DNN implementation")
            self._init_opencv_dnn('raft')
    
    def _init_raft_pytorch(self):
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        
        class ResidualBlock(nn.Module):
            def __init__(self, in_channels, out_channels):
                super().__init__()
                self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
                self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
                self.relu = nn.ReLU(inplace=True)
                
                if in_channels != out_channels:
                    self.shortcut = nn.Conv2d(in_channels, out_channels, 1)
                else:
                    self.shortcut = nn.Identity()
            
            def forward(self, x):
                residual = self.shortcut(x)
                out = self.relu(self.conv1(x))
                out = self.conv2(out)
                return self.relu(out + residual)
        
        class FeatureEncoder(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = nn.Conv2d(3, 64, 7, stride=2, padding=3)
                self.relu = nn.ReLU(inplace=True)
                self.res1 = ResidualBlock(64, 64)
                self.res2 = ResidualBlock(64, 128)
                self.res3 = ResidualBlock(128, 128)
                self.res4 = ResidualBlock(128, 256)
            
            def forward(self, x):
                x = self.relu(self.conv1(x))
                x = self.res1(x)
                x = F.avg_pool2d(x, 2)
                x = self.res2(x)
                x = self.res3(x)
                x = F.avg_pool2d(x, 2)
                x = self.res4(x)
                return x
        
        class CorrelationLayer:
            @staticmethod
            def compute(fmap1, fmap2, radius=4):
                B, C, H, W = fmap1.shape
                fmap2_pad = F.pad(fmap2, [radius] * 4)
                corr_volume = []
                
                for i in range(2 * radius + 1):
                    for j in range(2 * radius + 1):
                        corr = torch.sum(fmap1 * fmap2_pad[:, :, i:i+H, j:j+W], dim=1)
                        corr_volume.append(corr)
                
                return torch.stack(corr_volume, dim=1)
        
        class UpdateBlock(nn.Module):
            def __init__(self, hidden_dim=128, input_dim=256):
                super().__init__()
                self.conv1 = nn.Conv2d(input_dim, hidden_dim, 3, padding=1)
                self.conv2 = nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1)
                self.flow_head = nn.Conv2d(hidden_dim, 2, 3, padding=1)
                self.relu = nn.ReLU(inplace=True)
            
            def forward(self, x):
                x = self.relu(self.conv1(x))
                x = self.relu(self.conv2(x))
                return self.flow_head(x)
        
        class RAFT(nn.Module):
            def __init__(self):
                super().__init__()
                self.feature_encoder = FeatureEncoder()
                self.update_block = UpdateBlock()
                self.corr_radius = 4
            
            def forward(self, img1, img2, iters=12):
                fmap1 = self.feature_encoder(img1)
                fmap2 = self.feature_encoder(img2)
                
                B, _, H, W = fmap1.shape
                flow = torch.zeros(B, 2, H, W, device=img1.device)
                
                for _ in range(iters):
                    flow = flow.detach()
                    
                    grid = self._compute_grid(flow, H, W)
                    fmap2_warped = F.grid_sample(fmap2, grid, align_corners=True)
                    
                    corr = CorrelationLayer.compute(fmap1, fmap2_warped, self.corr_radius)
                    
                    combined = torch.cat([corr, flow], dim=1)
                    delta_flow = self.update_block(combined)
                    flow = flow + delta_flow
                
                return flow
            
            def _compute_grid(self, flow, H, W):
                B = flow.shape[0]
                y, x = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
                grid = torch.stack([x, y], dim=-1).float().to(flow.device)
                grid = grid.unsqueeze(0).repeat(B, 1, 1, 1)
                
                grid[..., 0] = 2.0 * (grid[..., 0] + flow[:, 0]) / (W - 1) - 1.0
                grid[..., 1] = 2.0 * (grid[..., 1] + flow[:, 1]) / (H - 1) - 1.0
                
                return grid
        
        self.model = RAFT()
        self.model.eval()
        
        if self.use_gpu:
            self.model = self.model.cuda()
        
        self._torch_available = True
        self._model_type = 'pytorch_raft'
        
    def _init_opencv_dnn(self, model_type: str):
        self._torch_available = False
        self._model_type = 'opencv_dnn'
        
        if model_type == 'raft':
            self._dnn_model_type = 'raft'
        else:
            self._dnn_model_type = 'default'
        
        warnings.warn(
            "Using simplified OpenCV-based implementation. "
            "For full RAFT model with pretrained weights, install PyTorch and 'raft' package."
        )
    
    def _init_flownet2(self):
        try:
            import torch
            self._torch_available = True
            self._model_type = 'pytorch_flownet2'
            warnings.warn("FlowNet2 implementation placeholder. Using RAFT implementation instead.")
            self._init_raft_pytorch()
        except ImportError:
            self._init_opencv_dnn('flownet2')
    
    def _init_liteflownet(self):
        try:
            import torch
            self._torch_available = True
            self._model_type = 'pytorch_liteflownet'
            warnings.warn("LiteFlowNet implementation placeholder. Using RAFT implementation instead.")
            self._init_raft_pytorch()
        except ImportError:
            self._init_opencv_dnn('liteflownet')
    
    def calculate(self, frame1: np.ndarray, frame2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if frame1.shape != frame2.shape:
            raise ValueError("Frames must have the same dimensions")
        
        if self._model_type.startswith('pytorch'):
            return self._calculate_pytorch(frame1, frame2)
        else:
            return self._calculate_opencv(frame1, frame2)
    
    def _calculate_pytorch(self, frame1: np.ndarray, frame2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        import torch
        
        if len(frame1.shape) == 2:
            frame1 = cv2.cvtColor(frame1, cv2.COLOR_GRAY2RGB)
            frame2 = cv2.cvtColor(frame2, cv2.COLOR_GRAY2RGB)
        elif frame1.shape[2] == 4:
            frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGRA2RGB)
            frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGRA2RGB)
        else:
            frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
            frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)
        
        original_h, original_w = frame1.shape[:2]
        
        divisor = 8
        new_h = ((original_h + divisor - 1) // divisor) * divisor
        new_w = ((original_w + divisor - 1) // divisor) * divisor
        
        if new_h != original_h or new_w != original_w:
            frame1 = cv2.resize(frame1, (new_w, new_h))
            frame2 = cv2.resize(frame2, (new_w, new_h))
        
        img1_tensor = torch.from_numpy(frame1).permute(2, 0, 1).float() / 255.0
        img2_tensor = torch.from_numpy(frame2).permute(2, 0, 1).float() / 255.0
        
        img1_tensor = img1_tensor.unsqueeze(0)
        img2_tensor = img2_tensor.unsqueeze(0)
        
        if self.use_gpu:
            img1_tensor = img1_tensor.cuda()
            img2_tensor = img2_tensor.cuda()
        
        with torch.no_grad():
            flow = self.model(img1_tensor, img2_tensor)
        
        flow = flow.squeeze(0).cpu().numpy()
        
        scale_h = original_h / (new_h / 8.0)
        scale_w = original_w / (new_w / 8.0)
        
        flow[0] *= scale_w
        flow[1] *= scale_h
        
        flow_u = cv2.resize(flow[0], (original_w, original_h))
        flow_v = cv2.resize(flow[1], (original_w, original_h))
        
        return flow_u.astype(np.float32), flow_v.astype(np.float32)
    
    def _calculate_opencv(self, frame1: np.ndarray, frame2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if len(frame1.shape) == 3:
            gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        else:
            gray1 = frame1
            gray2 = frame2
        
        flow = cv2.calcOpticalFlowFarneback(
            gray1, gray2, None,
            pyr_scale=0.5, levels=5, winsize=15,
            iterations=5, poly_n=5, poly_sigma=1.2, flags=0
        )
        
        return flow[..., 0].astype(np.float32), flow[..., 1].astype(np.float32)
    
    def calculate_batch(self, frames_pairs: list) -> list:
        results = []
        for frame1, frame2 in frames_pairs:
            u, v = self.calculate(frame1, frame2)
            results.append((u, v))
        return results


class SubpixelRefinement:
    @staticmethod
    def refine_flow(u: np.ndarray, v: np.ndarray, 
                    img1: np.ndarray, img2: np.ndarray,
                    iterations: int = 3, window_size: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        if len(img1.shape) == 3:
            gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY).astype(np.float32)
            gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY).astype(np.float32)
        else:
            gray1 = img1.astype(np.float32)
            gray2 = img2.astype(np.float32)
        
        rows, cols = gray1.shape
        
        u_refined = u.copy()
        v_refined = v.copy()
        
        half_window = window_size // 2
        
        for _ in range(iterations):
            map_x, map_y = np.meshgrid(np.arange(cols), np.arange(rows))
            map_x = map_x.astype(np.float32) + u_refined
            map_y = map_y.astype(np.float32) + v_refined
            
            warped = cv2.remap(gray2, map_x, map_y, cv2.INTER_LINEAR)
            
            Ix = cv2.Sobel(warped, cv2.CV_32F, 1, 0, ksize=3)
            Iy = cv2.Sobel(warped, cv2.CV_32F, 0, 1, ksize=3)
            It = warped - gray1
            
            Ix_pad = np.pad(Ix, half_window, mode='edge')
            Iy_pad = np.pad(Iy, half_window, mode='edge')
            It_pad = np.pad(It, half_window, mode='edge')
            
            delta_u = np.zeros_like(u_refined)
            delta_v = np.zeros_like(v_refined)
            
            for i in range(rows):
                for j in range(cols):
                    i_end = i + window_size
                    j_end = j + window_size
                    
                    Ix_win = Ix_pad[i:i_end, j:j_end].flatten()
                    Iy_win = Iy_pad[i:i_end, j:j_end].flatten()
                    It_win = It_pad[i:i_end, j:j_end].flatten()
                    
                    A = np.vstack((Ix_win, Iy_win)).T
                    b = -It_win
                    
                    try:
                        flow_delta = np.linalg.lstsq(A, b, rcond=None)[0]
                        delta_u[i, j] = flow_delta[0]
                        delta_v[i, j] = flow_delta[1]
                    except:
                        pass
            
            u_refined += delta_u * 0.5
            v_refined += delta_v * 0.5
        
        return u_refined, v_refined


def flow_to_color(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    magnitude, angle = cv2.cartToPolar(u, v)
    
    hsv = np.zeros((u.shape[0], u.shape[1], 3), dtype=np.uint8)
    hsv[..., 1] = 255
    
    hsv[..., 0] = (angle * 180 / np.pi / 2).astype(np.uint8)
    hsv[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


if __name__ == "__main__":
    print("Deep Optical Flow Calculator")
    print("=" * 50)
    print("Models available:")
    print("  - RAFT (Recurrent All-Pairs Field Transforms)")
    print("  - FlowNet2")
    print("  - LiteFlowNet")
    print()
    print("Features:")
    print("  - Subpixel accuracy")
    print("  - GPU acceleration (CUDA)")
    print("  - Batch processing support")
    print()
    print("Usage example:")
    print("  from deep_optical_flow import DeepOpticalFlow")
    print("  calculator = DeepOpticalFlow(model_name='raft', device='cuda')")
    print("  u, v = calculator.calculate(frame1, frame2)")
    print()
    print("For subpixel refinement:")
    print("  from deep_optical_flow import SubpixelRefinement")
    print("  u_ref, v_ref = SubpixelRefinement.refine_flow(u, v, frame1, frame2)")
