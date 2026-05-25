import numpy as np
import cv2
import os
import warnings
warnings.filterwarnings('ignore')

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("PyTorch not available. Deep learning features will be disabled.")


class SingularityNet(nn.Module):
    def __init__(self, num_cores=2, num_deltas=2, img_size=256):
        super(SingularityNet, self).__init__()
        self.num_cores = num_cores
        self.num_deltas = num_deltas
        self.img_size = img_size
        self.num_keypoints = num_cores + num_deltas
        
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5)
        
        self.fc1 = nn.Linear(256 * 16 * 16, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, self.num_keypoints * 3)
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        
        x = x.view(x.size(0), -1)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.fc3(x)
        
        batch_size = x.size(0)
        x = x.view(batch_size, self.num_keypoints, 3)
        
        keypoints = torch.sigmoid(x[:, :, :2])
        confidences = torch.sigmoid(x[:, :, 2:3])
        
        return torch.cat([keypoints, confidences], dim=2)


class SingularityHeatmapNet(nn.Module):
    def __init__(self, num_cores=2, num_deltas=2, img_size=256, heatmap_size=64):
        super(SingularityHeatmapNet, self).__init__()
        self.num_cores = num_cores
        self.num_deltas = num_deltas
        self.num_keypoints = num_cores + num_deltas
        self.img_size = img_size
        self.heatmap_size = heatmap_size
        
        self.down1 = self._make_down_block(1, 64)
        self.down2 = self._make_down_block(64, 128)
        self.down3 = self._make_down_block(128, 256)
        self.down4 = self._make_down_block(256, 512)
        
        self.bottleneck = nn.Sequential(
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        
        self.up1 = self._make_up_block(512, 256)
        self.up2 = self._make_up_block(256, 128)
        self.up3 = self._make_up_block(128, 64)
        
        self.final = nn.Conv2d(64, self.num_keypoints, 1)
    
    def _make_down_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
    
    def _make_up_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, 2, stride=2),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        
        bn = self.bottleneck(d4)
        
        u1 = self.up1(bn)
        u2 = self.up2(u1)
        u3 = self.up3(u2)
        
        heatmaps = torch.sigmoid(self.final(u3))
        
        return heatmaps


class SingularityDetector:
    def __init__(self, model_path=None, num_cores=2, num_deltas=2, img_size=256, use_heatmap=False):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for deep learning detection")
        
        self.num_cores = num_cores
        self.num_deltas = num_deltas
        self.img_size = img_size
        self.use_heatmap = use_heatmap
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        if use_heatmap:
            self.model = SingularityHeatmapNet(num_cores, num_deltas, img_size)
        else:
            self.model = SingularityNet(num_cores, num_deltas, img_size)
        
        self.model.to(self.device)
        
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
            print(f"Loaded model from {model_path}")
        
        self.model.eval()
    
    def load_model(self, model_path):
        state_dict = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
    
    def preprocess(self, image):
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        h, w = image.shape
        if h != self.img_size or w != self.img_size:
            image = cv2.resize(image, (self.img_size, self.img_size))
        
        image = image.astype(np.float32) / 255.0
        image = (image - 0.5) / 0.5
        
        tensor = torch.from_numpy(image).unsqueeze(0).unsqueeze(0)
        return tensor.to(self.device)
    
    def detect(self, image, confidence_threshold=0.5):
        if not TORCH_AVAILABLE:
            return [], []
        
        with torch.no_grad():
            input_tensor = self.preprocess(image)
            outputs = self.model(input_tensor)
            
            if self.use_heatmap:
                cores, deltas = self._parse_heatmaps(outputs[0], confidence_threshold)
            else:
                cores, deltas = self._parse_regression(outputs[0], confidence_threshold, image.shape[:2])
        
        return cores, deltas
    
    def _parse_regression(self, output, confidence_threshold, orig_shape):
        h, w = orig_shape[:2]
        scale_x = w / self.img_size
        scale_y = h / self.img_size
        
        cores = []
        deltas = []
        
        for i in range(self.num_cores):
            x = float(output[i, 0] * self.img_size * scale_x)
            y = float(output[i, 1] * self.img_size * scale_y)
            conf = float(output[i, 2])
            if conf >= confidence_threshold:
                cores.append((int(x), int(y), 1.0, conf))
        
        for i in range(self.num_deltas):
            idx = self.num_cores + i
            x = float(output[idx, 0] * self.img_size * scale_x)
            y = float(output[idx, 1] * self.img_size * scale_y)
            conf = float(output[idx, 2])
            if conf >= confidence_threshold:
                deltas.append((int(x), int(y), -1.0, conf))
        
        return cores, deltas
    
    def _parse_heatmaps(self, heatmaps, confidence_threshold):
        cores = []
        deltas = []
        
        heatmaps = heatmaps.cpu().numpy()
        
        for i in range(self.num_cores):
            hm = heatmaps[i]
            y, x = np.unravel_index(np.argmax(hm), hm.shape)
            conf = float(hm[y, x])
            if conf >= confidence_threshold:
                scale = self.img_size / heatmaps.shape[1]
                cores.append((int(x * scale), int(y * scale), 1.0, conf))
        
        for i in range(self.num_deltas):
            idx = self.num_cores + i
            hm = heatmaps[idx]
            y, x = np.unravel_index(np.argmax(hm), hm.shape)
            conf = float(hm[y, x])
            if conf >= confidence_threshold:
                scale = self.img_size / heatmaps.shape[1]
                deltas.append((int(x * scale), int(y * scale), -1.0, conf))
        
        return cores, deltas


def keypoint_loss(pred, target, target_mask=None):
    if target_mask is None:
        target_mask = torch.ones_like(target[:, :, 2:3])
    
    pred_xy = pred[:, :, :2]
    pred_conf = pred[:, :, 2:3]
    
    target_xy = target[:, :, :2]
    target_conf = target[:, :, 2:3]
    
    reg_loss = F.mse_loss(pred_xy * target_mask, target_xy * target_mask, reduction='sum')
    reg_loss = reg_loss / (target_mask.sum() + 1e-6)
    
    conf_loss = F.binary_cross_entropy(pred_conf, target_conf)
    
    return reg_loss + conf_loss


def heatmap_loss(pred, target, target_mask=None):
    if target_mask is None:
        target_mask = torch.ones_like(target)
    
    loss = F.mse_loss(pred * target_mask, target * target_mask)
    return loss


def generate_heatmap(x, y, img_size=256, heatmap_size=64, sigma=2):
    scale = heatmap_size / img_size
    x_scaled = int(x * scale)
    y_scaled = int(y * scale)
    
    heatmap = np.zeros((heatmap_size, heatmap_size), dtype=np.float32)
    
    if x_scaled < 0 or x_scaled >= heatmap_size or y_scaled < 0 or y_scaled >= heatmap_size:
        return heatmap
    
    yy, xx = np.mgrid[0:heatmap_size, 0:heatmap_size]
    dist = np.sqrt((xx - x_scaled)**2 + (yy - y_scaled)**2)
    heatmap = np.exp(-dist**2 / (2 * sigma**2))
    
    return heatmap


if __name__ == '__main__':
    if TORCH_AVAILABLE:
        print("Testing model...")
        model = SingularityNet(num_cores=2, num_deltas=2)
        test_input = torch.randn(2, 1, 256, 256)
        output = model(test_input)
        print(f"Input shape: {test_input.shape}")
        print(f"Output shape: {output.shape}")
        
        model_hm = SingularityHeatmapNet(num_cores=2, num_deltas=2)
        output_hm = model_hm(test_input)
        print(f"Heatmap output shape: {output_hm.shape}")
    else:
        print("PyTorch not available. Install with: pip install torch")
