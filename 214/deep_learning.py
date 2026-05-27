import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Tuple, Optional, Union
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import warnings


warnings.filterwarnings('ignore')


class SpectrumDataset(Dataset):
    """光谱数据集类"""
    
    def __init__(self, spectra: np.ndarray, labels: np.ndarray,
                 transform=None):
        """
        Args:
            spectra: 光谱数据，形状 (N, L)
            labels: 标签数据
            transform: 数据变换
        """
        self.spectra = torch.FloatTensor(spectra)
        self.labels = torch.LongTensor(labels) if labels.dtype == np.int64 else torch.FloatTensor(labels)
        self.transform = transform
    
    def __len__(self):
        return len(self.spectra)
    
    def __getitem__(self, idx):
        spectrum = self.spectra[idx].unsqueeze(0)
        label = self.labels[idx]
        
        if self.transform:
            spectrum = self.transform(spectrum)
        
        return spectrum, label


class SpectraAugmentation:
    """光谱数据增强"""
    
    def __init__(self, noise_level: float = 0.02,
                 intensity_shift_range: float = 0.1,
                 wavelength_shift_max: int = 3,
                 scaling_range: float = 0.1):
        self.noise_level = noise_level
        self.intensity_shift_range = intensity_shift_range
        self.wavelength_shift_max = wavelength_shift_max
        self.scaling_range = scaling_range
    
    def __call__(self, spectrum: torch.Tensor) -> torch.Tensor:
        """对光谱进行增强"""
        spectrum = spectrum.clone()
        
        if np.random.random() > 0.5:
            noise = torch.randn_like(spectrum) * self.noise_level
            spectrum += noise
        
        if np.random.random() > 0.5:
            shift = np.random.uniform(-self.intensity_shift_range, self.intensity_shift_range)
            spectrum += shift
        
        if np.random.random() > 0.5 and self.wavelength_shift_max > 0:
            shift = np.random.randint(-self.wavelength_shift_max, self.wavelength_shift_max)
            spectrum = torch.roll(spectrum, shifts=shift, dims=-1)
        
        if np.random.random() > 0.5:
            scale = np.random.uniform(1 - self.scaling_range, 1 + self.scaling_range)
            spectrum *= scale
        
        return spectrum


class Conv1DBlock(nn.Module):
    """1D卷积块"""
    
    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int = 3, stride: int = 1,
                 dropout: float = 0.2):
        super(Conv1DBlock, self).__init__()
        padding = kernel_size // 2
        
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, 1, padding)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        
        self.pool = nn.MaxPool1d(2)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        
        x = self.pool(x)
        return x


class Raman1DCNN(nn.Module):
    """1D-CNN光谱分类模型"""
    
    def __init__(self, num_classes: int = 10,
                 input_length: int = 1000,
                 dropout: float = 0.3):
        super(Raman1DCNN, self).__init__()
        
        self.features = nn.Sequential(
            Conv1DBlock(1, 32, 7, dropout=dropout),
            Conv1DBlock(32, 64, 5, dropout=dropout),
            Conv1DBlock(64, 128, 3, dropout=dropout),
            Conv1DBlock(128, 256, 3, dropout=dropout),
        )
        
        with torch.no_grad():
            dummy = torch.zeros(1, 1, input_length)
            features = self.features(dummy)
            feature_size = features.view(1, -1).size(1)
        
        self.classifier = nn.Sequential(
            nn.Linear(feature_size, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
    
    def extract_features(self, x):
        """提取特征向量"""
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return x


class ResidualBlock1D(nn.Module):
    """1D残差块"""
    
    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int = 3, stride: int = 1,
                 dropout: float = 0.2):
        super(ResidualBlock1D, self).__init__()
        
        padding = kernel_size // 2
        
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, 1, padding)
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        if in_channels != out_channels or stride != 1:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()
    
    def forward(self, x):
        identity = self.shortcut(x)
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out += identity
        out = self.relu(out)
        
        return out


class RamanResNet(nn.Module):
    """1D-ResNet光谱分类模型"""
    
    def __init__(self, num_classes: int = 10,
                 input_length: int = 1000,
                 dropout: float = 0.3):
        super(RamanResNet, self).__init__()
        
        self.stem = nn.Sequential(
            nn.Conv1d(1, 64, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(3, stride=2, padding=1)
        )
        
        self.layer1 = self._make_layer(64, 64, 2, dropout)
        self.layer2 = self._make_layer(64, 128, 2, dropout, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, dropout, stride=2)
        self.layer4 = self._make_layer(256, 512, 2, dropout, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        
        with torch.no_grad():
            dummy = torch.zeros(1, 1, input_length)
            features = self.stem(dummy)
            features = self.layer1(features)
            features = self.layer2(features)
            features = self.layer3(features)
            features = self.layer4(features)
            features = self.avgpool(features)
            feature_size = features.view(1, -1).size(1)
        
        self.classifier = nn.Sequential(
            nn.Linear(feature_size, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )
    
    def _make_layer(self, in_channels, out_channels, num_blocks, dropout, stride=1):
        layers = []
        layers.append(ResidualBlock1D(in_channels, out_channels, 3, stride, dropout))
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock1D(out_channels, out_channels, 3, 1, dropout))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
    
    def extract_features(self, x):
        """提取特征向量"""
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return x


class RamanMixupModel(nn.Module):
    """混合光谱分析模型（支持多标签分类）"""
    
    def __init__(self, num_components: int = 10,
                 input_length: int = 1000,
                 dropout: float = 0.3):
        super(RamanMixupModel, self).__init__()
        
        self.features = nn.Sequential(
            Conv1DBlock(1, 32, 7, dropout=dropout),
            Conv1DBlock(32, 64, 5, dropout=dropout),
            Conv1DBlock(64, 128, 3, dropout=dropout),
        )
        
        with torch.no_grad():
            dummy = torch.zeros(1, 1, input_length)
            features = self.features(dummy)
            feature_size = features.view(1, -1).size(1)
        
        self.encoder = nn.Sequential(
            nn.Linear(feature_size, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_components),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.encoder(x)
        x = self.classifier(x)
        return x
    
    def extract_features(self, x):
        """提取特征向量"""
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.encoder(x)
        return x
    
    def predict_components(self, x, threshold: float = 0.5):
        """预测混合光谱中的成分"""
        probs = self.forward(x)
        predictions = (probs > threshold).float()
        return probs, predictions


def generate_mixture_spectra(pure_spectra: Dict[str, np.ndarray],
                             num_mixtures: int = 1000,
                             max_components: int = 3,
                             noise_level: float = 0.02) -> Tuple[np.ndarray, np.ndarray, List[List[str]]]:
    """
    生成混合光谱数据用于训练
    
    Args:
        pure_spectra: 纯物质光谱字典 {名称: 强度数组}
        num_mixtures: 生成的混合光谱数量
        max_components: 混合光谱中的最大组分数
        noise_level: 噪声水平
    
    Returns:
        (混合光谱数组, 多标签编码数组, 成分名称列表)
    """
    names = list(pure_spectra.keys())
    num_components = len(names)
    spectrum_length = len(list(pure_spectra.values())[0])
    
    mixtures = []
    labels = []
    component_names = []
    
    for _ in range(num_mixtures):
        num_comp = np.random.randint(1, max_components + 1)
        selected = np.random.choice(num_components, num_comp, replace=False)
        
        weights = np.random.dirichlet(np.ones(num_comp))
        
        mixture = np.zeros(spectrum_length)
        for i, idx in enumerate(selected):
            mixture += weights[i] * pure_spectra[names[idx]]
        
        noise = np.random.normal(0, noise_level, spectrum_length)
        mixture += noise
        
        min_val = mixture.min()
        max_val = mixture.max()
        if max_val - min_val > 1e-10:
            mixture = (mixture - min_val) / (max_val - min_val)
        
        label = np.zeros(num_components)
        label[selected] = 1
        
        mixtures.append(mixture)
        labels.append(label)
        component_names.append([names[i] for i in selected])
    
    return np.array(mixtures), np.array(labels), component_names


class DeepLearningAnalyzer:
    """深度学习光谱分析器"""
    
    def __init__(self, model_type: str = 'cnn',
                 num_classes: int = 10,
                 input_length: int = 1000,
                 dropout: float = 0.3,
                 device: str = None):
        """
        Args:
            model_type: 模型类型 ('cnn', 'resnet', 'mixup')
            num_classes: 类别数（mixup模式下为成分数）
            input_length: 输入光谱长度
            dropout: Dropout率
            device: 计算设备
        """
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        self.model_type = model_type
        self.num_classes = num_classes
        self.input_length = input_length
        
        if model_type == 'cnn':
            self.model = Raman1DCNN(num_classes, input_length, dropout).to(self.device)
        elif model_type == 'resnet':
            self.model = RamanResNet(num_classes, input_length, dropout).to(self.device)
        elif model_type == 'mixup':
            self.model = RamanMixupModel(num_classes, input_length, dropout).to(self.device)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        self.class_names = []
        self.is_trained = False
        
        print(f"使用设备: {self.device}")
        print(f"模型类型: {model_type}")
    
    def prepare_data(self, spectra: np.ndarray, labels: np.ndarray,
                     batch_size: int = 32,
                     use_augmentation: bool = True,
                     train_ratio: float = 0.8):
        """
        准备训练数据
        
        Args:
            spectra: 光谱数据
            labels: 标签数据
            batch_size: 批大小
            use_augmentation: 是否使用数据增强
            train_ratio: 训练集比例
        
        Returns:
            (train_loader, val_loader)
        """
        X_train, X_val, y_train, y_val = train_test_split(
            spectra, labels, test_size=1 - train_ratio, random_state=42, stratify=labels if self.model_type != 'mixup' else None
        )
        
        transform = SpectraAugmentation() if use_augmentation else None
        
        train_dataset = SpectrumDataset(X_train, y_train, transform)
        val_dataset = SpectrumDataset(X_val, y_val, None)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        return train_loader, val_loader
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader,
              epochs: int = 50, lr: float = 0.001,
              weight_decay: float = 1e-4,
              early_stopping_patience: int = 10):
        """
        训练模型
        
        Args:
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            epochs: 训练轮数
            lr: 学习率
            weight_decay: 权重衰减
            early_stopping_patience: 早停耐心值
        """
        if self.model_type == 'mixup':
            criterion = nn.BCELoss()
        else:
            criterion = nn.CrossEntropyLoss()
        
        optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
        
        print("开始训练...")
        
        for epoch in range(epochs):
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for spectra, labels in train_loader:
                spectra = spectra.to(self.device)
                labels = labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(spectra)
                
                if self.model_type == 'mixup':
                    loss = criterion(outputs, labels)
                else:
                    loss = criterion(outputs, labels)
                
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                train_total += labels.size(0)
                
                if self.model_type == 'mixup':
                    predictions = (outputs > 0.5).float()
                    train_correct += (predictions == labels).all(dim=1).sum().item()
                else:
                    _, predicted = torch.max(outputs, 1)
                    train_correct += (predicted == labels).sum().item()
            
            train_loss /= len(train_loader)
            train_acc = 100 * train_correct / train_total
            
            self.model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for spectra, labels in val_loader:
                    spectra = spectra.to(self.device)
                    labels = labels.to(self.device)
                    
                    outputs = self.model(spectra)
                    
                    if self.model_type == 'mixup':
                        loss = criterion(outputs, labels)
                    else:
                        loss = criterion(outputs, labels)
                    
                    val_loss += loss.item()
                    val_total += labels.size(0)
                    
                    if self.model_type == 'mixup':
                        predictions = (outputs > 0.5).float()
                        val_correct += (predictions == labels).all(dim=1).sum().item()
                    else:
                        _, predicted = torch.max(outputs, 1)
                        val_correct += (predicted == labels).sum().item()
            
            val_loss /= len(val_loader)
            val_acc = 100 * val_correct / val_total
            
            scheduler.step()
            
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            
            print(f'Epoch [{epoch+1}/{epochs}] '
                  f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% '
                  f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(self.model.state_dict(), 'best_model.pth')
            else:
                patience_counter += 1
            
            if patience_counter >= early_stopping_patience:
                print(f"早停于 epoch {epoch+1}")
                break
        
        self.model.load_state_dict(torch.load('best_model.pth', map_location=self.device, weights_only=True))
        self.is_trained = True
        
        print("训练完成!")
        
        return history
    
    def predict(self, spectrum: np.ndarray) -> Dict:
        """
        预测光谱
        
        Args:
            spectrum: 光谱数据
        
        Returns:
            预测结果字典
        """
        self.model.eval()
        
        spectrum = torch.FloatTensor(spectrum).unsqueeze(0).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            if self.model_type == 'mixup':
                probs = self.model(spectrum)
                predictions = (probs > 0.5).float()
                
                probs = probs.cpu().numpy()[0]
                predictions = predictions.cpu().numpy()[0]
                
                detected_components = []
                for i, (prob, pred) in enumerate(zip(probs, predictions)):
                    if pred == 1 and i < len(self.class_names):
                        detected_components.append({
                            'name': self.class_names[i],
                            'probability': float(prob)
                        })
                
                return {
                    'probabilities': probs,
                    'predictions': predictions,
                    'detected_components': detected_components
                }
            else:
                outputs = self.model(spectrum)
                probabilities = torch.softmax(outputs, dim=1)
                confidence, predicted = torch.max(probabilities, 1)
                
                return {
                    'class': self.class_names[predicted.item()] if predicted.item() < len(self.class_names) else f'Class_{predicted.item()}',
                    'class_index': predicted.item(),
                    'confidence': confidence.item(),
                    'probabilities': probabilities.cpu().numpy()[0]
                }
    
    def predict_batch(self, spectra: np.ndarray) -> np.ndarray:
        """批量预测"""
        self.model.eval()
        
        spectra = torch.FloatTensor(spectra).unsqueeze(1).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(spectra)
            if self.model_type == 'mixup':
                predictions = (outputs > 0.5).float().cpu().numpy()
            else:
                _, predicted = torch.max(outputs, 1)
                predictions = predicted.cpu().numpy()
        
        return predictions
    
    def extract_features(self, spectrum: np.ndarray) -> np.ndarray:
        """提取特征向量"""
        self.model.eval()
        
        spectrum = torch.FloatTensor(spectrum).unsqueeze(0).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            features = self.model.extract_features(spectrum)
        
        return features.cpu().numpy()[0]
    
    def save_model(self, filepath: str):
        """保存模型"""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'model_type': self.model_type,
            'num_classes': self.num_classes,
            'input_length': self.input_length,
            'class_names': self.class_names
        }
        torch.save(checkpoint, filepath)
        print(f"模型已保存到: {filepath}")
    
    def load_model(self, filepath: str):
        """加载模型"""
        checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)
        
        self.model_type = checkpoint['model_type']
        self.num_classes = checkpoint['num_classes']
        self.input_length = checkpoint['input_length']
        self.class_names = checkpoint['class_names']
        
        if self.model_type == 'cnn':
            self.model = Raman1DCNN(self.num_classes, self.input_length).to(self.device)
        elif self.model_type == 'resnet':
            self.model = RamanResNet(self.num_classes, self.input_length).to(self.device)
        elif self.model_type == 'mixup':
            self.model = RamanMixupModel(self.num_classes, self.input_length).to(self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.is_trained = True
        
        print(f"模型已从 {filepath} 加载")
    
    def set_class_names(self, names: List[str]):
        """设置类别名称"""
        self.class_names = names
