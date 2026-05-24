import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("PyTorch not available. Deep prior methods will be disabled.")


if TORCH_AVAILABLE:
    
    class DoubleConv(nn.Module):
        """(Conv => BN => ReLU) * 2"""
        
        def __init__(self, in_channels, out_channels, mid_channels=None):
            super().__init__()
            if not mid_channels:
                mid_channels = out_channels
            self.double_conv = nn.Sequential(
                nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(mid_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
        
        def forward(self, x):
            return self.double_conv(x)
    
    
    class Down(nn.Module):
        """Downscaling with maxpool then double conv"""
        
        def __init__(self, in_channels, out_channels):
            super().__init__()
            self.maxpool_conv = nn.Sequential(
                nn.MaxPool2d(2),
                DoubleConv(in_channels, out_channels)
            )
        
        def forward(self, x):
            return self.maxpool_conv(x)
    
    
    class Up(nn.Module):
        """Upscaling then double conv"""
        
        def __init__(self, in_channels, out_channels, bilinear=True):
            super().__init__()
            
            if bilinear:
                self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
                self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
            else:
                self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
                self.conv = DoubleConv(in_channels, out_channels)
        
        def forward(self, x1, x2):
            x1 = self.up(x1)
            
            diffY = x2.size()[2] - x1.size()[2]
            diffX = x2.size()[3] - x1.size()[3]
            
            x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                            diffY // 2, diffY - diffY // 2])
            
            x = torch.cat([x2, x1], dim=1)
            return self.conv(x)
    
    
    class OutConv(nn.Module):
        def __init__(self, in_channels, out_channels):
            super(OutConv, self).__init__()
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        
        def forward(self, x):
            return self.conv(x)
    
    
    class UNet(nn.Module):
        """
        U-Net 网络结构，用于 Deep Image Prior
        
        输入: 随机噪声或低维特征
        输出: 复振幅分布（2通道：振幅和相位，或实部和虚部）
        """
        
        def __init__(self, n_channels=1, n_output=2, bilinear=True):
            super(UNet, self).__init__()
            self.n_channels = n_channels
            self.n_output = n_output
            self.bilinear = bilinear
            
            self.inc = DoubleConv(n_channels, 64)
            self.down1 = Down(64, 128)
            self.down2 = Down(128, 256)
            self.down3 = Down(256, 512)
            factor = 2 if bilinear else 1
            self.down4 = Down(512, 1024 // factor)
            self.up1 = Up(1024, 512 // factor, bilinear)
            self.up2 = Up(512, 256 // factor, bilinear)
            self.up3 = Up(256, 128 // factor, bilinear)
            self.up4 = Up(128, 64, bilinear)
            self.outc = OutConv(64, n_output)
        
        def forward(self, x):
            x1 = self.inc(x)
            x2 = self.down1(x1)
            x3 = self.down2(x2)
            x4 = self.down3(x3)
            x5 = self.down4(x4)
            x = self.up1(x5, x4)
            x = self.up2(x, x3)
            x = self.up3(x, x2)
            x = self.up4(x, x1)
            logits = self.outc(x)
            return logits
    
    
    class SmallUNet(nn.Module):
        """
        小型 U-Net，适合快速实验
        """
        
        def __init__(self, n_channels=1, n_output=2):
            super(SmallUNet, self).__init__()
            
            self.encoder = nn.Sequential(
                nn.Conv2d(n_channels, 32, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(32, 64, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(64, 128, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(128, 256, 3, padding=1),
                nn.ReLU(),
            )
            
            self.decoder = nn.Sequential(
                nn.Conv2d(256, 128, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(128, 64, 3, padding=1),
                nn.ReLU(),
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                nn.Conv2d(64, 32, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(32, n_output, 3, padding=1),
            )
        
        def forward(self, x):
            x = self.encoder(x)
            x = self.decoder(x)
            return x
    
    
    class ResidualBlock(nn.Module):
        def __init__(self, channels):
            super(ResidualBlock, self).__init__()
            self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
            self.bn1 = nn.BatchNorm2d(channels)
            self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
            self.bn2 = nn.BatchNorm2d(channels)
        
        def forward(self, x):
            residual = x
            out = F.relu(self.bn1(self.conv1(x)))
            out = self.bn2(self.conv2(out))
            out += residual
            out = F.relu(out)
            return out
    
    
    class ResNetPrior(nn.Module):
        """
        基于残差网络的先验模型
        """
        
        def __init__(self, n_channels=1, n_output=2, num_blocks=5):
            super(ResNetPrior, self).__init__()
            
            self.input_conv = nn.Sequential(
                nn.Conv2d(n_channels, 64, 3, padding=1),
                nn.ReLU()
            )
            
            self.residual_blocks = nn.Sequential(
                *[ResidualBlock(64) for _ in range(num_blocks)]
            )
            
            self.output_conv = nn.Conv2d(64, n_output, 3, padding=1)
        
        def forward(self, x):
            x = self.input_conv(x)
            x = self.residual_blocks(x)
            x = self.output_conv(x)
            return x
    
    
    class DeepPriorPhaseRetrieval:
        """
        基于 Deep Image Prior 的相位恢复算法
        
        使用未训练的神经网络作为隐式正则化器，
        通过网络结构约束来提高重建质量和信噪比。
        """
        
        def __init__(self, wavelength, pixel_size, distance,
                     method='angular_spectrum',
                     network_type='unet',
                     output_type='amp_phase',
                     device='cpu',
                     dtype=torch.float32):
            """
            初始化深度先验相位恢复
            
            参数:
                wavelength: 波长
                pixel_size: 像素大小
                distance: 传播距离
                method: 传播方法
                network_type: 网络类型 ('unet', 'small_unet', 'resnet')
                output_type: 输出类型 ('amp_phase' 振幅相位, 'real_imag' 实部虚部)
                device: 计算设备 ('cpu' 或 'cuda')
                dtype: 数据类型
            """
            self.wavelength = wavelength
            self.pixel_size = pixel_size
            self.distance = distance
            self.method = method
            self.output_type = output_type
            self.device = device
            self.dtype = dtype
            
            self.k = 2 * np.pi / wavelength
            
            if network_type == 'unet':
                self.network = UNet(n_channels=1, n_output=2)
            elif network_type == 'small_unet':
                self.network = SmallUNet(n_channels=1, n_output=2)
            elif network_type == 'resnet':
                self.network = ResNetPrior(n_channels=1, n_output=2)
            else:
                raise ValueError(f"未知网络类型: {network_type}")
            
            self.network = self.network.to(device=device, dtype=dtype)
            
            self.prop_kernel = None
            self.errors = []
        
        def _init_propagation(self, shape):
            """
            初始化传播核
            """
            ny, nx = shape
            fx = np.fft.fftfreq(nx, self.pixel_size)
            fy = np.fft.fftfreq(ny, self.pixel_size)
            FX, FY = np.meshgrid(fx, fy)
            
            k_squared = self.k ** 2
            kx = 2 * np.pi * FX
            ky = 2 * np.pi * FY
            
            sqrt_term = np.sqrt(np.maximum(k_squared - kx**2 - ky**2, 0))
            H = np.exp(1j * self.distance * sqrt_term)
            
            self.prop_kernel = torch.tensor(H, dtype=torch.complex64, device=self.device)
        
        def _propagate(self, field):
            """
            角谱传播（PyTorch版本）
            """
            field_ft = torch.fft.fftn(field, dim=(-2, -1))
            field_prop_ft = field_ft * self.prop_kernel
            field_prop = torch.fft.ifftn(field_prop_ft, dim=(-2, -1))
            return field_prop
        
        def _decode_output(self, output):
            """
            解码网络输出为复振幅
            """
            if self.output_type == 'amp_phase':
                amplitude = torch.abs(output[:, 0:1]) + 1e-8
                phase = output[:, 1:2]
                complex_field = amplitude * torch.exp(1j * phase)
            elif self.output_type == 'real_imag':
                real = output[:, 0:1]
                imag = output[:, 1:2]
                complex_field = torch.complex(real, imag)
            else:
                raise ValueError(f"未知输出类型: {self.output_type}")
            
            return complex_field
        
        def reconstruct(self, measured_intensity,
                        num_iterations=1000,
                        lr=0.01,
                        input_noise_std=0.1,
                        verbose=True,
                        print_interval=100):
            """
            执行深度先验相位恢复
            
            参数:
                measured_intensity: 测得的衍射强度图
                num_iterations: 迭代次数
                lr: 学习率
                input_noise_std: 输入噪声标准差
                verbose: 是否打印进度
                print_interval: 打印间隔
            
            返回:
                重建的复振幅分布（numpy数组）
            """
            if not TORCH_AVAILABLE:
                raise RuntimeError("PyTorch is not available")
            
            measured_intensity_torch = torch.tensor(
                measured_intensity, dtype=self.dtype, device=self.device
            )
            measured_intensity_torch = measured_intensity_torch.unsqueeze(0).unsqueeze(0)
            
            ny, nx = measured_intensity.shape
            self._init_propagation((ny, nx))
            
            input_noise = torch.randn(1, 1, ny, nx, device=self.device, dtype=self.dtype)
            input_noise = input_noise * input_noise_std
            input_noise.requires_grad = False
            
            optimizer = torch.optim.Adam(self.network.parameters(), lr=lr)
            
            self.errors = []
            
            for iteration in range(num_iterations):
                optimizer.zero_grad()
                
                network_output = self.network(input_noise)
                
                object_field = self._decode_output(network_output)
                
                diffraction_field = self._propagate(object_field)
                
                predicted_intensity = torch.abs(diffraction_field) ** 2
                
                loss = torch.mean((predicted_intensity - measured_intensity_torch) ** 2)
                
                loss.backward()
                optimizer.step()
                
                loss_value = loss.item()
                self.errors.append(loss_value)
                
                if verbose and (iteration + 1) % print_interval == 0:
                    print(f"Iteration {iteration + 1}/{num_iterations}, Loss: {loss_value:.6e}")
            
            with torch.no_grad():
                final_output = self.network(input_noise)
                final_field = self._decode_output(final_output)
                final_field_np = final_field[0, 0].cpu().numpy()
            
            return final_field_np
        
        def get_errors(self):
            return np.array(self.errors)
    
    
    class DeepPriorMultiDistance:
        """
        多距离深度先验相位恢复
        """
        
        def __init__(self, wavelength, pixel_size, distances,
                     method='angular_spectrum',
                     network_type='unet',
                     output_type='amp_phase',
                     weights=None,
                     device='cpu',
                     dtype=torch.float32):
            """
            初始化多距离深度先验相位恢复
            """
            self.wavelength = wavelength
            self.pixel_size = pixel_size
            self.distances = distances
            self.method = method
            self.output_type = output_type
            self.device = device
            self.dtype = dtype
            
            if weights is None:
                self.weights = np.ones(len(distances)) / len(distances)
            else:
                self.weights = np.array(weights) / np.sum(weights)
            
            self.k = 2 * np.pi / wavelength
            
            if network_type == 'unet':
                self.network = UNet(n_channels=1, n_output=2)
            elif network_type == 'small_unet':
                self.network = SmallUNet(n_channels=1, n_output=2)
            elif network_type == 'resnet':
                self.network = ResNetPrior(n_channels=1, n_output=2)
            else:
                raise ValueError(f"未知网络类型: {network_type}")
            
            self.network = self.network.to(device=device, dtype=dtype)
            
            self.prop_kernels = []
            self.errors = []
        
        def _init_propagation(self, shape):
            """
            初始化多个传播核
            """
            ny, nx = shape
            fx = np.fft.fftfreq(nx, self.pixel_size)
            fy = np.fft.fftfreq(ny, self.pixel_size)
            FX, FY = np.meshgrid(fx, fy)
            
            k_squared = self.k ** 2
            kx = 2 * np.pi * FX
            ky = 2 * np.pi * FY
            
            sqrt_term = np.sqrt(np.maximum(k_squared - kx**2 - ky**2, 0))
            
            self.prop_kernels = []
            for distance in self.distances:
                H = np.exp(1j * distance * sqrt_term)
                self.prop_kernels.append(torch.tensor(H, dtype=torch.complex64, device=self.device))
        
        def _propagate(self, field, kernel_idx):
            """
            使用指定核进行传播
            """
            field_ft = torch.fft.fftn(field, dim=(-2, -1))
            field_prop_ft = field_ft * self.prop_kernels[kernel_idx]
            field_prop = torch.fft.ifftn(field_prop_ft, dim=(-2, -1))
            return field_prop
        
        def _decode_output(self, output):
            if self.output_type == 'amp_phase':
                amplitude = torch.abs(output[:, 0:1]) + 1e-8
                phase = output[:, 1:2]
                complex_field = amplitude * torch.exp(1j * phase)
            elif self.output_type == 'real_imag':
                real = output[:, 0:1]
                imag = output[:, 1:2]
                complex_field = torch.complex(real, imag)
            else:
                raise ValueError(f"未知输出类型: {self.output_type}")
            
            return complex_field
        
        def reconstruct(self, measured_intensities,
                        num_iterations=1000,
                        lr=0.01,
                        input_noise_std=0.1,
                        verbose=True,
                        print_interval=100):
            """
            执行多距离深度先验相位恢复
            """
            if not TORCH_AVAILABLE:
                raise RuntimeError("PyTorch is not available")
            
            measured_tensors = []
            for intensity in measured_intensities:
                t = torch.tensor(intensity, dtype=self.dtype, device=self.device)
                measured_tensors.append(t.unsqueeze(0).unsqueeze(0))
            
            ny, nx = measured_intensities[0].shape
            self._init_propagation((ny, nx))
            
            input_noise = torch.randn(1, 1, ny, nx, device=self.device, dtype=self.dtype)
            input_noise = input_noise * input_noise_std
            input_noise.requires_grad = False
            
            optimizer = torch.optim.Adam(self.network.parameters(), lr=lr)
            
            self.errors = []
            
            for iteration in range(num_iterations):
                optimizer.zero_grad()
                
                network_output = self.network(input_noise)
                object_field = self._decode_output(network_output)
                
                total_loss = 0
                for idx, (measured, weight) in enumerate(zip(measured_tensors, self.weights)):
                    diffraction_field = self._propagate(object_field, idx)
                    predicted_intensity = torch.abs(diffraction_field) ** 2
                    loss = torch.mean((predicted_intensity - measured) ** 2)
                    total_loss += weight * loss
                
                total_loss.backward()
                optimizer.step()
                
                loss_value = total_loss.item()
                self.errors.append(loss_value)
                
                if verbose and (iteration + 1) % print_interval == 0:
                    print(f"Iteration {iteration + 1}/{num_iterations}, Loss: {loss_value:.6e}")
            
            with torch.no_grad():
                final_output = self.network(input_noise)
                final_field = self._decode_output(final_output)
                final_field_np = final_field[0, 0].cpu().numpy()
            
            return final_field_np
        
        def get_errors(self):
            return np.array(self.errors)
    
    
    class DeepPriorHybrid:
        """
        混合深度先验相位恢复
        
        结合传统迭代算法和深度先验
        使用传统算法提供初始估计，深度先验进行正则化
        """
        
        def __init__(self, wavelength, pixel_size, distance,
                     method='angular_spectrum',
                     network_type='small_unet',
                     device='cpu',
                     dtype=torch.float32):
            self.wavelength = wavelength
            self.pixel_size = pixel_size
            self.distance = distance
            self.method = method
            self.device = device
            self.dtype = dtype
            
            self.k = 2 * np.pi / wavelength
            
            self.network = SmallUNet(n_channels=2, n_output=2)
            self.network = self.network.to(device=device, dtype=dtype)
            
            self.prop_kernel = None
            self.errors = []
        
        def _init_propagation(self, shape):
            ny, nx = shape
            fx = np.fft.fftfreq(nx, self.pixel_size)
            fy = np.fft.fftfreq(ny, self.pixel_size)
            FX, FY = np.meshgrid(fx, fy)
            
            k_squared = self.k ** 2
            kx = 2 * np.pi * FX
            ky = 2 * np.pi * FY
            
            sqrt_term = np.sqrt(np.maximum(k_squared - kx**2 - ky**2, 0))
            H = np.exp(1j * self.distance * sqrt_term)
            
            self.prop_kernel = torch.tensor(H, dtype=torch.complex64, device=self.device)
        
        def _propagate(self, field):
            field_ft = torch.fft.fftn(field, dim=(-2, -1))
            field_prop_ft = field_ft * self.prop_kernel
            field_prop = torch.fft.ifftn(field_prop_ft, dim=(-2, -1))
            return field_prop
        
        def reconstruct(self, measured_intensity,
                        initial_guess=None,
                        num_iterations=500,
                        lr=0.001,
                        alpha=0.5,
                        verbose=True,
                        print_interval=50):
            """
            执行混合深度先验相位恢复
            
            参数:
                alpha: 物理损失权重 (0-1)，剩余权重分配给先验损失
            """
            if not TORCH_AVAILABLE:
                raise RuntimeError("PyTorch is not available")
            
            measured_intensity_torch = torch.tensor(
                measured_intensity, dtype=self.dtype, device=self.device
            )
            measured_intensity_torch = measured_intensity_torch.unsqueeze(0).unsqueeze(0)
            
            ny, nx = measured_intensity.shape
            self._init_propagation((ny, nx))
            
            if initial_guess is None:
                initial_guess_np = np.random.rand(ny, nx) + 1j * np.random.rand(ny, nx)
            else:
                initial_guess_np = initial_guess
            
            object_field = torch.tensor(
                np.stack([initial_guess_np.real, initial_guess_np.imag]),
                dtype=self.dtype, device=self.device
            ).unsqueeze(0)
            object_field.requires_grad = True
            
            optimizer = torch.optim.Adam([
                {'params': self.network.parameters()},
                {'params': object_field, 'lr': lr * 10}
            ], lr=lr)
            
            self.errors = []
            
            for iteration in range(num_iterations):
                optimizer.zero_grad()
                
                network_input = object_field
                network_output = self.network(network_input)
                
                refined_real = object_field[:, 0:1] + network_output[:, 0:1]
                refined_imag = object_field[:, 1:2] + network_output[:, 1:2]
                refined_field = torch.complex(refined_real, refined_imag)
                
                diffraction_field = self._propagate(refined_field)
                predicted_intensity = torch.abs(diffraction_field) ** 2
                
                physics_loss = torch.mean((predicted_intensity - measured_intensity_torch) ** 2)
                
                prior_loss = torch.mean(network_output ** 2)
                
                total_loss = alpha * physics_loss + (1 - alpha) * prior_loss
                
                total_loss.backward()
                optimizer.step()
                
                loss_value = total_loss.item()
                self.errors.append(loss_value)
                
                if verbose and (iteration + 1) % print_interval == 0:
                    print(f"Iteration {iteration + 1}/{num_iterations}, "
                          f"Total Loss: {loss_value:.6e}, "
                          f"Physics Loss: {physics_loss.item():.6e}")
            
            with torch.no_grad():
                network_input = object_field
                network_output = self.network(network_input)
                refined_real = object_field[:, 0:1] + network_output[:, 0:1]
                refined_imag = object_field[:, 1:2] + network_output[:, 1:2]
                final_field = torch.complex(refined_real, refined_imag)
                final_field_np = final_field[0, 0].cpu().numpy()
            
            return final_field_np
        
        def get_errors(self):
            return np.array(self.errors)

else:
    
    class DeepPriorPhaseRetrieval:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("PyTorch is not available. Please install PyTorch to use deep prior methods.")
    
    class DeepPriorMultiDistance:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("PyTorch is not available. Please install PyTorch to use deep prior methods.")
    
    class DeepPriorHybrid:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("PyTorch is not available. Please install PyTorch to use deep prior methods.")
