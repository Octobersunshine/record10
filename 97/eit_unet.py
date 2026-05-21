import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)

        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class EIT_UNet(nn.Module):
    def __init__(self, n_measurements=224, grid_size=64, bilinear=True):
        super().__init__()
        self.n_measurements = n_measurements
        self.grid_size = grid_size
        self.bilinear = bilinear

        self.fc_input = nn.Sequential(
            nn.Linear(n_measurements, 1024),
            nn.ReLU(),
            nn.Linear(1024, grid_size * grid_size),
            nn.ReLU()
        )

        self.inc = DoubleConv(1, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, 1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        batch_size = x.size(0)
        x = self.fc_input(x)
        x = x.view(batch_size, 1, self.grid_size, self.grid_size)

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
        output = torch.sigmoid(logits)
        return output


class EITDataset(Dataset):
    def __init__(self, measurements, conductivity_images, transform=None):
        self.measurements = measurements
        self.conductivity_images = conductivity_images
        self.transform = transform

    def __len__(self):
        return len(self.measurements)

    def __getitem__(self, idx):
        x = torch.FloatTensor(self.measurements[idx])
        y = torch.FloatTensor(self.conductivity_images[idx]).unsqueeze(0)

        if self.transform:
            y = self.transform(y)

        return x, y


class EITDataGenerator:
    def __init__(self, mesh, forward_solver, grid_size=64):
        self.mesh = mesh
        self.forward = forward_solver
        self.grid_size = grid_size
        self._create_grid_mask()

    def _create_grid_mask(self):
        xi = np.linspace(-1.1, 1.1, self.grid_size)
        yi = np.linspace(-1.1, 1.1, self.grid_size)
        XI, YI = np.meshgrid(xi, yi)
        self.mask = XI**2 + YI**2 <= 1.0
        self.XI, self.YI = XI, YI

    def sigma_to_image(self, sigma):
        from scipy.interpolate import griddata
        
        elem_centers = np.zeros((self.mesh.n_elements, 2))
        for e_idx in range(self.mesh.n_elements):
            elem_nodes = self.mesh.nodes[self.mesh.elements[e_idx]]
            elem_centers[e_idx] = np.mean(elem_nodes, axis=0)

        zi = griddata(elem_centers, sigma, (self.XI, self.YI), method='cubic', fill_value=1.0)
        zi[~self.mask] = 1.0
        return zi

    def generate_random_conductivity(self, n_anomalies_range=(1, 4)):
        sigma = np.ones(self.mesh.n_elements)
        
        elem_centers = np.zeros((self.mesh.n_elements, 2))
        for e_idx in range(self.mesh.n_elements):
            elem_nodes = self.mesh.nodes[self.mesh.elements[e_idx]]
            elem_centers[e_idx] = np.mean(elem_nodes, axis=0)

        n_anomalies = np.random.randint(n_anomalies_range[0], n_anomalies_range[1] + 1)
        
        for _ in range(n_anomalies):
            center_x = np.random.uniform(-0.7, 0.7)
            center_y = np.random.uniform(-0.7, 0.7)
            radius = np.random.uniform(0.1, 0.3)
            conductivity = np.random.uniform(0.2, 5.0)
            
            dist = np.sqrt((elem_centers[:, 0] - center_x)**2 + (elem_centers[:, 1] - center_y)**2)
            sigma[dist < radius] = conductivity

        return sigma

    def generate_dataset(self, n_samples=1000, use_cem=True, contact_impedance_std=0.0):
        measurements = []
        images = []

        for i in range(n_samples):
            sigma = self.generate_random_conductivity()
            
            if use_cem and contact_impedance_std > 0:
                z = np.ones(self.mesh.n_electrodes) * 0.1
                z *= (1 + contact_impedance_std * np.random.randn(self.mesh.n_electrodes))
                z = np.maximum(z, 0.01)
                v = self.forward.simulate_measurements(sigma, z)
            elif use_cem:
                z = np.ones(self.mesh.n_electrodes) * 0.1
                v = self.forward.simulate_measurements(sigma, z)
            else:
                v = self.forward.simulate_measurements_simple(sigma)

            noise = 0.005 * np.random.randn(len(v))
            v_noisy = v + noise

            v_normalized = (v_noisy - np.mean(v_noisy)) / (np.std(v_noisy) + 1e-8)
            
            img = self.sigma_to_image(sigma)
            img_normalized = (img - 1.0) / 4.0 + 0.5

            measurements.append(v_normalized)
            images.append(img_normalized)

            if (i + 1) % 100 == 0:
                print(f"  Generated {i + 1}/{n_samples} samples")

        return np.array(measurements), np.array(images)


class EITTrainer:
    def __init__(self, model, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model.to(device)
        self.device = device
        self.train_losses = []
        self.val_losses = []

    def train(self, train_loader, val_loader, num_epochs=50, lr=1e-4, save_path='eit_unet_model.pth'):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)

        best_val_loss = float('inf')

        for epoch in range(num_epochs):
            self.model.train()
            train_loss = 0.0

            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(self.device), target.to(self.device)
                
                optimizer.zero_grad()
                output = self.model(data)
                
                loss = F.mse_loss(output, target)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

            avg_train_loss = train_loss / len(train_loader)
            self.train_losses.append(avg_train_loss)

            val_loss = self.validate(val_loader)
            self.val_losses.append(val_loss)

            scheduler.step(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), save_path)

            print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.6f}, Val Loss: {val_loss:.6f}")

        self.model.load_state_dict(torch.load(save_path))
        return best_val_loss

    def validate(self, val_loader):
        self.model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = F.mse_loss(output, target)
                val_loss += loss.item()

        return val_loss / len(val_loader)

    def plot_training_history(self, save_path='training_history.png'):
        plt.figure(figsize=(10, 5))
        plt.plot(self.train_losses, label='Train Loss')
        plt.plot(self.val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('MSE Loss')
        plt.title('Training History')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()


class EITFastReconstructor:
    def __init__(self, model_path, mesh, grid_size=64, device='cpu'):
        self.mesh = mesh
        self.grid_size = grid_size
        self.device = device

        n_measurements = mesh.n_electrodes * (mesh.n_electrodes - 2)
        self.model = EIT_UNet(n_measurements=n_measurements, grid_size=grid_size)
        self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model.to(device)
        self.model.eval()

        self._create_grid()

    def _create_grid(self):
        xi = np.linspace(-1.1, 1.1, self.grid_size)
        yi = np.linspace(-1.1, 1.1, self.grid_size)
        self.XI, self.YI = np.meshgrid(xi, yi)
        self.mask = self.XI**2 + self.YI**2 <= 1.0

    def reconstruct(self, measurements):
        measurements_normalized = (measurements - np.mean(measurements)) / (np.std(measurements) + 1e-8)
        
        with torch.no_grad():
            x = torch.FloatTensor(measurements_normalized).unsqueeze(0).to(self.device)
            output = self.model(x)
            img_normalized = output.squeeze().cpu().numpy()

        img = (img_normalized - 0.5) * 4.0 + 1.0
        img[~self.mask] = 1.0

        return img

    def reconstruct_to_elements(self, measurements):
        from scipy.interpolate import griddata
        
        img = self.reconstruct(measurements)
        
        elem_centers = np.zeros((self.mesh.n_elements, 2))
        for e_idx in range(self.mesh.n_elements):
            elem_nodes = self.mesh.nodes[self.mesh.elements[e_idx]]
            elem_centers[e_idx] = np.mean(elem_nodes, axis=0)

        points = np.column_stack([self.XI[self.mask], self.YI[self.mask]])
        values = img[self.mask]
        
        sigma = griddata(points, values, elem_centers, method='linear', fill_value=1.0)
        return sigma


def plot_reconstruction_comparison(true_img, pred_img, save_path='comparison.png'):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    mask = np.ones_like(true_img)
    h, w = true_img.shape
    xi = np.linspace(-1.1, 1.1, w)
    yi = np.linspace(-1.1, 1.1, h)
    XI, YI = np.meshgrid(xi, yi)
    circle_mask = XI**2 + YI**2 <= 1.0

    true_display = np.where(circle_mask, true_img, np.nan)
    pred_display = np.where(circle_mask, pred_img, np.nan)
    error_display = np.where(circle_mask, np.abs(true_img - pred_img), np.nan)

    im1 = axes[0].pcolormesh(XI, YI, true_display, cmap='viridis', shading='auto')
    axes[0].set_title('真实分布', fontsize=14)
    axes[0].set_aspect('equal')
    plt.colorbar(im1, ax=axes[0])

    im2 = axes[1].pcolormesh(XI, YI, pred_display, cmap='viridis', shading='auto')
    axes[1].set_title('U-Net预测', fontsize=14)
    axes[1].set_aspect('equal')
    plt.colorbar(im2, ax=axes[1])

    im3 = axes[2].pcolormesh(XI, YI, error_display, cmap='hot', shading='auto')
    axes[2].set_title('绝对误差', fontsize=14)
    axes[2].set_aspect('equal')
    plt.colorbar(im3, ax=axes[2])

    from matplotlib.patches import Circle
    for ax in axes:
        circle = Circle((0, 0), 1.0, fill=False, color='black', linewidth=2)
        ax.add_patch(circle)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
