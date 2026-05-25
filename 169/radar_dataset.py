import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Optional, Callable
from generate_sample_data import generate_radar_sequence, add_noise, generate_convective_line
import os


class SyntheticRadarDataset(Dataset):
    def __init__(self, 
                 num_samples: int = 1000,
                 num_history_frames: int = 6,
                 num_future_frames: int = 20,
                 image_size: Tuple[int, int] = (128, 128),
                 transform: Optional[Callable] = None,
                 add_noise_level: float = 0.5,
                 include_convective_lines: bool = True):
        self.num_samples = num_samples
        self.num_history_frames = num_history_frames
        self.num_future_frames = num_future_frames
        self.total_frames = num_history_frames + num_future_frames
        self.image_size = image_size
        self.transform = transform
        self.add_noise_level = add_noise_level
        self.include_convective_lines = include_convective_lines
        
    def __len__(self) -> int:
        return self.num_samples
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.include_convective_lines and np.random.random() < 0.3:
            sequence = generate_convective_line(
                num_frames=self.total_frames,
                shape=self.image_size,
                velocity=np.random.uniform(1.0, 4.0)
            )
        else:
            sequence = generate_radar_sequence(
                num_frames=self.total_frames,
                shape=self.image_size,
                velocity=(np.random.uniform(-3, 3), np.random.uniform(-3, 3)),
                num_storms=np.random.randint(2, 6)
            )
        
        if self.add_noise_level > 0:
            sequence = add_noise(sequence, noise_level=self.add_noise_level)
        
        sequence = np.array(sequence, dtype=np.float32)
        
        sequence = sequence / 70.0
        
        history = sequence[:self.num_history_frames]
        future = sequence[self.num_history_frames:]
        
        history = torch.tensor(history, dtype=torch.float32).unsqueeze(1)
        future = torch.tensor(future, dtype=torch.float32).unsqueeze(1)
        
        if self.transform:
            history = self.transform(history)
            future = self.transform(future)
        
        return history, future


class NumpyRadarDataset(Dataset):
    def __init__(self, 
                 data_path: str,
                 num_history_frames: int = 6,
                 num_future_frames: int = 20,
                 transform: Optional[Callable] = None,
                 normalize: bool = True):
        self.data_path = data_path
        self.num_history_frames = num_history_frames
        self.num_future_frames = num_future_frames
        self.transform = transform
        self.normalize = normalize
        self.total_frames = num_history_frames + num_future_frames
        
        self.data = np.load(data_path)
        self.num_sequences = len(self.data) - self.total_frames + 1
        
    def __len__(self) -> int:
        return self.num_sequences
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        sequence = self.data[idx:idx + self.total_frames]
        
        if self.normalize:
            sequence = sequence / 70.0
        
        sequence = np.array(sequence, dtype=np.float32)
        
        history = sequence[:self.num_history_frames]
        future = sequence[self.num_history_frames:]
        
        history = torch.tensor(history, dtype=torch.float32).unsqueeze(1)
        future = torch.tensor(future, dtype=torch.float32).unsqueeze(1)
        
        if self.transform:
            history = self.transform(history)
            future = self.transform(future)
        
        return history, future


def create_dataloaders(batch_size: int = 4,
                       num_train_samples: int = 800,
                       num_val_samples: int = 100,
                       num_history_frames: int = 6,
                       num_future_frames: int = 20,
                       image_size: Tuple[int, int] = (128, 128),
                       num_workers: int = 0) -> Tuple[DataLoader, DataLoader]:
    
    train_dataset = SyntheticRadarDataset(
        num_samples=num_train_samples,
        num_history_frames=num_history_frames,
        num_future_frames=num_future_frames,
        image_size=image_size
    )
    
    val_dataset = SyntheticRadarDataset(
        num_samples=num_val_samples,
        num_history_frames=num_history_frames,
        num_future_frames=num_future_frames,
        image_size=image_size
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader


def generate_large_synthetic_dataset(output_path: str,
                                     num_sequences: int = 100,
                                     frames_per_sequence: int = 100,
                                     image_size: Tuple[int, int] = (256, 256)):
    all_data = []
    
    for i in range(num_sequences):
        if i % 10 == 0:
            print(f"Generating sequence {i}/{num_sequences}")
        
        if np.random.random() < 0.3:
            sequence = generate_convective_line(
                num_frames=frames_per_sequence,
                shape=image_size,
                velocity=np.random.uniform(0.5, 3.0)
            )
        else:
            sequence = generate_radar_sequence(
                num_frames=frames_per_sequence,
                shape=image_size,
                velocity=(np.random.uniform(-2.5, 2.5), np.random.uniform(-2.5, 2.5)),
                num_storms=np.random.randint(1, 5)
            )
        
        sequence = add_noise(sequence, noise_level=np.random.uniform(0.3, 0.8))
        all_data.append(sequence)
    
    all_data = np.array(all_data, dtype=np.float32)
    np.save(output_path, all_data)
    print(f"Dataset saved to {output_path}, shape: {all_data.shape}")


if __name__ == '__main__':
    print("Testing synthetic dataset...")
    
    dataset = SyntheticRadarDataset(
        num_samples=10,
        num_history_frames=6,
        num_future_frames=20,
        image_size=(128, 128)
    )
    
    history, future = dataset[0]
    print(f"History shape: {history.shape}")
    print(f"Future shape: {future.shape}")
    print(f"History value range: [{history.min():.3f}, {history.max():.3f}]")
    print(f"Future value range: [{future.min():.3f}, {future.max():.3f}]")
    
    train_loader, val_loader = create_dataloaders(batch_size=2)
    for batch_history, batch_future in train_loader:
        print(f"Batch history shape: {batch_history.shape}")
        print(f"Batch future shape: {batch_future.shape}")
        break
    
    print("Dataset test completed!")
