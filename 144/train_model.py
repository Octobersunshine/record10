import os
import sys
import argparse
import numpy as np
import cv2
import matplotlib.pyplot as plt
from datetime import datetime

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader
    from torch.optim.lr_scheduler import ReduceLROnPlateau
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("PyTorch not available. Please install with: pip install torch")
    sys.exit(1)

from deep_learning import (
    SingularityNet,
    SingularityHeatmapNet,
    keypoint_loss,
    heatmap_loss,
    SingularityDetector
)
from data_generator import (
    FingerprintDataset,
    generate_synthetic_fingerprint_with_keypoints,
    apply_wet_effect,
    apply_dry_effect,
    apply_blur_effect,
    apply_noise_effect
)


def train(args):
    print("=" * 60)
    print("Fingerprint Singularity Detection - Training")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if args.use_heatmap:
        print("Training Heatmap-based model...")
        model = SingularityHeatmapNet(
            num_cores=args.num_cores,
            num_deltas=args.num_deltas,
            img_size=args.img_size,
            heatmap_size=args.heatmap_size
        )
        criterion = heatmap_loss
    else:
        print("Training Regression-based model...")
        model = SingularityNet(
            num_cores=args.num_cores,
            num_deltas=args.num_deltas,
            img_size=args.img_size
        )
        criterion = keypoint_loss
    
    model.to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5, verbose=True)
    
    print(f"\nCreating training dataset with {args.num_train} samples...")
    train_dataset = FingerprintDataset(
        num_samples=args.num_train,
        img_size=args.img_size,
        num_cores=args.num_cores,
        num_deltas=args.num_deltas,
        augment=True,
        use_heatmap=args.use_heatmap,
        heatmap_size=args.heatmap_size
    )
    
    print(f"Creating validation dataset with {args.num_val} samples...")
    val_dataset = FingerprintDataset(
        num_samples=args.num_val,
        img_size=args.img_size,
        num_cores=args.num_cores,
        num_deltas=args.num_deltas,
        augment=False,
        use_heatmap=args.use_heatmap,
        heatmap_size=args.heatmap_size
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'checkpoints'), exist_ok=True)
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    print(f"\nStarting training for {args.epochs} epochs...")
    print("-" * 60)
    
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        
        for batch_idx, (images, targets) in enumerate(train_loader):
            images = images.to(device)
            targets = targets.to(device)
            
            optimizer.zero_grad()
            
            outputs = model(images)
            loss = criterion(outputs, targets)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item() * images.size(0)
            
            if (batch_idx + 1) % args.log_interval == 0:
                print(f"Epoch {epoch+1}/{args.epochs} | Batch {batch_idx+1}/{len(train_loader)} | Loss: {loss.item():.6f}")
        
        train_loss /= len(train_dataset)
        train_losses.append(train_loss)
        
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for images, targets in val_loader:
                images = images.to(device)
                targets = targets.to(device)
                
                outputs = model(images)
                loss = criterion(outputs, targets)
                
                val_loss += loss.item() * images.size(0)
        
        val_loss /= len(val_dataset)
        val_losses.append(val_loss)
        
        scheduler.step(val_loss)
        
        print(f"\nEpoch {epoch+1}/{args.epochs} Summary:")
        print(f"  Train Loss: {train_loss:.6f}")
        print(f"  Val Loss:   {val_loss:.6f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_path = os.path.join(args.output_dir, 'checkpoints', 'best_model.pth')
            torch.save(model.state_dict(), checkpoint_path)
            print(f"  Best model saved! (Val Loss: {best_val_loss:.6f})")
        
        if (epoch + 1) % args.save_interval == 0:
            checkpoint_path = os.path.join(args.output_dir, 'checkpoints', f'epoch_{epoch+1}.pth')
            torch.save(model.state_dict(), checkpoint_path)
        
        print("-" * 60)
    
    plot_training_curves(train_losses, val_losses, args.output_dir)
    
    final_model_path = os.path.join(args.output_dir, 'final_model.pth')
    torch.save(model.state_dict(), final_model_path)
    print(f"\nFinal model saved to: {final_model_path}")
    
    print("\nEvaluating on test conditions...")
    evaluate_robustness(model, device, args)
    
    print("\n" + "=" * 60)
    print("Training completed!")
    print("=" * 60)
    
    return model


def plot_training_curves(train_losses, val_losses, output_dir):
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss', linewidth=2)
    plt.plot(val_losses, label='Val Loss', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Curves')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    curve_path = os.path.join(output_dir, 'training_curves.png')
    plt.savefig(curve_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Training curves saved to: {curve_path}")


def evaluate_robustness(model, device, args):
    model.eval()
    
    detector = SingularityDetector(
        num_cores=args.num_cores,
        num_deltas=args.num_deltas,
        img_size=args.img_size,
        use_heatmap=args.use_heatmap
    )
    detector.model = model
    detector.device = device
    
    fp_types = ['loop_right', 'loop_left', 'whorl', 'arch']
    conditions = ['original', 'wet', 'dry', 'blur', 'noisy']
    
    results = {}
    
    for fp_type in fp_types:
        results[fp_type] = {}
        
        for condition in conditions:
            image, true_cores, true_deltas = generate_synthetic_fingerprint_with_keypoints(
                size=(args.img_size, args.img_size),
                fingerprint_type=fp_type
            )
            
            if condition == 'wet':
                image = apply_wet_effect(image, severity=0.7)
            elif condition == 'dry':
                image = apply_dry_effect(image, severity=0.7)
            elif condition == 'blur':
                image = apply_blur_effect(image, severity=0.6)
            elif condition == 'noisy':
                image = apply_noise_effect(image, severity=0.6)
            
            pred_cores, pred_deltas = detector.detect(image, confidence_threshold=0.3)
            
            core_dist_error = 0.0
            delta_dist_error = 0.0
            
            if len(pred_cores) > 0 and len(true_cores) > 0:
                for (px, py, _, _) in pred_cores:
                    min_dist = min([np.sqrt((px - tx)**2 + (py - ty)**2) for (tx, ty) in true_cores])
                    core_dist_error += min_dist
                core_dist_error /= len(pred_cores)
            
            if len(pred_deltas) > 0 and len(true_deltas) > 0:
                for (px, py, _, _) in pred_deltas:
                    min_dist = min([np.sqrt((px - tx)**2 + (py - ty)**2) for (tx, ty) in true_deltas])
                    delta_dist_error += min_dist
                delta_dist_error /= len(pred_deltas)
            
            results[fp_type][condition] = {
                'core_detection_rate': len(pred_cores) / max(1, len(true_cores)),
                'delta_detection_rate': len(pred_deltas) / max(1, len(true_deltas)),
                'core_dist_error': core_dist_error,
                'delta_dist_error': delta_dist_error
            }
    
    print("\nRobustness Evaluation Results:")
    print("-" * 80)
    print(f"{'Fingerprint':<15} {'Condition':<10} {'Core Rate':<12} {'Delta Rate':<12} {'Core Err':<12} {'Delta Err':<12}")
    print("-" * 80)
    
    for fp_type in fp_types:
        for condition in conditions:
            r = results[fp_type][condition]
            print(f"{fp_type:<15} {condition:<10} {r['core_detection_rate']:<12.2f} {r['delta_detection_rate']:<12.2f} {r['core_dist_error']:<12.1f} {r['delta_dist_error']:<12.1f}")
    
    return results


def quick_test(args):
    print("Quick test of data loading and model...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    dataset = FingerprintDataset(
        num_samples=10,
        img_size=args.img_size,
        num_cores=args.num_cores,
        num_deltas=args.num_deltas,
        augment=True,
        use_heatmap=args.use_heatmap
    )
    
    loader = DataLoader(dataset, batch_size=2)
    images, targets = next(iter(loader))
    
    print(f"Input shape: {images.shape}")
    print(f"Target shape: {targets.shape}")
    
    if args.use_heatmap:
        model = SingularityHeatmapNet(args.num_cores, args.num_deltas, args.img_size)
    else:
        model = SingularityNet(args.num_cores, args.num_deltas, args.img_size)
    
    model.to(device)
    images = images.to(device)
    
    output = model(images)
    print(f"Output shape: {output.shape}")
    
    print("Quick test passed!")


if __name__ == '__main__':
    if not TORCH_AVAILABLE:
        print("PyTorch is required for training.")
        print("Install with: pip install torch torchvision")
        sys.exit(1)
    
    parser = argparse.ArgumentParser(description='Train Singularity Detection Model')
    
    parser.add_argument('--num_train', type=int, default=5000, help='Number of training samples')
    parser.add_argument('--num_val', type=int, default=500, help='Number of validation samples')
    parser.add_argument('--img_size', type=int, default=256, help='Input image size')
    parser.add_argument('--num_cores', type=int, default=2, help='Maximum number of core points')
    parser.add_argument('--num_deltas', type=int, default=2, help='Maximum number of delta points')
    parser.add_argument('--use_heatmap', action='store_true', help='Use heatmap-based model')
    parser.add_argument('--heatmap_size', type=int, default=64, help='Heatmap size')
    
    parser.add_argument('--epochs', type=int, default=30, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--num_workers', type=int, default=0, help='Number of data loader workers')
    
    parser.add_argument('--output_dir', type=str, default='training_output', help='Output directory')
    parser.add_argument('--log_interval', type=int, default=10, help='Log interval')
    parser.add_argument('--save_interval', type=int, default=5, help='Model save interval')
    
    parser.add_argument('--quick_test', action='store_true', help='Run quick test')
    
    args = parser.parse_args()
    
    if args.quick_test:
        quick_test(args)
    else:
        train(args)
