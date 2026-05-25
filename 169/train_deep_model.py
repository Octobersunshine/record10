import os
import sys
import time
import argparse
import numpy as np
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from deep_extrapolation_model import RadarGAN, train_step
from radar_dataset import create_dataloaders
from radar_extrapolation_enhanced import calculate_csi, calculate_pod, calculate_far


def parse_args():
    parser = argparse.ArgumentParser(description='Train RadarGAN for precipitation nowcasting')
    
    parser.add_argument('--num_epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--lr_g', type=float, default=1e-4, help='Generator learning rate')
    parser.add_argument('--lr_d', type=float, default=1e-4, help='Discriminator learning rate')
    parser.add_argument('--num_history_frames', type=int, default=6, help='Number of history frames')
    parser.add_argument('--num_future_frames', type=int, default=20, help='Number of future frames to predict')
    parser.add_argument('--image_size', type=int, nargs=2, default=(128, 128), help='Image size (H W)')
    parser.add_argument('--hidden_dims', type=int, nargs='+', default=[64, 128, 256], help='Hidden dimensions')
    parser.add_argument('--lambda_adv', type=float, default=0.01, help='Adversarial loss weight')
    parser.add_argument('--lambda_perceptual', type=float, default=0.1, help='Perceptual loss weight')
    parser.add_argument('--num_train_samples', type=int, default=500, help='Number of training samples')
    parser.add_argument('--num_val_samples', type=int, default=50, help='Number of validation samples')
    parser.add_argument('--save_interval', type=int, default=5, help='Save model every N epochs')
    parser.add_argument('--output_dir', type=str, default='./checkpoints', help='Output directory')
    parser.add_argument('--use_cuda', action='store_true', help='Use CUDA if available')
    parser.add_argument('--resume', type=str, default=None, help='Path to resume checkpoint')
    
    return parser.parse_args()


def validate(model, val_loader, device, epoch, output_dir):
    model.eval()
    
    all_losses = []
    all_csi = []
    all_pod = []
    all_far = []
    
    radar_cmap = ListedColormap([
        '#FFFFFF', '#80FF80', '#00FF00', '#00C000',
        '#008000', '#FFFF00', '#FFC000', '#FF8000',
        '#FF0000', '#C00000', '#800000', '#FF00FF'
    ])
    
    with torch.no_grad():
        for batch_idx, (history, future) in enumerate(val_loader):
            history = history.to(device)
            future = future.to(device)
            
            g_losses = model.generator_loss(history, future)
            all_losses.append(g_losses['mse'].item())
            
            pred = g_losses['predictions']
            
            history_np = history.cpu().numpy() * 70
            future_np = future.cpu().numpy() * 70
            pred_np = pred.cpu().numpy() * 70
            
            for i in range(history_np.shape[0]):
                for t in range(0, min(10, future_np.shape[1]), 3):
                    true_frame = future_np[i, t, 0]
                    pred_frame = pred_np[i, t, 0]
                    
                    csi = calculate_csi(true_frame, pred_frame, threshold=20)
                    pod = calculate_pod(true_frame, pred_frame, threshold=20)
                    far = calculate_far(true_frame, pred_frame, threshold=20)
                    
                    all_csi.append(csi)
                    all_pod.append(pod)
                    all_far.append(far)
            
            if batch_idx == 0:
                fig, axes = plt.subplots(3, 6, figsize=(18, 9))
                
                for i in range(6):
                    if i < history_np.shape[1]:
                        axes[0, i].imshow(history_np[0, i, 0], cmap=radar_cmap, vmin=0, vmax=60)
                        axes[0, i].set_title(f'History t-{6-i}')
                    else:
                        axes[0, i].imshow(np.zeros_like(history_np[0, 0, 0]), cmap=radar_cmap, vmin=0, vmax=60)
                        axes[0, i].set_title('')
                    axes[0, i].axis('off')
                
                for i in range(6):
                    if i < future_np.shape[1]:
                        axes[1, i].imshow(future_np[0, i, 0], cmap=radar_cmap, vmin=0, vmax=60)
                        axes[1, i].set_title(f'True +{(i+1)*6}min')
                    else:
                        axes[1, i].imshow(np.zeros_like(future_np[0, 0, 0]), cmap=radar_cmap, vmin=0, vmax=60)
                        axes[1, i].set_title('')
                    axes[1, i].axis('off')
                
                for i in range(6):
                    if i < pred_np.shape[1]:
                        axes[2, i].imshow(pred_np[0, i, 0], cmap=radar_cmap, vmin=0, vmax=60)
                        axes[2, i].set_title(f'Pred +{(i+1)*6}min')
                    else:
                        axes[2, i].imshow(np.zeros_like(pred_np[0, 0, 0]), cmap=radar_cmap, vmin=0, vmax=60)
                        axes[2, i].set_title('')
                    axes[2, i].axis('off')
                
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, f'val_epoch_{epoch+1}.png'), dpi=100)
                plt.close()
    
    avg_loss = np.mean(all_losses)
    avg_csi = np.mean(all_csi)
    avg_pod = np.mean(all_pod)
    avg_far = np.mean(all_far)
    
    return {
        'mse': avg_loss,
        'csi': avg_csi,
        'pod': avg_pod,
        'far': avg_far
    }


def plot_training_curves(train_losses, val_metrics, output_dir):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    epochs = range(1, len(train_losses['g_mse']) + 1)
    
    axes[0, 0].plot(epochs, train_losses['g_mse'], label='Train MSE', marker='o')
    axes[0, 0].plot(epochs, val_metrics['mse'], label='Val MSE', marker='s')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('MSE Loss')
    axes[0, 0].set_title('MSE Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].plot(epochs, train_losses['g_total'], label='Generator Total', marker='o')
    axes[0, 1].plot(epochs, train_losses['d_total'], label='Discriminator Total', marker='s')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].set_title('GAN Losses')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[1, 0].plot(epochs, val_metrics['csi'], label='CSI', marker='o')
    axes[1, 0].plot(epochs, val_metrics['pod'], label='POD', marker='s')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Score')
    axes[1, 0].set_title('Validation Metrics (20dBZ)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_ylim(0, 1)
    
    axes[1, 1].plot(epochs, val_metrics['far'], label='FAR', marker='o', color='red')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('FAR')
    axes[1, 1].set_title('False Alarm Rate (20dBZ)')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_curves.png'), dpi=100)
    plt.close()


def main():
    args = parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    device = torch.device('cuda' if args.use_cuda and torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    print("Creating dataloaders...")
    train_loader, val_loader = create_dataloaders(
        batch_size=args.batch_size,
        num_train_samples=args.num_train_samples,
        num_val_samples=args.num_val_samples,
        num_history_frames=args.num_history_frames,
        num_future_frames=args.num_future_frames,
        image_size=tuple(args.image_size)
    )
    
    print("Creating model...")
    model = RadarGAN(
        input_dim=1,
        output_dim=1,
        hidden_dims=args.hidden_dims,
        num_history_frames=args.num_history_frames,
        num_future_frames=args.num_future_frames,
        lambda_adv=args.lambda_adv,
        lambda_perceptual=args.lambda_perceptual
    ).to(device)
    
    g_optimizer = optim.Adam(model.generator.parameters(), lr=args.lr_g, betas=(0.5, 0.999))
    d_optimizer = optim.Adam(model.discriminator.parameters(), lr=args.lr_d, betas=(0.5, 0.999))
    
    start_epoch = 0
    if args.resume and os.path.exists(args.resume):
        print(f"Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        g_optimizer.load_state_dict(checkpoint['g_optimizer_state_dict'])
        d_optimizer.load_state_dict(checkpoint['d_optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        print(f"Resumed from epoch {start_epoch}")
    
    writer = SummaryWriter(os.path.join(args.output_dir, 'logs'))
    
    train_losses = {
        'g_total': [],
        'g_mse': [],
        'g_adv': [],
        'g_perceptual': [],
        'd_total': []
    }
    
    val_metrics = {
        'mse': [],
        'csi': [],
        'pod': [],
        'far': []
    }
    
    print("Starting training...")
    print(f"Total epochs: {args.num_epochs}")
    print(f"Training batches per epoch: {len(train_loader)}")
    print(f"Validation batches per epoch: {len(val_loader)}")
    
    for epoch in range(start_epoch, args.num_epochs):
        epoch_start_time = time.time()
        
        model.train()
        epoch_losses = {k: [] for k in train_losses.keys()}
        
        for batch_idx, (history, future) in enumerate(train_loader):
            history = history.to(device)
            future = future.to(device)
            
            losses = train_step(model, history, future, g_optimizer, d_optimizer, device)
            
            for k in epoch_losses.keys():
                epoch_losses[k].append(losses[k])
            
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch+1}/{args.num_epochs}, Batch {batch_idx}/{len(train_loader)}: "
                      f"G_MSE={losses['g_mse']:.4f}, G_Total={losses['g_total']:.4f}, "
                      f"D_Total={losses['d_total']:.4f}")
        
        for k in train_losses.keys():
            train_losses[k].append(np.mean(epoch_losses[k]))
        
        val_results = validate(model, val_loader, device, epoch, args.output_dir)
        
        for k in val_metrics.keys():
            val_metrics[k].append(val_results[k])
        
        epoch_time = time.time() - epoch_start_time
        
        print(f"\nEpoch {epoch+1}/{args.num_epochs} completed in {epoch_time:.1f}s")
        print(f"  Train - G_MSE: {train_losses['g_mse'][-1]:.4f}, G_Total: {train_losses['g_total'][-1]:.4f}")
        print(f"  Val   - MSE: {val_results['mse']:.4f}, CSI: {val_results['csi']:.4f}, "
              f"POD: {val_results['pod']:.4f}, FAR: {val_results['far']:.4f}")
        print()
        
        for k, v in train_losses.items():
            writer.add_scalar(f'Train/{k}', v[-1], epoch)
        for k, v in val_metrics.items():
            writer.add_scalar(f'Val/{k}', v[-1], epoch)
        
        plot_training_curves(train_losses, val_metrics, args.output_dir)
        
        if (epoch + 1) % args.save_interval == 0 or epoch == args.num_epochs - 1:
            checkpoint_path = os.path.join(args.output_dir, f'checkpoint_epoch_{epoch+1}.pth')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'g_optimizer_state_dict': g_optimizer.state_dict(),
                'd_optimizer_state_dict': d_optimizer.state_dict(),
                'train_losses': train_losses,
                'val_metrics': val_metrics,
                'args': args
            }, checkpoint_path)
            print(f"Checkpoint saved: {checkpoint_path}")
    
    writer.close()
    print("Training completed!")
    
    final_model_path = os.path.join(args.output_dir, 'final_model.pth')
    torch.save(model.state_dict(), final_model_path)
    print(f"Final model saved: {final_model_path}")


if __name__ == '__main__':
    main()
