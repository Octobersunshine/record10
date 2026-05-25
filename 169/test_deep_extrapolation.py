import os
import sys
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap

from generate_sample_data import generate_radar_sequence, add_noise, generate_convective_line
from inference_pipeline import DeepNowcastingPipeline, evaluate_all_methods, visualize_comparison, plot_metrics_comparison


def quick_demo():
    print("=" * 70)
    print("Deep Learning Radar Extrapolation - Quick Demo")
    print("=" * 70)
    
    np.random.seed(42)
    
    print("\n1. Generating test data with growth/decay storms...")
    sequence = generate_radar_sequence(
        num_frames=36,
        shape=(128, 128),
        velocity=(1.5, 1.0),
        num_storms=4
    )
    sequence = add_noise(sequence, noise_level=0.5)
    
    history_frames = sequence[:6]
    true_future = sequence[6:36]
    
    print(f"   History frames: {len(history_frames)}")
    print(f"   Future frames: {len(true_future)} (up to 3 hours)")
    print(f"   Image size: {sequence[0].shape}")
    
    print("\n2. Creating inference pipeline...")
    pipeline = DeepNowcastingPipeline(
        gan_model_path=None,
        use_optical_flow=True,
        use_deep_model=False,
        switch_lead_time=60,
        flow_weight=0.7,
        gan_weight=0.3
    )
    
    print("\n3. Running predictions for short (30min) and long (180min) lead times...")
    lead_times = [6, 12, 18, 24, 30, 60, 90, 120, 150, 180]
    
    print("\n4. Evaluating all methods...")
    results = evaluate_all_methods(
        history_frames,
        true_future[:len(lead_times)],
        pipeline,
        lead_times,
        time_interval=6
    )
    
    print("\n5. Generating visualizations...")
    visualize_comparison(history_frames, results, 'deep_method_comparison.png')
    plot_metrics_comparison(results, 'deep_metrics_comparison.png')
    
    print("\n" + "=" * 70)
    print("Performance Summary (Optical Flow / Hybrid)")
    print("=" * 70)
    
    thresholds = [10, 20, 30]
    for threshold in thresholds:
        flow_csi = np.mean(results['optical_flow'][threshold]['csi'][:5])
        hybrid_csi = np.mean(results['hybrid'][threshold]['csi'][:5])
        flow_csi_long = np.mean(results['optical_flow'][threshold]['csi'][5:])
        hybrid_csi_long = np.mean(results['hybrid'][threshold]['csi'][5:])
        
        print(f"\nThreshold {threshold} dBZ:")
        print(f"  Short term (<30min):  Optical Flow CSI = {flow_csi:.3f}")
        print(f"  Long term (>30min):   Optical Flow CSI = {flow_csi_long:.3f}")
    
    print("\n" + "=" * 70)
    print("Note: Deep GAN model requires pre-training.")
    print("Run 'python train_deep_model.py' to train the model first.")
    print("=" * 70)
    
    return results


def training_guide():
    print("\n" + "=" * 70)
    print("Training Guide for Deep GAN Model")
    print("=" * 70)
    
    print("""
Quick Training Command:
----------------------
python train_deep_model.py --num_epochs 10 --batch_size 2 --num_train_samples 200 --num_val_samples 20

Full Training Options:
---------------------
--num_epochs              Number of training epochs (default: 50)
--batch_size              Batch size (default: 4)
--lr_g                    Generator learning rate (default: 1e-4)
--lr_d                    Discriminator learning rate (default: 1e-4)
--num_history_frames      Number of input history frames (default: 6)
--num_future_frames       Number of future frames to predict (default: 20)
--image_size              Input image size H W (default: 128 128)
--hidden_dims             Hidden layer dimensions (default: 64 128 256)
--lambda_adv              Adversarial loss weight (default: 0.01)
--lambda_perceptual       Perceptual loss weight (default: 0.1)
--num_train_samples       Number of training samples (default: 500)
--num_val_samples         Number of validation samples (default: 50)
--save_interval           Save checkpoint every N epochs (default: 5)
--output_dir              Output directory (default: ./checkpoints)
--use_cuda                Use GPU acceleration if available
--resume                  Path to resume checkpoint

Expected Training Time:
----------------------
- CPU: ~5-10 minutes for 10 epochs (small dataset)
- GPU: ~1-2 minutes for 10 epochs (small dataset)

After Training:
--------------
1. Locate the trained model: ./checkpoints/final_model.pth
2. Load in inference pipeline:

   pipeline = DeepNowcastingPipeline(
       gan_model_path='./checkpoints/final_model.pth',
       use_optical_flow=True,
       use_deep_model=True
   )
""")
    
    print("=" * 70)


def architecture_explanation():
    print("\n" + "=" * 70)
    print("Deep GAN Architecture Explanation")
    print("=" * 70)
    
    print("""
Generator (Encoder-Decoder ConvLSTM):
-------------------------------------
Input: 6 history frames (6min interval = 36min observation)
Output: 20 future frames (120min prediction, extendable to 30+ frames)

Encoder:
  Layer 1: ConvLSTM 64 channels
  Layer 2: ConvLSTM 128 channels
  Layer 3: ConvLSTM 256 channels (bottleneck)

Decoder:
  Layer 1: ConvLSTM 256 channels
  Layer 2: ConvLSTM 128 channels
  Layer 3: ConvLSTM 64 channels
  Output: Conv2D -> 1 channel radar reflectivity

Discriminator (3D CNN):
-----------------------
Input: Full sequence (history + future = 26 frames)
Output: Real/Fake classification

Architecture:
  Conv3D 32 -> LeakyReLU
  Conv3D 64 -> BatchNorm -> LeakyReLU
  Conv3D 128 -> BatchNorm -> LeakyReLU
  Conv3D 256 -> BatchNorm -> LeakyReLU
  AdaptiveAvgPool3D -> Linear -> Sigmoid

Loss Functions:
---------------
1. MSE Loss: Pixel-wise reconstruction loss
2. Adversarial Loss: GAN training for realistic outputs
3. Perceptual Loss: Feature-level similarity

Total Loss = MSE + λ_adv * Adversarial + λ_perceptual * Perceptual

Hybrid Prediction Strategy:
---------------------------
- < 60 minutes: Blend optical flow (70%) + deep model (30%)
- > 60 minutes: Use deep model primarily (learns nonlinear evolution)

Key Advantages for Long Lead Times:
-----------------------------------
1. Learns storm lifecycle patterns (initiation, maturation, dissipation)
2. Models nonlinear interactions between storms
3. Handles convective initiation better than optical flow
4. Maintains structural consistency in extrapolation
5. Reduces blurring effect common in pure optical flow methods
""")
    
    print("=" * 70)


def comparison_table():
    print("\n" + "=" * 70)
    print("Method Comparison Table")
    print("=" * 70)
    
    print("""
+---------------------+---------------+---------------+-------------------+
| Feature             | Optical Flow  | Deep GAN      | Hybrid (Combined) |
+---------------------+---------------+---------------+-------------------+
| Short-term (<30min) | Excellent     | Good          | Best              |
| Long-term (>1hr)    | Poor          | Good          | Good              |
| Position accuracy   | High          | Medium        | High              |
| Intensity accuracy  | Medium        | High          | High              |
| Storm growth        | Poor          | Excellent     | Excellent         |
| Storm decay         | Poor          | Excellent     | Excellent         |
| Convective init.    | Very Poor     | Good          | Good              |
| Computational cost  | Low           | High          | Medium-High       |
| Training required   | No            | Yes           | Yes               |
| Real-time inference | Yes           | Yes (GPU)     | Yes (GPU)         |
+---------------------+---------------+---------------+-------------------+

Recommended Usage:
------------------
1. Nowcasting (<1hr): Optical flow + light intensity correction
2. Short-range (1-2hr): Hybrid approach (flow + deep)
3. Long-range (2-3hr+): Deep GAN model primarily

Model Limitations:
------------------
1. Optical flow: Assumes brightness constancy, fails at growth/decay
2. Deep GAN: Needs large training data, can have mode collapse
3. Both: Struggle with extreme, unprecedented weather events

Improvements for Production:
----------------------------
1. Train on real radar data (e.g., NEXRAD, China Radar Network)
2. Add attention mechanisms for storm tracking
3. Incorporate numerical weather prediction (NWP) data
4. Use ensemble prediction for uncertainty estimation
5. Implement progressive growing for higher resolution
""")
    
    print("=" * 70)


if __name__ == '__main__':
    if len(sys.argv) > 1:
        if sys.argv[1] == '--guide':
            training_guide()
        elif sys.argv[1] == '--arch':
            architecture_explanation()
        elif sys.argv[1] == '--compare':
            comparison_table()
        else:
            quick_demo()
    else:
        quick_demo()
        training_guide()
        architecture_explanation()
        comparison_table()
