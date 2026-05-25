import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from radar_extrapolation_enhanced import EnhancedRadarExtrapolator, NowcastingPipeline
from radar_extrapolation_enhanced import calculate_csi, calculate_pod, calculate_far
from generate_sample_data import generate_radar_sequence, add_noise


def basic_usage_example():
    print("=" * 60)
    print("Basic Usage Example - Enhanced Radar Extrapolation")
    print("=" * 60)
    
    np.random.seed(42)
    
    print("\n1. Generating synthetic radar data...")
    sequence = generate_radar_sequence(num_frames=10, num_storms=4, velocity=(2.0, 1.0))
    sequence = add_noise(sequence, noise_level=0.5)
    
    history = sequence[:6]
    true_future = sequence[6:]
    
    print(f"   History frames: {len(history)}")
    print(f"   Future frames: {len(true_future)}")
    print(f"   Image shape: {sequence[0].shape}")
    
    print("\n2. Creating enhanced extrapolator with all improvements...")
    extrapolator = EnhancedRadarExtrapolator(
        flow_method='farneback',
        use_mass_conservation=True,
        use_intensity_correction=True,
        use_adaptive_weighting=True
    )
    
    print("\n3. Running extrapolation for 4 steps (24 minutes)...")
    predictions = extrapolator.extrapolate_enhanced(
        history, steps=4, time_interval=6
    )
    
    print("\n4. Evaluating predictions:")
    for i, (true, pred) in enumerate(zip(true_future[:4], predictions)):
        csi = calculate_csi(true, pred, threshold=20)
        pod = calculate_pod(true, pred, threshold=20)
        far = calculate_far(true, pred, threshold=20)
        print(f"   +{(i+1)*6}min: CSI={csi:.3f}, POD={pod:.3f}, FAR={far:.3f}")
    
    print("\n5. Accessing diagnostic information:")
    print(f"   Growth rate map shape: {extrapolator.growth_rate_map.shape}")
    print(f"   Divergence map shape: {extrapolator.divergence_map.shape}")
    
    print("\n6. Detecting growth/decay regions:")
    regions = extrapolator.detect_growth_decay_regions(history, threshold=0.3)
    print(f"   Growth pixels: {np.sum(regions['growth'])}")
    print(f"   Decay pixels: {np.sum(regions['decay'])}")
    print(f"   Stable pixels: {np.sum(regions['stable'])}")
    
    print("\n" + "=" * 60)
    return history, true_future, predictions, extrapolator


def pipeline_example():
    print("\n" + "=" * 60)
    print("Nowcasting Pipeline Example")
    print("=" * 60)
    
    np.random.seed(123)
    sequence = generate_radar_sequence(num_frames=12, num_storms=3, velocity=(1.5, 2.0))
    sequence = add_noise(sequence, noise_level=0.3)
    
    print("\nCreating nowcasting pipeline...")
    pipeline = NowcastingPipeline(
        num_history_frames=6,
        max_lead_time=60,
        time_interval=6,
        flow_method='farneback',
        use_mass_conservation=True,
        use_intensity_correction=True,
        use_adaptive_weighting=True
    )
    
    history = sequence[:6]
    lead_times = [6, 12, 18, 24, 30]
    
    print(f"Predicting for lead times: {lead_times} minutes")
    result = pipeline.predict(history, lead_times=lead_times)
    
    true_future = sequence[6:]
    
    print("\nPrediction results:")
    for lt in lead_times:
        pred = result['predictions'][lt]
        true_idx = (lt // 6) - 1
        if true_idx < len(true_future):
            true = true_future[true_idx]
            csi = calculate_csi(true, pred, threshold=15)
            print(f"  +{lt:3d}min: CSI(15dBZ) = {csi:.3f}")
    
    diagnostics = result['diagnostics']
    print(f"\nDiagnostics available: {list(diagnostics.keys())}")
    
    print("\n" + "=" * 60)
    return result


def ablation_study_example():
    print("\n" + "=" * 60)
    print("Ablation Study - Comparing Different Configurations")
    print("=" * 60)
    
    np.random.seed(42)
    sequence = generate_radar_sequence(num_frames=10, num_storms=4)
    sequence = add_noise(sequence, noise_level=0.5)
    
    history = sequence[:6]
    true_future = sequence[6:]
    
    configurations = [
        ('Baseline (Optical Flow only)', False, False, False),
        ('+ Mass Conservation', True, False, False),
        ('+ Intensity Correction', True, True, False),
        ('+ Adaptive Weighting (Full)', True, True, True),
    ]
    
    results = {}
    for name, use_mass, use_intensity, use_adaptive in configurations:
        print(f"\nTesting: {name}")
        ext = EnhancedRadarExtrapolator(
            flow_method='farneback',
            use_mass_conservation=use_mass,
            use_intensity_correction=use_intensity,
            use_adaptive_weighting=use_adaptive
        )
        preds = ext.extrapolate_enhanced(history, steps=4)
        
        csi_scores = []
        for true, pred in zip(true_future[:4], preds):
            csi = calculate_csi(true, pred, threshold=20)
            csi_scores.append(csi)
        
        avg_csi = np.mean(csi_scores)
        results[name] = avg_csi
        print(f"  Average CSI (20dBZ): {avg_csi:.4f}")
    
    print("\n" + "-" * 60)
    print("Summary:")
    baseline_csi = results['Baseline (Optical Flow only)']
    for name, csi in results.items():
        improvement = (csi - baseline_csi) / baseline_csi * 100
        print(f"  {name}: {csi:.4f} ({improvement:+.1f}%)")
    
    print("\n" + "=" * 60)
    return results


def visualize_results_example():
    print("\n" + "=" * 60)
    print("Visualization Example")
    print("=" * 60)
    
    np.random.seed(42)
    sequence = generate_radar_sequence(num_frames=10, num_storms=3)
    sequence = add_noise(sequence, noise_level=0.3)
    
    history = sequence[:6]
    true_future = sequence[6:]
    
    ext = EnhancedRadarExtrapolator(
        use_mass_conservation=True,
        use_intensity_correction=True,
        use_adaptive_weighting=True
    )
    predictions = ext.extrapolate_enhanced(history, steps=4)
    
    radar_cmap = ListedColormap([
        '#FFFFFF', '#80FF80', '#00FF00', '#00C000',
        '#008000', '#FFFF00', '#FFC000', '#FF8000',
        '#FF0000', '#C00000', '#800000', '#FF00FF'
    ])
    
    fig, axes = plt.subplots(3, 4, figsize=(16, 10))
    
    for i in range(4):
        axes[0, i].imshow(true_future[i], cmap=radar_cmap, vmin=0, vmax=60)
        axes[0, i].set_title(f'True +{(i+1)*6}min')
        axes[0, i].axis('off')
        
        axes[1, i].imshow(predictions[i], cmap=radar_cmap, vmin=0, vmax=60)
        axes[1, i].set_title(f'Predicted +{(i+1)*6}min')
        axes[1, i].axis('off')
        
        diff = np.abs(true_future[i] - predictions[i])
        im = axes[2, i].imshow(diff, cmap='hot', vmin=0, vmax=20)
        axes[2, i].set_title(f'Absolute Error')
        axes[2, i].axis('off')
        plt.colorbar(im, ax=axes[2, i], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    plt.savefig('quick_start_visualization.png', dpi=100)
    plt.close()
    print("\nSaved visualization: quick_start_visualization.png")
    
    print("\n" + "=" * 60)


if __name__ == '__main__':
    basic_usage_example()
    pipeline_example()
    ablation_study_example()
    visualize_results_example()
    
    print("\n" + "=" * 60)
    print("All examples completed successfully!")
    print("=" * 60)
