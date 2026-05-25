import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from radar_extrapolation import RadarExtrapolator
from radar_extrapolation_enhanced import EnhancedRadarExtrapolator, NowcastingPipeline
from radar_extrapolation_enhanced import calculate_csi, calculate_pod, calculate_far, calculate_bias
from generate_sample_data import generate_radar_sequence, add_noise, generate_convective_line


def generate_growth_decay_sequence(num_frames: int = 10, shape: tuple = (200, 200)):
    sequence = []
    h, w = shape
    
    storms = [
        {'x0': 60, 'y0': 80, 'sigma': 15, 'amplitude': 15, 'vx': 2, 'vy': 0, 'growth': 1.15},
        {'x0': 100, 'y0': 100, 'sigma': 20, 'amplitude': 40, 'vx': 1, 'vy': 1, 'growth': 1.0},
        {'x0': 140, 'y0': 120, 'sigma': 18, 'amplitude': 45, 'vx': 0.5, 'vy': 0, 'growth': 0.85},
    ]
    
    for t in range(num_frames):
        frame = np.zeros(shape, dtype=np.float32)
        for storm in storms:
            cx = storm['x0'] + storm['vx'] * t
            cy = storm['y0'] + storm['vy'] * t
            sigma = storm['sigma']
            amplitude = storm['amplitude'] * (storm['growth'] ** t)
            
            if 0 <= cx < w and 0 <= cy < h:
                y, x = np.ogrid[:h, :w]
                gaussian = np.exp(-((x - cx)**2 + (y - cy)**2) / (2 * sigma**2))
                frame += gaussian * amplitude
        
        frame = np.clip(frame, 0, 70)
        sequence.append(frame)
    
    return sequence


def compare_methods():
    radar_cmap = ListedColormap([
        '#FFFFFF', '#80FF80', '#00FF00', '#00C000',
        '#008000', '#FFFF00', '#FFC000', '#FF8000',
        '#FF0000', '#C00000', '#800000', '#FF00FF'
    ])
    
    np.random.seed(42)
    
    print('Generating test sequence with growth/decay storms...')
    sequence = generate_growth_decay_sequence(num_frames=12)
    sequence = add_noise(sequence, noise_level=0.5)
    
    hist_frames = sequence[:6]
    true_future = sequence[6:12]
    num_future = len(true_future)
    
    print('Running baseline extrapolation...')
    baseline_ext = RadarExtrapolator(method='farneback')
    baseline_flow = baseline_ext.compute_average_flow(hist_frames)
    baseline_pred = baseline_ext.extrapolate(
        hist_frames[-1], baseline_flow, steps=num_future, method='advection'
    )
    
    print('Running enhanced extrapolation (mass conservation)...')
    enhanced_mass = EnhancedRadarExtrapolator(
        flow_method='farneback',
        use_mass_conservation=True,
        use_intensity_correction=False,
        use_adaptive_weighting=False
    )
    enhanced_mass_pred = enhanced_mass.extrapolate_enhanced(
        hist_frames, steps=num_future
    )
    
    print('Running enhanced extrapolation (full)...')
    enhanced_full = EnhancedRadarExtrapolator(
        flow_method='farneback',
        use_mass_conservation=True,
        use_intensity_correction=True,
        use_adaptive_weighting=True
    )
    enhanced_full_pred = enhanced_full.extrapolate_enhanced(
        hist_frames, steps=num_future
    )
    
    print('\n=== Generating comparison plots ===')
    plot_comparison(hist_frames, true_future, baseline_pred, 
                   enhanced_mass_pred, enhanced_full_pred,
                   baseline_flow, enhanced_full, radar_cmap)
    
    print('\n=== Quantitative Evaluation ===')
    thresholds = [10, 20, 30]
    methods = {
        'Baseline (Optical Flow)': baseline_pred,
        'Enhanced (Mass Cons.)': enhanced_mass_pred,
        'Enhanced (Full)': enhanced_full_pred
    }
    
    results = {}
    for method_name, predictions in methods.items():
        results[method_name] = evaluate_method(true_future, predictions, method_name, thresholds)
    
    plot_metrics_comparison(results, thresholds, num_future)
    
    print('\n=== Generating diagnostics plots ===')
    plot_diagnostics(hist_frames, enhanced_full, radar_cmap)


def plot_comparison(history, true_future, baseline_pred, mass_pred, full_pred,
                    flow, enhanced_ext, cmap):
    num_hist = len(history)
    num_future = len(true_future)
    
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(5, max(num_hist, num_future) + 1, hspace=0.25, wspace=0.05)
    
    for i in range(num_hist):
        ax = fig.add_subplot(gs[0, i])
        ax.imshow(history[i], cmap=cmap, vmin=0, vmax=60)
        ax.set_title(f'History t-{num_hist-i}')
        ax.axis('off')
    
    ax_flow = fig.add_subplot(gs[0, num_hist])
    h, w = flow.shape[:2]
    y, x = np.mgrid[0:h:10, 0:w:10]
    u = flow[::10, ::10, 0]
    v = flow[::10, ::10, 1]
    ax_flow.quiver(x, y, u, v, scale=50, color='blue', alpha=0.7)
    ax_flow.set_title('Avg Flow')
    ax_flow.set_xlim(0, w)
    ax_flow.set_ylim(h, 0)
    ax_flow.axis('off')
    
    for i in range(num_future):
        ax = fig.add_subplot(gs[1, i])
        ax.imshow(true_future[i], cmap=cmap, vmin=0, vmax=60)
        ax.set_title(f'True +{(i+1)*6}min')
        ax.axis('off')
    
    for i in range(num_future):
        ax = fig.add_subplot(gs[2, i])
        ax.imshow(baseline_pred[i], cmap=cmap, vmin=0, vmax=60)
        ax.set_title(f'Baseline +{(i+1)*6}min')
        ax.axis('off')
    
    for i in range(num_future):
        ax = fig.add_subplot(gs[3, i])
        ax.imshow(mass_pred[i], cmap=cmap, vmin=0, vmax=60)
        ax.set_title(f'Mass Cons. +{(i+1)*6}min')
        ax.axis('off')
    
    for i in range(num_future):
        ax = fig.add_subplot(gs[4, i])
        ax.imshow(full_pred[i], cmap=cmap, vmin=0, vmax=60)
        ax.set_title(f'Full Enhanced +{(i+1)*6}min')
        ax.axis('off')
    
    fig.suptitle('Radar Echo Extrapolation Comparison - Growth/Decay Test Case', 
                 fontsize=14, y=0.98)
    plt.savefig('comparison_all_methods.png', dpi=100, bbox_inches='tight')
    plt.close()
    print('Saved: comparison_all_methods.png')


def evaluate_method(true_future, predictions, method_name, thresholds):
    print(f'\n--- {method_name} ---')
    
    all_metrics = {}
    for threshold in thresholds:
        csi_scores = []
        pod_scores = []
        far_scores = []
        bias_scores = []
        
        for t, p in zip(true_future, predictions):
            csi = calculate_csi(t, p, threshold=threshold)
            pod = calculate_pod(t, p, threshold=threshold)
            far = calculate_far(t, p, threshold=threshold)
            bias = calculate_bias(t, p, threshold=threshold)
            
            csi_scores.append(csi)
            pod_scores.append(pod)
            far_scores.append(far)
            bias_scores.append(bias)
        
        avg_csi = np.mean(csi_scores)
        avg_pod = np.mean(pod_scores)
        avg_far = np.mean(far_scores)
        avg_bias = np.mean(bias_scores)
        
        all_metrics[threshold] = {
            'csi': csi_scores,
            'pod': pod_scores,
            'far': far_scores,
            'bias': bias_scores,
            'avg_csi': avg_csi,
            'avg_pod': avg_pod,
            'avg_far': avg_far,
            'avg_bias': avg_bias
        }
        
        print(f'  Threshold {threshold} dBZ: CSI={avg_csi:.3f}, POD={avg_pod:.3f}, FAR={avg_far:.3f}, BIAS={avg_bias:.3f}')
    
    return all_metrics


def plot_metrics_comparison(results, thresholds, num_future):
    lead_times = [(i+1)*6 for i in range(num_future)]
    methods = list(results.keys())
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Forecast Metrics Comparison', fontsize=14, y=0.98)
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    markers = ['o', 's', '^']
    
    for idx, method in enumerate(methods):
        for threshold in thresholds:
            csi_scores = results[method][threshold]['csi']
            axes[0, 0].plot(lead_times, csi_scores, label=f'{method} ({threshold}dBZ)',
                           color=colors[idx], linestyle=['-', '--', ':'][thresholds.index(threshold)],
                           marker=markers[idx], markersize=4, alpha=0.8)
    
    axes[0, 0].set_xlabel('Lead Time (minutes)')
    axes[0, 0].set_ylabel('CSI')
    axes[0, 0].set_title('Critical Success Index')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend(fontsize=7, loc='lower left')
    
    for idx, method in enumerate(methods):
        avg_csi = [results[method][t]['avg_csi'] for t in thresholds]
        axes[0, 1].plot(thresholds, avg_csi, label=method,
                       color=colors[idx], marker=markers[idx], linewidth=2)
    
    axes[0, 1].set_xlabel('Threshold (dBZ)')
    axes[0, 1].set_ylabel('Average CSI')
    axes[0, 1].set_title('CSI vs Threshold')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()
    
    for idx, method in enumerate(methods):
        avg_pod = [results[method][t]['avg_pod'] for t in thresholds]
        avg_far = [results[method][t]['avg_far'] for t in thresholds]
        axes[1, 0].plot(avg_far, avg_pod, label=method,
                       color=colors[idx], marker=markers[idx], linewidth=2)
        
        for i, t in enumerate(thresholds):
            axes[1, 0].annotate(f'{t}dBZ', (avg_far[i], avg_pod[i]),
                               fontsize=8, xytext=(5, 5), textcoords='offset points')
    
    axes[1, 0].set_xlabel('FAR')
    axes[1, 0].set_ylabel('POD')
    axes[1, 0].set_title('ROC-like Diagram')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend()
    axes[1, 0].set_xlim(0, 1)
    axes[1, 0].set_ylim(0, 1)
    
    for idx, method in enumerate(methods):
        avg_bias = [results[method][t]['avg_bias'] for t in thresholds]
        axes[1, 1].plot(thresholds, avg_bias, label=method,
                       color=colors[idx], marker=markers[idx], linewidth=2)
    
    axes[1, 1].axhline(y=1, color='gray', linestyle='--', alpha=0.5)
    axes[1, 1].set_xlabel('Threshold (dBZ)')
    axes[1, 1].set_ylabel('Average BIAS')
    axes[1, 1].set_title('Frequency Bias')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig('metrics_comparison.png', dpi=100, bbox_inches='tight')
    plt.close()
    print('Saved: metrics_comparison.png')


def plot_diagnostics(history, enhanced_ext, cmap):
    h, w = history[0].shape
    
    regions = enhanced_ext.detect_growth_decay_regions(history)
    weights = enhanced_ext.compute_adaptive_weights(history, regions)
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle('Enhanced Extrapolation Diagnostics', fontsize=14, y=0.98)
    
    im = axes[0, 0].imshow(history[-1], cmap=cmap, vmin=0, vmax=60)
    axes[0, 0].set_title('Latest Observation')
    axes[0, 0].axis('off')
    plt.colorbar(im, ax=axes[0, 0], fraction=0.046, pad=0.04)
    
    growth_rate = enhanced_ext.growth_rate_map
    im = axes[0, 1].imshow(growth_rate, cmap='RdYlGn', vmin=-2, vmax=2)
    axes[0, 1].set_title('Growth Rate Map')
    axes[0, 1].axis('off')
    plt.colorbar(im, ax=axes[0, 1], fraction=0.046, pad=0.04)
    
    region_vis = np.zeros((h, w, 3))
    region_vis[regions['growth']] = [0, 1, 0]
    region_vis[regions['decay']] = [1, 0, 0]
    region_vis[regions['stable']] = [0.5, 0.5, 0.5]
    axes[0, 2].imshow(region_vis)
    axes[0, 2].set_title('Regions (Green=Growth, Red=Decay)')
    axes[0, 2].axis('off')
    
    divergence = enhanced_ext.divergence_map
    im = axes[0, 3].imshow(divergence, cmap='RdBu_r', vmin=-0.1, vmax=0.1)
    axes[0, 3].set_title('Flow Divergence')
    axes[0, 3].axis('off')
    plt.colorbar(im, ax=axes[0, 3], fraction=0.046, pad=0.04)
    
    im = axes[1, 0].imshow(weights['motion'], cmap='viridis', vmin=0, vmax=1)
    axes[1, 0].set_title('Motion Confidence')
    axes[1, 0].axis('off')
    plt.colorbar(im, ax=axes[1, 0], fraction=0.046, pad=0.04)
    
    im = axes[1, 1].imshow(weights['intensity'], cmap='viridis', vmin=0, vmax=1)
    axes[1, 1].set_title('Intensity Model Trust')
    axes[1, 1].axis('off')
    plt.colorbar(im, ax=axes[1, 1], fraction=0.046, pad=0.04)
    
    alpha = weights['intensity'] * weights['growth_weight']
    alpha = np.clip(alpha, 0.2, 0.8)
    im = axes[1, 2].imshow(alpha, cmap='coolwarm', vmin=0.2, vmax=0.8)
    axes[1, 2].set_title('Blend Weight (motion vs intensity)')
    axes[1, 2].axis('off')
    plt.colorbar(im, ax=axes[1, 2], fraction=0.046, pad=0.04)
    
    flow = enhanced_ext.compute_average_flow(history)
    y, x = np.mgrid[0:h:10, 0:w:10]
    u = flow[::10, ::10, 0]
    v = flow[::10, ::10, 1]
    magnitude = np.sqrt(flow[:, :, 0]**2 + flow[:, :, 1]**2)
    axes[1, 3].imshow(magnitude, cmap='jet', alpha=0.6)
    axes[1, 3].quiver(x, y, u, v, color='white', scale=50, alpha=0.8)
    axes[1, 3].set_title('Flow Field')
    axes[1, 3].axis('off')
    
    plt.tight_layout()
    plt.savefig('diagnostics.png', dpi=100, bbox_inches='tight')
    plt.close()
    print('Saved: diagnostics.png')


def demo_nowcasting_pipeline():
    print('\n=== Nowcasting Pipeline Demo ===')
    
    np.random.seed(123)
    sequence = generate_convective_line(num_frames=10, velocity=2.0)
    sequence = add_noise(sequence, noise_level=0.3)
    
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
    result = pipeline.predict(history, lead_times=[6, 12, 18, 24, 30, 36])
    
    print(f"Predicted lead times: {result['diagnostics']['lead_times']}")
    print(f"Flow field shape: {result['diagnostics']['flow_field'].shape}")
    print(f"Prediction keys: {list(result['predictions'].keys())}")
    
    true_future = sequence[6:12]
    for lt in [6, 18, 30, 36]:
        if lt <= 36:
            pred = result['predictions'][lt]
            true_idx = (lt // 6) - 1
            if true_idx < len(true_future):
                true = true_future[true_idx]
                csi = calculate_csi(true, pred, threshold=20)
                print(f"  +{lt}min CSI (20dBZ): {csi:.3f}")


if __name__ == '__main__':
    print('='*60)
    print('Enhanced Radar Extrapolation Test Suite')
    print('='*60)
    
    compare_methods()
    demo_nowcasting_pipeline()
    
    print('\n' + '='*60)
    print('All tests completed! Check generated PNG files for results.')
    print('='*60)
