import os
import numpy as np
import torch
from typing import List, Dict, Optional, Tuple
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap

from deep_extrapolation_model import RadarGAN, HybridExtrapolator
from radar_extrapolation_enhanced import EnhancedRadarExtrapolator
from radar_extrapolation_enhanced import calculate_csi, calculate_pod, calculate_far, calculate_bias


class DeepNowcastingPipeline:
    def __init__(self, 
                 gan_model_path: Optional[str] = None,
                 use_optical_flow: bool = True,
                 use_deep_model: bool = True,
                 switch_lead_time: int = 60,
                 flow_weight: float = 0.5,
                 gan_weight: float = 0.5,
                 device: str = 'cpu'):
        
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.use_optical_flow = use_optical_flow
        self.use_deep_model = use_deep_model
        self.switch_lead_time = switch_lead_time
        self.flow_weight = flow_weight
        self.gan_weight = gan_weight
        
        self.optical_flow_model = None
        self.gan_model = None
        
        if use_optical_flow:
            self.optical_flow_model = EnhancedRadarExtrapolator(
                flow_method='farneback',
                use_mass_conservation=True,
                use_intensity_correction=True,
                use_adaptive_weighting=True
            )
        
        if use_deep_model and gan_model_path is not None and os.path.exists(gan_model_path):
            self.gan_model = self._load_gan_model(gan_model_path)
        
        self.hybrid_extrapolator = HybridExtrapolator(
            optical_flow_model=self.optical_flow_model,
            gan_model=self.gan_model,
            flow_weight=flow_weight,
            gan_weight=gan_weight,
            switch_lead_time=switch_lead_time
        )
    
    def _load_gan_model(self, model_path: str) -> RadarGAN:
        checkpoint = torch.load(model_path, map_location=self.device)
        
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
            args = checkpoint.get('args', None)
            if args:
                model = RadarGAN(
                    input_dim=1,
                    output_dim=1,
                    hidden_dims=args.hidden_dims if hasattr(args, 'hidden_dims') else [64, 128, 256],
                    num_history_frames=args.num_history_frames if hasattr(args, 'num_history_frames') else 6,
                    num_future_frames=args.num_future_frames if hasattr(args, 'num_future_frames') else 20
                )
            else:
                model = RadarGAN()
            model.load_state_dict(state_dict)
        else:
            model = RadarGAN()
            model.load_state_dict(checkpoint)
        
        model = model.to(self.device)
        model.eval()
        return model
    
    def predict(self, 
                history_frames: List[np.ndarray], 
                lead_times: List[int],
                time_interval: int = 6) -> Dict[int, np.ndarray]:
        
        if not self.use_optical_flow and not self.use_deep_model:
            raise ValueError("At least one model (optical_flow or deep_model) must be enabled")
        
        return self.hybrid_extrapolator.predict(
            history_frames, lead_times, time_interval
        )
    
    def predict_deep_only(self, 
                         history_frames: List[np.ndarray], 
                         num_steps: int) -> List[np.ndarray]:
        if self.gan_model is None:
            raise ValueError("GAN model is not loaded")
        
        self.gan_model.eval()
        with torch.no_grad():
            history_tensor = torch.tensor(np.array(history_frames), dtype=torch.float32) / 70.0
            history_tensor = history_tensor.unsqueeze(1).unsqueeze(0).to(self.device)
            
            predictions = self.gan_model.generator(history_tensor, future_length=num_steps)
            predictions = predictions.squeeze(0).squeeze(1).cpu().numpy() * 70.0
            
            return [predictions[i] for i in range(num_steps)]
    
    def predict_optical_flow_only(self,
                                  history_frames: List[np.ndarray],
                                  num_steps: int,
                                  time_interval: int = 6) -> List[np.ndarray]:
        if self.optical_flow_model is None:
            raise ValueError("Optical flow model is not enabled")
        
        return self.optical_flow_model.extrapolate_enhanced(
            history_frames, steps=num_steps, time_interval=time_interval
        )


def evaluate_all_methods(history_frames: List[np.ndarray],
                         true_future: List[np.ndarray],
                         pipeline: DeepNowcastingPipeline,
                         lead_times: List[int],
                         time_interval: int = 6) -> Dict:
    
    results = {
        'lead_times': lead_times,
        'optical_flow': {},
        'deep_model': {},
        'hybrid': {}
    }
    
    flow_predictions = pipeline.predict_optical_flow_only(
        history_frames, num_steps=len(true_future), time_interval=time_interval
    )
    
    try:
        deep_predictions = pipeline.predict_deep_only(
            history_frames, num_steps=len(true_future)
        )
    except:
        deep_predictions = None
    
    hybrid_predictions_dict = pipeline.predict(history_frames, lead_times, time_interval)
    hybrid_predictions = [hybrid_predictions_dict[lt] for lt in lead_times]
    
    thresholds = [10, 20, 30]
    
    for threshold in thresholds:
        results['optical_flow'][threshold] = {
            'csi': [], 'pod': [], 'far': [], 'bias': []
        }
        results['deep_model'][threshold] = {
            'csi': [], 'pod': [], 'far': [], 'bias': []
        }
        results['hybrid'][threshold] = {
            'csi': [], 'pod': [], 'far': [], 'bias': []
        }
        
        for i, true_frame in enumerate(true_future):
            flow_pred = flow_predictions[i] if i < len(flow_predictions) else flow_predictions[-1]
            
            results['optical_flow'][threshold]['csi'].append(
                calculate_csi(true_frame, flow_pred, threshold)
            )
            results['optical_flow'][threshold]['pod'].append(
                calculate_pod(true_frame, flow_pred, threshold)
            )
            results['optical_flow'][threshold]['far'].append(
                calculate_far(true_frame, flow_pred, threshold)
            )
            results['optical_flow'][threshold]['bias'].append(
                calculate_bias(true_frame, flow_pred, threshold)
            )
            
            hybrid_pred = hybrid_predictions[i] if i < len(hybrid_predictions) else hybrid_predictions[-1]
            
            results['hybrid'][threshold]['csi'].append(
                calculate_csi(true_frame, hybrid_pred, threshold)
            )
            results['hybrid'][threshold]['pod'].append(
                calculate_pod(true_frame, hybrid_pred, threshold)
            )
            results['hybrid'][threshold]['far'].append(
                calculate_far(true_frame, hybrid_pred, threshold)
            )
            results['hybrid'][threshold]['bias'].append(
                calculate_bias(true_frame, hybrid_pred, threshold)
            )
            
            if deep_predictions is not None:
                deep_pred = deep_predictions[i] if i < len(deep_predictions) else deep_predictions[-1]
                
                results['deep_model'][threshold]['csi'].append(
                    calculate_csi(true_frame, deep_pred, threshold)
                )
                results['deep_model'][threshold]['pod'].append(
                    calculate_pod(true_frame, deep_pred, threshold)
                )
                results['deep_model'][threshold]['far'].append(
                    calculate_far(true_frame, deep_pred, threshold)
                )
                results['deep_model'][threshold]['bias'].append(
                    calculate_bias(true_frame, deep_pred, threshold)
                )
    
    results['flow_predictions'] = flow_predictions
    results['deep_predictions'] = deep_predictions
    results['hybrid_predictions'] = hybrid_predictions
    results['true_future'] = true_future
    
    return results


def visualize_comparison(history_frames: List[np.ndarray],
                         results: Dict,
                         output_path: str = 'method_comparison.png'):
    
    radar_cmap = ListedColormap([
        '#FFFFFF', '#80FF80', '#00FF00', '#00C000',
        '#008000', '#FFFF00', '#FFC000', '#FF8000',
        '#FF0000', '#C00000', '#800000', '#FF00FF'
    ])
    
    num_history = len(history_frames)
    num_future = len(results['true_future'])
    display_steps = min(6, num_future)
    
    has_deep = results['deep_predictions'] is not None
    
    n_rows = 4 if has_deep else 3
    
    fig, axes = plt.subplots(n_rows, display_steps, figsize=(3*display_steps, 3*n_rows))
    
    for i in range(display_steps):
        hist_idx = num_history - display_steps + i
        if hist_idx >= 0:
            axes[0, i].imshow(history_frames[hist_idx], cmap=radar_cmap, vmin=0, vmax=60)
            axes[0, i].set_title(f'History t-{display_steps-i}')
        else:
            axes[0, i].imshow(np.zeros_like(history_frames[0]), cmap=radar_cmap, vmin=0, vmax=60)
        axes[0, i].axis('off')
    
    for i in range(display_steps):
        axes[1, i].imshow(results['true_future'][i], cmap=radar_cmap, vmin=0, vmax=60)
        axes[1, i].set_title(f'True +{(i+1)*6}min')
        axes[1, i].axis('off')
    
    for i in range(display_steps):
        axes[2, i].imshow(results['hybrid_predictions'][i], cmap=radar_cmap, vmin=0, vmax=60)
        axes[2, i].set_title(f'Hybrid +{(i+1)*6}min')
        axes[2, i].axis('off')
    
    if has_deep:
        for i in range(display_steps):
            axes[3, i].imshow(results['deep_predictions'][i], cmap=radar_cmap, vmin=0, vmax=60)
            axes[3, i].set_title(f'Deep +{(i+1)*6}min')
            axes[3, i].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=100)
    plt.close()
    print(f"Comparison visualization saved: {output_path}")


def plot_metrics_comparison(results: Dict, output_path: str = 'metrics_comparison_full.png'):
    thresholds = [10, 20, 30]
    methods = ['optical_flow', 'hybrid']
    if results['deep_predictions'] is not None:
        methods.append('deep_model')
    
    method_names = {
        'optical_flow': 'Optical Flow',
        'deep_model': 'Deep GAN',
        'hybrid': 'Hybrid'
    }
    
    colors = ['#1f77b4', '#2ca02c', '#ff7f0e']
    markers = ['o', 's', '^']
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    lead_times = results['lead_times']
    
    for method_idx, method in enumerate(methods):
        for threshold in thresholds:
            csi_scores = results[method][threshold]['csi']
            axes[0, 0].plot(lead_times, csi_scores, 
                           label=f'{method_names[method]} ({threshold}dBZ)',
                           color=colors[method_idx],
                           linestyle=['-', '--', ':'][thresholds.index(threshold)],
                           marker=markers[method_idx],
                           markersize=4, alpha=0.8)
    
    axes[0, 0].set_xlabel('Lead Time (minutes)')
    axes[0, 0].set_ylabel('CSI')
    axes[0, 0].set_title('Critical Success Index')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend(fontsize=7, loc='lower left')
    
    for method_idx, method in enumerate(methods):
        avg_csi = [np.mean(results[method][t]['csi']) for t in thresholds]
        axes[0, 1].plot(thresholds, avg_csi, label=method_names[method],
                       color=colors[method_idx], marker=markers[method_idx], linewidth=2)
    
    axes[0, 1].set_xlabel('Threshold (dBZ)')
    axes[0, 1].set_ylabel('Average CSI')
    axes[0, 1].set_title('CSI vs Threshold')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()
    
    for method_idx, method in enumerate(methods):
        avg_pod = [np.mean(results[method][t]['pod']) for t in thresholds]
        avg_far = [np.mean(results[method][t]['far']) for t in thresholds]
        axes[1, 0].plot(avg_far, avg_pod, label=method_names[method],
                       color=colors[method_idx], marker=markers[method_idx], linewidth=2)
        
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
    
    for method_idx, method in enumerate(methods):
        avg_bias = [np.mean(results[method][t]['bias']) for t in thresholds]
        axes[1, 1].plot(thresholds, avg_bias, label=method_names[method],
                       color=colors[method_idx], marker=markers[method_idx], linewidth=2)
    
    axes[1, 1].axhline(y=1, color='gray', linestyle='--', alpha=0.5)
    axes[1, 1].set_xlabel('Threshold (dBZ)')
    axes[1, 1].set_ylabel('Average BIAS')
    axes[1, 1].set_title('Frequency Bias')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=100)
    plt.close()
    print(f"Metrics comparison saved: {output_path}")
