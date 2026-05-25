import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from radar_extrapolation import RadarExtrapolator, calculate_csi, calculate_pod, calculate_far
from generate_sample_data import generate_radar_sequence, add_noise


def main():
    radar_cmap = ListedColormap([
        '#FFFFFF', '#80FF80', '#00FF00', '#00C000',
        '#008000', '#FFFF00', '#FFC000', '#FF8000',
        '#FF0000', '#C00000', '#800000', '#FF00FF'
    ])

    np.random.seed(42)
    total_frames = 10
    sequence = generate_radar_sequence(num_frames=total_frames, num_storms=4, velocity=(2.5, 1.5))
    sequence = add_noise(sequence, noise_level=0.8)

    hist_frames = sequence[:6]
    true_future = sequence[6:]

    methods = ['farneback', 'dis']
    extrapolation_methods = ['warp', 'advection']

    for flow_method in methods:
        extrapolator = RadarExtrapolator(method=flow_method)

        avg_flow = extrapolator.compute_average_flow(hist_frames)

        for ext_method in extrapolation_methods:
            extrapolated = extrapolator.extrapolate(
                hist_frames[-1], avg_flow, steps=len(true_future), method=ext_method
            )

            plot_results(hist_frames, true_future, extrapolated, avg_flow, 
                        f'{flow_method.upper()} + {ext_method}', radar_cmap)

            evaluate_predictions(true_future, extrapolated, f'{flow_method.upper()} + {ext_method}')


def plot_results(history, true_future, predicted, flow, title, cmap):
    num_hist = len(history)
    num_future = len(true_future)

    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, max(num_hist, num_future) + 1, hspace=0.3, wspace=0.1)

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
    ax_flow.set_title('Average Flow')
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
        ax.imshow(predicted[i], cmap=cmap, vmin=0, vmax=60)
        ax.set_title(f'Pred +{(i+1)*6}min')
        ax.axis('off')

    fig.suptitle(f'Radar Echo Extrapolation - {title}', fontsize=14, y=0.98)
    plt.tight_layout()
    filename = f'extrapolation_{title.lower().replace(" ", "_").replace("+", "_")}.png'
    plt.savefig(filename, dpi=100, bbox_inches='tight')
    plt.close()
    print(f'Plot saved: {filename}')


def evaluate_predictions(true_future, predicted, title):
    print(f'\n=== Evaluation: {title} ===')
    thresholds = [10, 20, 30]

    for threshold in thresholds:
        csi_scores = []
        pod_scores = []
        far_scores = []

        for t, p in zip(true_future, predicted):
            csi = calculate_csi(t, p, threshold=threshold)
            pod = calculate_pod(t, p, threshold=threshold)
            far = calculate_far(t, p, threshold=threshold)
            csi_scores.append(csi)
            pod_scores.append(pod)
            far_scores.append(far)

        avg_csi = np.mean(csi_scores)
        avg_pod = np.mean(pod_scores)
        avg_far = np.mean(far_scores)

        print(f'Threshold {threshold} dBZ:')
        print(f'  CSI: {avg_csi:.3f} | POD: {avg_pod:.3f} | FAR: {avg_far:.3f}')

    return csi_scores, pod_scores, far_scores


def visualize_flow_field():
    np.random.seed(42)
    sequence = generate_radar_sequence(num_frames=3, num_storms=3, velocity=(3, 2))

    extrapolator = RadarExtrapolator(method='farneback')
    flow = extrapolator._compute_flow(sequence[0], sequence[1])

    h, w = flow.shape[:2]
    y, x = np.mgrid[0:h:8, 0:w:8]
    u = flow[::8, ::8, 0]
    v = flow[::8, ::8, 1]

    magnitude = np.sqrt(flow[:, :, 0]**2 + flow[:, :, 1]**2)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(sequence[0], cmap='gray')
    axes[0].set_title('Frame t')
    axes[0].axis('off')

    axes[1].imshow(sequence[1], cmap='gray')
    axes[1].set_title('Frame t+1')
    axes[1].axis('off')

    im = axes[2].imshow(magnitude, cmap='jet')
    axes[2].quiver(x, y, u, v, color='white', scale=30, alpha=0.8)
    axes[2].set_title('Flow Field (magnitude + vectors)')
    axes[2].axis('off')
    plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig('flow_field_visualization.png', dpi=100)
    plt.close()
    print('Flow field visualization saved: flow_field_visualization.png')


if __name__ == '__main__':
    print('Starting radar extrapolation test...')
    visualize_flow_field()
    main()
    print('\nAll tests completed!')
