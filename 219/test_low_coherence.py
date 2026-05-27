import numpy as np
import matplotlib.pyplot as plt
import time
from phase_unwrapping import (
    wrap,
    create_residues,
    branch_cut_unwrap,
    least_squares_unwrap,
    quality_guided_unwrap,
    estimate_coherence,
    improved_branch_cut_unwrap,
    graph_cut_unwrap,
    snaphu_style_unwrap,
    improved_residue_connection,
)


def generate_low_coherence_data(size=64, noise_level=0.5):
    x = np.linspace(-5, 5, size)
    y = np.linspace(-5, 5, size)
    X, Y = np.meshgrid(x, y)

    true_phase = X**2 + Y**2 + 2 * np.sin(X) * np.cos(Y)

    coherence = np.ones((size, size))

    low_coh_regions = [
        (slice(int(size*0.2), int(size*0.4)), slice(int(size*0.2), int(size*0.4))),
        (slice(int(size*0.6), int(size*0.8)), slice(int(size*0.6), int(size*0.8))),
        (slice(int(size*0.1), int(size*0.3)), slice(int(size*0.7), int(size*0.9))),
    ]

    for i, j in low_coh_regions:
        coherence[i, j] = 0.2 + np.random.rand(len(range(*i.indices(size))), len(range(*j.indices(size)))) * 0.3

    noise = np.random.normal(0, noise_level, true_phase.shape)
    for i, j in low_coh_regions:
        noise[i, j] *= 3

    true_phase += noise

    wrapped_phase = wrap(true_phase)

    return true_phase, wrapped_phase, coherence


def calculate_error(unwrapped, true_phase):
    diff = unwrapped - true_phase
    k = np.round(diff / (2 * np.pi))
    diff_corrected = diff - 2 * np.pi * k
    rmse = np.sqrt(np.mean(diff_corrected**2))
    return rmse, diff_corrected


def main():
    print("=" * 70)
    print("Low Coherence Region Phase Unwrapping Comparison")
    print("=" * 70)

    np.random.seed(42)

    size = 48
    true_phase, wrapped_phase, coherence = generate_low_coherence_data(size, noise_level=0.3)

    print(f"\nData size: {size}x{size}")
    print(f"Mean coherence: {np.mean(coherence):.4f}")
    print(f"Min coherence: {np.min(coherence):.4f}")

    residues_old = create_residues(wrapped_phase)
    residues_new, branch_cuts_new = improved_residue_connection(wrapped_phase, coherence)

    print(f"\nNumber of residues: {np.sum(np.abs(residues_old)):.0f}")

    algorithms = [
        ("Original Branch Cut", branch_cut_unwrap, {}),
        ("Improved Branch Cut", improved_branch_cut_unwrap, {"coherence": coherence}),
        ("Least Squares", least_squares_unwrap, {}),
        ("Quality Guided", quality_guided_unwrap, {}),
        ("SNAPHU Style", snaphu_style_unwrap, {"coherence": coherence}),
        ("Graph Cut", graph_cut_unwrap, {"coherence": coherence, "num_labels": 8}),
    ]

    results = {}

    fig, axes = plt.subplots(3, 3, figsize=(18, 15))

    im0 = axes[0, 0].imshow(true_phase, cmap="jet")
    axes[0, 0].set_title("True Phase")
    plt.colorbar(im0, ax=axes[0, 0])

    im1 = axes[0, 1].imshow(wrapped_phase, cmap="jet")
    axes[0, 1].set_title("Wrapped Phase")
    plt.colorbar(im1, ax=axes[0, 1])

    im2 = axes[0, 2].imshow(coherence, cmap="gray", vmin=0, vmax=1)
    axes[0, 2].set_title("Coherence Map")
    plt.colorbar(im2, ax=axes[0, 2])

    plot_idx = 1
    for name, func, kwargs in algorithms:
        print(f"\n{'-' * 60}")
        print(f"Testing: {name}")

        try:
            start_time = time.time()

            if name == "Improved Branch Cut":
                unwrapped, bc, res = func(wrapped_phase, **kwargs)
            elif name == "Graph Cut":
                unwrapped, labels = func(wrapped_phase, **kwargs)
            else:
                unwrapped = func(wrapped_phase, **kwargs)

            elapsed = time.time() - start_time

            rmse, error_map = calculate_error(unwrapped, true_phase)

            low_coh_mask = coherence < 0.5
            low_coh_rmse = np.sqrt(np.mean(error_map[low_coh_mask]**2))
            high_coh_mask = coherence >= 0.5
            high_coh_rmse = np.sqrt(np.mean(error_map[high_coh_mask]**2))

            print(f"  Total RMSE: {rmse:.4f} rad")
            print(f"  Low coherence RMSE: {low_coh_rmse:.4f} rad")
            print(f"  High coherence RMSE: {high_coh_rmse:.4f} rad")
            print(f"  Time: {elapsed:.2f} s")

            results[name] = {
                'rmse': rmse,
                'low_rmse': low_coh_rmse,
                'high_rmse': high_coh_rmse,
                'time': elapsed,
                'unwrapped': unwrapped,
            }

            row = plot_idx // 3 + 1
            col = plot_idx % 3

            if plot_idx < 6:
                im = axes[row, col].imshow(unwrapped, cmap="jet")
                axes[row, col].set_title(f"{name}\nRMSE: {rmse:.3f}")
                plt.colorbar(im, ax=axes[row, col])

                if name == "Improved Branch Cut":
                    axes[row, col].contour(bc, colors='red', linewidths=0.5)

                plot_idx += 1

        except Exception as e:
            print(f"  FAILED: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")
    print(f"{'Algorithm':<25} {'Total RMSE':<15} {'Low RMSE':<15} {'Time(s)':<10}")
    print(f"{'-' * 70}")
    for name, res in sorted(results.items(), key=lambda x: x[1]['rmse']):
        print(f"{name:<25} {res['rmse']:<15.4f} {res['low_rmse']:<15.4f} {res['time']:<10.2f}")

    print(f"\nImprovement in low coherence regions:")
    if 'Original Branch Cut' in results and 'Improved Branch Cut' in results:
        orig_low = results['Original Branch Cut']['low_rmse']
        impr_low = results['Improved Branch Cut']['low_rmse']
        improvement = (orig_low - impr_low) / orig_low * 100
        print(f"  Branch Cut improvement: {improvement:.1f}%")

    plt.tight_layout()
    plt.savefig("low_coherence_comparison.png", dpi=150)
    print(f"\nVisualization saved to low_coherence_comparison.png")
    plt.close()

    print(f"\n{'=' * 70}")

    if results['Improved Branch Cut']['low_rmse'] < results['Original Branch Cut']['low_rmse']:
        print("SUCCESS: Improved algorithm performs better in low coherence regions!")
    else:
        print("Note: Results may vary depending on noise characteristics.")


if __name__ == "__main__":
    main()
