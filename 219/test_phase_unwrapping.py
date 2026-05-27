import numpy as np
import matplotlib.pyplot as plt
from phase_unwrapping import (
    wrap,
    branch_cut_unwrap,
    least_squares_unwrap,
    quality_guided_unwrap,
    create_residues,
)


def generate_simulation_data(size=64, noise=0.0):
    x = np.linspace(-5, 5, size)
    y = np.linspace(-5, 5, size)
    X, Y = np.meshgrid(x, y)

    true_phase = X**2 + Y**2

    if noise > 0:
        true_phase += np.random.normal(0, noise, true_phase.shape)

    wrapped_phase = wrap(true_phase)

    return true_phase, wrapped_phase


def generate_ramp_phase(size=64, slope=1.0):
    x = np.linspace(0, size - 1, size)
    y = np.linspace(0, size - 1, size)
    X, Y = np.meshgrid(x, y)

    true_phase = slope * (X + Y)
    wrapped_phase = wrap(true_phase)

    return true_phase, wrapped_phase


def calculate_error(unwrapped, true_phase):
    diff = unwrapped - true_phase
    k = np.round(diff / (2 * np.pi))
    diff_corrected = diff - 2 * np.pi * k
    rmse = np.sqrt(np.mean(diff_corrected**2))
    return rmse


def test_branch_cut():
    print("Testing Branch Cut algorithm...")
    true_phase, wrapped_phase = generate_ramp_phase(size=32, slope=0.5)

    unwrapped = branch_cut_unwrap(wrapped_phase)

    error = calculate_error(unwrapped, true_phase)
    print(f"  RMSE: {error:.6f} rad")

    residues = create_residues(wrapped_phase)
    n_residues = np.sum(np.abs(residues))
    print(f"  Number of residues: {n_residues}")

    return error < 0.1


def test_least_squares():
    print("\nTesting Least Squares algorithm...")
    true_phase, wrapped_phase = generate_ramp_phase(size=32, slope=0.5)

    unwrapped = least_squares_unwrap(wrapped_phase)

    error = calculate_error(unwrapped, true_phase)
    print(f"  RMSE: {error:.6f} rad")

    return error < 0.1


def test_quality_guided():
    print("\nTesting Quality Guided algorithm...")
    true_phase, wrapped_phase = generate_ramp_phase(size=32, slope=0.5)

    unwrapped = quality_guided_unwrap(wrapped_phase)

    error = calculate_error(unwrapped, true_phase)
    print(f"  RMSE: {error:.6f} rad")

    return error < 0.1


def test_noisy_data():
    print("\nTesting with noisy data...")
    np.random.seed(42)
    true_phase, wrapped_phase = generate_simulation_data(size=32, noise=0.1)

    unwrapped_ls = least_squares_unwrap(wrapped_phase)
    error_ls = calculate_error(unwrapped_ls, true_phase)
    print(f"  Least Squares RMSE: {error_ls:.6f} rad")

    unwrapped_qg = quality_guided_unwrap(wrapped_phase)
    error_qg = calculate_error(unwrapped_qg, true_phase)
    print(f"  Quality Guided RMSE: {error_qg:.6f} rad")

    return error_ls < 3.0 and error_qg < 1.0


def visualize_results():
    print("\nGenerating visualization...")

    true_phase, wrapped_phase = generate_simulation_data(size=64, noise=0.05)

    unwrapped_bc = branch_cut_unwrap(wrapped_phase)
    unwrapped_ls = least_squares_unwrap(wrapped_phase)
    unwrapped_qg = quality_guided_unwrap(wrapped_phase)

    residues = create_residues(wrapped_phase)

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    im0 = axes[0, 0].imshow(true_phase, cmap="jet")
    axes[0, 0].set_title("True Phase")
    plt.colorbar(im0, ax=axes[0, 0])

    im1 = axes[0, 1].imshow(wrapped_phase, cmap="jet")
    axes[0, 1].set_title("Wrapped Phase")
    plt.colorbar(im1, ax=axes[0, 1])

    im2 = axes[0, 2].imshow(residues, cmap="bwr", vmin=-1, vmax=1)
    axes[0, 2].set_title("Residues")
    plt.colorbar(im2, ax=axes[0, 2])

    im3 = axes[1, 0].imshow(unwrapped_bc, cmap="jet")
    axes[1, 0].set_title("Branch Cut Unwrapped")
    plt.colorbar(im3, ax=axes[1, 0])

    im4 = axes[1, 1].imshow(unwrapped_ls, cmap="jet")
    axes[1, 1].set_title("Least Squares Unwrapped")
    plt.colorbar(im4, ax=axes[1, 1])

    im5 = axes[1, 2].imshow(unwrapped_qg, cmap="jet")
    axes[1, 2].set_title("Quality Guided Unwrapped")
    plt.colorbar(im5, ax=axes[1, 2])

    plt.tight_layout()
    plt.savefig("phase_unwrapping_results.png", dpi=150)
    print("  Visualization saved to phase_unwrapping_results.png")
    plt.close()


def main():
    print("=" * 60)
    print("Phase Unwrapping Algorithm Tests")
    print("=" * 60)

    all_passed = True

    try:
        all_passed &= test_branch_cut()
    except Exception as e:
        print(f"  Branch Cut test failed: {e}")
        all_passed = False

    try:
        all_passed &= test_least_squares()
    except Exception as e:
        print(f"  Least Squares test failed: {e}")
        all_passed = False

    try:
        all_passed &= test_quality_guided()
    except Exception as e:
        print(f"  Quality Guided test failed: {e}")
        all_passed = False

    try:
        all_passed &= test_noisy_data()
    except Exception as e:
        print(f"  Noisy data test failed: {e}")
        all_passed = False

    try:
        visualize_results()
    except Exception as e:
        print(f"  Visualization failed: {e}")

    print("\n" + "=" * 60)
    if all_passed:
        print("All tests PASSED!")
    else:
        print("Some tests FAILED!")
    print("=" * 60)


if __name__ == "__main__":
    main()
