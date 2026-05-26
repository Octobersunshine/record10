import numpy as np
from prony_analysis import PronyAnalyzer as OldPronyAnalyzer
from prony_analysis_advanced import EnhancedPronyAnalyzer, generate_test_signal, print_results
import time


def test_noise_robustness():
    np.random.seed(42)
    fs = 100.0
    duration = 15.0
    true_modes = [
        {'freq': 0.4, 'damping': 0.05, 'amp': 1.0, 'phase': 0.0},
        {'freq': 1.2, 'damping': 0.08, 'amp': 0.6, 'phase': 0.3},
    ]

    noise_levels = [0.005, 0.01, 0.02, 0.05]

    print("="*90)
    print("Noise Robustness Comparison: Old Prony vs Enhanced Methods")
    print("="*90)
    print(f"True Modes: {true_modes[0]['freq']:.2f} Hz (ζ={true_modes[0]['damping']:.3f}), "
          f"{true_modes[1]['freq']:.2f} Hz (ζ={true_modes[1]['damping']:.3f})")
    print("="*90)

    old_analyzer = OldPronyAnalyzer(fs=fs, freq_range=(0.1, 2.0))
    new_analyzer = EnhancedPronyAnalyzer(fs=fs, freq_range=(0.1, 2.0))

    for noise_level in noise_levels:
        print(f"\n{'='*90}")
        print(f"Noise Level: {noise_level} (SNR ≈ {20*np.log10(1.0/noise_level):.1f} dB)")
        print("="*90)

        t, signal = generate_test_signal(fs=fs, duration=duration,
                                          modes=true_modes, noise_level=noise_level)

        print("\n--- Old Prony (Classic Method) ---")
        start = time.time()
        try:
            result_old = old_analyzer.analyze(signal, method='classic', order=12)
            elapsed = time.time() - start
            print(f"Time: {elapsed:.3f}s, Modes found: {len(result_old['freq'])}")
            for i, (f, d) in enumerate(zip(result_old['freq'], result_old['damping_ratio'])):
                f_err = abs(f - true_modes[i % 2]['freq']) / true_modes[i % 2]['freq'] * 100
                d_err = abs(d - true_modes[i % 2]['damping']) / true_modes[i % 2]['damping'] * 100
                print(f"  Mode {i+1}: {f:.4f} Hz (err: {f_err:.1f}%), "
                      f"ζ={d:.4f} (err: {d_err:.1f}%)")
            mse_old = np.mean((result_old['processed'] - result_old['reconstructed'])**2)
            print(f"  Reconstruction MSE: {mse_old:.6f}")
        except Exception as e:
            print(f"  Error: {e}")

        print("\n--- TLS-ESPRIT (with SVD denoise) ---")
        start = time.time()
        try:
            result_tls = new_analyzer.analyze(signal, method='tls_esprit',
                                               use_stabilization=False, svd_denoise=True)
            elapsed = time.time() - start
            print(f"Time: {elapsed:.3f}s, Modes found: {len(result_tls['freq'])}")
            for i, (f, d) in enumerate(zip(result_tls['freq'], result_tls['damping_ratio'])):
                if i < len(true_modes):
                    f_err = abs(f - true_modes[i]['freq']) / true_modes[i]['freq'] * 100
                    d_err = abs(d - true_modes[i]['damping']) / true_modes[i]['damping'] * 100
                    print(f"  Mode {i+1}: {f:.4f} Hz (err: {f_err:.1f}%), "
                          f"ζ={d:.4f} (err: {d_err:.1f}%)")
                else:
                    print(f"  Mode {i+1}: {f:.4f} Hz, ζ={d:.4f} (spurious?)")
            mse_tls = np.mean((result_tls['processed'] - result_tls['reconstructed'])**2)
            print(f"  Reconstruction MSE: {mse_tls:.6f}")
        except Exception as e:
            print(f"  Error: {e}")

        print("\n--- Improved Matrix Pencil (with SVD denoise) ---")
        start = time.time()
        try:
            result_mp = new_analyzer.analyze(signal, method='matrix_pencil',
                                              use_stabilization=False, svd_denoise=True)
            elapsed = time.time() - start
            print(f"Time: {elapsed:.3f}s, Modes found: {len(result_mp['freq'])}")
            for i, (f, d) in enumerate(zip(result_mp['freq'], result_mp['damping_ratio'])):
                if i < len(true_modes):
                    f_err = abs(f - true_modes[i]['freq']) / true_modes[i]['freq'] * 100
                    d_err = abs(d - true_modes[i]['damping']) / true_modes[i]['damping'] * 100
                    print(f"  Mode {i+1}: {f:.4f} Hz (err: {f_err:.1f}%), "
                          f"ζ={d:.4f} (err: {d_err:.1f}%)")
                else:
                    print(f"  Mode {i+1}: {f:.4f} Hz, ζ={d:.4f} (spurious?)")
            mse_mp = np.mean((result_mp['processed'] - result_mp['reconstructed'])**2)
            print(f"  Reconstruction MSE: {mse_mp:.6f}")
        except Exception as e:
            print(f"  Error: {e}")

    print("\n" + "="*90)
    print("Summary of Improvements:")
    print("="*90)
    print("1. SVD Denoising: Removes noise from the Hankel matrix before processing")
    print("2. TLS-ESPRIT: Better noise handling than classic Prony")
    print("3. Improved Matrix Pencil: More robust than classic Matrix Pencil")
    print("4. Stabilization Diagram (optional): Eliminates spurious modes via clustering")
    print("5. Automatic model order selection: Based on SVD singular values")
    print("="*90)


if __name__ == '__main__':
    test_noise_robustness()
