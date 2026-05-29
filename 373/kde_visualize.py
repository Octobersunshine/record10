import numpy as np
import matplotlib.pyplot as plt
from kde import (
    kde, kde_cdf,
    scott_bandwidth, silverman_bandwidth, robust_bandwidth, cv_bandwidth,
    KERNEL_NAMES
)


def plot_kernels_comparison(data, title="Kernel Comparison", save_path=None):
    plt.figure(figsize=(12, 6))
    plt.hist(data, bins=50, density=True, alpha=0.2, color='gray', label='Histogram')
    
    styles = {'gaussian': 'b-', 'epanechnikov': 'g--', 'triangular': 'r-.'}
    labels = {'gaussian': 'Gaussian', 'epanechnikov': 'Epanechnikov', 'triangular': 'Triangular'}
    
    for kn in KERNEL_NAMES:
        x, density = kde(data, kernel=kn, bandwidth='silverman')
        plt.plot(x, density, styles[kn], linewidth=2, label=labels[kn])
    
    plt.xlabel('x')
    plt.ylabel('Density')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Chart saved: {save_path}")
    plt.show()


def plot_adaptive_vs_fixed(data, title="Adaptive vs Fixed Bandwidth", save_path=None):
    plt.figure(figsize=(12, 6))
    plt.hist(data, bins=50, density=True, alpha=0.2, color='gray', label='Histogram')
    
    x_fixed, d_fixed = kde(data, kernel='gaussian', bandwidth='silverman')
    x_adapt, d_adapt = kde(data, kernel='gaussian', adaptive=True, alpha=0.5)
    
    plt.plot(x_fixed, d_fixed, 'b-', linewidth=2, label=f'Fixed (Silverman)')
    plt.plot(x_adapt, d_adapt, 'r--', linewidth=2, label=f'Adaptive (alpha=0.5')
    
    plt.xlabel('x')
    plt.ylabel('Density')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Chart saved: {save_path}")
    plt.show()
    
    print(f"  Fixed peak: {np.max(d_fixed):.4f}")
    print(f"  Adaptive peak: {np.max(d_adapt):.4f}")


def plot_cdf_comparison(data, title="CDF Comparison", save_path=None):
    plt.figure(figsize=(12, 6))
    
    styles = {'gaussian': ('b-', 'Gaussian'), 
              'epanechnikov': ('g--', 'Epanechnikov'), 
              'triangular': ('r-.', 'Triangular')}
    
    for kn in KERNEL_NAMES:
        x, cdf = kde_cdf(data, kernel=kn, bandwidth='silverman')
        ls, label = styles[kn]
        plt.plot(x, cdf, ls, linewidth=2, label=label)
    
    sorted_data = np.sort(data)
    ecdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
    plt.step(sorted_data, ecdf, 'k:', linewidth=1, alpha=0.6, label='Empirical CDF')
    
    plt.xlabel('x')
    plt.ylabel('CDF')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(-0.05, 1.05)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Chart saved: {save_path}")
    plt.show()


if __name__ == "__main__":
    np.random.seed(42)
    
    data = np.concatenate([np.random.normal(-1.5, 0.5, 500), np.random.normal(2, 0.7, 800)])
    
    print("=" * 65)
    print("KDE Visualization Demo")
    print("=" * 65)
    
    print("\n[1] Bandwidth comparison:")
    h_s = silverman_bandwidth(data)
    h_r = robust_bandwidth(data)
    h_cv = cv_bandwidth(data)
    print(f"  Silverman: {h_s:.4f}")
    print(f"  Robust:    {h_r:.4f}")
    print(f"  LSCV:      {h_cv:.4f}")
    
    print("\n[2] Kernel functions:")
    for kn in KERNEL_NAMES:
        _, d = kde(data, kernel=kn, bandwidth='silverman')
        print(f"  {kn}: peak density = {np.max(d):.4f}")
    
    try:
        print("\n[3] Generating charts...")
        plot_kernels_comparison(data, "Kernel Functions Comparison", 'kernel_comparison.png')
        plot_adaptive_vs_fixed(data, "Fixed vs Adaptive Bandwidth KDE", 'adaptive_comparison.png')
        plot_cdf_comparison(data, "CDF by Different Kernels", 'cdf_comparison.png')
    except Exception as e:
        print(f"Visualization skipped: {e}")
    
    print("\n" + "=" * 65)
    print("Usage examples:")
    print("=" * 65)
    print("""
  # Different kernels:
  x, d = kde(data, kernel='epanechnikov')
  x, d = kde(data, kernel='triangular')
  
  # Adaptive bandwidth:
  x, d = kde(data, adaptive=True, alpha=0.5)
  
  # CDF:
  x, cdf = kde_cdf(data, kernel='gaussian')
  x, cdf = kde_cdf(data, adaptive=True)
""")
