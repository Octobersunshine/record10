import numpy as np
from scipy.optimize import minimize_scalar
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm


KERNEL_REGISTRY = {}


def register_kernel(name):
    def decorator(fn):
        KERNEL_REGISTRY[name] = fn
        return fn
    return decorator


@register_kernel('gaussian')
def gaussian_kernel(x):
    return (1 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * x ** 2)


@register_kernel('epanechnikov')
def epanechnikov_kernel(x):
    result = np.zeros_like(x, dtype=float)
    mask = np.abs(x) <= 1
    result[mask] = 0.75 * (1 - x[mask] ** 2)
    return result


@register_kernel('triangular')
def triangular_kernel(x):
    result = np.zeros_like(x, dtype=float)
    mask = np.abs(x) <= 1
    result[mask] = 1 - np.abs(x[mask])
    return result


@register_kernel('uniform')
def uniform_kernel(x):
    result = np.zeros_like(x, dtype=float)
    mask = np.abs(x) <= 0.5
    result[mask] = 1.0
    return result


def get_kernel(name):
    if callable(name):
        return name
    if name not in KERNEL_REGISTRY:
        available = ', '.join(KERNEL_REGISTRY.keys())
        raise ValueError(f"Unknown kernel '{name}'. Available: {available}")
    return KERNEL_REGISTRY[name]


def _kernel_integral(kernel_name, t):
    kernel = get_kernel(kernel_name)
    u = np.linspace(-50, t, 10000)
    ku = kernel(u)
    return np.trapz(ku, u)


def scott_bandwidth(data):
    data = np.asarray(data)
    if data.ndim == 1:
        data = data.reshape(-1, 1)
    n, d = data.shape
    bandwidths = np.zeros(d)
    for j in range(d):
        std = np.std(data[:, j], ddof=1)
        bandwidths[j] = std * n ** (-1 / (d + 4))
    return bandwidths if d > 1 else bandwidths[0]


def silverman_bandwidth(data):
    data = np.asarray(data)
    if data.ndim == 1:
        data = data.reshape(-1, 1)
    n, d = data.shape
    bandwidths = np.zeros(d)
    for j in range(d):
        std = np.std(data[:, j], ddof=1)
        iqr = np.percentile(data[:, j], 75) - np.percentile(data[:, j], 25)
        sigma = min(std, iqr / 1.34)
        bandwidths[j] = sigma * (n ** (-1 / (d + 4))) * (4 / (d + 2)) ** (1 / (d + 4))
    return bandwidths if d > 1 else bandwidths[0]


def _kde_loo_density(data, h, kernel_name='gaussian'):
    data = np.asarray(data).flatten()
    n = len(data)
    kernel = get_kernel(kernel_name)
    loo_densities = np.zeros(n)
    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        x_i = data[i]
        data_without_i = data[mask]
        kernel_values = kernel((x_i - data_without_i) / h)
        loo_densities[i] = np.sum(kernel_values) / ((n - 1) * h)
    return loo_densities


def lcv_criterion(h, data, kernel_name='gaussian'):
    h = max(h, 1e-6)
    loo_densities = _kde_loo_density(data, h, kernel_name)
    log_densities = np.log(np.maximum(loo_densities, 1e-10))
    return -np.mean(log_densities)


def lcv_bandwidth(data, h_min=None, h_max=None, kernel_name='gaussian'):
    data = np.asarray(data).flatten()
    n = len(data)
    if n < 2:
        return 0.1
    h_silverman = silverman_bandwidth(data)
    if h_min is None:
        h_min = h_silverman * 0.2
    if h_max is None:
        h_max = h_silverman * 3.0
    result = minimize_scalar(
        lcv_criterion,
        args=(data, kernel_name),
        bounds=(h_min, h_max),
        method='bounded'
    )
    return max(result.x, 1e-6)


def _estimate_n_modes(data, h=None, threshold=0.01):
    data = np.asarray(data).flatten()
    if h is None:
        h = silverman_bandwidth(data)
    x_min, x_max = np.min(data) - 2 * h, np.max(data) + 2 * h
    x = np.linspace(x_min, x_max, 500)
    n = len(data)
    densities = np.zeros_like(x)
    for i, xi in enumerate(x):
        densities[i] = np.sum(gaussian_kernel((xi - data) / h)) / (n * h)
    kernel_std = h
    density_threshold = np.max(densities) * threshold
    peaks = []
    for i in range(1, len(densities) - 1):
        if densities[i] > densities[i-1] and densities[i] > densities[i+1]:
            if densities[i] > density_threshold:
                peaks.append((x[i], densities[i]))
    if len(peaks) < 2:
        return 1, peaks
    peaks.sort(key=lambda p: p[1], reverse=True)
    significant_peaks = [peaks[0]]
    for peak in peaks[1:]:
        is_significant = True
        for sp in significant_peaks:
            distance = abs(peak[0] - sp[0])
            if distance < 2 * kernel_std:
                is_significant = False
                break
        if is_significant:
            significant_peaks.append(peak)
    return len(significant_peaks), significant_peaks


def adaptive_bandwidth(data, method='auto'):
    data = np.asarray(data).flatten()
    n_modes, peaks = _estimate_n_modes(data)
    h_silverman = silverman_bandwidth(data)
    if method == 'auto':
        if n_modes >= 2:
            std = np.std(data, ddof=1)
            peak_positions = [p[0] for p in peaks]
            mode_spread = np.std(peak_positions) if len(peak_positions) > 1 else 0
            if mode_spread > 0.5 * std:
                return h_silverman * 1.3
            else:
                return h_silverman * 1.15
        else:
            return h_silverman
    elif method == 'conservative':
        return h_silverman * 1.2
    else:
        return h_silverman


def _resolve_1d_bandwidth(data, bandwidth, kernel_name='gaussian'):
    data = np.asarray(data).flatten()
    if isinstance(bandwidth, str):
        bw = bandwidth.lower()
        if bw == 'scott':
            return scott_bandwidth(data)
        elif bw == 'silverman':
            return silverman_bandwidth(data)
        elif bw == 'lcv':
            return lcv_bandwidth(data, kernel_name=kernel_name)
        elif bw == 'auto':
            return adaptive_bandwidth(data, method='auto')
        elif bw == 'conservative':
            return adaptive_bandwidth(data, method='conservative')
        else:
            raise ValueError(
                "bandwidth string must be 'scott', 'silverman', 'lcv', 'auto', or 'conservative'"
            )
    elif isinstance(bandwidth, (int, float)):
        h = float(bandwidth)
        if h <= 0:
            raise ValueError("bandwidth must be positive")
        return h
    else:
        raise ValueError("bandwidth must be a string or a positive numeric value")


def kernel_density_estimation(data, bandwidth='auto', kernel='gaussian',
                               num_samples=1000, return_cdf=False):
    data = np.asarray(data)
    if data.ndim > 1 and data.shape[1] > 1:
        return kde_nd(data, bandwidth=bandwidth, kernel=kernel,
                      num_samples=num_samples, return_cdf=return_cdf)

    data = data.flatten()
    kernel_fn = get_kernel(kernel)
    h = _resolve_1d_bandwidth(data, bandwidth, kernel)

    x_min = np.min(data) - 3 * h
    x_max = np.max(data) + 3 * h
    x_samples = np.linspace(x_min, x_max, num_samples)

    n = len(data)
    densities = np.zeros(num_samples)
    for i, x in enumerate(x_samples):
        kernel_values = kernel_fn((x - data) / h)
        densities[i] = np.sum(kernel_values) / (n * h)

    if return_cdf:
        cdf = np.cumsum(densities) * (x_samples[1] - x_samples[0])
        cdf = np.minimum(cdf, 1.0)
        return x_samples, densities, h, cdf

    return x_samples, densities, h


def kde_nd(data, bandwidth='scott', kernel='gaussian', num_samples=100,
            return_cdf=False):
    data = np.asarray(data)
    if data.ndim == 1:
        data = data.reshape(-1, 1)
    n, d = data.shape

    if d < 2 or d > 3:
        raise ValueError(f"kde_nd supports 2D or 3D data, got {d}D")

    kernel_fn = get_kernel(kernel)

    if isinstance(bandwidth, str):
        bw_method = bandwidth.lower()
        if bw_method in ('scott', 'silverman'):
            h_vec = scott_bandwidth(data) if bw_method == 'scott' else silverman_bandwidth(data)
        else:
            h_vec = scott_bandwidth(data)
    elif isinstance(bandwidth, (int, float)):
        h_vec = np.full(d, float(bandwidth))
    elif isinstance(bandwidth, (list, tuple, np.ndarray)):
        h_vec = np.asarray(bandwidth, dtype=float)
        if len(h_vec) != d:
            raise ValueError(f"bandwidth vector length {len(h_vec)} != data dimension {d}")
    else:
        raise ValueError("bandwidth must be a string, number, or array-like")

    grids = []
    for j in range(d):
        lo = np.min(data[:, j]) - 3 * h_vec[j]
        hi = np.max(data[:, j]) + 3 * h_vec[j]
        grids.append(np.linspace(lo, hi, num_samples))

    mesh = np.meshgrid(*grids, indexing='ij')
    grid_shape = mesh[0].shape
    grid_points = np.column_stack([m.ravel() for m in mesh])

    densities = np.zeros(len(grid_points))
    for i, pt in enumerate(grid_points):
        scaled = (pt - data) / h_vec
        k_vals = kernel_fn(scaled)
        prod_kernel = np.prod(k_vals, axis=1)
        densities[i] = np.sum(prod_kernel) / (n * np.prod(h_vec))

    density_grid = densities.reshape(grid_shape)

    result = {
        'grids': grids,
        'density': density_grid,
        'bandwidth': h_vec,
    }

    if return_cdf:
        if d == 2:
            dx = grids[0][1] - grids[0][0]
            dy = grids[1][1] - grids[1][0]
            cdf_grid = np.cumsum(np.cumsum(density_grid, axis=0), axis=1) * dx * dy
            cdf_grid = np.minimum(cdf_grid, 1.0)
        elif d == 3:
            dx = grids[0][1] - grids[0][0]
            dy = grids[1][1] - grids[1][0]
            dz = grids[2][1] - grids[2][0]
            cdf_grid = np.cumsum(
                np.cumsum(np.cumsum(density_grid, axis=0), axis=1), axis=2
            ) * dx * dy * dz
            cdf_grid = np.minimum(cdf_grid, 1.0)
        result['cdf'] = cdf_grid

    return result


def kde_cdf(data, bandwidth='auto', kernel='gaussian', num_samples=1000):
    data = np.asarray(data)
    if data.ndim > 1 and data.shape[1] > 1:
        raise ValueError("kde_cdf only supports 1D data. Use kde_nd with return_cdf=True for multi-dim.")
    data = data.flatten()

    result = kernel_density_estimation(data, bandwidth=bandwidth, kernel=kernel,
                                        num_samples=num_samples, return_cdf=True)
    x_samples, densities, h, cdf = result
    return x_samples, cdf, h


def count_local_maxima(x, densities):
    maxima_count = 0
    maxima_positions = []
    for i in range(1, len(densities) - 1):
        if densities[i] > densities[i-1] and densities[i] > densities[i+1]:
            left_slope = densities[i] - densities[i-1]
            right_slope = densities[i+1] - densities[i]
            if left_slope > 1e-6 or right_slope < -1e-6:
                maxima_count += 1
                maxima_positions.append(x[i])
    return maxima_count, maxima_positions


def plot_kde_1d(data, bandwidth='auto', kernel='gaussian', save_path='kde_1d.png'):
    x, density, h = kernel_density_estimation(data, bandwidth=bandwidth, kernel=kernel)
    x_c, cdf, _ = kde_cdf(data, bandwidth=bandwidth, kernel=kernel)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={'height_ratios': [2, 1]})

    ax1.hist(data, bins=50, density=True, alpha=0.3, color='gray', label='Histogram')
    ax1.plot(x, density, 'b-', linewidth=2, label=f'KDE ({kernel}, h={h:.3f})')
    ax1.set_ylabel('Density')
    ax1.set_title(f'KDE with {kernel} kernel (bandwidth={h:.4f})')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(x_c, cdf, 'r-', linewidth=2, label='CDF')
    ax2.set_xlabel('x')
    ax2.set_ylabel('CDF')
    ax2.set_title('Cumulative Distribution Function')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"1D KDE图已保存至: {save_path}")


def plot_kde_2d(data, bandwidth='scott', kernel='gaussian', save_path='kde_2d.png'):
    result = kde_nd(data, bandwidth=bandwidth, kernel=kernel, num_samples=80)
    X, Y = np.meshgrid(result['grids'][0], result['grids'][1], indexing='ij')
    Z = result['density']
    h = result['bandwidth']

    fig = plt.figure(figsize=(14, 6))

    ax1 = fig.add_subplot(121)
    c = ax1.contourf(X, Y, Z, levels=20, cmap='viridis')
    fig.colorbar(c, ax=ax1, label='Density')
    ax1.scatter(data[:, 0], data[:, 1], s=5, c='white', alpha=0.3)
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_title(f'2D KDE ({kernel} kernel, h={h})')
    ax1.set_aspect('equal')

    ax2 = fig.add_subplot(122, projection='3d')
    ax2.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8, edgecolor='none')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Density')
    ax2.set_title(f'2D KDE Surface ({kernel})')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"2D KDE图已保存至: {save_path}")


def plot_kde_kernels_comparison(data, bandwidth='auto', save_path='kde_kernels.png'):
    kernels = list(KERNEL_REGISTRY.keys())
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={'height_ratios': [2, 1]})

    for k_name in kernels:
        x, density, h = kernel_density_estimation(data, bandwidth=bandwidth, kernel=k_name)
        x_c, cdf, _ = kde_cdf(data, bandwidth=bandwidth, kernel=k_name)
        ax1.plot(x, density, linewidth=2, label=f'{k_name} (h={h:.3f})')
        ax2.plot(x_c, cdf, linewidth=2, label=f'{k_name}')

    ax1.hist(data, bins=50, density=True, alpha=0.2, color='gray')
    ax1.set_ylabel('Density')
    ax1.set_title('KDE with Different Kernels')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel('x')
    ax2.set_ylabel('CDF')
    ax2.set_title('CDF with Different Kernels')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"核函数对比图已保存至: {save_path}")


if __name__ == "__main__":
    print("=" * 70)
    print("测试1: 1D KDE - 多核函数对比")
    print("=" * 70)
    np.random.seed(42)
    data1d = np.concatenate([
        np.random.normal(-2, 0.5, 300),
        np.random.normal(2, 0.8, 500)
    ])

    for k_name in KERNEL_REGISTRY:
        x, density, h = kernel_density_estimation(data1d, bandwidth='auto', kernel=k_name)
        n_maxima, _ = count_local_maxima(x, density)
        print(f"  {k_name.upper():15s} -> 带宽: {h:.4f}, 峰数: {n_maxima}, 最大密度: {np.max(density):.4f}")

    print("\n" + "=" * 70)
    print("测试2: 1D CDF估计")
    print("=" * 70)
    x_cdf, cdf, h_cdf = kde_cdf(data1d, bandwidth='auto', kernel='gaussian')
    print(f"  CDF范围: [{cdf[0]:.6f}, {cdf[-1]:.6f}]")
    print(f"  CDF在x=0处: {np.interp(0, x_cdf, cdf):.4f}")

    print("\n" + "=" * 70)
    print("测试3: 2D KDE")
    print("=" * 70)
    np.random.seed(42)
    data2d = np.column_stack([
        np.concatenate([np.random.normal(-1, 0.5, 200), np.random.normal(1.5, 0.6, 300)]),
        np.concatenate([np.random.normal(0, 0.8, 200), np.random.normal(1, 0.5, 300)])
    ])
    print(f"  数据维度: {data2d.shape}")

    for k_name in ['gaussian', 'epanechnikov']:
        result2d = kde_nd(data2d, bandwidth='scott', kernel=k_name, num_samples=50)
        print(f"  {k_name.upper():15s} -> 密度网格: {result2d['density'].shape}, "
              f"带宽: {result2d['bandwidth']}, 最大密度: {np.max(result2d['density']):.4f}")

    result2d_cdf = kde_nd(data2d, bandwidth='scott', kernel='gaussian', num_samples=50, return_cdf=True)
    print(f"  CDF网格: {result2d_cdf['cdf'].shape}, CDF范围: [{result2d_cdf['cdf'][0,0]:.6f}, {result2d_cdf['cdf'][-1,-1]:.6f}]")

    print("\n" + "=" * 70)
    print("测试4: 3D KDE")
    print("=" * 70)
    np.random.seed(42)
    data3d = np.column_stack([
        np.random.normal(0, 1, 200),
        np.random.normal(0, 1, 200),
        np.random.normal(0, 1, 200)
    ])
    result3d = kde_nd(data3d, bandwidth='scott', kernel='gaussian', num_samples=30)
    print(f"  数据维度: {data3d.shape}")
    print(f"  密度网格: {result3d['density'].shape}")
    print(f"  带宽: {result3d['bandwidth']}")
    print(f"  最大密度: {np.max(result3d['density']):.4f}")

    result3d_cdf = kde_nd(data3d, bandwidth='scott', kernel='gaussian', num_samples=30, return_cdf=True)
    print(f"  CDF范围: [{result3d_cdf['cdf'][0,0,0]:.6f}, {result3d_cdf['cdf'][-1,-1,-1]:.6f}]")

    print("\n" + "=" * 70)
    print("生成可视化图...")
    print("=" * 70)
    try:
        plot_kde_1d(data1d, bandwidth='auto', kernel='gaussian', save_path='kde_1d.png')
        plot_kde_kernels_comparison(data1d, bandwidth='auto', save_path='kde_kernels.png')
        plot_kde_2d(data2d, bandwidth='scott', kernel='gaussian', save_path='kde_2d.png')
    except Exception as e:
        print(f"  可视化跳过: {e}")

    print("\n" + "=" * 70)
    print("使用示例:")
    print("=" * 70)
    print('''
# 1D KDE (多种核函数)
x, density, h = kernel_density_estimation(data, bandwidth='auto', kernel='gaussian')
x, density, h = kernel_density_estimation(data, bandwidth='auto', kernel='epanechnikov')

# 1D CDF
x, cdf, h = kde_cdf(data, bandwidth='auto', kernel='gaussian')

# 2D KDE
result = kde_nd(data_2d, bandwidth='scott', kernel='gaussian', num_samples=80)
result = kde_nd(data_2d, bandwidth='scott', kernel='gaussian', num_samples=80, return_cdf=True)

# 3D KDE
result = kde_nd(data_3d, bandwidth='scott', kernel='epanechnikov', num_samples=30)

# 可用核函数: gaussian, epanechnikov, triangular, uniform
# 可用带宽: 'scott', 'silverman', 'auto', 'conservative', 'lcv', 数值
''')
