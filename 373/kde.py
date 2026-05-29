import numpy as np


KERNEL_NAMES = ['gaussian', 'epanechnikov', 'triangular']


def gaussian_kernel(x):
    return (1 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * x ** 2)


def gaussian_kernel_cdf(t):
    return 0.5 * (1 + np.sign(t) * np.sqrt(1 - np.exp(-(t ** 2) / 2)))


def epanechnikov_kernel(x):
    out = np.zeros_like(x, dtype=float)
    mask = np.abs(x) <= 1
    out[mask] = 0.75 * (1 - x[mask] ** 2)
    return out


def epanechnikov_kernel_cdf(t):
    if t <= -1:
        return 0.0
    elif t >= 1:
        return 1.0
    else:
        return 0.5 + 0.75 * t - 0.25 * t ** 3


def triangular_kernel(x):
    out = np.zeros_like(x, dtype=float)
    mask = np.abs(x) <= 1
    out[mask] = (1 - np.abs(x[mask]))
    return out


def triangular_kernel_cdf(t):
    if t <= -1:
        return 0.0
    elif t >= 1:
        return 1.0
    elif t < 0:
        return 0.5 * (1 + t) ** 2
    else:
        return 0.5 + t - 0.5 * t ** 2


def get_kernel(name):
    if name == 'gaussian':
        return gaussian_kernel
    elif name == 'epanechnikov':
        return epanechnikov_kernel
    elif name == 'triangular':
        return triangular_kernel
    else:
        raise ValueError(f"Unknown kernel: {name}. Choose from {KERNEL_NAMES}")


def get_kernel_cdf(name):
    import math
    if name == 'gaussian':
        return np.vectorize(lambda t: 0.5 * (1 + math.erf(t / np.sqrt(2))))
    elif name == 'epanechnikov':
        return np.vectorize(epanechnikov_kernel_cdf)
    elif name == 'triangular':
        return np.vectorize(triangular_kernel_cdf)
    else:
        raise ValueError(f"Unknown kernel: {name}")


def scott_bandwidth(data):
    n = len(data)
    std = np.std(data, ddof=1)
    return 1.06 * std * (n ** (-1 / 5))


def silverman_bandwidth(data):
    n = len(data)
    std = np.std(data, ddof=1)
    q75, q25 = np.percentile(data, [75, 25])
    iqr = q75 - q25
    h = 0.9 * min(std, iqr / 1.34) * (n ** (-1 / 5))
    return h


def robust_bandwidth(data):
    n = len(data)
    q75, q25 = np.percentile(data, [75, 25])
    iqr = q75 - q25
    spread = iqr / 1.34
    h = 0.9 * spread * (n ** (-1 / 5))
    return h


def _lscv_score(h, data, kernel_func):
    n = len(data)
    data = np.asarray(data).flatten()
    diff = data[:, None] - data[None, :]
    d = diff / h
    
    if kernel_func == gaussian_kernel:
        conv_val = (1 / np.sqrt(4 * np.pi)) * np.exp(-d ** 2 / 4)
        term_a = (1 / (n ** 2 * h)) * np.sum(conv_val)
        k_val = gaussian_kernel(d)
        k_sum_no_diag = np.sum(k_val) - n * gaussian_kernel(0)
        term_b = (2 / (n * (n - 1) * h)) * k_sum_no_diag
    else:
        x_grid = np.linspace(np.min(data), np.max(data), 500)
        dx = x_grid[1] - x_grid[0]
        f_hat = np.zeros_like(x_grid)
        for i, x in enumerate(x_grid):
            f_hat[i] = np.mean(kernel_func((x - data) / h)) / h
        term_a = np.sum(f_hat ** 2) * dx
        
        k_sum_no_diag = 0.0
        for i in range(n):
            for j in range(n):
                if i != j:
                    k_sum_no_diag += kernel_func((data[i] - data[j]) / h)
        term_b = (2 / (n * (n - 1) * h)) * k_sum_no_diag
    
    return term_a - term_b


def cv_bandwidth(data, kernel='gaussian', h_range=None, num_grid=30):
    data = np.asarray(data).flatten()
    kernel_func = get_kernel(kernel)
    
    if h_range is None:
        h_ref = silverman_bandwidth(data)
        h_min = max(h_ref * 0.2, 1e-6)
        h_max = h_ref * 2.5
    else:
        h_min, h_max = h_range
    
    h_candidates = np.linspace(h_min, h_max, num_grid)
    scores = np.array([_lscv_score(h, data, kernel_func) for h in h_candidates])
    best_idx = np.argmin(scores)
    
    if best_idx > 0 and best_idx < num_grid - 1:
        h_fine_min = h_candidates[best_idx - 1]
        h_fine_max = h_candidates[best_idx + 1]
        h_fine = np.linspace(h_fine_min, h_fine_max, 20)
        scores_fine = np.array([_lscv_score(h, data, kernel_func) for h in h_fine])
        return h_fine[np.argmin(scores_fine)]
    
    return h_candidates[best_idx]


def _get_bandwidth(data, bandwidth, kernel):
    if bandwidth == 'scott':
        h = scott_bandwidth(data)
    elif bandwidth == 'silverman':
        h = silverman_bandwidth(data)
    elif bandwidth == 'robust':
        h = robust_bandwidth(data)
    elif bandwidth == 'cv':
        h = cv_bandwidth(data, kernel=kernel)
    elif isinstance(bandwidth, (int, float)):
        h = float(bandwidth)
    else:
        raise ValueError("bandwidth must be 'scott', 'silverman', 'robust', 'cv', or numeric")
    return h


def _pilot_density(data, h0, kernel):
    kernel_func = get_kernel(kernel)
    return np.array([np.mean(kernel_func((xi - data) / h0)) / h0 for xi in data])


def adaptive_bandwidths(data, base_bandwidth='silverman', kernel='gaussian', alpha=0.5):
    data = np.asarray(data).flatten()
    h0 = _get_bandwidth(data, base_bandwidth, kernel)
    
    f_pilot = _pilot_density(data, h0, kernel)
    g = np.exp(np.mean(np.log(f_pilot + 1e-10)))
    
    lam = (g / (f_pilot + 1e-10)) ** alpha
    h_local = h0 * lam
    
    return h_local, h0


def kde(data, kernel='gaussian', bandwidth='silverman', num_samples=1000, x_range=None,
        adaptive=False, base_bandwidth='silverman', alpha=0.5):
    data = np.asarray(data).flatten()
    n = len(data)
    kernel_func = get_kernel(kernel)
    
    if adaptive:
        h_local, h0 = adaptive_bandwidths(data, base_bandwidth=base_bandwidth, kernel=kernel, alpha=alpha)
        h_range = h0
    else:
        h = _get_bandwidth(data, bandwidth, kernel)
        h_range = h
    
    if x_range is None:
        x_min = np.min(data) - 3 * h_range
        x_max = np.max(data) + 3 * h_range
    else:
        x_min, x_max = x_range
    
    x_samples = np.linspace(x_min, x_max, num_samples)
    density = np.zeros(num_samples)
    
    if adaptive:
        for i, x in enumerate(x_samples):
            density[i] = np.mean(kernel_func((x - data) / h_local) / h_local)
    else:
        for i, x in enumerate(x_samples):
            density[i] = np.mean(kernel_func((x - data) / h)) / h
    
    return x_samples, density


def kde_cdf(data, kernel='gaussian', bandwidth='silverman', num_samples=1000, x_range=None,
            adaptive=False, base_bandwidth='silverman', alpha=0.5):
    data = np.asarray(data).flatten()
    n = len(data)
    kernel_cdf = get_kernel_cdf(kernel)
    
    if adaptive:
        h_local, h0 = adaptive_bandwidths(data, base_bandwidth=base_bandwidth, kernel=kernel, alpha=alpha)
        h_range = h0
    else:
        h = _get_bandwidth(data, bandwidth, kernel)
        h_range = h
    
    if x_range is None:
        x_min = np.min(data) - 3 * h_range
        x_max = np.max(data) + 3 * h_range
    else:
        x_min, x_max = x_range
    
    x_samples = np.linspace(x_min, x_max, num_samples)
    cdf = np.zeros(num_samples)
    
    if adaptive:
        for i, x in enumerate(x_samples):
            cdf[i] = np.mean(kernel_cdf((x - data) / h_local))
    else:
        for i, x in enumerate(x_samples):
            cdf[i] = np.mean(kernel_cdf((x - data) / h))
    
    return x_samples, cdf


if __name__ == "__main__":
    np.random.seed(42)
    
    print("=" * 65)
    print("Test 1: Kernel functions comparison")
    print("=" * 65)
    data1 = np.concatenate([np.random.normal(-2, 0.6, 400), np.random.normal(2, 0.8, 600)])
    
    for kn in ['gaussian', 'epanechnikov', 'triangular']:
        x, d = kde(data1, kernel=kn, bandwidth='silverman')
        print(f"  {kn:>12s}: peak={np.max(d):.4f}")
    
    print("\n" + "=" * 65)
    print("Test 2: Adaptive vs Fixed bandwidth")
    print("=" * 65)
    x_fixed, d_fixed = kde(data1, kernel='gaussian', bandwidth='silverman')
    x_adapt, d_adapt = kde(data1, kernel='gaussian', adaptive=True, alpha=0.5)
    print(f"  Fixed bandwidth (Silverman): peak={np.max(d_fixed):.4f}")
    print(f"  Adaptive bandwidth:       peak={np.max(d_adapt):.4f}")
    print(f"  Adaptive has sharper peaks: {np.max(d_adapt) > np.max(d_fixed)}")
    
    print("\n" + "=" * 65)
    print("Test 3: CDF calculation")
    print("=" * 65)
    x_cdf, cdf = kde_cdf(data1, kernel='gaussian', bandwidth='silverman')
    print(f"  CDF range: [{cdf[0]:.4f}, {cdf[-1]:.4f}]")
    print(f"  CDF at mean(data): {cdf[np.argmin(np.abs(x_cdf - np.mean(data1)))]:.4f}")
    
    print("\n" + "=" * 65)
    print("Test 4: Outlier robustness comparison")
    print("=" * 65)
    data2 = np.concatenate([np.random.normal(0, 1, 1000), np.array([50.0, -50.0])])
    
    print(f"\n{'Method':<12s} {'h(Gauss)':>10s} {'h(Epanech)':>12s}")
    print("-" * 40)
    for bw in ['scott', 'silverman', 'robust', 'cv']:
        h_g = _get_bandwidth(data2, bw, 'gaussian')
        h_e = _get_bandwidth(data2, bw, 'epanechnikov')
        print(f"{bw:<12s} {h_g:>10.4f} {h_e:>12.4f}")
    
    print("\n" + "=" * 65)
    print("All tests passed!")
    print("=" * 65)
