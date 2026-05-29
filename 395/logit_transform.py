import math
import warnings
import numpy as np

try:
    from scipy.stats import norm
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    warnings.warn("scipy未安装，Probit变换将使用近似算法", ImportWarning)

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    warnings.warn("matplotlib未安装，绘图功能不可用", ImportWarning)

EPS = 1e-12
MAX_LOGIT = math.log((1 - EPS) / EPS)


def _check_array_input(p):
    return isinstance(p, (np.ndarray, list, tuple))


def _to_numpy(p):
    if isinstance(p, (list, tuple)):
        return np.array(p, dtype=np.float64)
    return p


def logit_transform(p, clip=True):
    is_array = _check_array_input(p)
    p = _to_numpy(p)
    
    if is_array:
        if np.any((p < 0) | (p > 1)):
            raise ValueError("概率p必须在[0, 1]之间")
        
        if clip:
            mask = (p <= 0) | (p >= 1)
            if np.any(mask):
                warnings.warn(
                    f"输入概率包含超出(0, 1)范围的值，已裁剪至[{EPS}, {1-EPS}]",
                    UserWarning
                )
            p_clipped = np.clip(p, EPS, 1 - EPS)
            return np.log(p_clipped / (1 - p_clipped))
        else:
            result = np.empty_like(p, dtype=np.float64)
            mask_valid = (p > 0) & (p < 1)
            result[mask_valid] = np.log(p[mask_valid] / (1 - p[mask_valid]))
            result[p <= 0] = -np.inf
            result[p >= 1] = np.inf
            return result
    else:
        if not (0 <= p <= 1):
            raise ValueError("概率p必须在[0, 1]之间")
        
        if p <= 0 or p >= 1:
            if clip:
                original_p = p
                p = max(EPS, min(1 - EPS, p))
                warnings.warn(
                    f"输入概率p={original_p}超出(0, 1)范围，已裁剪至p={p:.12f}",
                    UserWarning
                )
            else:
                if p == 0:
                    return -float('inf')
                elif p == 1:
                    return float('inf')
        
        return math.log(p / (1 - p))


def sigmoid(logit_value, clip=True):
    is_array = _check_array_input(logit_value)
    logit_value = _to_numpy(logit_value)
    
    if is_array:
        if clip:
            logit_clipped = np.clip(logit_value, -MAX_LOGIT, MAX_LOGIT)
            return 1 / (1 + np.exp(-logit_clipped))
        else:
            result = np.empty_like(logit_value, dtype=np.float64)
            mask_low = logit_value < -MAX_LOGIT
            mask_high = logit_value > MAX_LOGIT
            mask_valid = ~mask_low & ~mask_high
            result[mask_valid] = 1 / (1 + np.exp(-logit_value[mask_valid]))
            result[mask_low] = 0.0
            result[mask_high] = 1.0
            return result
    else:
        if clip:
            logit_value = max(-MAX_LOGIT, min(MAX_LOGIT, logit_value))
        else:
            if logit_value > MAX_LOGIT:
                return 1.0
            elif logit_value < -MAX_LOGIT:
                return 0.0
        
        return 1 / (1 + math.exp(-logit_value))


def _probit_approx(p):
    a0 = 2.50662823884
    a1 = -18.61500062529
    a2 = 41.39119773534
    a3 = -25.44106049637
    b0 = -8.47351093090
    b1 = 23.08336743743
    b2 = -21.06224101826
    b3 = 3.13082909833
    c0 = 0.3374754822726147
    c1 = 0.9761690190917186
    c2 = 0.1607979714918209
    c3 = 0.0276438810333863
    c4 = 0.0038405729373609
    c5 = 0.0003951896511919
    c6 = 0.0000321767881768
    c7 = 0.0000002888167364
    c8 = 0.0000003960315187
    
    q = p - 0.5
    if abs(q) < 0.42:
        r = q * q
        return q * (((a3 * r + a2) * r + a1) * r + a0) / ((((b3 * r + b2) * r + b1) * r + b0) * r + 1.0)
    else:
        r = p if q < 0 else 1 - p
        r = math.log(-math.log(r))
        result = c0 + r * (c1 + r * (c2 + r * (c3 + r * (c4 + r * (c5 + r * (c6 + r * (c7 + r * c8)))))))
        return -result if q < 0 else result


def probit_transform(p, clip=True):
    is_array = _check_array_input(p)
    p = _to_numpy(p)
    
    if is_array:
        if np.any((p < 0) | (p > 1)):
            raise ValueError("概率p必须在[0, 1]之间")
        
        if clip:
            mask = (p <= 0) | (p >= 1)
            if np.any(mask):
                warnings.warn(
                    f"输入概率包含超出(0, 1)范围的值，已裁剪至[{EPS}, {1-EPS}]",
                    UserWarning
                )
            p_clipped = np.clip(p, EPS, 1 - EPS)
        else:
            p_clipped = p
        
        if SCIPY_AVAILABLE:
            return norm.ppf(p_clipped)
        else:
            vec_probit = np.vectorize(_probit_approx)
            return vec_probit(p_clipped)
    else:
        if not (0 <= p <= 1):
            raise ValueError("概率p必须在[0, 1]之间")
        
        if p <= 0 or p >= 1:
            if clip:
                original_p = p
                p = max(EPS, min(1 - EPS, p))
                warnings.warn(
                    f"输入概率p={original_p}超出(0, 1)范围，已裁剪至p={p:.12f}",
                    UserWarning
                )
            else:
                if p == 0:
                    return -float('inf')
                elif p == 1:
                    return float('inf')
        
        if SCIPY_AVAILABLE:
            return norm.ppf(p)
        else:
            return _probit_approx(p)


def probit_inv(z):
    is_array = _check_array_input(z)
    z = _to_numpy(z)
    
    if is_array:
        if SCIPY_AVAILABLE:
            return norm.cdf(z)
        else:
            vec_cdf = np.vectorize(lambda x: 0.5 * (1 + math.erf(x / math.sqrt(2))))
            return vec_cdf(z)
    else:
        if SCIPY_AVAILABLE:
            return norm.cdf(z)
        else:
            return 0.5 * (1 + math.erf(z / math.sqrt(2)))


def cloglog_transform(p, clip=True):
    is_array = _check_array_input(p)
    p = _to_numpy(p)
    
    if is_array:
        if np.any((p < 0) | (p > 1)):
            raise ValueError("概率p必须在[0, 1]之间")
        
        if clip:
            mask = (p <= 0) | (p >= 1)
            if np.any(mask):
                warnings.warn(
                    f"输入概率包含超出(0, 1)范围的值，已裁剪至[{EPS}, {1-EPS}]",
                    UserWarning
                )
            p_clipped = np.clip(p, EPS, 1 - EPS)
            return np.log(-np.log(1 - p_clipped))
        else:
            result = np.empty_like(p, dtype=np.float64)
            mask_valid = (p > 0) & (p < 1)
            result[mask_valid] = np.log(-np.log(1 - p[mask_valid]))
            result[p <= 0] = -np.inf
            result[p >= 1] = np.inf
            return result
    else:
        if not (0 <= p <= 1):
            raise ValueError("概率p必须在[0, 1]之间")
        
        if p <= 0 or p >= 1:
            if clip:
                original_p = p
                p = max(EPS, min(1 - EPS, p))
                warnings.warn(
                    f"输入概率p={original_p}超出(0, 1)范围，已裁剪至p={p:.12f}",
                    UserWarning
                )
            else:
                if p == 0:
                    return -float('inf')
                elif p == 1:
                    return float('inf')
        
        return math.log(-math.log(1 - p))


def cloglog_inv(y):
    is_array = _check_array_input(y)
    y = _to_numpy(y)
    
    if is_array:
        return 1 - np.exp(-np.exp(y))
    else:
        return 1 - math.exp(-math.exp(y))


def generate_distribution_data(n_samples=10000, seed=42):
    np.random.seed(seed)
    
    p_beta = np.random.beta(a=2, b=5, size=n_samples)
    p_uniform = np.random.uniform(0, 1, size=n_samples)
    p_bimodal = np.concatenate([
        np.random.beta(a=5, b=15, size=n_samples//2),
        np.random.beta(a=15, b=5, size=n_samples//2)
    ])
    
    datasets = {
        'Beta(2,5)': p_beta,
        'Uniform(0,1)': p_uniform,
        'Bimodal': p_bimodal
    }
    
    results = {}
    for name, p in datasets.items():
        results[name] = {
            'original': p,
            'logit': logit_transform(p, clip=True),
            'probit': probit_transform(p, clip=True),
            'cloglog': cloglog_transform(p, clip=True)
        }
    
    return results


def plot_distribution_comparison(data, save_path=None, show=True):
    if not MATPLOTLIB_AVAILABLE:
        warnings.warn("matplotlib未安装，无法绘图")
        return None
    
    n_datasets = len(data)
    transforms = ['original', 'logit', 'probit', 'cloglog']
    transform_names = ['Original (p)', 'Logit', 'Probit', 'Cloglog']
    
    fig, axes = plt.subplots(n_datasets, 4, figsize=(16, 4 * n_datasets))
    if n_datasets == 1:
        axes = axes.reshape(1, -1)
    
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']
    
    for i, (dataset_name, transform_data) in enumerate(data.items()):
        for j, (trans, trans_name) in enumerate(zip(transforms, transform_names)):
            ax = axes[i, j]
            values = transform_data[trans]
            
            valid_values = values[np.isfinite(values)]
            
            ax.hist(valid_values, bins=50, density=True, alpha=0.7, 
                   color=colors[j], edgecolor='black', linewidth=0.5)
            ax.set_title(f'{dataset_name} - {trans_name}', fontsize=10, fontweight='bold')
            ax.set_xlabel('Value', fontsize=9)
            ax.set_ylabel('Density', fontsize=9)
            ax.grid(True, alpha=0.3, linestyle='--')
            
            ax.axvline(np.mean(valid_values), color='red', linestyle='--', 
                      linewidth=1.5, label=f'Mean: {np.mean(valid_values):.2f}')
            ax.legend(fontsize=8)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"图表已保存至: {save_path}")
    
    if show:
        plt.show()
    
    return fig


if __name__ == "__main__":
    print("=" * 60)
    print("概率变换函数测试")
    print("=" * 60)
    
    test_probabilities = [0, 1e-15, 0.1, 0.3, 0.5, 0.7, 0.9, 1 - 1e-15, 1]
    
    print("\n1. 单值变换测试")
    print("-" * 60)
    
    for p in test_probabilities:
        print(f"\np = {p}")
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            logit_val = logit_transform(p, clip=True)
            probit_val = probit_transform(p, clip=True)
            cloglog_val = cloglog_transform(p, clip=True)
            if w:
                for warning in w:
                    print(f"  警告: {warning.message}")
        
        print(f"  Logit:   {logit_val:.6f} -> Sigmoid: {sigmoid(logit_val):.12f}")
        print(f"  Probit:  {probit_val:.6f} -> Inv:     {probit_inv(probit_val):.12f}")
        print(f"  Cloglog: {cloglog_val:.6f} -> Inv:     {cloglog_inv(cloglog_val):.12f}")
    
    print("\n2. 批量变换测试 (numpy数组)")
    print("-" * 60)
    
    p_array = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    print(f"输入数组: {p_array}")
    print(f"Logit:    {logit_transform(p_array)}")
    print(f"Probit:   {probit_transform(p_array)}")
    print(f"Cloglog:  {cloglog_transform(p_array)}")
    
    print("\n3. 分布数据生成与对比")
    print("-" * 60)
    
    dist_data = generate_distribution_data(n_samples=5000)
    for name, transforms in dist_data.items():
        print(f"\n{name} 分布:")
        for trans_name, values in transforms.items():
            valid = values[np.isfinite(values)]
            print(f"  {trans_name:8s}: mean={np.mean(valid):.4f}, std={np.std(valid):.4f}, "
                  f"min={np.min(valid):.4f}, max={np.max(valid):.4f}")
    
    print("\n4. 绘制分布对比图")
    print("-" * 60)
    
    if MATPLOTLIB_AVAILABLE:
        fig = plot_distribution_comparison(dist_data, save_path='distribution_comparison.png', show=False)
        print("图表已生成并保存为 distribution_comparison.png")
    else:
        print("matplotlib未安装，跳过绘图")
    
    print("\n" + "=" * 60)
    print("测试完成!")
    print("=" * 60)
