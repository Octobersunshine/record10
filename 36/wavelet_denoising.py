import numpy as np
import pywt
from collections.abc import Callable


def shannon_entropy(x: np.ndarray) -> float:
    x_abs = np.abs(x) + 1e-12
    x_norm = x_abs / np.sum(x_abs)
    return -np.sum(x_norm * np.log2(x_norm))


def log_energy_entropy(x: np.ndarray) -> float:
    x_sq = x**2 + 1e-12
    return np.sum(np.log(x_sq))


def threshold_entropy(x: np.ndarray, threshold: float = 0.1) -> float:
    return np.sum(np.abs(x) > threshold)


def sure_entropy(x: np.ndarray, sigma: float = 1.0) -> float:
    n = len(x)
    x_sorted = np.sort(np.abs(x))
    x_sq = x_sorted**2
    sure = n - 2 * np.sum(x_sq <= sigma**2) + np.sum(x_sq[x_sq <= sigma**2]) / sigma**2
    return sure


COST_FUNCTIONS = {
    'shannon': shannon_entropy,
    'log_energy': log_energy_entropy,
    'threshold': threshold_entropy,
    'sure': sure_entropy
}


def smooth_threshold(x, threshold, method='garrote', alpha=1.0):
    if method == 'garrote':
        abs_x = np.abs(x)
        mask = abs_x >= threshold
        result = np.zeros_like(x)
        result[mask] = np.sign(x[mask]) * (abs_x[mask] - threshold**2 / abs_x[mask])
        return result
    elif method == 'sigmoid':
        return x * (1 / (1 + np.exp(-alpha * (np.abs(x) - threshold))))
    elif method == 'improved_soft':
        abs_x = np.abs(x)
        t = threshold
        mask1 = abs_x >= 2 * t
        mask2 = (abs_x < 2 * t) & (abs_x >= t)
        mask3 = abs_x < t
        
        result = np.zeros_like(x)
        result[mask1] = np.sign(x[mask1]) * (abs_x[mask1] - t)
        result[mask2] = np.sign(x[mask2]) * (
            (abs_x[mask2] - t)**2 / (2 * t)
        )
        return result
    else:
        return pywt.threshold(x, threshold, mode='soft')


def wavelet_denoise(noisy_signal, wavelet='db4', level=3, mode='symmetric', 
                   threshold_method='garrote', alpha=1.0):
    coeffs = pywt.wavedec(noisy_signal, wavelet, level=level, mode=mode)
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745
    threshold = sigma * np.sqrt(2 * np.log(len(noisy_signal)))
    
    coeffs_thresh = [coeffs[0]]
    for c in coeffs[1:]:
        coeffs_thresh.append(smooth_threshold(c, threshold, method=threshold_method, alpha=alpha))
    
    denoised_signal = pywt.waverec(coeffs_thresh, wavelet, mode=mode)
    return denoised_signal[:len(noisy_signal)]


def wavelet_packet_denoise(noisy_signal, wavelet='db4', max_level=4, mode='symmetric',
                           threshold_method='garrote', alpha=1.0,
                           cost_function='shannon', 
                           custom_cost: Callable[[np.ndarray], float] = None):
    wp = pywt.WaveletPacket(data=noisy_signal, wavelet=wavelet, mode=mode, maxlevel=max_level)
    
    if custom_cost is not None:
        cost_func = custom_cost
    else:
        cost_func = COST_FUNCTIONS.get(cost_function, shannon_entropy)
    
    def node_cost(node):
        if node.data is None or len(node.data) == 0:
            return 0
        return cost_func(np.abs(node.data))
    
    wp.evaluate(node_cost, 'optimal')
    
    best_basis = wp.get_level(max_level, 'natural')
    
    sigma = np.median(np.abs(best_basis[-1].data)) / 0.6745 if best_basis[-1].data is not None else 0.1
    threshold = sigma * np.sqrt(2 * np.log(len(noisy_signal)))
    
    for node in best_basis:
        if node.data is not None:
            node.data = smooth_threshold(node.data, threshold, method=threshold_method, alpha=alpha)
    
    denoised_signal = wp.reconstruct(update=True)
    return denoised_signal[:len(noisy_signal)], wp


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    
    t = np.linspace(0, 1, 1000)
    clean_signal = np.sin(2 * np.pi * 5 * t) + 0.5 * np.sin(2 * np.pi * 10 * t)
    noise = np.random.normal(0, 0.3, len(t))
    noisy_signal = clean_signal + noise
    
    dwt_result = wavelet_denoise(noisy_signal, threshold_method='garrote')
    
    cost_functions = ['shannon', 'log_energy', 'threshold', 'sure']
    wpt_results = {}
    wp_objects = {}
    
    for cost in cost_functions:
        wpt_results[cost], wp_objects[cost] = wavelet_packet_denoise(
            noisy_signal, max_level=4, threshold_method='garrote', cost_function=cost
        )
    
    plt.figure(figsize=(16, 12))
    
    plt.subplot(4, 2, 1)
    plt.plot(t, clean_signal)
    plt.title('Clean Signal')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(4, 2, 2)
    plt.plot(t, noisy_signal)
    plt.title('Noisy Signal')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(4, 2, 3)
    plt.plot(t, dwt_result)
    plt.title('DWT Denoised (Garrote)')
    plt.grid(True, alpha=0.3)
    
    for i, cost in enumerate(cost_functions):
        plt.subplot(4, 2, i + 5)
        plt.plot(t, wpt_results[cost])
        plt.title(f'WPT Denoised ({cost} entropy)')
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    def my_custom_cost(x: np.ndarray) -> float:
        return np.sum(np.log1p(np.abs(x)))
    
    custom_result, wp_custom = wavelet_packet_denoise(
        noisy_signal, max_level=4, threshold_method='garrote',
        cost_function='shannon',
        custom_cost=my_custom_cost
    )
    
    print("=== 方法说明 ===")
    print("DWT: 离散小波变换 - 仅分解低频分量，频带划分粗糙")
    print("WPT: 小波包变换 - 同时分解高低频，频带划分更精细")
    print("\n=== 代价函数说明 ===")
    print("shannon: 香农熵 - 衡量信息含量，选择最能集中能量的基")
    print("log_energy: 对数能量熵 - 强调大系数贡献")
    print("threshold: 阈值熵 - 计算超过阈值的系数数量")
    print("sure: Stein无偏风险估计 - 最小化估计风险")
    print("\n=== 自定义代价函数 ===")
    print("可通过 custom_cost 参数传入自定义函数，输入为系数数组，返回代价值")
    print("示例: def my_cost(x): return np.sum(np.log1p(np.abs(x)))")
    
    mse_dwt = np.mean((clean_signal - dwt_result)**2)
    print(f"\n=== 均方误差 (MSE) ===")
    print(f"DWT: {mse_dwt:.6f}")
    for cost in cost_functions:
        mse = np.mean((clean_signal - wpt_results[cost])**2)
        print(f"WPT ({cost}): {mse:.6f}")
