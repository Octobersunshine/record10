import numpy as np
from scipy.sparse import spdiags, eye
from scipy.sparse.linalg import spsolve
from scipy.signal import savgol_filter, find_peaks


def asls_baseline(y, lam=1e6, p=0.001, niter=100, tol=1e-6):
    """
    非对称最小二乘（AsLS）基线校正

    参数:
        y: 拉曼光谱强度数组 (1D numpy array)
        lam: 平滑参数 (lambda), 通常在1e2到1e9之间
        p: 非对称权重参数, 通常在0.001到0.01之间
        niter: 最大迭代次数
        tol: 收敛阈值

    返回:
        z: 拟合的基线
        y_corrected: 扣除基线后的光谱 (y - z)
    """
    y = np.asarray(y)
    n = len(y)

    e = eye(n, format='csr')
    d = np.diff(e, n=2, axis=0)

    w = np.ones(n)
    z_old = np.zeros(n)

    for i in range(niter):
        W = spdiags(w, 0, n, n)
        Z = W + lam * d.T @ d
        z = spsolve(Z, w * y)

        w_new = p * (y > z) + (1 - p) * (y <= z)

        if np.max(np.abs(z - z_old)) < tol:
            break

        w = w_new
        z_old = z.copy()

    y_corrected = y - z
    return z, y_corrected


def asls_baseline_2d(spectra, lam=1e6, p=0.001, niter=100, tol=1e-6):
    """
    对二维光谱数组（每行是一条光谱）进行AsLS基线校正

    参数:
        spectra: 二维光谱数组 (n_samples x n_wavenumbers)
        lam: 平滑参数
        p: 非对称权重参数
        niter: 最大迭代次数
        tol: 收敛阈值

    返回:
        baselines: 拟合的基线数组
        corrected_spectra: 扣除基线后的光谱数组
    """
    spectra = np.asarray(spectra)
    if spectra.ndim == 1:
        baseline, corrected = asls_baseline(spectra, lam, p, niter, tol)
        return baseline.reshape(1, -1), corrected.reshape(1, -1)

    n_samples, n_wavenumbers = spectra.shape
    baselines = np.zeros_like(spectra)
    corrected_spectra = np.zeros_like(spectra)

    for i in range(n_samples):
        baselines[i], corrected_spectra[i] = asls_baseline(spectra[i], lam, p, niter, tol)

    return baselines, corrected_spectra


def detect_raman_peaks(y_corrected, x=None, smooth_window=7, poly_order=2,
                       height_threshold=None, prominence=0.5, width_range=(1, 50)):
    """
    检测拉曼峰并返回峰参数

    参数:
        y_corrected: 基线校正后的光谱强度
        x: 波数/波长数组 (可选)，如果为None则使用索引
        smooth_window: Savitzky-Golay平滑窗口大小（奇数）
        poly_order: Savitzky-Golay多项式阶数
        height_threshold: 峰高阈值，默认为噪声水平的3倍
        prominence: 峰突出度阈值
        width_range: 峰宽范围 (min_width, max_width)

    返回:
        peaks: 字典数组，每个字典包含峰参数:
            - position: 峰位置（x坐标或索引）
            - intensity: 峰强度
            - index: 峰位置的索引
            - fwhm: 半高宽
            - left_base: 左基线位置索引
            - right_base: 右基线位置索引
    """
    y_corrected = np.asarray(y_corrected)
    n = len(y_corrected)

    if x is None:
        x = np.arange(n)

    y_smooth = savgol_filter(y_corrected, smooth_window, poly_order)

    noise = np.std(y_smooth[:min(100, n//10)])
    if height_threshold is None:
        height_threshold = 3 * noise

    peaks_idx, properties = find_peaks(
        y_smooth,
        height=height_threshold,
        prominence=prominence,
        width=width_range
    )

    peak_heights = properties['peak_heights']
    widths = properties['widths']
    left_bases = properties['left_bases']
    right_bases = properties['right_bases']

    peaks = []
    for i, peak_idx in enumerate(peaks_idx):
        peak_position = x[peak_idx]
        peak_intensity = peak_heights[i]
        fwhm = widths[i] * (x[1] - x[0]) if len(x) > 1 else widths[i]

        peaks.append({
            'index': int(peak_idx),
            'position': float(peak_position),
            'intensity': float(peak_intensity),
            'fwhm': float(fwhm),
            'left_base': int(left_bases[i]),
            'right_base': int(right_bases[i])
        })

    peaks.sort(key=lambda p: p['position'])

    return peaks


def detect_peaks_second_derivative(y_corrected, x=None, smooth_window=7, poly_order=2,
                                   threshold=-0.1, min_distance=5):
    """
    通过二阶导数找谷的方法检测拉曼峰

    参数:
        y_corrected: 基线校正后的光谱强度
        x: 波数/波长数组 (可选)
        smooth_window: 平滑窗口大小
        poly_order: 多项式阶数
        threshold: 二阶导数谷值阈值（负数，越负表示谷越深）
        min_distance: 峰之间最小距离（点数）

    返回:
        peaks: 字典数组，每个字典包含峰参数
    """
    y_corrected = np.asarray(y_corrected)
    n = len(y_corrected)

    if x is None:
        x = np.arange(n)

    y_smooth = savgol_filter(y_corrected, smooth_window, poly_order)

    second_deriv = savgol_filter(y_smooth, smooth_window, poly_order, deriv=2)

    valleys_idx, _ = find_peaks(-second_deriv, height=-threshold, distance=min_distance)

    peaks = []
    dx = x[1] - x[0] if len(x) > 1 else 1

    for valley_idx in valleys_idx:
        left_idx = max(0, valley_idx - 10)
        right_idx = min(n - 1, valley_idx + 10)

        peak_idx = np.argmax(y_smooth[left_idx:right_idx + 1]) + left_idx

        peak_height = y_smooth[peak_idx]

        half_height = peak_height / 2

        left_hm = peak_idx
        while left_hm > 0 and y_smooth[left_hm] > half_height:
            left_hm -= 1

        right_hm = peak_idx
        while right_hm < n - 1 and y_smooth[right_hm] > half_height:
            right_hm += 1

        fwhm = (right_hm - left_hm) * dx

        if second_deriv[valley_idx] < 0:
            peaks.append({
                'index': int(peak_idx),
                'position': float(x[peak_idx]),
                'intensity': float(peak_height),
                'fwhm': float(fwhm),
                'valley_index': int(valley_idx),
                'second_derivative': float(second_deriv[valley_idx])
            })

    peaks.sort(key=lambda p: p['position'])

    return peaks


def correct_and_detect_peaks(y, x=None, lam=1e6, p=0.001, niter=100, tol=1e-6,
                             method='second_derivative', **peak_kwargs):
    """
    综合功能：基线校正 + 峰检测

    参数:
        y: 原始光谱强度
        x: 波数/波长数组 (可选)
        lam: AsLS平滑参数
        p: AsLS非对称权重参数
        niter: 最大迭代次数
        tol: 收敛阈值
        method: 峰检测方法 ('second_derivative' 或 'find_peaks')
        **peak_kwargs: 传递给峰检测函数的额外参数

    返回:
        baseline: 拟合的基线
        y_corrected: 校正后的光谱
        peaks: 检测到的峰参数列表
    """
    baseline, y_corrected = asls_baseline(y, lam, p, niter, tol)

    if method == 'second_derivative':
        peaks = detect_peaks_second_derivative(y_corrected, x, **peak_kwargs)
    else:
        peaks = detect_raman_peaks(y_corrected, x, **peak_kwargs)

    return baseline, y_corrected, peaks


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    x = np.linspace(0, 100, 1000)
    y_true = np.zeros_like(x)
    for peak_pos, peak_height, peak_width in [(20, 10, 2), (50, 15, 3), (80, 8, 1.5)]:
        y_true += peak_height * np.exp(-(x - peak_pos)**2 / (2 * peak_width**2))

    baseline = 5 + 0.05 * x + 0.0002 * x**2
    y_noisy = y_true + baseline + 0.3 * np.random.randn(len(x))

    baseline_fit, y_corrected, peaks = correct_and_detect_peaks(
        y_noisy, x, lam=1e6, p=0.001, method='second_derivative'
    )

    print("检测到的拉曼峰：")
    for i, peak in enumerate(peaks):
        print(f"峰 {i+1}: 位置={peak['position']:.2f}, 强度={peak['intensity']:.2f}, "
              f"半高宽={peak['fwhm']:.2f}")

    fig = plt.figure(figsize=(15, 10))

    ax1 = plt.subplot(2, 2, 1)
    ax1.plot(x, y_noisy, label='原始光谱')
    ax1.plot(x, baseline_fit, label='拟合基线', linewidth=2)
    ax1.legend()
    ax1.set_title('AsLS基线拟合')

    ax2 = plt.subplot(2, 2, 2)
    ax2.plot(x, y_corrected, label='校正后光谱')
    ax2.plot(x, y_true, '--', label='真实信号', alpha=0.7)

    peak_positions = [p['position'] for p in peaks]
    peak_intensities = [p['intensity'] for p in peaks]
    ax2.scatter(peak_positions, peak_intensities, color='red', s=100,
                marker='v', label='检测到的峰', zorder=5)

    ax2.legend()
    ax2.set_title('基线校正与峰检测结果')

    ax3 = plt.subplot(2, 2, 3)
    second_deriv = savgol_filter(y_corrected, 7, 2, deriv=2)
    ax3.plot(x, second_deriv, label='二阶导数', color='green')
    ax3.axhline(y=0, color='k', linestyle='--', alpha=0.5)

    valley_positions = [x[p['valley_index']] for p in peaks]
    valley_values = [p['second_derivative'] for p in peaks]
    ax3.scatter(valley_positions, valley_values, color='orange', s=100,
                marker='o', label='二阶导数谷', zorder=5)

    ax3.legend()
    ax3.set_title('二阶导数找谷')

    ax4 = plt.subplot(2, 2, 4)
    if peaks:
        peak_data = peaks[0]
        idx = peak_data['index']
        half_width = int(peak_data['fwhm'] / (x[1] - x[0]) * 2)
        plot_start = max(0, idx - half_width)
        plot_end = min(len(x), idx + half_width)

        ax4.plot(x[plot_start:plot_end], y_corrected[plot_start:plot_end],
                 label='校正后光谱', linewidth=2)
        ax4.axhline(y=peak_data['intensity'] / 2, color='r', linestyle='--',
                    label='半高')
        ax4.axvline(x=peak_data['position'], color='g', linestyle='--',
                    label='峰中心')

        fwhm_left = peak_data['position'] - peak_data['fwhm'] / 2
        fwhm_right = peak_data['position'] + peak_data['fwhm'] / 2
        ax4.annotate('', xy=(fwhm_left, peak_data['intensity'] / 2),
                     xytext=(fwhm_right, peak_data['intensity'] / 2),
                     arrowprops=dict(arrowstyle='<->', color='blue', lw=2))
        ax4.text(peak_data['position'], peak_data['intensity'] / 2.2,
                 f'FWHM = {peak_data["fwhm"]:.2f}', ha='center', color='blue')

        ax4.legend()
        ax4.set_title(f'第一个峰的半高宽 (位置: {peak_data["position"]:.1f})')
    else:
        ax4.text(0.5, 0.5, '未检测到峰', ha='center', va='center')
        ax4.set_title('峰检测结果')

    plt.tight_layout()
    plt.show()
