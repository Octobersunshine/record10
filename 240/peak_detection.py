from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np


@dataclass
class PeaksResult:
    """
    峰值检测结果的数据容器。

    支持元组解包: pos, val = result  (向后兼容)
    也支持完整访问: result.positions, result.fwhm, result.prominences 等
    """
    positions: List[int] = field(default_factory=list)
    amplitudes: List[float] = field(default_factory=list)
    fwhm: List[float] = field(default_factory=list)
    prominences: List[float] = field(default_factory=list)
    noise_level: float = 0.0

    def __iter__(self):
        yield self.positions
        yield self.amplitudes

    def __len__(self):
        return len(self.positions)

    def __getitem__(self, idx):
        return self.positions[idx], self.amplitudes[idx]


def estimate_noise(signal):
    """
    估计信号的噪声标准差，使用一阶差分的MAD（中位数绝对偏差）方法。
    该方法对信号中的峰值具有鲁棒性。

    参数:
        signal (np.ndarray): 输入信号

    返回:
        float: 估计的噪声标准差
    """
    signal = np.asarray(signal, dtype=np.float64)
    if len(signal) < 2:
        return 0.0
    diff = np.diff(signal)
    mad = np.median(np.abs(diff - np.median(diff)))
    noise_std = 1.4826 * mad / np.sqrt(2.0)
    return max(noise_std, 1e-12)


def compute_prominences(signal, peak_positions):
    """
    计算每个峰值的显著性(prominence)。

    显著性定义为峰值高度减去其两侧最低谷的较高值。
    即：prominence = peak_val - max(left_base, right_base)
    其中 left_base 是峰值与左侧更高峰之间的最低点，
    right_base 是峰值与右侧更高峰之间的最低点。

    参数:
        signal (np.ndarray): 输入信号
        peak_positions (list): 峰值位置索引列表

    返回:
        list: 每个峰值的显著性
    """
    signal = np.asarray(signal, dtype=np.float64)
    n = len(signal)
    prominences = []

    for p in peak_positions:
        left_base = signal[p]
        min_val = signal[p]
        for k in range(p - 1, -1, -1):
            min_val = min(min_val, signal[k])
            if signal[k] > signal[p]:
                break
            left_base = min_val

        right_base = signal[p]
        min_val = signal[p]
        for k in range(p + 1, n):
            min_val = min(min_val, signal[k])
            if signal[k] > signal[p]:
                break
            right_base = min_val

        prominence = signal[p] - max(left_base, right_base)
        prominences.append(max(prominence, 0.0))

    return prominences


def compute_fwhm(signal, peak_positions, prominences):
    """
    计算每个峰值的半高宽（Full Width at Half Maximum）。

    FWHM 定义为信号在半最大值处的宽度：
    half_level = peak_val - prominence / 2
    在 half_level 处从峰值向两侧搜索，直到信号低于 half_level，
    通过线性插值获取亚采样精度的交叉点。

    参数:
        signal (np.ndarray): 输入信号
        peak_positions (list): 峰值位置索引列表
        prominences (list): 每个峰值的显著性

    返回:
        list: 每个峰值的FWHM值（单位为采样点数）
    """
    signal = np.asarray(signal, dtype=np.float64)
    n = len(signal)
    fwhm_list = []

    for p, prom in zip(peak_positions, prominences):
        if prom < 1e-12:
            fwhm_list.append(0.0)
            continue

        half_level = signal[p] - prom / 2.0

        left_cross = 0.0
        found = False
        for k in range(p - 1, -1, -1):
            if signal[k] < half_level:
                denom = signal[k + 1] - signal[k]
                if abs(denom) > 1e-12:
                    frac = (half_level - signal[k]) / denom
                else:
                    frac = 0.5
                left_cross = k + frac
                found = True
                break
        if not found:
            left_cross = 0.0

        right_cross = float(n - 1)
        found = False
        for k in range(p + 1, n):
            if signal[k] < half_level:
                denom = signal[k - 1] - signal[k]
                if abs(denom) > 1e-12:
                    frac = (half_level - signal[k]) / denom
                else:
                    frac = 0.5
                right_cross = k - frac
                found = True
                break
        if not found:
            right_cross = float(n - 1)

        fwhm_list.append(max(right_cross - left_cross, 0.0))

    return fwhm_list


def detect_peaks(signal, radius, delta=0.0, include_boundaries=True,
                 strict_monotonic=False, include_plateau_midpoint=True,
                 sort_by='none', auto_delta=False, noise_factor=3.0):
    """
    从一维信号中检测峰值（局部极大值），支持排序、自动阈值和FWHM计算。

    参数:
        signal (array_like): 输入的一维信号数组
        radius (int): 邻域半径，表示左右各radius个点作为比较范围
        delta (float): 相对高度阈值，峰值需比左右邻域内的点至少高delta，默认为0
        include_boundaries (bool): 是否包含边界点作为峰值，默认为True
        strict_monotonic (bool): 是否仅检测严格单调上升后下降的点，默认为False
        include_plateau_midpoint (bool): 是否将平坦区域（平台）的中点视为峰值，默认为True
        sort_by (str): 峰值排序方式，可选:
            'none' - 按位置顺序（默认）
            'amplitude' - 按幅值降序
            'significance' - 按显著性(prominence)降序
        auto_delta (bool): 是否根据噪声水平自动设置delta阈值，默认为False
            当为True时，delta = noise_factor * noise_std
        noise_factor (float): 噪声因子，auto_delta时使用，默认为3.0

    返回:
        PeaksResult: 峰值检测结果对象，包含:
            .positions  - 峰值位置索引列表
            .amplitudes - 峰值幅值列表
            .fwhm       - 峰值半高宽列表
            .prominences - 峰值显著性列表
            .noise_level - 估计的噪声标准差

        支持元组解包: pos, val = detect_peaks(signal, radius)
    """
    signal = np.asarray(signal, dtype=np.float64)
    n = len(signal)

    noise_std = estimate_noise(signal)

    if n == 0 or radius < 0:
        return PeaksResult(noise_level=noise_std)

    if radius == 0:
        return PeaksResult(
            positions=list(range(n)),
            amplitudes=signal.tolist(),
            fwhm=[0.0] * n,
            prominences=[0.0] * n,
            noise_level=noise_std
        )

    effective_delta = delta
    if auto_delta and noise_std > 0:
        effective_delta = noise_factor * noise_std

    peaks_pos = []
    peaks_val = []

    i = 0
    while i < n:
        j = i
        while j + 1 < n and abs(signal[j + 1] - signal[i]) < 1e-12:
            j += 1

        plateau_start = i
        plateau_end = j

        window_left = max(0, plateau_start - radius)
        window_right = min(n, plateau_end + radius + 1)

        plateau_val = signal[plateau_start]
        window_max = np.max(signal[window_left:window_right])

        left_min = np.min(signal[window_left:plateau_start]) if plateau_start > window_left else -np.inf
        right_min = np.min(signal[plateau_end + 1:window_right]) if plateau_end + 1 < window_right else -np.inf

        is_peak = True

        if plateau_val + 1e-12 < window_max:
            is_peak = False

        if is_peak and effective_delta > 0:
            if plateau_start > window_left and plateau_val - left_min < effective_delta - 1e-12:
                is_peak = False
            if is_peak and plateau_end + 1 < window_right and plateau_val - right_min < effective_delta - 1e-12:
                is_peak = False

        if is_peak and strict_monotonic:
            if plateau_start > 0:
                k = plateau_start - 1
                prev_val = signal[k]
                k -= 1
                while k >= window_left and abs(signal[k] - prev_val) < 1e-12:
                    k -= 1
                if k >= window_left and signal[k] > prev_val + 1e-12:
                    is_peak = False

            if is_peak and plateau_end < n - 1:
                k = plateau_end + 1
                next_val = signal[k]
                k += 1
                while k < window_right and abs(signal[k] - next_val) < 1e-12:
                    k += 1
                if k < window_right and signal[k] > next_val + 1e-12:
                    is_peak = False

        if is_peak:
            if plateau_start == plateau_end:
                peak_pos = plateau_start
            else:
                if include_plateau_midpoint:
                    peak_pos = (plateau_start + plateau_end) // 2
                else:
                    i = plateau_end + 1
                    continue

            if include_boundaries or (peak_pos >= radius and peak_pos < n - radius):
                peaks_pos.append(peak_pos)
                peaks_val.append(signal[peak_pos])

        i = plateau_end + 1

    if not peaks_pos:
        return PeaksResult(noise_level=noise_std)

    filtered_pos = [peaks_pos[0]]
    filtered_val = [peaks_val[0]]
    for p, v in zip(peaks_pos[1:], peaks_val[1:]):
        if p - filtered_pos[-1] > radius:
            filtered_pos.append(p)
            filtered_val.append(v)

    prominences = compute_prominences(signal, filtered_pos)
    fwhm_list = compute_fwhm(signal, filtered_pos, prominences)

    if sort_by == 'amplitude':
        order = sorted(range(len(filtered_pos)), key=lambda k: -filtered_val[k])
    elif sort_by == 'significance':
        order = sorted(range(len(filtered_pos)), key=lambda k: -prominences[k])
    else:
        order = list(range(len(filtered_pos)))

    result = PeaksResult(
        positions=[filtered_pos[k] for k in order],
        amplitudes=[filtered_val[k] for k in order],
        fwhm=[fwhm_list[k] for k in order],
        prominences=[prominences[k] for k in order],
        noise_level=noise_std
    )

    return result
