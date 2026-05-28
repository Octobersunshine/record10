import numpy as np
from scipy import stats


def _compute_features(window, window_size, feature_names, fs=1.0, power_bands=None):
    actual_size = len(window)
    features = []

    if feature_names is None or 'mean' in feature_names:
        features.append(np.nanmean(window))
    if feature_names is None or 'std' in feature_names:
        features.append(np.nanstd(window, ddof=1) if actual_size > 1 else np.nan)
    if feature_names is None or 'min' in feature_names:
        features.append(np.nanmin(window))
    if feature_names is None or 'max' in feature_names:
        features.append(np.nanmax(window))
    if feature_names is None or 'slope' in feature_names:
        x = np.arange(actual_size)
        valid = ~np.isnan(window)
        if np.sum(valid) >= 2:
            slope, _ = np.polyfit(x[valid], window[valid], 1)
            features.append(slope)
        else:
            features.append(np.nan)
    if feature_names is None or 'kurtosis' in feature_names:
        features.append(stats.kurtosis(window, nan_policy='omit') if actual_size > 3 else np.nan)
    if feature_names is None or 'skewness' in feature_names:
        features.append(stats.skew(window, nan_policy='omit') if actual_size > 2 else np.nan)
    if feature_names is None or 'energy' in feature_names:
        features.append(np.nansum(window ** 2))
    if feature_names is None or 'median' in feature_names:
        features.append(np.nanmedian(window))
    if feature_names is None or 'range' in feature_names:
        features.append(np.nanmax(window) - np.nanmin(window))
    if feature_names is None or 'rms' in feature_names:
        features.append(np.sqrt(np.nanmean(window ** 2)))

    if (feature_names is None or
            any(f in feature_names for f in ['dominant_freq', 'spectral_entropy', 'power_band'])):
        valid_fft = ~np.isnan(window)
        n_valid = np.sum(valid_fft)
        if n_valid >= 4:
            fft_vals = np.fft.fft(window[valid_fft])
            fft_mag = np.abs(fft_vals[:n_valid // 2])
            freqs = np.fft.fftfreq(n_valid, d=1.0 / fs)[:n_valid // 2]

            if np.sum(fft_mag) > 0:
                if feature_names is None or 'dominant_freq' in feature_names:
                    features.append(freqs[np.argmax(fft_mag)])
                if feature_names is None or 'spectral_entropy' in feature_names:
                    psd = fft_mag ** 2
                    psd_norm = psd / np.sum(psd)
                    psd_norm = psd_norm[psd_norm > 0]
                    features.append(-np.sum(psd_norm * np.log2(psd_norm)))
                if feature_names is None or 'power_band' in feature_names:
                    if power_bands is None:
                        power_bands = [(0, 0.1), (0.1, 0.3), (0.3, 0.5)]
                    total_power = np.sum(fft_mag ** 2)
                    if total_power > 0:
                        for (low, high) in power_bands:
                            band_mask = (freqs >= low) & (freqs < high)
                            band_power = np.sum(fft_mag[band_mask] ** 2) / total_power
                            features.append(band_power)
                    else:
                        for _ in power_bands:
                            features.append(np.nan)
            else:
                n_freq_features = 0
                if feature_names is None or 'dominant_freq' in feature_names:
                    features.append(np.nan)
                    n_freq_features += 1
                if feature_names is None or 'spectral_entropy' in feature_names:
                    features.append(np.nan)
                    n_freq_features += 1
                if feature_names is None or 'power_band' in feature_names:
                    if power_bands is None:
                        power_bands = [(0, 0.1), (0.1, 0.3), (0.3, 0.5)]
                    for _ in power_bands:
                        features.append(np.nan)
        else:
            if feature_names is None or 'dominant_freq' in feature_names:
                features.append(np.nan)
            if feature_names is None or 'spectral_entropy' in feature_names:
                features.append(np.nan)
            if feature_names is None or 'power_band' in feature_names:
                if power_bands is None:
                    power_bands = [(0, 0.1), (0.1, 0.3), (0.3, 0.5)]
                for _ in power_bands:
                    features.append(np.nan)

    return features


def extract_sliding_window_features(
    time_series,
    window_size,
    step_size=1,
    feature_names=None,
    handle_incomplete='drop',
    fs=1.0,
    power_bands=None
):
    """
    对时间序列进行滑动窗口特征提取

    参数:
        time_series: 一维数组或列表，时间序列数据
        window_size: int，滑动窗口大小，必须为正整数
        step_size: int，滑动步长，默认为1，必须满足 1 <= step_size <= window_size
        feature_names: list，要计算的特征名称列表，默认为None（计算所有特征）
        handle_incomplete: str，末尾不完整窗口的处理方式:
            - 'drop': 丢弃不完整窗口（默认）
            - 'pad': 用NaN填充至窗口大小后计算
            - 'compute': 直接用剩余数据计算（特征值基于较短窗口）
        fs: float，采样频率，用于频域特征计算，默认为1.0
        power_bands: list of tuples，功率带定义，如 [(0, 0.1), (0.1, 0.3)]，
            默认为 [(0, 0.1), (0.1, 0.3), (0.3, 0.5)]

    返回:
        features: numpy数组，形状为 (n_windows, n_features)，特征矩阵
        feature_labels: list，特征名称列表
    """
    time_series = np.asarray(time_series, dtype=float).flatten()
    n_samples = len(time_series)

    if not isinstance(window_size, int) or window_size < 1:
        raise ValueError(f"窗口大小必须为正整数，当前值: {window_size}")
    if window_size > n_samples:
        raise ValueError(f"窗口大小 {window_size} 大于时间序列长度 {n_samples}")
    if not isinstance(step_size, int) or step_size < 1:
        raise ValueError(f"步长必须为正整数，当前值: {step_size}")
    if step_size > window_size:
        raise ValueError(
            f"步长 {step_size} 大于窗口大小 {window_size}，"
            f"会导致窗口间跳过数据而遗漏。请设置步长 <= 窗口大小。"
        )
    if handle_incomplete not in ('drop', 'pad', 'compute'):
        raise ValueError(
            f"handle_incomplete 必须为 'drop'、'pad' 或 'compute'，"
            f"当前值: '{handle_incomplete}'"
        )
    if fs <= 0:
        raise ValueError(f"采样频率 fs 必须为正数，当前值: {fs}")
    if power_bands is not None:
        if not isinstance(power_bands, (list, tuple)) or len(power_bands) == 0:
            raise ValueError("power_bands 必须为非空列表或元组")
        for band in power_bands:
            if not isinstance(band, (list, tuple)) or len(band) != 2:
                raise ValueError(f"每个功率带必须是包含两个元素的元组，当前值: {band}")
            if band[0] < 0 or band[1] <= band[0]:
                raise ValueError(f"功率带必须满足 0 <= low < high，当前值: {band}")

    window_starts = np.arange(0, n_samples - window_size + 1, step_size)

    last_complete_end = (window_starts[-1] + window_size) if len(window_starts) > 0 else 0
    has_incomplete = last_complete_end < n_samples

    if has_incomplete and handle_incomplete != 'drop':
        window_starts = np.append(window_starts, last_complete_end)

    all_features = []
    feature_labels = None

    for start in window_starts:
        end = start + window_size
        if end <= n_samples:
            window = time_series[start:end]
        elif handle_incomplete == 'pad':
            window = np.full(window_size, np.nan)
            window[:n_samples - start] = time_series[start:]
        elif handle_incomplete == 'compute':
            window = time_series[start:]
        else:
            continue

        window_features = _compute_features(
            window, window_size, feature_names, fs=fs, power_bands=power_bands
        )

        if feature_labels is None:
            feature_labels = _build_feature_labels(feature_names, power_bands)

        all_features.append(window_features)

    return np.array(all_features), feature_labels


def _build_feature_labels(feature_names, power_bands=None):
    all_labels = ['mean', 'std', 'min', 'max', 'slope', 'kurtosis',
                  'skewness', 'energy', 'median', 'range', 'rms']
    freq_labels = []
    if feature_names is None or 'dominant_freq' in feature_names:
        freq_labels.append('dominant_freq')
    if feature_names is None or 'spectral_entropy' in feature_names:
        freq_labels.append('spectral_entropy')
    if feature_names is None or 'power_band' in feature_names:
        if power_bands is None:
            power_bands = [(0, 0.1), (0.1, 0.3), (0.3, 0.5)]
        for i, (low, high) in enumerate(power_bands):
            freq_labels.append(f'power_band_{low}_{high}')
    all_labels.extend(freq_labels)
    if feature_names is None:
        return all_labels
    filtered = []
    for l in all_labels:
        if l in feature_names:
            filtered.append(l)
        elif l.startswith('power_band_') and 'power_band' in feature_names:
            filtered.append(l)
    return filtered


def select_features_by_variance(features, feature_labels, threshold=0.01):
    """
    基于方差筛选特征，移除方差低于阈值的特征

    参数:
        features: numpy数组，形状为 (n_samples, n_features)，特征矩阵
        feature_labels: list，特征名称列表
        threshold: float，方差阈值，默认为0.01

    返回:
        selected_features: numpy数组，筛选后的特征矩阵
        selected_labels: list，筛选后的特征名称列表
        variances: numpy数组，各特征的方差
    """
    if features.ndim != 2:
        raise ValueError(f"features 必须是二维数组，当前维度: {features.ndim}")
    if features.shape[1] != len(feature_labels):
        raise ValueError(
            f"特征数 {features.shape[1]} 与标签数 {len(feature_labels)} 不匹配"
        )
    if threshold < 0:
        raise ValueError(f"阈值必须非负，当前值: {threshold}")

    variances = np.nanvar(features, axis=0, ddof=1)
    valid_mask = ~np.isnan(variances)
    variances_clean = np.where(valid_mask, variances, 0.0)
    selected_mask = variances_clean >= threshold

    selected_features = features[:, selected_mask]
    selected_labels = [label for i, label in enumerate(feature_labels) if selected_mask[i]]

    return selected_features, selected_labels, variances


def select_features_by_correlation(
    features,
    feature_labels,
    target=None,
    max_corr=0.9,
    method='pearson'
):
    """
    基于相关性筛选特征，移除高度相关的冗余特征

    参数:
        features: numpy数组，形状为 (n_samples, n_features)，特征矩阵
        feature_labels: list，特征名称列表
        target: numpy数组，形状为 (n_samples,)，目标变量（用于有监督特征选择）。
            如果为None，则只基于特征间相关性筛选。
        max_corr: float，特征间最大允许相关系数，默认为0.9
        method: str，相关系数计算方法，'pearson' 或 'spearman'，默认为'pearson'

    返回:
        selected_features: numpy数组，筛选后的特征矩阵
        selected_labels: list，筛选后的特征名称列表
        corr_matrix: numpy数组，特征间相关系数矩阵
    """
    if features.ndim != 2:
        raise ValueError(f"features 必须是二维数组，当前维度: {features.ndim}")
    if features.shape[1] != len(feature_labels):
        raise ValueError(
            f"特征数 {features.shape[1]} 与标签数 {len(feature_labels)} 不匹配"
        )
    if not (0 < max_corr <= 1):
        raise ValueError(f"max_corr 必须在 (0, 1] 范围内，当前值: {max_corr}")
    if method not in ('pearson', 'spearman'):
        raise ValueError(f"method 必须为 'pearson' 或 'spearman'，当前值: '{method}'")

    n_features = features.shape[1]

    if method == 'pearson':
        features_clean = np.nan_to_num(features, nan=0.0)
        features_centered = features_clean - np.mean(features_clean, axis=0, keepdims=True)
        features_std = np.std(features_centered, axis=0, ddof=1)
        features_std = np.where(features_std < 1e-10, 1.0, features_std)
        features_normalized = features_centered / features_std
        cov_matrix = features_normalized.T @ features_normalized / (features_normalized.shape[0] - 1)
        std_matrix = features_std[:, None] @ features_std[None, :]
        corr_matrix = cov_matrix / std_matrix
    else:
        rank_features = np.zeros_like(features)
        for i in range(n_features):
            col = features[:, i]
            valid = ~np.isnan(col)
            ranks = np.zeros_like(col)
            ranks[valid] = stats.rankdata(col[valid])
            rank_features[:, i] = ranks
        rank_centered = rank_features - np.mean(rank_features, axis=0, keepdims=True)
        rank_std = np.std(rank_centered, axis=0, ddof=1)
        rank_std = np.where(rank_std < 1e-10, 1.0, rank_std)
        rank_normalized = rank_centered / rank_std
        cov_matrix = rank_normalized.T @ rank_normalized / (rank_normalized.shape[0] - 1)
        std_matrix = rank_std[:, None] @ rank_std[None, :]
        corr_matrix = cov_matrix / std_matrix

    corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)
    corr_matrix = np.clip(corr_matrix, -1.0, 1.0)
    np.fill_diagonal(corr_matrix, 1.0)

    if target is not None:
        target = np.asarray(target).flatten()
        if len(target) != features.shape[0]:
            raise ValueError(
                f"目标变量长度 {len(target)} 与样本数 {features.shape[0]} 不匹配"
            )
        if method == 'pearson':
            target_corr = np.array([
                np.corrcoef(features[:, i], target)[0, 1]
                for i in range(n_features)
            ])
        else:
            target_corr = np.array([
                stats.spearmanr(features[:, i], target)[0]
                for i in range(n_features)
            ])
        target_corr = np.nan_to_num(np.abs(target_corr), nan=0.0)
        feature_importance = target_corr
    else:
        feature_importance = np.nanvar(features, axis=0, ddof=1)
        feature_importance = np.nan_to_num(feature_importance, nan=0.0)

    sorted_indices = np.argsort(-feature_importance)

    selected_indices = []
    for i in sorted_indices:
        is_redundant = False
        for j in selected_indices:
            if abs(corr_matrix[i, j]) > max_corr:
                is_redundant = True
                break
        if not is_redundant:
            selected_indices.append(i)

    selected_indices.sort()
    selected_features = features[:, selected_indices]
    selected_labels = [feature_labels[i] for i in selected_indices]

    return selected_features, selected_labels, corr_matrix


if __name__ == "__main__":
    np.random.seed(42)
    n_samples = 200
    fs = 10.0
    t = np.arange(n_samples) / fs
    time_series = (np.random.randn(n_samples) * 0.5 +
                   np.sin(2 * np.pi * 0.8 * t) * 2 +
                   np.sin(2 * np.pi * 0.2 * t) * 3)
    target = np.sin(2 * np.pi * 0.2 * t[:19]) + np.random.randn(19) * 0.1

    window_size = 20
    step_size = 10
    power_bands = [(0, 0.3), (0.3, 0.6), (0.6, 1.0)]

    print("=" * 60)
    print("1. 特征提取（含频域特征）")
    print("=" * 60)
    features, labels = extract_sliding_window_features(
        time_series, window_size=window_size, step_size=step_size,
        fs=fs, power_bands=power_bands
    )
    print(f"时间序列长度: {len(time_series)}, 采样频率: {fs} Hz")
    print(f"窗口大小: {window_size}, 步长: {step_size}")
    print(f"窗口数量: {features.shape[0]}, 特征数: {features.shape[1]}")
    print("特征名称:")
    for i, label in enumerate(labels):
        print(f"  {i + 1:2d}. {label}")
    print(f"\n特征矩阵形状: {features.shape}")
    print(f"第一个窗口的频域特征:")
    print(f"  主频: {features[0, labels.index('dominant_freq')]:.4f} Hz")
    print(f"  谱熵: {features[0, labels.index('spectral_entropy')]:.4f}")
    for band in power_bands:
        key = f'power_band_{band[0]}_{band[1]}'
        print(f"  功率带 {band}: {features[0, labels.index(key)]:.4f}")

    print("\n" + "=" * 60)
    print("2. 基于方差的特征选择")
    print("=" * 60)
    selected_feat, selected_labels, variances = select_features_by_variance(
        features, labels, threshold=0.1
    )
    print(f"原始特征数: {features.shape[1]}")
    print(f"筛选后特征数: {selected_feat.shape[1]}")
    print(f"各特征方差:")
    for label, var in zip(labels, variances):
        marker = "[x]" if label in selected_labels else "[ ]"
        print(f"  {marker} {label}: {var:.6f}")

    print("\n" + "=" * 60)
    print("3. 基于相关性的特征选择（无监督）")
    print("=" * 60)
    corr_feat, corr_labels, corr_matrix = select_features_by_correlation(
        features, labels, max_corr=0.8
    )
    print(f"原始特征数: {features.shape[1]}")
    print(f"筛选后特征数: {corr_feat.shape[1]}")
    print(f"保留的特征: {corr_labels}")

    print("\n" + "=" * 60)
    print("4. 基于相关性的特征选择（有监督，结合目标变量）")
    print("=" * 60)
    target_feat, target_labels, _ = select_features_by_correlation(
        features, labels, target=target, max_corr=0.8
    )
    print(f"原始特征数: {features.shape[1]}")
    print(f"筛选后特征数: {target_feat.shape[1]}")
    print(f"保留的特征: {target_labels}")

    print("\n" + "=" * 60)
    print("5. 指定部分特征计算")
    print("=" * 60)
    partial_features, partial_labels = extract_sliding_window_features(
        time_series, window_size=window_size, step_size=step_size,
        feature_names=['mean', 'std', 'dominant_freq', 'power_band'],
        fs=fs, power_bands=power_bands
    )
    print(f"计算的特征: {partial_labels}")
    print(f"特征矩阵形状: {partial_features.shape}")

    print("\n" + "=" * 60)
    print("6. 参数校验 - 步长 > 窗口大小")
    print("=" * 60)
    try:
        extract_sliding_window_features(
            time_series, window_size=10, step_size=15
        )
    except ValueError as e:
        print(f"捕获异常: {e}")

    print("\n" + "=" * 60)
    print("7. 参数校验 - 非法功率带")
    print("=" * 60)
    try:
        extract_sliding_window_features(
            time_series, window_size=10, power_bands=[(0.5, 0.1)]
        )
    except ValueError as e:
        print(f"捕获异常: {e}")

    print("\n" + "=" * 60)
    print("8. 特征工程完整流程示例")
    print("=" * 60)
    print("步骤1: 提取所有特征")
    all_features, all_labels = extract_sliding_window_features(
        time_series, window_size=window_size, step_size=step_size,
        fs=fs, power_bands=power_bands
    )
    print(f"  原始特征: {all_features.shape[1]} 维")

    print("步骤2: 基于方差筛选")
    feat_v, labels_v, _ = select_features_by_variance(
        all_features, all_labels, threshold=0.05
    )
    print(f"  方差筛选后: {feat_v.shape[1]} 维")

    print("步骤3: 基于相关性去冗余")
    feat_final, labels_final, _ = select_features_by_correlation(
        feat_v, labels_v, target=target, max_corr=0.75, method='spearman'
    )
    print(f"  相关性筛选后: {feat_final.shape[1]} 维")
    print(f"  最终特征: {labels_final}")
    print(f"  最终特征矩阵形状: {feat_final.shape}")
