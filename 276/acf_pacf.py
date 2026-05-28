import numpy as np
import math
from typing import Tuple, Dict, List, Optional


def calculate_acf(series: np.ndarray, max_lag: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    计算自相关函数(ACF)

    Parameters
    ----------
    series : np.ndarray
        输入时间序列
    max_lag : int
        最大滞后阶数

    Returns
    -------
    acf_values : np.ndarray
        各滞后阶数的自相关系数，长度为max_lag + 1（包含滞后0）
    conf_interval : np.ndarray
        95%置信区间的上下界，形状为(max_lag + 1, 2)
    """
    n = len(series)
    if max_lag >= n:
        raise ValueError(f"max_lag ({max_lag}) 必须小于序列长度 ({n})")

    series = np.asarray(series, dtype=np.float64)
    mean = np.mean(series)
    centered = series - mean
    variance = np.sum(centered ** 2)

    if variance == 0:
        raise ValueError("序列方差为0，无法计算自相关")

    acf_values = np.zeros(max_lag + 1)

    for lag in range(max_lag + 1):
        if lag == 0:
            cov = np.sum(centered * centered)
        else:
            cov = np.sum(centered[:n - lag] * centered[lag:])
        acf_values[lag] = cov / variance

    conf_interval = np.zeros((max_lag + 1, 2))
    for lag in range(max_lag + 1):
        if lag == 0:
            se = 0
        else:
            var_sum = 1 + 2 * np.sum(acf_values[1:lag] ** 2)
            se = np.sqrt(var_sum / n)
        margin = 1.96 * se
        conf_interval[lag, 0] = acf_values[lag] - margin
        conf_interval[lag, 1] = acf_values[lag] + margin

    return acf_values, conf_interval


def calculate_pacf(series: np.ndarray, max_lag: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    使用Levinson-Durbin算法计算偏自相关函数(PACF)

    Parameters
    ----------
    series : np.ndarray
        输入时间序列
    max_lag : int
        最大滞后阶数

    Returns
    -------
    pacf_values : np.ndarray
        各滞后阶数的偏自相关系数，长度为max_lag + 1（包含滞后0）
    conf_interval : np.ndarray
        95%置信区间的上下界，形状为(max_lag + 1, 2)
    """
    n = len(series)
    if max_lag >= n:
        raise ValueError(f"max_lag ({max_lag}) 必须小于序列长度 ({n})")

    series = np.asarray(series, dtype=np.float64)
    acf_values, _ = calculate_acf(series, max_lag)

    pacf_values = np.zeros(max_lag + 1)
    pacf_values[0] = 1.0

    if max_lag == 0:
        conf_interval = np.zeros((1, 2))
        conf_interval[0] = [1.0, 1.0]
        return pacf_values, conf_interval

    phi = np.zeros((max_lag + 1, max_lag + 1))
    phi[1, 1] = acf_values[1]
    pacf_values[1] = acf_values[1]

    for k in range(2, max_lag + 1):
        num = acf_values[k] - np.sum(phi[k - 1, 1:k] * acf_values[1:k][::-1])
        den = 1 - np.sum(phi[k - 1, 1:k] * acf_values[1:k])
        phi[k, k] = num / den
        pacf_values[k] = phi[k, k]

        for j in range(1, k):
            phi[k, j] = phi[k - 1, j] - phi[k, k] * phi[k - 1, k - j]

    conf_interval = np.zeros((max_lag + 1, 2))
    se = 1.0 / np.sqrt(n)
    margin = 1.96 * se

    for lag in range(max_lag + 1):
        if lag == 0:
            conf_interval[lag] = [1.0, 1.0]
        else:
            conf_interval[lag, 0] = pacf_values[lag] - margin
            conf_interval[lag, 1] = pacf_values[lag] + margin

    return pacf_values, conf_interval


def compute_acf_pacf(series: np.ndarray, max_lag: int) -> Dict[str, np.ndarray]:
    """
    同时计算ACF和PACF，返回统一的结果字典

    Parameters
    ----------
    series : np.ndarray
        输入时间序列
    max_lag : int
        最大滞后阶数

    Returns
    -------
    result : Dict[str, np.ndarray]
        包含以下键的字典：
        - 'lags': 滞后阶数数组
        - 'acf': ACF值
        - 'acf_ci_lower': ACF置信区间下界
        - 'acf_ci_upper': ACF置信区间上界
        - 'pacf': PACF值
        - 'pacf_ci_lower': PACF置信区间下界
        - 'pacf_ci_upper': PACF置信区间上界
    """
    acf_values, acf_ci = calculate_acf(series, max_lag)
    pacf_values, pacf_ci = calculate_pacf(series, max_lag)

    lags = np.arange(max_lag + 1)

    return {
        'lags': lags,
        'acf': acf_values,
        'acf_ci_lower': acf_ci[:, 0],
        'acf_ci_upper': acf_ci[:, 1],
        'pacf': pacf_values,
        'pacf_ci_lower': pacf_ci[:, 0],
        'pacf_ci_upper': pacf_ci[:, 1]
    }


def test_significance(result: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """
    基于Bartlett公式进行ACF和PACF的显著性检验

    Parameters
    ----------
    result : Dict[str, np.ndarray]
        compute_acf_pacf返回的结果字典

    Returns
    -------
    significance : Dict[str, np.ndarray]
        包含显著性检验结果的字典：
        - 'acf_significant': ACF是否显著的布尔数组
        - 'pacf_significant': PACF是否显著的布尔数组
        - 'acf_p_values': ACF的近似p值
        - 'pacf_p_values': PACF的近似p值
    """
    acf = result['acf']
    pacf = result['pacf']
    acf_ci_lower = result['acf_ci_lower']
    acf_ci_upper = result['acf_ci_upper']
    pacf_ci_lower = result['pacf_ci_lower']
    pacf_ci_upper = result['pacf_ci_upper']
    n = len(result['lags']) - 1

    acf_significant = np.zeros_like(acf, dtype=bool)
    pacf_significant = np.zeros_like(pacf, dtype=bool)
    acf_p_values = np.ones_like(acf)
    pacf_p_values = np.ones_like(pacf)

    for i in range(len(acf)):
        if i == 0:
            acf_significant[i] = True
            pacf_significant[i] = True
            acf_p_values[i] = 0.0
            pacf_p_values[i] = 0.0
        else:
            acf_significant[i] = not (acf_ci_lower[i] <= 0 <= acf_ci_upper[i])
            pacf_significant[i] = not (pacf_ci_lower[i] <= 0 <= pacf_ci_upper[i])

            acf_margin = (acf_ci_upper[i] - acf_ci_lower[i]) / 2
            if acf_margin > 0:
                z_score = abs(acf[i]) / (acf_margin / 1.96)
                acf_p_values[i] = 2 * (1 - 0.5 * (1 + math.erf(z_score / np.sqrt(2))))

            pacf_margin = (pacf_ci_upper[i] - pacf_ci_lower[i]) / 2
            if pacf_margin > 0:
                z_score = abs(pacf[i]) / (pacf_margin / 1.96)
                pacf_p_values[i] = 2 * (1 - 0.5 * (1 + math.erf(z_score / np.sqrt(2))))

    return {
        'acf_significant': acf_significant,
        'pacf_significant': pacf_significant,
        'acf_p_values': acf_p_values,
        'pacf_p_values': pacf_p_values
    }


def identify_cutoff_decay(values: np.ndarray, significant: np.ndarray,
                          max_lag: int, threshold: float = 0.05) -> Dict:
    """
    识别序列的截尾或拖尾特征

    Parameters
    ----------
    values : np.ndarray
        ACF或PACF值
    significant : np.ndarray
        显著性布尔数组
    max_lag : int
        最大滞后阶数
    threshold : float
        显著性水平，默认0.05

    Returns
    -------
    result : Dict
        包含模式识别结果的字典：
        - 'pattern': 模式类型 ('cutoff'=截尾, 'decay'=拖尾, 'none'=无显著自相关)
        - 'cutoff_lag': 截尾滞后阶数（如果是截尾模式）
        - 'significant_lags': 显著的滞后阶数列表
        - 'explanation': 解释说明
    """
    significant = significant.copy()
    significant[0] = False

    significant_lags = np.where(significant)[0].tolist()

    if not significant_lags:
        return {
            'pattern': 'none',
            'cutoff_lag': None,
            'significant_lags': [],
            'explanation': '没有显著的自相关系数，序列可能是白噪声'
        }

    max_sig_lag = max(significant_lags)

    decay_count = 0
    cutoff_candidate = None

    for lag in range(1, max_lag + 1):
        if significant[lag]:
            decay_count = 0
            cutoff_candidate = lag
        else:
            decay_count += 1
            if decay_count >= 2 and cutoff_candidate is not None:
                remaining_sig = any(significant[lag + 1:])
                if not remaining_sig:
                    return {
                        'pattern': 'cutoff',
                        'cutoff_lag': cutoff_candidate,
                        'significant_lags': significant_lags,
                        'explanation': f'在滞后{cutoff_candidate}阶后截尾，连续2阶不再显著'
                    }

    if max_sig_lag >= max_lag - 1:
        return {
            'pattern': 'decay',
            'cutoff_lag': None,
            'significant_lags': significant_lags,
            'explanation': '呈现拖尾特征，自相关系数逐渐衰减'
        }

    if len(significant_lags) <= 2 and max_sig_lag <= 3:
        return {
            'pattern': 'cutoff',
            'cutoff_lag': max_sig_lag,
            'significant_lags': significant_lags,
            'explanation': f'仅在低阶滞后显著，判定为在滞后{max_sig_lag}阶截尾'
        }

    return {
        'pattern': 'decay',
        'cutoff_lag': None,
        'significant_lags': significant_lags,
        'explanation': '呈现拖尾特征'
    }


def suggest_arima_order(result: Dict[str, np.ndarray],
                        significance: Optional[Dict[str, np.ndarray]] = None,
                        max_p: int = 5, max_q: int = 5) -> Dict:
    """
    根据ACF和PACF的模式给出ARIMA模型阶数(p,q)的初步建议

    Parameters
    ----------
    result : Dict[str, np.ndarray]
        compute_acf_pacf返回的结果字典
    significance : Optional[Dict[str, np.ndarray]]
        test_significance返回的显著性结果，如果为None则自动计算
    max_p : int
        建议的p的最大值，默认5
    max_q : int
        建议的q的最大值，默认5

    Returns
    -------
    suggestion : Dict
        包含ARIMA阶数建议的字典：
        - 'acf_pattern': ACF模式
        - 'pacf_pattern': PACF模式
        - 'suggested_p': 建议的AR阶数p
        - 'suggested_q': 建议的MA阶数q
        - 'suggested_models': 建议的模型列表
        - 'reasoning': 建议理由
    """
    if significance is None:
        significance = test_significance(result)

    max_lag = len(result['lags']) - 1

    acf_pattern = identify_cutoff_decay(
        result['acf'], significance['acf_significant'], max_lag
    )
    pacf_pattern = identify_cutoff_decay(
        result['pacf'], significance['pacf_significant'], max_lag
    )

    suggested_p = 0
    suggested_q = 0
    reasoning = []
    suggested_models = []

    if acf_pattern['pattern'] == 'none' and pacf_pattern['pattern'] == 'none':
        reasoning.append('ACF和PACF均无显著自相关，序列可能是白噪声')
        suggested_models.append((0, 0))
    elif acf_pattern['pattern'] == 'cutoff' and pacf_pattern['pattern'] == 'decay':
        suggested_q = min(acf_pattern['cutoff_lag'], max_q)
        reasoning.append(f'ACF在滞后{acf_pattern["cutoff_lag"]}阶截尾，PACF拖尾 → MA({suggested_q})')
        suggested_models.append((0, suggested_q))
        if suggested_q > 1:
            suggested_models.append((0, suggested_q - 1))
    elif acf_pattern['pattern'] == 'decay' and pacf_pattern['pattern'] == 'cutoff':
        suggested_p = min(pacf_pattern['cutoff_lag'], max_p)
        reasoning.append(f'PACF在滞后{pacf_pattern["cutoff_lag"]}阶截尾，ACF拖尾 → AR({suggested_p})')
        suggested_models.append((suggested_p, 0))
        if suggested_p > 1:
            suggested_models.append((suggested_p - 1, 0))
    elif acf_pattern['pattern'] == 'decay' and pacf_pattern['pattern'] == 'decay':
        reasoning.append('ACF和PACF均拖尾 → 建议尝试ARMA模型')
        acf_sig = acf_pattern['significant_lags']
        pacf_sig = pacf_pattern['significant_lags']
        p_candidate = min(max(pacf_sig) if pacf_sig else 1, max_p)
        q_candidate = min(max(acf_sig) if acf_sig else 1, max_q)
        suggested_p = min(p_candidate, 2)
        suggested_q = min(q_candidate, 2)
        for p in range(suggested_p + 1):
            for q in range(suggested_q + 1):
                if p + q > 0:
                    suggested_models.append((p, q))
    elif acf_pattern['pattern'] == 'cutoff' and pacf_pattern['pattern'] == 'cutoff':
        reasoning.append('ACF和PACF均截尾 → 建议尝试低阶ARMA模型')
        p_candidate = min(pacf_pattern['cutoff_lag'] or 1, max_p)
        q_candidate = min(acf_pattern['cutoff_lag'] or 1, max_q)
        suggested_p = min(p_candidate, 2)
        suggested_q = min(q_candidate, 2)
        suggested_models.append((suggested_p, suggested_q))
        suggested_models.append((min(suggested_p, 1), min(suggested_q, 1)))
    else:
        acf_sig = acf_pattern['significant_lags']
        pacf_sig = pacf_pattern['significant_lags']
        if acf_sig and pacf_sig:
            suggested_p = min(max(pacf_sig), max_p)
            suggested_q = min(max(acf_sig), max_q)
            suggested_models.append((min(suggested_p, 1), min(suggested_q, 1)))
            reasoning.append('模式不明确，建议尝试低阶ARMA模型')
        elif acf_sig:
            suggested_q = min(max(acf_sig), max_q)
            suggested_models.append((0, suggested_q))
            reasoning.append('仅ACF显著，建议尝试MA模型')
        elif pacf_sig:
            suggested_p = min(max(pacf_sig), max_p)
            suggested_models.append((suggested_p, 0))
            reasoning.append('仅PACF显著，建议尝试AR模型')

    suggested_models = list(dict.fromkeys(suggested_models))[:5]

    return {
        'acf_pattern': acf_pattern,
        'pacf_pattern': pacf_pattern,
        'suggested_p': suggested_p,
        'suggested_q': suggested_q,
        'suggested_models': suggested_models,
        'reasoning': reasoning
    }


def print_results(result: Dict[str, np.ndarray],
                  show_significance: bool = True,
                  show_suggestion: bool = True) -> None:
    """
    打印ACF和PACF的计算结果，包括显著性检验和模型建议

    Parameters
    ----------
    result : Dict[str, np.ndarray]
        compute_acf_pacf返回的结果字典
    show_significance : bool
        是否显示显著性标记
    show_suggestion : bool
        是否显示ARIMA模型阶数建议
    """
    significance = None
    if show_significance or show_suggestion:
        significance = test_significance(result)

    if show_significance:
        print(f"{'Lag':>4} | {'ACF':>10} | {'Sig':>4} | {'p-value':>10} | "
              f"{'PACF':>10} | {'Sig':>4} | {'p-value':>10}")
        print("-" * 78)

        for i, lag in enumerate(result['lags']):
            acf = result['acf'][i]
            pacf = result['pacf'][i]
            acf_sig = significance['acf_significant'][i]
            pacf_sig = significance['pacf_significant'][i]
            acf_p = significance['acf_p_values'][i]
            pacf_p = significance['pacf_p_values'][i]

            acf_sig_str = '***' if acf_sig and lag > 0 else ''
            pacf_sig_str = '***' if pacf_sig and lag > 0 else ''

            acf_p_str = f'{acf_p:.4f}' if acf_p < 0.0001 else f'{acf_p:.4f}'
            pacf_p_str = f'{pacf_p:.4f}' if pacf_p < 0.0001 else f'{pacf_p:.4f}'

            print(
                f"{lag:>4} | {acf:>10.6f} | {acf_sig_str:>4} | {acf_p_str:>10} | "
                f"{pacf:>10.6f} | {pacf_sig_str:>4} | {pacf_p_str:>10}"
            )
        print("\n显著性标记: *** 表示在95%置信水平下显著")
    else:
        print(f"{'Lag':>4} | {'ACF':>10} | {'ACF 95% CI':>22} | {'PACF':>10} | {'PACF 95% CI':>22}")
        print("-" * 78)

        for i, lag in enumerate(result['lags']):
            acf = result['acf'][i]
            acf_low = result['acf_ci_lower'][i]
            acf_high = result['acf_ci_upper'][i]
            pacf = result['pacf'][i]
            pacf_low = result['pacf_ci_lower'][i]
            pacf_high = result['pacf_ci_upper'][i]

            print(
                f"{lag:>4} | {acf:>10.6f} | [{acf_low:>10.6f}, {acf_high:>10.6f}] | "
                f"{pacf:>10.6f} | [{pacf_low:>10.6f}, {pacf_high:>10.6f}]"
            )

    if show_suggestion and significance is not None:
        print("\n" + "=" * 78)
        print("模式识别与ARIMA模型建议")
        print("=" * 78)

        suggestion = suggest_arima_order(result, significance)

        print(f"\nACF模式: {suggestion['acf_pattern']['pattern']}")
        print(f"  说明: {suggestion['acf_pattern']['explanation']}")
        if suggestion['acf_pattern']['significant_lags']:
            print(f"  显著滞后阶数: {suggestion['acf_pattern']['significant_lags']}")

        print(f"\nPACF模式: {suggestion['pacf_pattern']['pattern']}")
        print(f"  说明: {suggestion['pacf_pattern']['explanation']}")
        if suggestion['pacf_pattern']['significant_lags']:
            print(f"  显著滞后阶数: {suggestion['pacf_pattern']['significant_lags']}")

        print(f"\n推荐阶数: AR(p={suggestion['suggested_p']}), MA(q={suggestion['suggested_q']})")
        print(f"\n建议尝试的模型 (p, q):")
        for i, (p, q) in enumerate(suggestion['suggested_models'], 1):
            print(f"  {i}. ARIMA({p}, 0, {q})")

        if suggestion['reasoning']:
            print(f"\n判断依据:")
            for reason in suggestion['reasoning']:
                print(f"  - {reason}")

        print("\n说明:")
        print("  - 截尾(Cutoff): 自相关系数在某阶后突然降至置信区间内")
        print("  - 拖尾(Decay): 自相关系数逐渐衰减，持续多个滞后阶数")
        print("  - AR(p)特征: PACF截尾，ACF拖尾")
        print("  - MA(q)特征: ACF截尾，PACF拖尾")
        print("  - ARMA(p,q)特征: ACF和PACF均拖尾")


if __name__ == "__main__":
    np.random.seed(42)
    n = 200

    print("=" * 78)
    print("测试1: 随机白噪声序列")
    print("=" * 78)
    random_series = np.random.normal(0, 1, n)
    result1 = compute_acf_pacf(random_series, max_lag=12)
    print_results(result1, show_significance=True, show_suggestion=True)

    print("\n" + "=" * 78)
    print("测试2: AR(1)过程 (phi = 0.7)")
    print("=" * 78)
    ar1_series = np.zeros(n)
    ar1_series[0] = np.random.normal()
    for t in range(1, n):
        ar1_series[t] = 0.7 * ar1_series[t - 1] + np.random.normal()
    result2 = compute_acf_pacf(ar1_series, max_lag=12)
    print_results(result2, show_significance=True, show_suggestion=True)

    print("\n" + "=" * 78)
    print("测试3: MA(1)过程 (theta = 0.5)")
    print("=" * 78)
    errors = np.random.normal(0, 1, n + 1)
    ma1_series = errors[1:] + 0.5 * errors[:-1]
    result3 = compute_acf_pacf(ma1_series, max_lag=12)
    print_results(result3, show_significance=True, show_suggestion=True)

    print("\n" + "=" * 78)
    print("测试4: ARMA(1,1)过程 (phi = 0.6, theta = 0.4)")
    print("=" * 78)
    errors_arma = np.random.normal(0, 1, n + 1)
    arma11_series = np.zeros(n)
    arma11_series[0] = errors_arma[0]
    for t in range(1, n):
        arma11_series[t] = 0.6 * arma11_series[t - 1] + errors_arma[t] + 0.4 * errors_arma[t - 1]
    result4 = compute_acf_pacf(arma11_series, max_lag=12)
    print_results(result4, show_significance=True, show_suggestion=True)

    print("\n" + "=" * 78)
    print("测试5: 均值非零的AR(1)过程 (均值 = 10, phi = 0.6)")
    print("=" * 78)
    mean = 10.0
    ar1_mean_series = np.zeros(n)
    ar1_mean_series[0] = mean + np.random.normal()
    for t in range(1, n):
        ar1_mean_series[t] = mean + 0.6 * (ar1_mean_series[t - 1] - mean) + np.random.normal()
    print(f"序列均值: {np.mean(ar1_mean_series):.6f}")
    result5 = compute_acf_pacf(ar1_mean_series, max_lag=12)
    print_results(result5, show_significance=True, show_suggestion=True)
    print(f"\n验证滞后0处ACF = {result5['acf'][0]:.10f} (应等于1.0)")

    print("\n" + "=" * 78)
    print("测试6: AR(2)过程 (phi1 = 0.5, phi2 = 0.3)")
    print("=" * 78)
    ar2_series = np.zeros(n)
    ar2_series[0] = np.random.normal()
    ar2_series[1] = np.random.normal()
    for t in range(2, n):
        ar2_series[t] = 0.5 * ar2_series[t - 1] + 0.3 * ar2_series[t - 2] + np.random.normal()
    result6 = compute_acf_pacf(ar2_series, max_lag=12)
    print_results(result6, show_significance=True, show_suggestion=True)

    print("\n" + "=" * 78)
    print("API使用示例")
    print("=" * 78)
    series_example = np.random.normal(0, 1, 100)
    result_example = compute_acf_pacf(series_example, max_lag=8)
    sig_example = test_significance(result_example)
    suggestion_example = suggest_arima_order(result_example, sig_example)

    print(f"\n显著的ACF滞后阶数: {np.where(sig_example['acf_significant'])[0].tolist()}")
    print(f"显著的PACF滞后阶数: {np.where(sig_example['pacf_significant'])[0].tolist()}")
    print(f"建议的ARIMA模型: {suggestion_example['suggested_models']}")
    print(f"推荐阶数: p={suggestion_example['suggested_p']}, q={suggestion_example['suggested_q']}")
