import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import adfuller, kpss

_ADF_MIN_SAMPLE_SIZE = 10
_ADF_DEFAULT_MAXLAG = 1


def _compute_adf_maxlag(n):
    return max(1, int(12 * (n / 100) ** (1 / 4)))


def adf_test(series, significance_level=0.05, autolag='AIC'):
    series = np.asarray(series, dtype=float).flatten()
    n = len(series)

    if n < _ADF_MIN_SAMPLE_SIZE:
        result = adfuller(series, maxlag=_ADF_DEFAULT_MAXLAG, regression='c')
        lag_used = _ADF_DEFAULT_MAXLAG
        lag_selection = 'default (sample too small)'
    else:
        maxlag = _compute_adf_maxlag(n)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = adfuller(series, maxlag=maxlag, regression='c', autolag=autolag)
        lag_used = result[2]
        lag_selection = f'autolag={autolag} (maxlag={maxlag})'

    adf_statistic = result[0]
    p_value = result[1]
    critical_values = result[4]

    is_stationary = p_value < significance_level

    return {
        'test_name': 'ADF Test',
        'statistic': adf_statistic,
        'p_value': p_value,
        'used_lag': lag_used,
        'lag_selection': lag_selection,
        'critical_values': critical_values,
        'significance_level': significance_level,
        'is_stationary': is_stationary,
        'conclusion': '平稳' if is_stationary else '非平稳'
    }


def kpss_test(series, significance_level=0.05):
    series = np.asarray(series, dtype=float).flatten()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = kpss(series, regression='c', nlags='auto')

    kpss_statistic = result[0]
    p_value = result[1]
    critical_values = result[3]

    is_stationary = p_value > significance_level

    return {
        'test_name': 'KPSS Test',
        'statistic': kpss_statistic,
        'p_value': p_value,
        'critical_values': critical_values,
        'significance_level': significance_level,
        'is_stationary': is_stationary,
        'conclusion': '平稳' if is_stationary else '非平稳'
    }


def recommend_diff_order(series, max_d=3, significance_level=0.05):
    series = np.asarray(series, dtype=float).flatten()

    diff_steps = []
    current = series.copy()

    for d in range(max_d + 1):
        adf_result = adf_test(current, significance_level)
        step = {
            'd': d,
            'adf_statistic': adf_result['statistic'],
            'adf_p_value': adf_result['p_value'],
            'is_stationary': adf_result['is_stationary'],
            'series_length': len(current)
        }
        diff_steps.append(step)

        if adf_result['is_stationary']:
            return {
                'recommended_d': d,
                'is_stationary_after_diff': True,
                'diff_steps': diff_steps
            }

        current = np.diff(current)

    return {
        'recommended_d': max_d,
        'is_stationary_after_diff': False,
        'diff_steps': diff_steps
    }


def stl_decomposition(series, period=None):
    series = np.asarray(series, dtype=float).flatten()
    n = len(series)

    if period is None:
        if n >= 14:
            period = max(2, int(n / 2))
            if period > n // 2:
                period = n // 2
        else:
            return None

    if period < 2 or n < 2 * period:
        return None

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            stl = STL(series, period=period, robust=True)
            fit = stl.fit()

        trend_var = np.var(fit.trend + fit.resid)
        seasonal_var = np.var(fit.seasonal + fit.resid)
        resid_var = np.var(fit.resid)

        trend_strength = 1 - resid_var / trend_var if trend_var > 0 else 0
        seasonal_strength = 1 - resid_var / seasonal_var if seasonal_var > 0 else 0

        trend_strength = max(0.0, min(1.0, trend_strength))
        seasonal_strength = max(0.0, min(1.0, seasonal_strength))

        return {
            'trend': fit.trend,
            'seasonal': fit.seasonal,
            'resid': fit.resid,
            'period': period,
            'trend_strength': trend_strength,
            'seasonal_strength': seasonal_strength,
            'trend_dominant': trend_strength > 0.6,
            'seasonal_dominant': seasonal_strength > 0.6
        }
    except Exception:
        return None


def stationarity_recommendation(series, significance_level=0.05, period=None):
    series = np.asarray(series, dtype=float).flatten()
    n = len(series)

    adf_result = adf_test(series, significance_level)
    kpss_result = kpss_test(series, significance_level)
    diff_result = recommend_diff_order(series, max_d=3, significance_level=significance_level)
    stl_result = stl_decomposition(series, period=period)

    adf_stationary = adf_result['is_stationary']
    kpss_stationary = kpss_result['is_stationary']

    if adf_stationary and kpss_stationary:
        overall_conclusion = '序列平稳'
        overall_stationary = True
    elif not adf_stationary and not kpss_stationary:
        overall_conclusion = '序列非平稳（存在单位根）'
        overall_stationary = False
    elif adf_stationary and not kpss_stationary:
        overall_conclusion = '差分平稳（需要差分处理）'
        overall_stationary = False
    else:
        overall_conclusion = '趋势平稳（去除趋势后可平稳）'
        overall_stationary = False

    recommendations = []

    if adf_stationary and kpss_stationary:
        recommendations.append('序列已平稳，无需变换')
    else:
        if diff_result['is_stationary_after_diff']:
            d = diff_result['recommended_d']
            if d > 0:
                recommendations.append(f'进行 {d} 阶差分（d={d}）')
        else:
            recommendations.append('差分阶数超过3仍未平稳，请检查序列是否存在结构性变化')

        if stl_result is not None:
            if stl_result['trend_dominant']:
                recommendations.append('趋势成分显著，建议去趋势处理（减去STL趋势成分）')
            if stl_result['seasonal_dominant']:
                recommendations.append('季节成分显著，建议季节性调整（减去STL季节成分）')

        positive = series[series > 0]
        if len(positive) > n * 0.8:
            first_third = positive[:len(positive) // 3]
            last_third = positive[-(len(positive) // 3):]
            if len(first_third) > 1 and len(last_third) > 1:
                cv_first = np.std(first_third) / (np.mean(first_third) + 1e-10)
                cv_last = np.std(last_third) / (np.mean(last_third) + 1e-10)
                if cv_last > cv_first * 1.5 and np.mean(last_third) > np.mean(first_third) * 1.2:
                    recommendations.append('方差随水平增长，建议先取对数变换再差分')

    return {
        'adf_test': adf_result,
        'kpss_test': kpss_result,
        'diff_recommendation': diff_result,
        'stl_decomposition': stl_result,
        'overall_stationary': overall_stationary,
        'overall_conclusion': overall_conclusion,
        'recommendations': recommendations
    }


def comprehensive_stationarity_test(series, significance_level=0.05):
    return stationarity_recommendation(series, significance_level)


def print_test_results(results):
    print("=" * 60)
    print("ADF检验结果:")
    print("=" * 60)
    adf = results['adf_test']
    print(f"  检验统计量: {adf['statistic']:.6f}")
    print(f"  P值: {adf['p_value']:.6f}")
    print(f"  使用滞后阶数: {adf['used_lag']}")
    print(f"  滞后选择方式: {adf['lag_selection']}")
    print(f"  结论: {adf['conclusion']}")

    print("\n" + "=" * 60)
    print("KPSS检验结果:")
    print("=" * 60)
    kpss = results['kpss_test']
    print(f"  检验统计量: {kpss['statistic']:.6f}")
    print(f"  P值: {kpss['p_value']:.6f}")
    print(f"  结论: {kpss['conclusion']}")

    print("\n" + "=" * 60)
    print("综合判断:")
    print("=" * 60)
    print(f"  最终结论: {results['overall_conclusion']}")
    print(f"  是否平稳: {'是' if results['overall_stationary'] else '否'}")

    if 'diff_recommendation' in results:
        diff = results['diff_recommendation']
        print("\n" + "=" * 60)
        print("差分阶数推荐:")
        print("=" * 60)
        print(f"  推荐差分阶数 d = {diff['recommended_d']}")
        print(f"  差分后是否平稳: {'是' if diff['is_stationary_after_diff'] else '否'}")
        print("  各阶差分ADF检验:")
        for step in diff['diff_steps']:
            status = '✓ 平稳' if step['is_stationary'] else '✗ 非平稳'
            print(f"    d={step['d']}: ADF统计量={step['adf_statistic']:.4f}, "
                  f"p值={step['adf_p_value']:.4f}, {status}")

    if 'stl_decomposition' in results and results['stl_decomposition'] is not None:
        stl = results['stl_decomposition']
        print("\n" + "=" * 60)
        print("STL分解结果:")
        print("=" * 60)
        print(f"  季节周期: {stl['period']}")
        print(f"  趋势强度: {stl['trend_strength']:.4f}"
              f"{' (显著)' if stl['trend_dominant'] else ''}")
        print(f"  季节强度: {stl['seasonal_strength']:.4f}"
              f"{' (显著)' if stl['seasonal_dominant'] else ''}")

    if 'recommendations' in results:
        recs = results['recommendations']
        if recs:
            print("\n" + "=" * 60)
            print("平稳化建议:")
            print("=" * 60)
            for i, rec in enumerate(recs, 1):
                print(f"  {i}. {rec}")


if __name__ == "__main__":
    np.random.seed(42)

    print("=" * 70)
    print("示例1: 平稳序列（白噪声, n=100）")
    print("=" * 70)
    stationary_series = np.random.randn(100)
    results1 = stationarity_recommendation(stationary_series)
    print_test_results(results1)

    print("\n" + "#" * 70 + "\n")

    print("=" * 70)
    print("示例2: 非平稳序列（随机游走, n=100）")
    print("=" * 70)
    rw_series = np.cumsum(np.random.randn(100))
    results2 = stationarity_recommendation(rw_series)
    print_test_results(results2)

    print("\n" + "#" * 70 + "\n")

    print("=" * 70)
    print("示例3: 带趋势的序列（n=100）")
    print("=" * 70)
    trend_series = np.arange(100) + np.random.randn(100) * 5
    results3 = stationarity_recommendation(trend_series)
    print_test_results(results3)

    print("\n" + "#" * 70 + "\n")

    print("=" * 70)
    print("示例4: 指数增长序列（方差随水平增长, n=100）")
    print("=" * 70)
    exp_series = np.exp(np.linspace(0, 4, 100)) + np.random.randn(100) * 2
    results4 = stationarity_recommendation(exp_series)
    print_test_results(results4)

    print("\n" + "#" * 70 + "\n")

    print("=" * 70)
    print("示例5: 季节性序列（n=120, period=12）")
    print("=" * 70)
    t = np.arange(120)
    seasonal_series = 5 * np.sin(2 * np.pi * t / 12) + np.random.randn(120)
    results5 = stationarity_recommendation(seasonal_series, period=12)
    print_test_results(results5)
