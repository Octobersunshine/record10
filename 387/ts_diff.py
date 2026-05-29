import pandas as pd
import numpy as np


def time_series_changes(series, name="value", fill_strategy=None, lag=1, seasonal_lag=None):
    if not isinstance(series, pd.Series):
        series = pd.Series(series, name=name)

    fill_series = series.copy()
    fill_info = None

    if fill_strategy is not None:
        nan_count_before = fill_series.isna().sum()
        if nan_count_before > 0:
            if fill_strategy == "ffill":
                fill_series = fill_series.ffill()
            elif fill_strategy == "interpolate":
                fill_series = fill_series.interpolate(method="linear")
            else:
                raise ValueError(
                    f"未知的填充策略: {fill_strategy}。可选值: None, 'ffill', 'interpolate'"
                )
            nan_count_after = fill_series.isna().sum()
            fill_info = {
                "strategy": fill_strategy,
                "nan_before": nan_count_before,
                "nan_after": nan_count_after,
            }

    first_diff = fill_series.diff(1)
    second_diff = first_diff.diff(1)
    pct_change = fill_series.pct_change(1) * 100

    result = pd.DataFrame({
        series.name or name: series,
        "一阶差分": first_diff,
        "二阶差分": second_diff,
        "百分比变化(%)": pct_change,
    })

    if lag != 1:
        lag_diff = fill_series.diff(lag)
        lag_pct = fill_series.pct_change(lag) * 100
        result[f"滞后{lag}期差分"] = lag_diff
        result[f"滞后{lag}期变化(%)"] = lag_pct

    if seasonal_lag is not None:
        seasonal_diff = fill_series.diff(seasonal_lag)
        seasonal_pct = fill_series.pct_change(seasonal_lag) * 100
        result[f"季节差分(S={seasonal_lag})"] = seasonal_diff
        result[f"季节变化(%)"] = seasonal_pct

    stats = result.describe()

    variance = result.var()
    zero_ratio = (result == 0).sum() / len(result)

    extra_stats = pd.DataFrame({
        "方差": variance,
        "零值比例": zero_ratio,
    })

    return result, stats, extra_stats, fill_info


def _print_block(title, df, stats, extra_stats, fill_info):
    print("=" * 60)
    print(title)
    print("=" * 60)
    if fill_info:
        print(f"\n【缺失值填充】策略: {fill_info['strategy']}, "
              f"填充前NaN: {fill_info['nan_before']}, 填充后NaN: {fill_info['nan_after']}")
    print("\n【变化序列】")
    print(df.round(4).to_string())
    print("\n【统计摘要】")
    print(stats.round(4).to_string())
    print("\n【方差与零值比例】")
    print(extra_stats.round(4).to_string())
    print("\n")


if __name__ == "__main__":
    np.random.seed(42)
    data = np.cumsum(np.random.randn(12) * 5 + 2) + 100
    dates = pd.date_range("2025-01", periods=12, freq="MS")
    ts = pd.Series(data, index=dates, name="销售额")

    ts_with_nan = ts.copy()
    ts_with_nan.iloc[[2, 5, 8]] = np.nan

    df_base, stats_base, extra_base, info_base = time_series_changes(
        ts_with_nan, fill_strategy=None, lag=3, seasonal_lag=6
    )
    _print_block("基础分析 (滞后3期+季节差分S=6)", df_base, stats_base, extra_base, info_base)

    df_ffill, stats_ffill, extra_ffill, info_ffill = time_series_changes(
        ts_with_nan, fill_strategy="ffill", lag=3, seasonal_lag=6
    )
    _print_block("前向填充分析 (滞后3期+季节差分S=6)", df_ffill, stats_ffill, extra_ffill, info_ffill)
