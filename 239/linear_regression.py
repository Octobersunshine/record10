import warnings

import numpy as np
from scipy import stats


def linear_regression(x, y, alpha=0.05):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    n = len(x)
    if n != len(y):
        raise ValueError("x 和 y 的长度必须一致")
    if n < 3:
        raise ValueError("至少需要 3 个数据点才能进行回归和统计推断")

    x_mean = np.mean(x)
    y_mean = np.mean(y)

    sxx = np.sum((x - x_mean) ** 2)
    sxy = np.sum((x - x_mean) * (y - y_mean))

    if sxx == 0:
        warnings.warn(
            f"自变量方差为零（所有 x = {x_mean}），斜率为无穷大，拟合垂直直线 x = {x_mean}",
            RuntimeWarning,
            stacklevel=2,
        )
        residuals = y - y_mean
        return {
            "slope": float("inf"),
            "intercept": float("nan"),
            "r_squared": float("nan"),
            "residuals": residuals.tolist(),
            "vertical_line": True,
            "x_const": float(x_mean),
        }

    slope = sxy / sxx
    intercept = y_mean - slope * x_mean

    y_pred = intercept + slope * x
    residuals = y - y_pred

    ss_tot = np.sum((y - y_mean) ** 2)
    ss_res = np.sum(residuals ** 2)

    r_squared = 1 - ss_res / ss_tot if ss_tot != 0 else 0.0
    adjusted_r_squared = 1 - (1 - r_squared) * (n - 1) / (n - 2)

    df = n - 2
    mse = ss_res / df
    se = np.sqrt(mse)

    se_slope = se / np.sqrt(sxx)
    se_intercept = se * np.sqrt(1 / n + x_mean ** 2 / sxx)

    t_slope = slope / se_slope
    t_intercept = intercept / se_intercept

    p_slope = 2 * (1 - stats.t.cdf(abs(t_slope), df))
    p_intercept = 2 * (1 - stats.t.cdf(abs(t_intercept), df))

    t_crit = stats.t.ppf(1 - alpha / 2, df)
    ci_slope = [slope - t_crit * se_slope, slope + t_crit * se_slope]
    ci_intercept = [intercept - t_crit * se_intercept, intercept + t_crit * se_intercept]

    sorted_residuals = np.sort(residuals)
    theoretical_quantiles = stats.norm.ppf(np.linspace(0.5 / n, 1 - 0.5 / n, n))

    diagnostics = {
        "residual_vs_fitted": {
            "fitted_values": y_pred.tolist(),
            "residuals": residuals.tolist(),
        },
        "qq_plot": {
            "theoretical_quantiles": theoretical_quantiles.tolist(),
            "sample_quantiles": sorted_residuals.tolist(),
        },
    }

    return {
        "slope": float(slope),
        "intercept": float(intercept),
        "r_squared": float(r_squared),
        "adjusted_r_squared": float(adjusted_r_squared),
        "residuals": residuals.tolist(),
        "fitted_values": y_pred.tolist(),
        "std_error": float(se),
        "df": int(df),
        "mse": float(mse),
        "coefficients": {
            "slope": {
                "value": float(slope),
                "std_error": float(se_slope),
                "t_stat": float(t_slope),
                "p_value": float(p_slope),
                "confidence_interval": [float(ci_slope[0]), float(ci_slope[1])],
            },
            "intercept": {
                "value": float(intercept),
                "std_error": float(se_intercept),
                "t_stat": float(t_intercept),
                "p_value": float(p_intercept),
                "confidence_interval": [float(ci_intercept[0]), float(ci_intercept[1])],
            },
        },
        "diagnostics": diagnostics,
        "alpha": alpha,
    }


if __name__ == "__main__":
    x = [1, 2, 3, 4, 5]
    y = [2.2, 3.8, 6.5, 9.0, 11.3]

    result = linear_regression(x, y)

    print("=== 一元线性回归结果 ===")
    print(f"斜率 (slope):     {result['slope']:.6f}")
    print(f"截距 (intercept): {result['intercept']:.6f}")
    print(f"R² 决定系数:       {result['r_squared']:.6f}")
    print(f"调整 R²:           {result['adjusted_r_squared']:.6f}")
    print(f"残差标准误 (SE):   {result['std_error']:.6f}")
    print(f"自由度 (df):       {result['df']}")
    print(f"均方误差 (MSE):    {result['mse']:.6f}")
    print(f"残差序列:          {[round(r, 6) for r in result['residuals']]}")

    print("\n--- 系数统计推断 ---")
    for name, coeff in result["coefficients"].items():
        print(f"\n{name}:")
        print(f"  估计值:      {coeff['value']:.6f}")
        print(f"  标准误:      {coeff['std_error']:.6f}")
        print(f"  t 统计量:    {coeff['t_stat']:.6f}")
        print(f"  p 值:        {coeff['p_value']:.6f}")
        print(f"  95% 置信区间: [{coeff['confidence_interval'][0]:.6f}, {coeff['confidence_interval'][1]:.6f}]")

    print("\n--- 残差诊断图数据 ---")
    print("残差 vs 拟合值:")
    rvf = result["diagnostics"]["residual_vs_fitted"]
    for f, r in zip(rvf["fitted_values"], rvf["residuals"]):
        print(f"  拟合值: {f:.4f}, 残差: {r:.4f}")

    print("\nQ-Q 图 (理论分位数 vs 样本分位数):")
    qq = result["diagnostics"]["qq_plot"]
    for t, s in zip(qq["theoretical_quantiles"], qq["sample_quantiles"]):
        print(f"  理论分位数: {t:.4f}, 样本分位数: {s:.4f}")

    print("\n--- 自变量方差为零的测试 ---")
    x_const = [3, 3, 3, 3]
    y_const = [1, 4, 7, 10]

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        try:
            result_const = linear_regression(x_const, y_const)
        except ValueError as e:
            print(f"注意: {e}")
            x_const = [3, 3, 3, 3, 3, 3]
            y_const = [1, 4, 7, 10, 5, 8]
            result_const = linear_regression(x_const, y_const)

    if w:
        print(f"警告: {w[0].message}")

    print(f"斜率 (slope):     {result_const['slope']}")
    print(f"截距 (intercept): {result_const['intercept']}")
    print(f"R² 决定系数:       {result_const['r_squared']}")
    print(f"残差序列:          {[round(r, 6) for r in result_const['residuals']]}")
    print(f"垂直直线:          {result_const.get('vertical_line', False)}")
    print(f"x 常量值:          {result_const.get('x_const', 'N/A')}")
