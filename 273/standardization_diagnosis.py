import pandas as pd
import numpy as np
from scipy import stats


def diagnose_before_standardization(df):
    if not isinstance(df, pd.DataFrame):
        raise TypeError("输入必须是 pandas DataFrame")

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        raise ValueError("DataFrame 中没有数值型特征")

    records = []
    for col in numeric_cols:
        series = df[col]
        mean_val = series.mean()
        std_val = series.std(ddof=1)
        min_val = series.min()
        max_val = series.max()
        missing_count = series.isna().sum()
        valid_series = series.dropna()
        n_valid = len(valid_series)

        if n_valid >= 3:
            skew_val = valid_series.skew()
        else:
            skew_val = None

        if n_valid >= 4:
            kurt_val = valid_series.kurtosis()
        else:
            kurt_val = None

        outlier_ratio = None
        if n_valid >= 4:
            q1 = valid_series.quantile(0.25)
            q3 = valid_series.quantile(0.75)
            iqr = q3 - q1
            if iqr > 0:
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                outliers = (valid_series < lower_bound) | (valid_series > upper_bound)
                outlier_ratio = outliers.sum() / n_valid
            else:
                outlier_ratio = 0.0

        normality_test = None
        normality_p = None
        if n_valid >= 3 and std_val > 0:
            try:
                if 3 <= n_valid < 5000:
                    stat, p_val = stats.shapiro(valid_series)
                    normality_test = "Shapiro-Wilk"
                else:
                    stat, p_val = stats.normaltest(valid_series)
                    normality_test = "D'Agostino"
                normality_p = p_val
            except Exception:
                normality_test = None
                normality_p = None

        range_val = max_val - min_val
        need_std = False
        reasons = []

        if std_val == 0:
            need_std = False
            reasons.append("标准差为0，常量特征无需标准化")
        else:
            cv = abs(std_val / mean_val) if mean_val != 0 else float("inf")

            if range_val > 100:
                need_std = True
                reasons.append(f"值域过大(={range_val:.2f})")

            if cv > 1.0:
                need_std = True
                reasons.append(f"变异系数较大(CV={cv:.2f})")

            if skew_val is not None and abs(skew_val) > 1.0:
                need_std = True
                reasons.append(f"偏度较大(skew={skew_val:.2f})")
            elif skew_val is None:
                reasons.append(f"有效样本量不足({n_valid}<3)，偏度无法计算")

            if kurt_val is not None and abs(kurt_val) > 3.0:
                need_std = True
                reasons.append(f"峰度较大(kurt={kurt_val:.2f})")
            elif kurt_val is None:
                reasons.append(f"有效样本量不足({n_valid}<4)，峰度无法计算")

            if outlier_ratio is not None and outlier_ratio > 0.05:
                need_std = True
                reasons.append(f"异常值比例过高({outlier_ratio:.2%})")
            elif outlier_ratio is None:
                reasons.append(f"有效样本量不足({n_valid}<4)，异常值无法计算")

            if normality_p is not None and normality_p < 0.05:
                reasons.append(f"{normality_test}检验拒绝正态性假设(p={normality_p:.4f})")
            elif normality_p is not None and normality_p >= 0.05:
                reasons.append(f"{normality_test}检验符合正态性假设(p={normality_p:.4f})")
            elif normality_p is None and n_valid < 3:
                reasons.append(f"有效样本量不足({n_valid}<3)，正态性检验无法计算")

            for other_col in numeric_cols:
                if other_col == col:
                    continue
                other_std = df[other_col].std(ddof=1)
                if other_std > 0 and (std_val / other_std > 10 or other_std / std_val > 10):
                    need_std = True
                    reasons.append(f"与特征'{other_col}'量级差异大")
                    break

        if not reasons:
            if need_std:
                reasons.append("特征分布需要标准化处理")
            else:
                reasons.append("特征分布较均匀，标准化可选")

        scaler_recommendation = "不标准化"
        scaler_reason = ""
        if std_val == 0:
            scaler_recommendation = "不标准化"
            scaler_reason = "常量特征无需标准化"
        elif n_valid < 3:
            scaler_recommendation = "不标准化"
            scaler_reason = "样本量不足，暂不建议标准化"
        else:
            has_high_outliers = outlier_ratio is not None and outlier_ratio > 0.05
            has_high_skew = skew_val is not None and abs(skew_val) > 1.5
            is_normal = normality_p is not None and normality_p >= 0.05
            has_low_skew = skew_val is not None and abs(skew_val) <= 1.0
            has_low_outliers = outlier_ratio is not None and outlier_ratio <= 0.05

            if has_high_outliers or has_high_skew:
                scaler_recommendation = "RobustScaler"
                scaler_reason = "对异常值和偏度具有鲁棒性"
            elif is_normal and has_low_skew and has_low_outliers:
                scaler_recommendation = "Z-score"
                scaler_reason = "接近正态分布，适合标准差标准化"
            elif range_val <= 100 and (outlier_ratio is None or outlier_ratio <= 0.05):
                scaler_recommendation = "MinMaxScaler"
                scaler_reason = "值域适中且无显著异常值，适合区间缩放"
            else:
                scaler_recommendation = "RobustScaler"
                scaler_reason = "分布特性复杂，优先使用鲁棒性缩放"

        records.append(
            {
                "特征": col,
                "均值": round(mean_val, 4),
                "标准差": round(std_val, 4),
                "最小值": round(min_val, 4),
                "最大值": round(max_val, 4),
                "缺失值数量": int(missing_count),
                "偏度": round(skew_val, 4) if skew_val is not None else None,
                "峰度": round(kurt_val, 4) if kurt_val is not None else None,
                "异常值比例": round(outlier_ratio, 4) if outlier_ratio is not None else None,
                "正态性检验": normality_test,
                "正态性P值": round(normality_p, 4) if normality_p is not None else None,
                "建议标准化": "是" if need_std else "否",
                "推荐方案": scaler_recommendation,
                "方案依据": scaler_reason,
                "建议原因": "; ".join(reasons),
            }
        )

    result = pd.DataFrame(records)
    return result


if __name__ == "__main__":
    np.random.seed(42)
    demo = pd.DataFrame(
        {
            "age": np.random.normal(45, 12, 500),
            "income": np.random.lognormal(10, 1.2, 500),
            "score": np.random.uniform(0, 100, 500),
            "height_cm": np.random.normal(170, 8, 500),
            "tiny_feature": np.random.uniform(0.001, 0.01, 500),
            "constant_feature": np.full(500, 3.14),
            "feature_with_nan": np.where(
                np.random.rand(500) < 0.1, np.nan, np.random.normal(50, 15, 500)
            ),
            "skewed_feature": np.random.exponential(5, 500),
            "small_sample": [1.0, 2.0] + [np.nan] * 498,
            "tiny_sample": [5.0] + [np.nan] * 499,
            "feature_with_outliers": np.concatenate(
                [np.random.normal(50, 10, 475), np.random.uniform(200, 500, 25)]
            ),
        }
    )

    report = diagnose_before_standardization(demo)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 300)
    pd.set_option("display.max_colwidth", 80)
    print(report.to_string(index=False))
