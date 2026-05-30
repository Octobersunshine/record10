import json
import math
import statistics
from typing import List, Dict, Union, Tuple


def _is_missing(value) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    return False


def handle_missing_values(
    data: List[Union[float, None]],
    strategy: str = "ignore"
) -> Dict:
    total = len(data)
    missing_count = sum(1 for v in data if _is_missing(v))
    missing_ratio = round(missing_count / total, 4) if total > 0 else 0.0

    valid_values = [v for v in data if not _is_missing(v)]

    if not valid_values:
        return {
            "cleaned": [],
            "missing_count": missing_count,
            "missing_ratio": missing_ratio
        }

    if strategy == "ignore":
        cleaned = valid_values
    elif strategy == "fill_mean":
        fill_val = statistics.mean(valid_values)
        cleaned = [fill_val if _is_missing(v) else v for v in data]
    elif strategy == "fill_median":
        fill_val = statistics.median(valid_values)
        cleaned = [fill_val if _is_missing(v) else v for v in data]
    else:
        raise ValueError(f"不支持的缺失值处理策略: {strategy}，可选: ignore, fill_mean, fill_median")

    return {
        "cleaned": cleaned,
        "missing_count": missing_count,
        "missing_ratio": missing_ratio
    }


def calculate_five_number_summary(data: List[float]) -> Dict[str, float]:
    sorted_data = sorted(data)
    n = len(sorted_data)

    min_val = sorted_data[0]
    max_val = sorted_data[-1]

    if n % 2 == 0:
        median = (sorted_data[n // 2 - 1] + sorted_data[n // 2]) / 2
    else:
        median = sorted_data[n // 2]

    lower_half = sorted_data[:n // 2]
    upper_half = sorted_data[(n + 1) // 2:]

    q1 = statistics.median(lower_half)
    q3 = statistics.median(upper_half)

    return {
        "min": min_val,
        "q1": q1,
        "median": median,
        "q3": q3,
        "max": max_val
    }


def detect_outliers_iqr(
    data: List[float],
    q1: float,
    q3: float,
    k: float = 1.5
) -> Dict[str, Union[List[float], float]]:
    iqr = q3 - q1
    lower_bound = q1 - k * iqr
    upper_bound = q3 + k * iqr

    outliers_low = [v for v in data if v < lower_bound]
    outliers_high = [v for v in data if v > upper_bound]
    outliers = sorted(outliers_low + outliers_high)

    non_outliers = [v for v in data if lower_bound <= v <= upper_bound]

    if non_outliers:
        lower_whisker = min(non_outliers)
        upper_whisker = max(non_outliers)
    else:
        lower_whisker = lower_bound
        upper_whisker = upper_bound

    return {
        "outliers": outliers,
        "outliers_low": sorted(outliers_low),
        "outliers_high": sorted(outliers_high),
        "outlier_count": len(outliers),
        "lower_bound": lower_bound,
        "upper_bound": upper_bound,
        "lower_whisker": lower_whisker,
        "upper_whisker": upper_whisker,
        "iqr": iqr,
        "k": k
    }


def calculate_statistics(
    data: List[Union[float, None]],
    missing: str = "ignore",
    outlier_k: float = 1.5
) -> Dict[str, Union[float, int, List[float]]]:
    if not data:
        raise ValueError("数据不能为空")

    result = handle_missing_values(data, strategy=missing)
    cleaned = result["cleaned"]

    if len(cleaned) < 2:
        raise ValueError("有效数据不足，至少需要2个数据点才能计算标准差")

    five_num = calculate_five_number_summary(cleaned)
    mean = statistics.mean(cleaned)
    std_dev = statistics.stdev(cleaned)

    outlier_result = detect_outliers_iqr(
        cleaned,
        q1=five_num["q1"],
        q3=five_num["q3"],
        k=outlier_k
    )

    boxplot_data = {
        "min": five_num["min"],
        "q1": five_num["q1"],
        "median": five_num["median"],
        "q3": five_num["q3"],
        "max": five_num["max"],
        "lower_whisker": outlier_result["lower_whisker"],
        "upper_whisker": outlier_result["upper_whisker"],
        "lower_bound": outlier_result["lower_bound"],
        "upper_bound": outlier_result["upper_bound"],
        "iqr": outlier_result["iqr"]
    }

    return {
        **five_num,
        "mean": mean,
        "std_dev": std_dev,
        "count": len(cleaned),
        "missing_count": result["missing_count"],
        "missing_ratio": result["missing_ratio"],
        "outliers": outlier_result["outliers"],
        "outliers_low": outlier_result["outliers_low"],
        "outliers_high": outlier_result["outliers_high"],
        "outlier_count": outlier_result["outlier_count"],
        "boxplot": boxplot_data
    }


def batch_statistics(
    datasets: Dict[str, List[Union[float, None]]],
    missing: str = "ignore",
    outlier_k: float = 1.5
) -> Dict[str, Dict[str, Union[float, int, List[float]]]]:
    results = {}
    for name, data in datasets.items():
        try:
            results[name] = calculate_statistics(data, missing=missing, outlier_k=outlier_k)
        except Exception as e:
            results[name] = {"error": str(e)}
    return results


def group_statistics(
    values: List[Union[float, None]],
    categories: List,
    missing: str = "ignore",
    outlier_k: float = 1.5
) -> Dict[str, Dict[str, Union[float, int, List[float]]]]:
    if len(values) != len(categories):
        raise ValueError("数值列表和类别列表长度必须相等")

    grouped: Dict[str, List[Union[float, None]]] = {}
    for val, cat in zip(values, categories):
        cat_key = str(cat)
        if cat_key not in grouped:
            grouped[cat_key] = []
        grouped[cat_key].append(val)

    return batch_statistics(grouped, missing=missing, outlier_k=outlier_k)


def statistics_summary(
    data: Union[
        List[Union[float, None]],
        Dict[str, List[Union[float, None]]],
        Tuple[List[Union[float, None]], List]
    ],
    missing: str = "ignore",
    outlier_k: float = 1.5,
    pretty: bool = True
) -> str:
    if isinstance(data, tuple):
        if len(data) != 2:
            raise ValueError("分组数据应为 (values, categories) 元组")
        summary = group_statistics(data[0], data[1], missing=missing, outlier_k=outlier_k)
    elif isinstance(data, dict):
        summary = batch_statistics(data, missing=missing, outlier_k=outlier_k)
    else:
        summary = {"dataset": calculate_statistics(data, missing=missing, outlier_k=outlier_k)}

    indent = 2 if pretty else None
    return json.dumps(summary, ensure_ascii=False, indent=indent, default=str)


if __name__ == "__main__":
    data_with_outliers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 50, 100, -20, 0.5, 5.5]
    data_with_missing = [12, None, 3, 15, float('nan'), 10, 5, 2, 9, None, 6, 4, 13, 1, 14]

    print("=== 异常值检测测试 ===")
    print(statistics_summary(data_with_outliers, missing="ignore"))
    print()

    print("=== 含缺失值 + 异常值检测 ===")
    print(statistics_summary(data_with_missing, missing="fill_mean"))
    print()

    values = [85, 92, 78, 90, 88, 76, 95, 89, 82, 91, 75, 87, 93, 80, 86]
    categories = ["A班", "B班", "A班", "B班", "A班", "B班", "A班", "B班",
                  "A班", "B班", "A班", "B班", "A班", "B班", "A班"]

    print("=== 按班级分组统计 ===")
    print(statistics_summary((values, categories), missing="ignore"))
    print()

    batch_data = {
        "销售部": [5000, 6000, 5500, 7000, 6500, 15000, 4800, 5200],
        "技术部": [8000, 9000, 8500, 9500, 8800, 9200, 50000, 8700],
        "市场部": [4500, 4800, 5200, 4900, 5100, 4700, 4600, 20000]
    }

    print("=== 多部门批量统计（含异常值）===")
    print(statistics_summary(batch_data, missing="ignore"))
