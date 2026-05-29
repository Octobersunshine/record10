import math
import warnings
from collections import defaultdict
from decimal import Decimal, getcontext
from typing import Any, Callable, Dict, List, Tuple, Union

Number = Union[int, float]
DecimalOrFloat = Union[float, Decimal]


def cumulative_sum(seq: List[Number]) -> List[float]:
    result = []
    total = 0.0
    for x in seq:
        total += float(x)
        result.append(total)
    return result


def _cumulative_product_log_domain(
    seq: List[Number],
) -> Tuple[List[float], List[str]]:
    log_abs_cumsum = []
    sign_product = 1
    result = []
    warns = []
    zero_hit = False

    for i, x in enumerate(seq):
        fx = float(x)

        if zero_hit:
            result.append(0.0)
            continue

        if fx == 0.0:
            zero_hit = True
            log_abs_cumsum.append(float("-inf"))
            result.append(0.0)
            warns.append(f"[i={i}] 零值出现，后续累计积恒为 0")
            continue

        if fx < 0:
            sign_product *= -1
            log_abs = math.log(abs(fx))
        else:
            log_abs = math.log(fx)

        prev = log_abs_cumsum[-1] if log_abs_cumsum else 0.0
        cum_log = prev + log_abs
        log_abs_cumsum.append(cum_log)

        if cum_log < -708:
            warns.append(
                f"[i={i}] 对数域值 {cum_log:.2f} 低于 float64 下界 (≈-708)，"
                f"结果下溢为 0，建议使用 decimal 模式"
            )
            result.append(0.0 * sign_product)
            continue

        if cum_log > 709:
            warns.append(
                f"[i={i}] 对数域值 {cum_log:.2f} 超过 float64 上界 (≈709)，"
                f"结果溢出为 inf，建议使用 decimal 模式"
            )
            result.append(math.copysign(float("inf"), sign_product))
            continue

        val = math.exp(cum_log) * sign_product
        result.append(val)

    return result, warns


def _cumulative_product_decimal(
    seq: List[Number], precision: int = 50
) -> Tuple[List[Decimal], List[str]]:
    original_precision = getcontext().prec
    getcontext().prec = precision
    warns = []
    result = []
    cum = Decimal(1)

    for i, x in enumerate(seq):
        d = Decimal(str(x))
        cum *= d
        result.append(cum)

        if cum != 0 and abs(cum) < Decimal("1e-" + str(precision - 5)):
            warns.append(
                f"[i={i}] Decimal 累计积 {cum} 接近当前精度下界，"
                f"可提高 precision 参数 (当前 {precision})"
            )

    getcontext().prec = original_precision
    return result, warns


def cumulative_product(
    seq: List[Number],
    mode: str = "auto",
    decimal_precision: int = 50,
    return_decimal: bool = False,
) -> Tuple[List[DecimalOrFloat], List[str]]:
    if mode not in ("auto", "log", "decimal"):
        raise ValueError(f"mode 必须为 'auto'/'log'/'decimal'，收到 '{mode}'")

    if not seq:
        return [], []

    has_zero = any(float(x) == 0.0 for x in seq)
    has_negative = any(float(x) < 0 for x in seq)

    if mode == "auto":
        if has_zero and has_negative:
            mode = "decimal"
            warnings.warn(
                "序列同时包含零和负数，自动切换至 decimal 模式以避免精度损失",
                RuntimeWarning,
                stacklevel=2,
            )
        else:
            mode = "log"

    if mode == "decimal":
        dec_result, dec_warns = _cumulative_product_decimal(
            seq, precision=decimal_precision
        )
        if return_decimal:
            result = dec_result
        else:
            result = [float(v) for v in dec_result]
        warns = dec_warns
        if has_negative:
            warns.insert(0, "使用 decimal 模式处理含负数序列，符号已精确追踪")
    else:
        result, warns = _cumulative_product_log_domain(seq)
        if has_negative:
            warns.insert(0, "使用对数域计算，负数通过独立符号追踪处理")

    return result, warns


def cumulative_max(seq: List[Number]) -> List[float]:
    if not seq:
        return []
    result = [float(seq[0])]
    for x in seq[1:]:
        result.append(max(result[-1], float(x)))
    return result


def cumulative_min(seq: List[Number]) -> List[float]:
    if not seq:
        return []
    result = [float(seq[0])]
    for x in seq[1:]:
        result.append(min(result[-1], float(x)))
    return result


def cumulative_mean(seq: List[Number]) -> List[float]:
    if not seq:
        return []
    result = []
    n = 0
    total = 0.0
    for x in seq:
        n += 1
        total += float(x)
        result.append(total / n)
    return result


def cumulative_std(seq: List[Number], ddof: int = 0) -> List[float]:
    if not seq:
        return []
    result = []
    n = 0
    sum_x = 0.0
    sum_x2 = 0.0
    for x in seq:
        n += 1
        fx = float(x)
        sum_x += fx
        sum_x2 += fx * fx
        if n == 1:
            result.append(0.0)
        else:
            var = (sum_x2 - sum_x * sum_x / n) / max(n - ddof, 1)
            result.append(math.sqrt(max(var, 0.0)))
    return result


_CUM_FUNC_MAP: Dict[str, Callable] = {
    "sum": cumulative_sum,
    "max": cumulative_max,
    "min": cumulative_min,
    "mean": cumulative_mean,
    "std": cumulative_std,
}


def groupby_cumulative(
    values: List[Number],
    keys: List[Any],
    metrics: Union[str, List[str]] = "sum",
    product_mode: str = "auto",
) -> Dict[str, Dict[str, List[float]]]:
    if len(values) != len(keys):
        raise ValueError("values 和 keys 长度必须相同")

    if isinstance(metrics, str):
        metrics = [metrics]

    grouped_values: Dict[Any, List[float]] = defaultdict(list)
    grouped_indices: Dict[Any, List[int]] = defaultdict(list)
    for idx, (v, k) in enumerate(zip(values, keys)):
        grouped_values[k].append(float(v))
        grouped_indices[k].append(idx)

    result: Dict[str, Dict[str, List[float]]] = {}
    global_size = len(values)

    for key in grouped_values:
        group_vals = grouped_values[key]
        group_result: Dict[str, List[float]] = {}
        indices = grouped_indices[key]

        for metric in metrics:
            if metric == "prod":
                cum_vals, _ = cumulative_product(group_vals, mode=product_mode)
            elif metric in _CUM_FUNC_MAP:
                cum_vals = _CUM_FUNC_MAP[metric](group_vals)
            else:
                raise ValueError(
                    f"未知 metric '{metric}'，可选值: {list(_CUM_FUNC_MAP.keys()) + ['prod']}"
                )

            aligned = [0.0] * global_size
            for pos, idx in enumerate(indices):
                aligned[idx] = cum_vals[pos]
            group_result[metric] = aligned

        result[str(key)] = group_result

    return result


def prepare_visualization_data(
    seq: List[Number],
    x_labels: Union[List[str], None] = None,
    metrics: Union[str, List[str]] = "all",
) -> Dict[str, Any]:
    if x_labels is None:
        x_labels = [str(i) for i in range(len(seq))]

    if metrics == "all":
        metrics = ["sum", "mean", "std", "max", "min"]
    elif isinstance(metrics, str):
        metrics = [metrics]

    datasets: Dict[str, List[float]] = {}
    for metric in metrics:
        if metric == "sum":
            datasets["cumulative_sum"] = cumulative_sum(seq)
        elif metric == "mean":
            datasets["cumulative_mean"] = cumulative_mean(seq)
        elif metric == "std":
            datasets["cumulative_std"] = cumulative_std(seq)
        elif metric == "max":
            datasets["cumulative_max"] = cumulative_max(seq)
        elif metric == "min":
            datasets["cumulative_min"] = cumulative_min(seq)
        else:
            raise ValueError(
                f"未知 metric '{metric}'，可选值: sum, mean, std, max, min"
            )

    min_val = min(seq) if seq else 0
    max_val = max(seq) if seq else 0
    cum_sum = cumulative_sum(seq)

    return {
        "x_labels": x_labels,
        "original": [float(v) for v in seq],
        "datasets": datasets,
        "summary": {
            "length": len(seq),
            "min": min_val,
            "max": max_val,
            "mean": sum(float(v) for v in seq) / len(seq) if seq else 0,
            "total_sum": cum_sum[-1] if cum_sum else 0,
        },
    }


def cumulative_stats(
    seq: List[Number], product_mode: str = "auto", decimal_precision: int = 50
) -> dict:
    cumsum = cumulative_sum(seq)
    cumprod, cumprod_warns = cumulative_product(
        seq, mode=product_mode, decimal_precision=decimal_precision
    )
    cummean = cumulative_mean(seq)
    cumstd = cumulative_std(seq)
    cummax = cumulative_max(cumsum)
    cummin = cumulative_min(cumsum)
    return {
        "cumsum": cumsum,
        "cumprod": cumprod,
        "cummean": cummean,
        "cumstd": cumstd,
        "cummax": cummax,
        "cummin": cummin,
        "warnings": cumprod_warns,
    }


def _fmt(v):
    if v == 0:
        return "0.0"
    if isinstance(v, Decimal):
        s = str(v)
        return s if len(s) <= 15 else s[:15] + "..."
    if abs(v) >= 1e6 or (abs(v) < 1e-4 and v != 0):
        return f"{v:.6e}"
    return f"{v:.6f}"


def _print_aligned(values: List[Any], width: int = 12, sep: str = "|"):
    return sep + sep.join(f"{str(v)[:width-2]:^{width}}" for v in values) + sep


if __name__ == "__main__":
    print("=" * 70)
    print("场景 1: 新增累计均值 & 累计标准差")
    print("=" * 70)
    returns = [0.05, -0.02, 0.03, 0.04, -0.01, 0.02, 0.03, -0.015, 0.01, 0.025]
    cum_mean = cumulative_mean(returns)
    cum_std = cumulative_std(returns)
    print(f"日收益率: {returns}")
    print(f"累计均值: [{', '.join(_fmt(v) for v in cum_mean)}]")
    print(f"累计标准差: [{', '.join(_fmt(v) for v in cum_std)}]")

    print()
    print("=" * 70)
    print("场景 2: 分组累计 — 按股票代码分组计算累计收益")
    print("=" * 70)
    stocks = ["AAPL", "AAPL", "GOOG", "AAPL", "GOOG", "GOOG", "MSFT", "MSFT"]
    daily_ret = [0.02, -0.01, 0.03, 0.015, -0.02, 0.025, 0.01, -0.005]
    grouped = groupby_cumulative(daily_ret, stocks, metrics=["sum", "mean"])

    print(f"股票: {stocks}")
    print(f"收益率: {daily_ret}")
    print()
    for ticker in grouped:
        print(f"  {ticker}:")
        for metric in grouped[ticker]:
            vals = [_fmt(v) if v != 0 else "  --  " for v in grouped[ticker][metric]]
            print(f"    {metric:5s}: [{', '.join(vals)}]")

    print()
    print("=" * 70)
    print("场景 3: 可视化数据准备")
    print("=" * 70)
    viz_data = prepare_visualization_data(
        [0.05, -0.02, 0.03, 0.04, -0.01, 0.02],
        x_labels=["Day1", "Day2", "Day3", "Day4", "Day5", "Day6"],
        metrics=["sum", "mean", "std"],
    )
    print(f"x 轴标签: {viz_data['x_labels']}")
    print(f"原始数据: {viz_data['original']}")
    print(f"可用数据集: {list(viz_data['datasets'].keys())}")
    for name, data in viz_data["datasets"].items():
        print(f"  {name}: [{', '.join(_fmt(v) for v in data)}]")
    print(f"统计摘要: {viz_data['summary']}")

    print()
    print("=" * 70)
    print("场景 4: cumulative_stats 完整版 (含新指标)")
    print("=" * 70)
    stats = cumulative_stats([0.05, -0.02, 0.03, 0.04, -0.01])
    print(f"cumsum:   [{', '.join(_fmt(v) for v in stats['cumsum'])}]")
    print(f"cumprod:  [{', '.join(_fmt(v) for v in stats['cumprod'])}]")
    print(f"cummean:  [{', '.join(_fmt(v) for v in stats['cummean'])}]")
    print(f"cumstd:   [{', '.join(_fmt(v) for v in stats['cumstd'])}]")
    print(f"cummax:   [{', '.join(_fmt(v) for v in stats['cummax'])}]")
    print(f"cummin:   [{', '.join(_fmt(v) for v in stats['cummin'])}]")
    if stats["warnings"]:
        print("⚠ 警告:")
        for warn in stats["warnings"]:
            print(f"  {warn}")

    print()
    print("=" * 70)
    print("场景 5: 分组累计积 (带符号)")
    print("=" * 70)
    factors = ["F1", "F1", "F2", "F1", "F2"]
    values = [1.2, -0.9, 1.5, 1.1, 0.8]
    grouped_prod = groupby_cumulative(values, factors, metrics="prod")
    print(f"因子: {factors}")
    print(f"数值: {values}")
    for key in grouped_prod:
        vals = [_fmt(v) if v != 0 else "  --  " for v in grouped_prod[key]["prod"]]
        print(f"  {key} 累计积: [{', '.join(vals)}]")
