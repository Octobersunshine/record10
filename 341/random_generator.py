import math
import random
import uuid
import statistics


DISTRIBUTIONS = {"uniform", "normal", "exponential"}


def generate_histogram(numbers, bins=10):
    if not numbers:
        return {"bins": [], "counts": [], "bin_edges": []}

    lo = min(numbers)
    hi = max(numbers)

    if lo == hi:
        return {
            "bins": [f"{lo:.2f}"],
            "counts": [len(numbers)],
            "bin_edges": [lo, hi],
        }

    step = (hi - lo) / bins
    bin_edges = [lo + step * i for i in range(bins + 1)]
    counts = [0] * bins

    for v in numbers:
        idx = int((v - lo) / step)
        if idx >= bins:
            idx = bins - 1
        counts[idx] += 1

    bin_labels = [
        f"{bin_edges[i]:.2f}-{bin_edges[i + 1]:.2f}" for i in range(bins)
    ]

    return {"bins": bin_labels, "counts": counts, "bin_edges": bin_edges}


def generate_random_numbers(
    data_type="int",
    low=0,
    high=100,
    count=1,
    seed=None,
    distribution="uniform",
    distribution_params=None,
    bins=10,
):
    if count < 1 or count > 10000:
        raise ValueError("count must be between 1 and 10000")
    if distribution not in DISTRIBUTIONS:
        raise ValueError(f"distribution must be one of {DISTRIBUTIONS}")

    if seed is None:
        seed = uuid.uuid4().int

    rng = random.Random(seed)

    if distribution_params is None:
        distribution_params = {}

    if distribution == "uniform":
        a = distribution_params.get("low", low)
        b = distribution_params.get("high", high)
        numbers = [rng.uniform(a, b) for _ in range(count)]
    elif distribution == "normal":
        mu = distribution_params.get("mu", 0)
        sigma = distribution_params.get("sigma", 1)
        if sigma <= 0:
            raise ValueError("sigma must be positive for normal distribution")
        numbers = [rng.gauss(mu, sigma) for _ in range(count)]
    elif distribution == "exponential":
        lambd = distribution_params.get("lambda", 1)
        if lambd <= 0:
            raise ValueError("lambda must be positive for exponential distribution")
        numbers = [rng.expovariate(lambd) for _ in range(count)]

    if data_type == "int":
        numbers = [int(round(v)) for v in numbers]

    stats = {
        "min": min(numbers),
        "max": max(numbers),
        "mean": statistics.mean(numbers),
        "median": statistics.median(numbers),
        "stdev": statistics.stdev(numbers) if count > 1 else 0.0,
    }

    histogram = generate_histogram(numbers, bins)

    return numbers, stats, seed, histogram


def main():
    print("=== 随机数生成器 ===")
    data_type = input("类型 (int/float) [int]: ").strip().lower() or "int"
    distribution = (
        input("分布 (uniform/normal/exponential) [uniform]: ").strip().lower()
        or "uniform"
    )

    distribution_params = {}
    if distribution == "normal":
        distribution_params["mu"] = float(input("均值 mu [0]: ").strip() or 0)
        distribution_params["sigma"] = float(input("标准差 sigma [1]: ").strip() or 1)
    elif distribution == "exponential":
        distribution_params["lambda"] = float(input("lambda [1]: ").strip() or 1)
    else:
        low = float(input("最小值 [0]: ").strip() or 0)
        high = float(input("最大值 [100]: ").strip() or 100)
        distribution_params["low"] = low
        distribution_params["high"] = high

    count = int(input("数量 (1-10000) [10]: ").strip() or 10)
    bins = int(input("直方图分箱数 [10]: ").strip() or 10)
    seed_input = input("随机种子 (留空自动生成) [空]: ").strip()
    seed = int(seed_input) if seed_input else None

    numbers, stats, seed, histogram = generate_random_numbers(
        data_type=data_type,
        low=distribution_params.get("low", 0),
        high=distribution_params.get("high", 100),
        count=count,
        seed=seed,
        distribution=distribution,
        distribution_params=distribution_params,
        bins=bins,
    )

    print(f"\n生成 {count} 个 {data_type} 类型随机数，分布: {distribution}")
    print(f"  使用种子: {seed}")
    if count <= 20:
        print(f"  数值: {numbers}")
    else:
        print(f"  前10个: {numbers[:10]} ... 后10个: {numbers[-10:]}")

    print("\n统计信息:")
    print(f"  最小值: {stats['min']}")
    print(f"  最大值: {stats['max']}")
    print(f"  均值:   {stats['mean']:.4f}")
    print(f"  中位数: {stats['median']:.4f}")
    print(f"  标准差: {stats['stdev']:.4f}")

    print(f"\n频率直方图 ({bins} 箱):")
    max_count = max(histogram["counts"]) if histogram["counts"] else 1
    for label, cnt in zip(histogram["bins"], histogram["counts"]):
        bar = "█" * int(cnt / max_count * 30)
        print(f"  {label:>25s} | {cnt:>5d} {bar}")


if __name__ == "__main__":
    main()
