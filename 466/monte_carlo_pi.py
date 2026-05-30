import random
import math
import time
import multiprocessing
from dataclasses import dataclass


def _worker_batch(args: tuple[int, int | None]) -> int:
    num_points, seed = args
    rng = random.Random(seed)
    inside = 0
    for _ in range(num_points):
        x = rng.random()
        y = rng.random()
        if (x - 0.5) ** 2 + (y - 0.5) ** 2 <= 0.25:
            inside += 1
    return inside


def estimate_pi(num_points: int, seed: int | None = None) -> tuple[float, float]:
    if seed is not None:
        random.seed(seed)

    inside_circle = 0

    for _ in range(num_points):
        x = random.uniform(0, 1)
        y = random.uniform(0, 1)
        if (x - 0.5) ** 2 + (y - 0.5) ** 2 <= 0.25:
            inside_circle += 1

    pi_estimate = 4 * (inside_circle / num_points)
    error = abs(pi_estimate - math.pi)

    return pi_estimate, error


def estimate_pi_parallel(
    num_points: int,
    num_workers: int | None = None,
    seed: int | None = None
) -> tuple[float, float]:
    if num_workers is None:
        num_workers = multiprocessing.cpu_count()

    batch_size = num_points // num_workers
    remainder = num_points % num_workers

    tasks = []
    for i in range(num_workers):
        count = batch_size + (1 if i < remainder else 0)
        if seed is not None:
            task_seed = seed + i * 10007
        else:
            task_seed = int(time.time() * 1000) + i * 10007
        tasks.append((count, task_seed))

    with multiprocessing.Pool(processes=num_workers) as pool:
        results = pool.map(_worker_batch, tasks)

    total_inside = sum(results)
    total_points = sum(t[0] for t in tasks)
    pi_estimate = 4 * (total_inside / total_points)
    error = abs(pi_estimate - math.pi)

    return pi_estimate, error


@dataclass
class ConvergenceResult:
    pi_estimate: float
    error: float
    total_points: int
    iterations: int
    curve_points: list[tuple[int, float]]


def estimate_pi_to_precision(
    epsilon: float = 0.001,
    initial_points: int = 10000,
    growth_factor: int = 4,
    max_points: int = 10_000_000_000,
    seed: int | None = None,
    num_workers: int | None = None,
    use_parallel: bool = False
) -> ConvergenceResult:
    rng = random.Random(seed)
    total_inside = 0
    total_points = 0
    curve_points: list[tuple[int, float]] = []
    current_batch = initial_points
    iterations = 0

    while True:
        batch_inside = 0
        if use_parallel:
            pi_est, _ = estimate_pi_parallel(current_batch, num_workers, seed)
            batch_inside = int(pi_est / 4 * current_batch)
        else:
            for _ in range(current_batch):
                x = rng.random()
                y = rng.random()
                if (x - 0.5) ** 2 + (y - 0.5) ** 2 <= 0.25:
                    batch_inside += 1

        total_inside += batch_inside
        total_points += current_batch

        pi_estimate = 4 * (total_inside / total_points)
        error = abs(pi_estimate - math.pi)
        curve_points.append((total_points, pi_estimate))
        iterations += 1

        if error < epsilon or total_points >= max_points:
            return ConvergenceResult(
                pi_estimate=pi_estimate,
                error=error,
                total_points=total_points,
                iterations=iterations,
                curve_points=curve_points,
            )

        current_batch = min(current_batch * growth_factor, max_points - total_points)


def estimate_pi_average(
    num_points: int,
    num_runs: int = 10,
    seed: int | None = None
) -> tuple[float, float, float]:
    pi_estimates = []

    for i in range(num_runs):
        if seed is not None:
            run_seed = seed + i
        else:
            run_seed = int(time.time() * 1000) + i

        pi, _ = estimate_pi(num_points, run_seed)
        pi_estimates.append(pi)

    pi_avg = sum(pi_estimates) / num_runs
    pi_variance = sum((p - pi_avg) ** 2 for p in pi_estimates) / num_runs
    pi_std = math.sqrt(pi_variance)
    error_avg = abs(pi_avg - math.pi)

    return pi_avg, error_avg, pi_std


if __name__ == "__main__":
    print("=" * 65)
    print("1. 基本估算 (单线程)")
    print("=" * 65)
    for n in [1_000, 10_000, 100_000]:
        t0 = time.perf_counter()
        pi_est, err = estimate_pi(n)
        dt = time.perf_counter() - t0
        print(f"  n={n:<10} π={pi_est:.10f} 误差={err:.10f} 耗时={dt:.4f}s")

    print()
    print("=" * 65)
    print("2. 并行估算 (多进程)")
    print("=" * 65)
    workers = multiprocessing.cpu_count()
    print(f"  CPU核心数: {workers}")
    for n in [100_000, 1_000_000, 10_000_000]:
        t0 = time.perf_counter()
        pi_est, err = estimate_pi_parallel(n)
        dt = time.perf_counter() - t0
        print(f"  n={n:<12} π={pi_est:.10f} 误差={err:.10f} 耗时={dt:.4f}s")

    print()
    print("=" * 65)
    print("3. 精度自适应停止 (误差 < 0.001)")
    print("=" * 65)
    for eps in [0.01, 0.001]:
        t0 = time.perf_counter()
        result = estimate_pi_to_precision(epsilon=eps, use_parallel=True)
        dt = time.perf_counter() - t0
        print(f"  ε={eps}")
        print(f"    π={result.pi_estimate:.10f} 误差={result.error:.10f}")
        print(f"    总投点={result.total_points:,} 迭代次数={result.iterations} 耗时={dt:.4f}s")

    print()
    print("=" * 65)
    print("4. 收敛曲线数据 (投点数 vs 估算值, ε=0.001)")
    print("=" * 65)
    result = estimate_pi_to_precision(epsilon=0.001, use_parallel=True)
    print(f"  {'投点数':<18} {'π估算值':<18} {'误差':<15}")
    print(f"  {'-' * 50}")
    for points, pi_val in result.curve_points:
        err = abs(pi_val - math.pi)
        print(f"  {points:<18,} {pi_val:<18.10f} {err:<15.10f}")
    print(f"\n  真实π值: {math.pi:.10f}")
