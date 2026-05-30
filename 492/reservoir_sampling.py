import heapq
import math
import os
import random
from typing import Dict, Iterator, List, Optional, Tuple, TypeVar, Union

T = TypeVar('T')
WeightedItem = Tuple[T, float]
SampledItem = Tuple[float, T]
DistributedSample = List[SampledItem]


def reservoir_sampling(stream: Iterator[T], k: int, seed: Optional[int] = None) -> List[T]:
    """
    蓄水池采样算法：从未知长度的数据流中随机抽取k个元素，
    保证每个元素被选中的概率相等（k/N）。

    Args:
        stream: 数据流迭代器
        k: 需要采样的元素数量
        seed: 随机种子，指定后可复现结果；为None时使用系统熵源确保随机性

    Returns:
        包含k个随机采样元素的列表
    """
    if seed is not None:
        rng = random.Random(seed)
    else:
        rng = random.Random(os.urandom(32))

    reservoir = []

    for i, item in enumerate(stream):
        if i < k:
            reservoir.append(item)
        else:
            j = rng.randint(0, i)
            if j < k:
                reservoir[j] = item

    return reservoir


def weighted_reservoir_sampling(
    stream: Iterator[WeightedItem],
    k: int,
    seed: Optional[int] = None,
    return_keys: bool = False
) -> Union[List[T], DistributedSample]:
    """
    加权蓄水池采样算法（A-Res）：从未知长度的加权数据流中随机抽取k个元素，
    每个元素被选中的概率与其权重成正比。

    Args:
        stream: 带权重的数据流迭代器，每个元素为 (item, weight)
        k: 需要采样的元素数量
        seed: 随机种子，指定后可复现结果；为None时使用系统熵源确保随机性
        return_keys: 是否返回采样key（用于分布式合并）

    Returns:
        如果return_keys为False，返回包含k个采样元素的列表；
        如果return_keys为True，返回包含(key, item)元组的列表，用于分布式合并
    """
    if seed is not None:
        rng = random.Random(seed)
    else:
        rng = random.Random(os.urandom(32))

    heap: List[Tuple[float, int, T]] = []

    for idx, (item, weight) in enumerate(stream):
        if weight <= 0:
            continue

        u = rng.random()
        while u == 0:
            u = rng.random()
        key = -math.log(u) / weight

        if len(heap) < k:
            heapq.heappush(heap, (-key, idx, item))
        elif key < -heap[0][0]:
            heapq.heapreplace(heap, (-key, idx, item))

    if return_keys:
        return [(-neg_key, item) for (neg_key, _, item) in sorted(heap)]
    else:
        return [item for (_, _, item) in heap]


def merge_distributed_samples(
    node_samples: List[DistributedSample],
    k: int
) -> DistributedSample:
    """
    合并多节点分布式蓄水池采样结果。
    每个节点独立进行加权蓄水池采样并返回(key, item)，
    合并时选择全局key最大的k个元素。

    Args:
        node_samples: 各节点的采样结果列表，每个元素为(key, item)元组列表
        k: 最终需要采样的元素数量

    Returns:
        合并后的采样结果，包含(key, item)元组，按key升序排列（key越小权重越高）
    """
    merged: List[Tuple[float, T]] = []
    for sample in node_samples:
        merged.extend(sample)

    merged.sort(key=lambda x: x[0])
    return merged[:k]


def verify_weighted_sampling():
    """
    验证加权蓄水池采样算法的正确性：
    运行多次采样，统计每个元素被选中的频率，验证概率与权重成正比。
    """
    N = 10
    k = 3
    num_tests = 2000000

    items = list(range(N))
    weights = [float(i + 1) for i in range(N)]
    total_weight = sum(weights)

    expected_probs = [w * k / total_weight for w in weights]

    rng = random.Random(42)
    counts = [0] * N

    for _ in range(num_tests):
        stream = iter([(items[i], weights[i]) for i in range(N)])
        sample = weighted_reservoir_sampling(stream, k, seed=rng.randint(0, 2**63))
        for item in sample:
            counts[item] += 1

    print(f"总权重: {total_weight:.1f}, k={k}")
    print(f"{'元素':<6}{'权重':<8}{'期望概率':<10}{'实际频率':<10}{'误差':<10}")
    print("-" * 50)
    max_error = 0.0
    for i in range(N):
        actual = counts[i] / num_tests
        error = abs(actual - expected_probs[i])
        max_error = max(max_error, error)
        print(f"{i:<6}{weights[i]:<8.1f}{expected_probs[i]:<10.4f}{actual:<10.4f}{error:<10.4f}")

    print(f"\n最大误差: {max_error:.4f}")
    print(f"验证通过: {max_error < 0.05}")

    return max_error < 0.05


def verify_distributed_sampling():
    """
    验证分布式蓄水池采样的正确性：
    比较分布式合并结果与单节点全量采样结果的分布一致性。
    """
    N = 50
    k = 5
    num_nodes = 5
    num_tests = 20000

    items = list(range(N))
    weights = [1.0 + (i % 5) for i in range(N)]
    total_weight = sum(weights)
    expected_probs = [w * k / total_weight for w in weights]

    rng = random.Random(123)

    def split_stream(node_id: int, num_nodes: int) -> Iterator[WeightedItem]:
        for i in range(node_id, N, num_nodes):
            yield (items[i], weights[i])

    dist_counts = [0] * N
    single_counts = [0] * N

    for test_i in range(num_tests):
        base_seed = rng.randint(0, 2**63)

        node_samples = []
        for node_id in range(num_nodes):
            stream = split_stream(node_id, num_nodes)
            sample = weighted_reservoir_sampling(
                stream, k, seed=base_seed + node_id, return_keys=True
            )
            node_samples.append(sample)

        dist_result = merge_distributed_samples(node_samples, k)
        for _, item in dist_result:
            dist_counts[item] += 1

        full_stream = iter([(items[i], weights[i]) for i in range(N)])
        single_sample = weighted_reservoir_sampling(
            full_stream, k, seed=base_seed + 999
        )
        for item in single_sample:
            single_counts[item] += 1

    print(f"元素数: {N}, 节点数: {num_nodes}, k={k}, 测试次数: {num_tests}")
    print(f"{'元素':<6}{'期望':<10}{'分布式':<10}{'单节点':<10}")
    print("-" * 40)
    max_dist_error = 0.0
    max_single_error = 0.0
    for i in range(min(15, N)):
        dist_freq = dist_counts[i] / num_tests
        single_freq = single_counts[i] / num_tests
        expected = expected_probs[i]
        max_dist_error = max(max_dist_error, abs(dist_freq - expected))
        max_single_error = max(max_single_error, abs(single_freq - expected))
        print(f"{i:<6}{expected:<10.4f}{dist_freq:<10.4f}{single_freq:<10.4f}")

    print(f"\n分布式最大误差: {max_dist_error:.4f}")
    print(f"单节点最大误差: {max_single_error:.4f}")
    dist_ok = max_dist_error < 0.05
    single_ok = max_single_error < 0.05
    print(f"分布式验证通过: {dist_ok}")
    print(f"单节点验证通过: {single_ok}")

    return dist_ok and single_ok


def verify_reservoir_sampling():
    """
    验证蓄水池采样算法的正确性：
    运行多次采样，统计每个元素被选中的频率，验证概率是否接近 k/N。
    """
    N = 100
    k = 10
    num_tests = 100000

    rng = random.Random(42)

    counts = [0] * N

    for test_i in range(num_tests):
        stream = iter(range(N))
        sample = reservoir_sampling(stream, k, seed=rng.randint(0, 2**63))
        for item in sample:
            counts[item] += 1

    expected_prob = k / N
    print(f"期望概率: {expected_prob:.4f}")
    print(f"各元素被选中的频率（前15个）:")
    for i in range(15):
        freq = counts[i] / num_tests
        print(f"  元素 {i:2d}: {freq:.4f}")

    avg_freq = sum(counts) / (N * num_tests)
    print(f"\n平均频率: {avg_freq:.4f}")
    print(f"验证通过: {abs(avg_freq - expected_prob) < 0.01}")


if __name__ == "__main__":
    print("=" * 60)
    print("1. 基础蓄水池采样")
    print("=" * 60)

    data_stream = iter(range(1, 21))
    k = 5
    result = reservoir_sampling(data_stream, k)
    print(f"从1-20的数据流中采样 {k} 个元素（随机种子）: {result}")

    print("\n指定种子复现结果:")
    result_a = reservoir_sampling(iter(range(1, 21)), k, seed=123)
    result_b = reservoir_sampling(iter(range(1, 21)), k, seed=123)
    print(f"  seed=123 第1次: {result_a}")
    print(f"  seed=123 第2次: {result_b}")
    print(f"  结果一致: {result_a == result_b}")

    print("\n" + "=" * 60)
    print("2. 加权蓄水池采样")
    print("=" * 60)

    weighted_data = [(f"item_{i}", float(i + 1)) for i in range(10)]
    print("数据流 (元素, 权重):", weighted_data)
    k_weighted = 3

    weighted_result = weighted_reservoir_sampling(
        iter(weighted_data), k_weighted, seed=456
    )
    print(f"\n加权采样结果 (k={k_weighted}): {weighted_result}")

    print("\n指定种子复现加权采样:")
    wr_a = weighted_reservoir_sampling(iter(weighted_data), k_weighted, seed=789)
    wr_b = weighted_reservoir_sampling(iter(weighted_data), k_weighted, seed=789)
    print(f"  seed=789 第1次: {wr_a}")
    print(f"  seed=789 第2次: {wr_b}")
    print(f"  结果一致: {wr_a == wr_b}")

    print("\n" + "=" * 60)
    print("3. 分布式蓄水池采样")
    print("=" * 60)

    num_nodes = 3
    N_dist = 30
    k_dist = 5

    def node_stream(node_id: int) -> Iterator[WeightedItem]:
        for i in range(node_id, N_dist, num_nodes):
            yield (f"data_{i}", 1.0 + (i % 4))

    base_seed = 1000
    node_samples = []
    for node_id in range(num_nodes):
        stream = node_stream(node_id)
        sample = weighted_reservoir_sampling(
            stream, k_dist, seed=base_seed + node_id, return_keys=True
        )
        node_samples.append(sample)
        print(f"节点 {node_id} 采样 (key, item): {[(round(k, 4), v) for k, v in sample]}")

    merged = merge_distributed_samples(node_samples, k_dist)
    print(f"\n合并后结果 (key, item): {[(round(k, 4), v) for k, v in merged]}")

    print("\n" + "=" * 60)
    print("4. 基础采样正确性验证")
    print("=" * 60)
    verify_reservoir_sampling()

    print("\n" + "=" * 60)
    print("5. 加权采样权重验证")
    print("=" * 60)
    verify_weighted_sampling()

    print("\n" + "=" * 60)
    print("6. 分布式采样验证")
    print("=" * 60)
    verify_distributed_sampling()
