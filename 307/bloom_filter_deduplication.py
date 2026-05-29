import math
import hashlib
from typing import Any, List, Tuple, Generator, Dict, Optional
from dataclasses import dataclass
import time
from collections import OrderedDict


@dataclass
class DeduplicationStats:
    total_processed: int = 0
    unique_count: int = 0
    duplicate_count: int = 0
    false_positive_estimated: float = 0.0
    processing_time: float = 0.0
    filter_count: int = 1
    expansion_count: int = 0
    total_capacity: int = 0
    memory_bytes: int = 0
    shard_count: int = 1
    time_window_buckets: int = 0
    expired_count: int = 0


class BloomFilter:
    def __init__(self, capacity: int, false_positive_rate: float = 0.01):
        if capacity <= 0:
            raise ValueError("Capacity must be a positive integer")
        if not (0 < false_positive_rate < 1):
            raise ValueError("False positive rate must be between 0 and 1")

        self.capacity = capacity
        self.false_positive_rate = false_positive_rate

        self.bit_size = self._calculate_bit_size(capacity, false_positive_rate)
        self.hash_count = self._calculate_hash_count(self.bit_size, capacity)

        self.bit_array = bytearray((self.bit_size + 7) // 8)
        self.items_added = 0

    @staticmethod
    def _calculate_bit_size(n: int, p: float) -> int:
        m = -(n * math.log(p)) / (math.log(2) ** 2)
        return int(math.ceil(m))

    @staticmethod
    def _calculate_hash_count(m: int, n: int) -> int:
        k = (m / n) * math.log(2)
        return max(1, int(math.ceil(k)))

    def _get_hashes(self, item: Any) -> List[int]:
        item_str = str(item).encode('utf-8')
        hashes = []

        for i in range(self.hash_count):
            hash_obj = hashlib.md5(item_str + str(i).encode('utf-8'))
            hash_val = int(hash_obj.hexdigest(), 16)
            hashes.append(hash_val % self.bit_size)

        return hashes

    def add(self, item: Any) -> None:
        for hash_val in self._get_hashes(item):
            byte_idx = hash_val // 8
            bit_idx = hash_val % 8
            self.bit_array[byte_idx] |= (1 << bit_idx)
        self.items_added += 1

    def contains(self, item: Any) -> bool:
        for hash_val in self._get_hashes(item):
            byte_idx = hash_val // 8
            bit_idx = hash_val % 8
            if not (self.bit_array[byte_idx] & (1 << bit_idx)):
                return False
        return True

    def is_full(self) -> bool:
        return self.items_added >= self.capacity

    def get_current_false_positive_rate(self) -> float:
        if self.items_added == 0:
            return 0.0
        return (1 - math.exp(-self.hash_count * self.items_added / self.bit_size)) ** self.hash_count

    def get_memory_bytes(self) -> int:
        return len(self.bit_array)


class ScalableBloomFilter:
    TIGHTENING_RATIO = 0.5
    CAPACITY_GROWTH_FACTOR = 2

    def __init__(self, initial_capacity: int, false_positive_rate: float = 0.01,
                 capacity_growth_factor: int = 2, tightening_ratio: float = 0.5):
        if initial_capacity <= 0:
            raise ValueError("Initial capacity must be a positive integer")
        if not (0 < false_positive_rate < 1):
            raise ValueError("False positive rate must be between 0 and 1")
        if capacity_growth_factor < 1:
            raise ValueError("Capacity growth factor must be >= 1")
        if not (0 < tightening_ratio <= 1):
            raise ValueError("Tightening ratio must be between 0 and 1")

        self.initial_capacity = initial_capacity
        self.initial_fpr = false_positive_rate
        self.capacity_growth_factor = capacity_growth_factor
        self.tightening_ratio = tightening_ratio

        self.filters: List[BloomFilter] = []
        self.expansion_count = 0

        self._add_new_filter(initial_capacity, false_positive_rate)

    def _add_new_filter(self, capacity: int, fpr: float) -> None:
        new_filter = BloomFilter(capacity, fpr)
        self.filters.append(new_filter)
        if len(self.filters) > 1:
            self.expansion_count += 1

    def add(self, item: Any) -> None:
        if self.filters[-1].is_full():
            new_capacity = self.filters[-1].capacity * self.capacity_growth_factor
            new_fpr = self.filters[-1].false_positive_rate * self.tightening_ratio
            self._add_new_filter(new_capacity, new_fpr)
        self.filters[-1].add(item)

    def contains(self, item: Any) -> bool:
        for f in self.filters:
            if f.contains(item):
                return True
        return False

    def get_current_false_positive_rate(self) -> float:
        if not self.filters:
            return 0.0
        overall_fpr = 1.0
        for f in self.filters:
            fp = f.get_current_false_positive_rate()
            overall_fpr *= (1 - fp)
        return 1.0 - overall_fpr

    @property
    def total_items(self) -> int:
        return sum(f.items_added for f in self.filters)

    @property
    def total_bit_size(self) -> int:
        return sum(f.bit_size for f in self.filters)

    @property
    def filter_count(self) -> int:
        return len(self.filters)

    @property
    def total_capacity(self) -> int:
        return sum(f.capacity for f in self.filters)

    def get_memory_bytes(self) -> int:
        return sum(f.get_memory_bytes() for f in self.filters)

    def get_filter_details(self) -> List[dict]:
        details = []
        for i, f in enumerate(self.filters):
            details.append({
                "index": i + 1,
                "capacity": f.capacity,
                "items_added": f.items_added,
                "target_fpr": f.false_positive_rate,
                "current_fpr": f.get_current_false_positive_rate(),
                "bit_size": f.bit_size,
                "hash_count": f.hash_count,
                "is_full": f.is_full(),
            })
        return details


class TimeWindowBloomFilter:
    def __init__(self, window_seconds: float, bucket_count: int,
                 bucket_capacity: int, false_positive_rate: float = 0.01):
        if window_seconds <= 0:
            raise ValueError("Window seconds must be positive")
        if bucket_count <= 0:
            raise ValueError("Bucket count must be positive")

        self.window_seconds = window_seconds
        self.bucket_count = bucket_count
        self.bucket_interval = window_seconds / bucket_count
        self.bucket_capacity = bucket_capacity
        self.false_positive_rate = false_positive_rate

        self.buckets: OrderedDict[int, BloomFilter] = OrderedDict()
        self.expired_count = 0

    def _get_bucket_key(self, timestamp: float) -> int:
        return int(timestamp // self.bucket_interval)

    def _cleanup_expired(self, current_time: float) -> None:
        cutoff_key = self._get_bucket_key(current_time - self.window_seconds)
        expired_keys = [k for k in self.buckets.keys() if k <= cutoff_key]
        for k in expired_keys:
            self.expired_count += self.buckets[k].items_added
            del self.buckets[k]

    def add(self, item: Any, timestamp: Optional[float] = None) -> None:
        if timestamp is None:
            timestamp = time.time()
        self._cleanup_expired(timestamp)

        bucket_key = self._get_bucket_key(timestamp)
        if bucket_key not in self.buckets:
            self.buckets[bucket_key] = BloomFilter(
                self.bucket_capacity, self.false_positive_rate
            )

        self.buckets[bucket_key].add(item)

    def contains(self, item: Any, timestamp: Optional[float] = None) -> bool:
        if timestamp is None:
            timestamp = time.time()
        self._cleanup_expired(timestamp)

        for bf in self.buckets.values():
            if bf.contains(item):
                return True
        return False

    def get_current_false_positive_rate(self) -> float:
        if not self.buckets:
            return 0.0
        overall_fpr = 1.0
        for bf in self.buckets.values():
            fp = bf.get_current_false_positive_rate()
            overall_fpr *= (1 - fp)
        return 1.0 - overall_fpr

    def get_memory_bytes(self) -> int:
        return sum(bf.get_memory_bytes() for bf in self.buckets.values())

    @property
    def active_bucket_count(self) -> int:
        return len(self.buckets)

    @property
    def total_items(self) -> int:
        return sum(bf.items_added for bf in self.buckets.values())

    @property
    def total_capacity(self) -> int:
        return sum(bf.capacity for bf in self.buckets.values())


class ConsistentHash:
    def __init__(self, num_replicas: int = 100):
        self.num_replicas = num_replicas
        self.ring: Dict[int, str] = {}
        self._sorted_keys: List[int] = []

    def _hash(self, key: str) -> int:
        return int(hashlib.md5(key.encode('utf-8')).hexdigest(), 16)

    def add_node(self, node_id: str) -> None:
        for i in range(self.num_replicas):
            replica_key = f"{node_id}:{i}"
            hash_val = self._hash(replica_key)
            self.ring[hash_val] = node_id
        self._sorted_keys = sorted(self.ring.keys())

    def remove_node(self, node_id: str) -> None:
        for i in range(self.num_replicas):
            replica_key = f"{node_id}:{i}"
            hash_val = self._hash(replica_key)
            if hash_val in self.ring:
                del self.ring[hash_val]
        self._sorted_keys = sorted(self.ring.keys())

    def get_node(self, key: Any) -> str:
        if not self.ring:
            raise ValueError("Hash ring is empty")

        key_str = str(key)
        hash_val = self._hash(key_str)

        for ring_key in self._sorted_keys:
            if ring_key >= hash_val:
                return self.ring[ring_key]

        return self.ring[self._sorted_keys[0]]


class DistributedBloomFilter:
    def __init__(self, shard_count: int, shard_capacity: int,
                 false_positive_rate: float = 0.01, scalable: bool = True,
                 num_replicas: int = 100):
        self.shard_count = shard_count
        self.shard_capacity = shard_capacity
        self.false_positive_rate = false_positive_rate
        self.scalable = scalable

        self.hash_ring = ConsistentHash(num_replicas=num_replicas)
        self.shards: Dict[str, Any] = {}

        for i in range(shard_count):
            node_id = f"shard_{i}"
            self.hash_ring.add_node(node_id)
            if scalable:
                self.shards[node_id] = ScalableBloomFilter(
                    shard_capacity, false_positive_rate
                )
            else:
                self.shards[node_id] = BloomFilter(
                    shard_capacity, false_positive_rate
                )

    def _get_shard(self, item: Any) -> Any:
        node_id = self.hash_ring.get_node(item)
        return self.shards[node_id]

    def add(self, item: Any) -> None:
        shard = self._get_shard(item)
        shard.add(item)

    def contains(self, item: Any) -> bool:
        shard = self._get_shard(item)
        return shard.contains(item)

    def get_current_false_positive_rate(self) -> float:
        if not self.shards:
            return 0.0
        fprs = [s.get_current_false_positive_rate() for s in self.shards.values()]
        return sum(fprs) / len(fprs)

    def get_memory_bytes(self) -> int:
        return sum(s.get_memory_bytes() for s in self.shards.values())

    @property
    def total_items(self) -> int:
        return sum(s.total_items if hasattr(s, 'total_items') else s.items_added
                   for s in self.shards.values())

    @property
    def total_capacity(self) -> int:
        return sum(s.total_capacity if hasattr(s, 'total_capacity') else s.capacity
                   for s in self.shards.values())

    def get_shard_distribution(self) -> Dict[str, int]:
        return {
            node_id: (shard.total_items if hasattr(shard, 'total_items') else shard.items_added)
            for node_id, shard in self.shards.items()
        }


class StreamDeduplicator:
    MODE_BASIC = "basic"
    MODE_SCALABLE = "scalable"
    MODE_TIME_WINDOW = "time_window"
    MODE_DISTRIBUTED = "distributed"

    def __init__(self, mode: str = "scalable", **kwargs):
        self.mode = mode
        self.filter: Any = None
        self.stats = DeduplicationStats()

        if mode == self.MODE_BASIC:
            self.filter = BloomFilter(
                kwargs.get('capacity', 10000),
                kwargs.get('false_positive_rate', 0.01)
            )
        elif mode == self.MODE_SCALABLE:
            self.filter = ScalableBloomFilter(
                kwargs.get('initial_capacity', 10000),
                kwargs.get('false_positive_rate', 0.01),
                kwargs.get('capacity_growth_factor', 2),
                kwargs.get('tightening_ratio', 0.5)
            )
        elif mode == self.MODE_TIME_WINDOW:
            self.filter = TimeWindowBloomFilter(
                kwargs.get('window_seconds', 3600),
                kwargs.get('bucket_count', 60),
                kwargs.get('bucket_capacity', 10000),
                kwargs.get('false_positive_rate', 0.01)
            )
        elif mode == self.MODE_DISTRIBUTED:
            self.filter = DistributedBloomFilter(
                kwargs.get('shard_count', 4),
                kwargs.get('shard_capacity', 10000),
                kwargs.get('false_positive_rate', 0.01),
                kwargs.get('scalable', True),
                kwargs.get('num_replicas', 100)
            )
        else:
            raise ValueError(f"Unknown mode: {mode}")

    def process_stream(self, data_stream: Generator[Any, None, None] | List[Any]) -> Tuple[List[Any], DeduplicationStats]:
        unique_items = []
        start_time = time.time()

        for item in data_stream:
            self.stats.total_processed += 1

            if not self.filter.contains(item):
                self.filter.add(item)
                unique_items.append(item)
                self.stats.unique_count += 1
            else:
                self.stats.duplicate_count += 1

        self.stats.processing_time = time.time() - start_time
        self._update_stats()

        return unique_items, self.stats

    def process_item(self, item: Any, timestamp: Optional[float] = None) -> Tuple[bool, DeduplicationStats]:
        self.stats.total_processed += 1

        if self.mode == self.MODE_TIME_WINDOW and timestamp is not None:
            contains = self.filter.contains(item, timestamp)
        else:
            contains = self.filter.contains(item)

        if not contains:
            if self.mode == self.MODE_TIME_WINDOW and timestamp is not None:
                self.filter.add(item, timestamp)
            else:
                self.filter.add(item)
            self.stats.unique_count += 1
            is_unique = True
        else:
            self.stats.duplicate_count += 1
            is_unique = False

        self._update_stats()

        return is_unique, self.stats

    def _update_stats(self) -> None:
        self.stats.false_positive_estimated = self.filter.get_current_false_positive_rate()
        self.stats.memory_bytes = self.filter.get_memory_bytes()

        if self.mode == self.MODE_SCALABLE:
            self.stats.filter_count = self.filter.filter_count
            self.stats.expansion_count = self.filter.expansion_count
            self.stats.total_capacity = self.filter.total_capacity
        elif self.mode == self.MODE_TIME_WINDOW:
            self.stats.time_window_buckets = self.filter.active_bucket_count
            self.stats.total_capacity = self.filter.total_capacity
            self.stats.expired_count = self.filter.expired_count
        elif self.mode == self.MODE_DISTRIBUTED:
            self.stats.shard_count = len(self.filter.shards)
            self.stats.total_capacity = self.filter.total_capacity
        else:
            self.stats.total_capacity = self.filter.capacity


def generate_sample_data(total_items: int, duplicate_ratio: float = 0.3) -> Generator[str, None, None]:
    import random
    unique_count = int(total_items * (1 - duplicate_ratio))
    unique_items = [f"event_{i}" for i in range(unique_count)]
    for _ in range(total_items):
        if random.random() < duplicate_ratio and unique_items:
            yield random.choice(unique_items)
        else:
            yield f"event_{random.randint(0, unique_count - 1)}"


def generate_timed_sample_data(total_items: int, time_span_seconds: float,
                               duplicate_ratio: float = 0.3) -> Generator[Tuple[str, float], None, None]:
    import random
    unique_count = int(total_items * (1 - duplicate_ratio))
    unique_items = [f"event_{i}" for i in range(unique_count)]
    start_time = time.time()
    for _ in range(total_items):
        if random.random() < duplicate_ratio and unique_items:
            item = random.choice(unique_items)
        else:
            item = f"event_{random.randint(0, unique_count - 1)}"
        timestamp = start_time + random.random() * time_span_seconds
        yield item, timestamp


def print_memory_accuracy_tradeoff():
    print("\n" + "=" * 80)
    print("内存占用 vs 准确率 权衡分析")
    print("=" * 80)

    print(f"\n  {'目标FPR':>10} {'容量':>10} {'位数组大小':>14} {'内存(KB)':>10} {'哈希函数':>10}")
    print(f"  {'-' * 60}")

    capacities = [1000, 10000, 100000]
    fprs = [0.1, 0.05, 0.01, 0.001, 0.0001]

    for cap in capacities:
        print(f"\n  容量 = {cap}:")
        for fpr in fprs:
            m = BloomFilter._calculate_bit_size(cap, fpr)
            k = BloomFilter._calculate_hash_count(m, cap)
            mem_kb = m / 8 / 1024
            print(f"    {fpr*100:>8.2f}% {cap:>10} {m:>14} {mem_kb:>10.2f} {k:>10}")

    print(f"\n  公式说明:")
    print(f"    m = -n * ln(p) / (ln(2))^2  （位数组大小）")
    print(f"    k = (m / n) * ln(2)          （哈希函数数量）")
    print(f"    其中 n=容量, p=目标假阳性率")

    print(f"\n  关键结论:")
    print(f"    - FPR 降低 10 倍，内存约增加 1.7 倍")
    print(f"    - 容量增加 10 倍，内存约增加 10 倍（线性关系）")
    print(f"    - 哈希函数数量随 FPR 降低而增加，影响计算速度")


def main():
    print("=" * 80)
    print("布隆过滤器流式数据去重 - 增强版（时间窗口+分布式）")
    print("=" * 80)

    print_memory_accuracy_tradeoff()

    print("\n" + "=" * 80)
    print("演示1: 固定容量 vs 可扩容（容量超限场景）")
    print("=" * 80)

    capacity = 1000
    total_items = 5000

    basic_dedup = StreamDeduplicator(mode="basic", capacity=capacity, false_positive_rate=0.01)
    scalable_dedup = StreamDeduplicator(mode="scalable", initial_capacity=capacity, false_positive_rate=0.01)

    data = list(generate_sample_data(total_items, duplicate_ratio=0.3))

    _, basic_stats = basic_dedup.process_stream(iter(data))
    _, scalable_stats = scalable_dedup.process_stream(iter(data))

    print(f"\n  {'模式':>12} {'总处理':>8} {'唯一项':>8} {'FPR(%)':>10} {'内存(KB)':>10} {'扩容次数':>10}")
    print(f"  {'-' * 65}")
    print(f"  {'固定容量':>12} {basic_stats.total_processed:>8} {basic_stats.unique_count:>8} "
          f"{basic_stats.false_positive_estimated*100:>10.4f} {basic_stats.memory_bytes/1024:>10.2f} {'-':>10}")
    print(f"  {'可扩容':>12} {scalable_stats.total_processed:>8} {scalable_stats.unique_count:>8} "
          f"{scalable_stats.false_positive_estimated*100:>10.4f} {scalable_stats.memory_bytes/1024:>10.2f} {scalable_stats.expansion_count:>10}")

    print("\n" + "=" * 80)
    print("演示2: 基于时间窗口的去重（滑动窗口 + 自动过期）")
    print("=" * 80)

    window_seconds = 10
    bucket_count = 5
    bucket_capacity = 500

    tw_dedup = StreamDeduplicator(
        mode="time_window",
        window_seconds=window_seconds,
        bucket_count=bucket_count,
        bucket_capacity=bucket_capacity,
        false_positive_rate=0.01
    )

    timed_data = list(generate_timed_sample_data(2000, time_span_seconds=20, duplicate_ratio=0.3))
    timed_data.sort(key=lambda x: x[1])

    for item, ts in timed_data:
        tw_dedup.process_item(item, ts)

    tw_stats = tw_dedup.stats
    print(f"\n  窗口大小: {window_seconds} 秒")
    print(f"  分桶数量: {bucket_count}")
    print(f"  每桶容量: {bucket_capacity}")
    print(f"  活跃桶数: {tw_stats.time_window_buckets}")
    print(f"  总处理项: {tw_stats.total_processed}")
    print(f"  唯一项数: {tw_stats.unique_count}")
    print(f"  已过期项: {tw_stats.expired_count}")
    print(f"  估算 FPR: {tw_stats.false_positive_estimated * 100:.4f}%")
    print(f"  内存占用: {tw_stats.memory_bytes / 1024:.2f} KB")

    print("\n" + "=" * 80)
    print("演示3: 分布式分片去重（一致性哈希 + 多节点）")
    print("=" * 80)

    shard_count = 4
    shard_capacity = 2000

    dist_dedup = StreamDeduplicator(
        mode="distributed",
        shard_count=shard_count,
        shard_capacity=shard_capacity,
        false_positive_rate=0.01,
        scalable=True
    )

    dist_data = list(generate_sample_data(10000, duplicate_ratio=0.3))
    _, dist_stats = dist_dedup.process_stream(iter(dist_data))

    print(f"\n  分片数量: {shard_count}")
    print(f"  每片容量: {shard_capacity} (可扩容)")
    print(f"  总处理项: {dist_stats.total_processed}")
    print(f"  唯一项数: {dist_stats.unique_count}")
    print(f"  估算 FPR: {dist_stats.false_positive_estimated * 100:.4f}%")
    print(f"  总内存: {dist_stats.memory_bytes / 1024:.2f} KB")

    print(f"\n  各分片负载分布:")
    dist = dist_dedup.filter.get_shard_distribution()
    for shard_id, count in dist.items():
        print(f"    {shard_id}: {count} 项")

    print("\n" + "=" * 80)
    print("各模式对比总结")
    print("=" * 80)

    print(f"\n  {'模式':>15} {'适用场景':<30} {'优点':<30} {'缺点':<20}")
    print(f"  {'-' * 100}")
    print(f"  {'固定容量':>15} {'数据量已知':<30} {'简单、快速':<30} {'超限后FPR飙升':<20}")
    print(f"  {'可扩容':>15} {'数据量未知、持续增长':<30} {'FPR可控':<30} {'内存逐步增加':<20}")
    print(f"  {'时间窗口':>15} {'仅需去重最近N小时':<30} {'自动过期、内存恒定':<30} {'需时间戳':<20}")
    print(f"  {'分布式':>15} {'海量数据、高并发':<30} {'水平扩展、负载均衡':<30} {'一致性哈希开销':<20}")

    print("\n完成!")


if __name__ == "__main__":
    main()
