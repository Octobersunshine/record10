import random
import heapq
import math
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Set


class BloomFilter:
    def __init__(self, capacity: int, false_positive_rate: float = 0.01):
        self.capacity = capacity
        self.false_positive_rate = false_positive_rate
        self.bit_size = self._calculate_bit_size(capacity, false_positive_rate)
        self.num_hash_functions = self._calculate_num_hashes(capacity, self.bit_size)
        self.bit_array = bytearray(self.bit_size)
        self.seeds = [random.randint(0, 1000000) for _ in range(self.num_hash_functions)]
        self.inserted_count = 0

    @staticmethod
    def _calculate_bit_size(n: int, p: float) -> int:
        m = -(n * math.log(p)) / (math.log(2) ** 2)
        return max(1, int(m))

    @staticmethod
    def _calculate_num_hashes(n: int, m: int) -> int:
        k = (m / n) * math.log(2)
        return max(1, int(k))

    def _hash(self, item: int, seed: int) -> int:
        h = (item * 1103515245 + seed * 12345) & 0x7FFFFFFF
        return h % self.bit_size

    def add(self, item: int) -> None:
        for seed in self.seeds:
            idx = self._hash(item, seed)
            self.bit_array[idx] = 1
        self.inserted_count += 1

    def add_all(self, items: Set[int]) -> None:
        for item in items:
            self.add(item)

    def contains(self, item: int) -> bool:
        for seed in self.seeds:
            idx = self._hash(item, seed)
            if self.bit_array[idx] == 0:
                return False
        return True

    def __contains__(self, item: int) -> bool:
        return self.contains(item)

    def __repr__(self):
        return (f"BloomFilter(capacity={self.capacity}, fpr={self.false_positive_rate}, "
                f"bit_size={self.bit_size}, hashes={self.num_hash_functions}, "
                f"inserted={self.inserted_count})")


@dataclass
class BloomFilterStats:
    total_checks: int = 0
    true_negative: int = 0
    false_positive: int = 0
    true_positive: int = 0
    saved_lookups: int = 0

    @property
    def hit_rate(self) -> float:
        if self.total_checks == 0:
            return 0.0
        return (self.true_positive + self.false_positive) / self.total_checks

    @property
    def true_negative_rate(self) -> float:
        if self.total_checks == 0:
            return 0.0
        return self.true_negative / self.total_checks

    @property
    def false_positive_rate_actual(self) -> float:
        total_negatives = self.true_negative + self.false_positive
        if total_negatives == 0:
            return 0.0
        return self.false_positive / total_negatives

    @property
    def lookup_savings_rate(self) -> float:
        if self.total_checks == 0:
            return 0.0
        return self.saved_lookups / self.total_checks

    def __repr__(self):
        return (f"BloomFilterStats(checks={self.total_checks}, "
                f"hit_rate={self.hit_rate:.2%}, "
                f"true_negative_rate={self.true_negative_rate:.2%}, "
                f"fpr_actual={self.false_positive_rate_actual:.2%}, "
                f"saved_lookups={self.saved_lookups})")


@dataclass
class SSTable:
    sstable_id: int
    size: int
    min_key: int
    max_key: int
    level: int
    key_count: int = 0
    keys: Set[int] = field(default_factory=set)
    bloom_filter: Optional[BloomFilter] = None

    def __repr__(self):
        has_bf = "BF" if self.bloom_filter else "NoBF"
        return f"SSTable(id={self.sstable_id}, size={self.size}, level={self.level}, keys=[{self.min_key},{self.max_key}], {has_bf})"


@dataclass
class Level:
    level_num: int
    sstables: List[SSTable] = field(default_factory=list)
    target_size: int = 0

    def total_size(self) -> int:
        return sum(sst.size for sst in self.sstables)

    def total_keys(self) -> int:
        return sum(sst.key_count for sst in self.sstables)

    def __repr__(self):
        return f"Level(level={self.level_num}, sstables={len(self.sstables)}, size={self.total_size()}, target={self.target_size})"


@dataclass
class Metrics:
    total_writes: int = 0
    total_reads: int = 0
    bytes_written: int = 0
    bytes_read: int = 0
    compaction_count: int = 0
    compaction_bytes_read: int = 0
    compaction_bytes_written: int = 0
    sstable_merged: int = 0
    read_amplification_samples: List[int] = field(default_factory=list)
    read_amplification_with_bf_samples: List[int] = field(default_factory=list)
    bloom_filter_stats: BloomFilterStats = field(default_factory=BloomFilterStats)
    use_bloom_filter: bool = True

    @property
    def write_amplification(self) -> float:
        if self.total_writes == 0:
            return 0.0
        return self.bytes_written / (self.total_writes * 100)

    @property
    def read_amplification(self) -> float:
        if not self.read_amplification_samples:
            return 0.0
        return sum(self.read_amplification_samples) / len(self.read_amplification_samples)

    @property
    def read_amplification_with_bf(self) -> float:
        if not self.read_amplification_with_bf_samples:
            return 0.0
        return sum(self.read_amplification_with_bf_samples) / len(self.read_amplification_with_bf_samples)

    @property
    def read_amplification_improvement(self) -> float:
        ra = self.read_amplification
        ra_bf = self.read_amplification_with_bf
        if ra == 0:
            return 0.0
        return (ra - ra_bf) / ra * 100

    def __repr__(self):
        if self.use_bloom_filter:
            return (f"Metrics(write_amp={self.write_amplification:.2f}, "
                    f"read_amp(no BF)={self.read_amplification:.2f}, "
                    f"read_amp(with BF)={self.read_amplification_with_bf:.2f}, "
                    f"improvement={self.read_amplification_improvement:.1f}%, "
                    f"compactions={self.compaction_count}, "
                    f"total_bytes={self.bytes_written})")
        else:
            return (f"Metrics(write_amp={self.write_amplification:.2f}, "
                    f"read_amp={self.read_amplification:.2f}, "
                    f"compactions={self.compaction_count}, "
                    f"total_bytes={self.bytes_written})")


class CompactionStrategy:
    def __init__(self, max_levels: int = 7, sstable_size: int = 2000,
                 use_bloom_filter: bool = True,
                 bloom_filter_fpr: float = 0.01):
        self.max_levels = max_levels
        self.sstable_size = sstable_size
        self.use_bloom_filter = use_bloom_filter
        self.bloom_filter_fpr = bloom_filter_fpr
        self.next_sstable_id = 0

    def create_sstable(self, keys: Set[int], level: int) -> SSTable:
        self.next_sstable_id += 1
        size = len(keys) * 100

        bloom_filter = None
        if self.use_bloom_filter:
            bloom_filter = BloomFilter(
                capacity=max(1, len(keys)),
                false_positive_rate=self.bloom_filter_fpr
            )
            bloom_filter.add_all(keys)

        return SSTable(
            sstable_id=self.next_sstable_id,
            size=size,
            min_key=min(keys) if keys else 0,
            max_key=max(keys) if keys else 0,
            level=level,
            key_count=len(keys),
            keys=keys,
            bloom_filter=bloom_filter
        )

    def should_compact(self, levels: List[Level]) -> Optional[int]:
        raise NotImplementedError

    def do_compaction(self, levels: List[Level], metrics: Metrics) -> None:
        raise NotImplementedError

    def get_level_target_size(self, level: int) -> int:
        raise NotImplementedError


class LeveledCompaction(CompactionStrategy):
    def __init__(self, max_levels: int = 7, sstable_size: int = 2000,
                 max_bytes_for_level_base: int = 8000,
                 max_bytes_for_level_multiplier: int = 10,
                 use_bloom_filter: bool = True,
                 bloom_filter_fpr: float = 0.01):
        super().__init__(max_levels, sstable_size, use_bloom_filter, bloom_filter_fpr)
        self.max_bytes_for_level_base = max_bytes_for_level_base
        self.max_bytes_for_level_multiplier = max_bytes_for_level_multiplier

    def get_level_target_size(self, level: int) -> int:
        if level == 0:
            return self.sstable_size * 4
        return int(self.max_bytes_for_level_base * (self.max_bytes_for_level_multiplier ** (level - 1)))

    def should_compact(self, levels: List[Level]) -> Optional[int]:
        for i, level in enumerate(levels):
            if level.total_size() > self.get_level_target_size(i):
                return i
        return None

    def _overlapping_sstables(self, sst: SSTable, sstables: List[SSTable]) -> List[SSTable]:
        return [other for other in sstables
                if not (other.max_key < sst.min_key or other.min_key > sst.max_key)]

    def _merge_keys(self, sstables: List[SSTable]) -> Set[int]:
        all_keys: Set[int] = set()
        for sst in sstables:
            all_keys.update(sst.keys)
        return all_keys

    def _split_into_sstables(self, keys: Set[int], target_level: int) -> List[SSTable]:
        if not keys:
            return []

        sstables = []
        keys_per_sst = max(1, self.sstable_size // 100)
        sorted_keys = sorted(keys)

        for i in range(0, len(sorted_keys), keys_per_sst):
            chunk = set(sorted_keys[i:i + keys_per_sst])
            sstables.append(self.create_sstable(chunk, target_level))

        return sstables

    def do_compaction(self, levels: List[Level], metrics: Metrics) -> None:
        compact_level = self.should_compact(levels)
        if compact_level is None or compact_level >= self.max_levels - 1:
            return

        level = levels[compact_level]
        if compact_level + 1 >= len(levels):
            levels.append(Level(level_num=compact_level + 1,
                                target_size=self.get_level_target_size(compact_level + 1)))
        next_level = levels[compact_level + 1]

        if compact_level == 0:
            sst_to_compact = level.sstables[:]
        else:
            if not level.sstables:
                return
            sst_to_compact = [level.sstables[0]]

        overlapping = []
        for sst in sst_to_compact:
            overlapping.extend(self._overlapping_sstables(sst, next_level.sstables))
        overlapping = list({s.sstable_id: s for s in overlapping}.values())

        all_sstables = sst_to_compact + overlapping
        bytes_read = sum(sst.size for sst in all_sstables)
        metrics.compaction_bytes_read += bytes_read
        metrics.bytes_read += bytes_read

        merged_keys = self._merge_keys(all_sstables)
        new_sstables = self._split_into_sstables(merged_keys, compact_level + 1)

        bytes_written = sum(sst.size for sst in new_sstables)
        metrics.compaction_bytes_written += bytes_written
        metrics.bytes_written += bytes_written
        metrics.sstable_merged += len(all_sstables)
        metrics.compaction_count += 1

        for sst in sst_to_compact:
            if sst in level.sstables:
                level.sstables.remove(sst)
        for sst in overlapping:
            if sst in next_level.sstables:
                next_level.sstables.remove(sst)

        next_level.sstables.extend(new_sstables)
        next_level.sstables.sort(key=lambda s: s.min_key)


class SizeTieredCompaction(CompactionStrategy):
    def __init__(self, max_levels: int = 7, sstable_size: int = 2000,
                 min_threshold: int = 4, max_threshold: int = 32,
                 bucket_high: float = 1.5, bucket_low: float = 0.5,
                 use_bloom_filter: bool = True,
                 bloom_filter_fpr: float = 0.01):
        super().__init__(max_levels, sstable_size, use_bloom_filter, bloom_filter_fpr)
        self.min_threshold = min_threshold
        self.max_threshold = max_threshold
        self.bucket_high = bucket_high
        self.bucket_low = bucket_low

    def get_level_target_size(self, level: int) -> int:
        return self.sstable_size * (2 ** level) * self.min_threshold

    def _bucket_sstables(self, sstables: List[SSTable]) -> List[List[SSTable]]:
        if not sstables:
            return []

        sorted_sst = sorted(sstables, key=lambda s: s.size)
        buckets = []
        current_bucket = [sorted_sst[0]]
        current_avg = sorted_sst[0].size

        for sst in sorted_sst[1:]:
            if self.bucket_low * current_avg <= sst.size <= self.bucket_high * current_avg:
                current_bucket.append(sst)
                current_avg = sum(s.size for s in current_bucket) / len(current_bucket)
            else:
                buckets.append(current_bucket)
                current_bucket = [sst]
                current_avg = sst.size

        if current_bucket:
            buckets.append(current_bucket)

        return buckets

    def should_compact(self, levels: List[Level]) -> Optional[int]:
        for i, level in enumerate(levels):
            if i >= self.max_levels - 1:
                continue
            buckets = self._bucket_sstables(level.sstables)
            for bucket in buckets:
                if len(bucket) >= self.min_threshold:
                    return i
        return None

    def _merge_keys(self, sstables: List[SSTable]) -> Set[int]:
        all_keys: Set[int] = set()
        for sst in sstables:
            all_keys.update(sst.keys)
        return all_keys

    def _split_into_sstables(self, keys: Set[int], target_level: int) -> List[SSTable]:
        if not keys:
            return []

        sstables = []
        keys_per_sst = max(1, self.sstable_size // 100)
        sorted_keys = sorted(keys)

        for i in range(0, len(sorted_keys), keys_per_sst):
            chunk = set(sorted_keys[i:i + keys_per_sst])
            sstables.append(self.create_sstable(chunk, target_level))

        return sstables

    def do_compaction(self, levels: List[Level], metrics: Metrics) -> None:
        compact_level = self.should_compact(levels)
        if compact_level is None or compact_level >= self.max_levels - 1:
            return

        level = levels[compact_level]
        buckets = self._bucket_sstables(level.sstables)

        target_bucket = None
        for bucket in buckets:
            if len(bucket) >= self.min_threshold:
                target_bucket = bucket[:min(self.max_threshold, len(bucket))]
                break

        if not target_bucket:
            return

        if compact_level + 1 >= len(levels):
            levels.append(Level(level_num=compact_level + 1,
                                target_size=self.get_level_target_size(compact_level + 1)))
        next_level = levels[compact_level + 1]

        bytes_read = sum(sst.size for sst in target_bucket)
        metrics.compaction_bytes_read += bytes_read
        metrics.bytes_read += bytes_read

        merged_keys = self._merge_keys(target_bucket)
        new_sstables = self._split_into_sstables(merged_keys, compact_level + 1)

        bytes_written = sum(sst.size for sst in new_sstables)
        metrics.compaction_bytes_written += bytes_written
        metrics.bytes_written += bytes_written
        metrics.sstable_merged += len(target_bucket)
        metrics.compaction_count += 1

        for sst in target_bucket:
            if sst in level.sstables:
                level.sstables.remove(sst)

        next_level.sstables.extend(new_sstables)


@dataclass
class WorkloadStats:
    window_size: int = 100
    write_times: List[float] = field(default_factory=list)
    read_latencies: List[int] = field(default_factory=list)
    recent_writes: List[int] = field(default_factory=list)
    recent_reads: List[int] = field(default_factory=list)
    strategy_switches: List[Tuple[int, str, str]] = field(default_factory=list)

    def record_write(self, timestamp: int):
        self.recent_writes.append(timestamp)
        self._trim_old_records()

    def record_read_latency(self, latency: int, timestamp: int):
        self.read_latencies.append(latency)
        self.recent_reads.append(timestamp)
        self._trim_old_records()

    def _trim_old_records(self):
        cutoff = len(self.recent_writes) - self.window_size
        if cutoff > 0:
            self.recent_writes = self.recent_writes[cutoff:]
        cutoff = len(self.recent_reads) - self.window_size
        if cutoff > 0:
            self.recent_reads = self.recent_reads[cutoff:]
            self.read_latencies = self.read_latencies[cutoff:]

    def get_write_rate(self, current_time: int) -> float:
        if not self.recent_writes:
            return 0.0
        time_span = current_time - self.recent_writes[0] + 1
        return len(self.recent_writes) / time_span

    def get_avg_read_latency(self) -> float:
        if not self.read_latencies:
            return 0.0
        return sum(self.read_latencies) / len(self.read_latencies)

    def get_read_ratio(self, current_time: int) -> float:
        total_ops = len(self.recent_writes) + len(self.recent_reads)
        if total_ops == 0:
            return 0.5
        return len(self.recent_reads) / total_ops

    def record_switch(self, timestamp: int, from_strategy: str, to_strategy: str):
        self.strategy_switches.append((timestamp, from_strategy, to_strategy))

    @property
    def total_switches(self) -> int:
        return len(self.strategy_switches)


class AdaptiveCompaction(CompactionStrategy):
    def __init__(self, max_levels: int = 7, sstable_size: int = 2000,
                 use_bloom_filter: bool = True,
                 bloom_filter_fpr: float = 0.01,
                 switch_threshold: float = 0.6,
                 min_ops_before_switch: int = 50,
                 max_bytes_for_level_base: int = 8000,
                 max_bytes_for_level_multiplier: int = 10,
                 min_threshold: int = 4,
                 max_threshold: int = 32):
        super().__init__(max_levels, sstable_size, use_bloom_filter, bloom_filter_fpr)

        self.leveled = LeveledCompaction(
            max_levels=max_levels,
            sstable_size=sstable_size,
            max_bytes_for_level_base=max_bytes_for_level_base,
            max_bytes_for_level_multiplier=max_bytes_for_level_multiplier,
            use_bloom_filter=use_bloom_filter,
            bloom_filter_fpr=bloom_filter_fpr
        )

        self.size_tiered = SizeTieredCompaction(
            max_levels=max_levels,
            sstable_size=sstable_size,
            min_threshold=min_threshold,
            max_threshold=max_threshold,
            use_bloom_filter=use_bloom_filter,
            bloom_filter_fpr=bloom_filter_fpr
        )

        self.current_strategy = 'leveled'
        self.active_strategy = self.leveled
        self.switch_threshold = switch_threshold
        self.min_ops_before_switch = min_ops_before_switch
        self.ops_since_switch = 0
        self.workload_stats = WorkloadStats(window_size=200)
        self.operation_counter = 0

        self.leveled_params = {
            'max_bytes_for_level_base': max_bytes_for_level_base,
            'max_bytes_for_level_multiplier': max_bytes_for_level_multiplier
        }
        self.size_tiered_params = {
            'min_threshold': min_threshold,
            'max_threshold': max_threshold
        }

    def on_write(self):
        self.operation_counter += 1
        self.ops_since_switch += 1
        self.workload_stats.record_write(self.operation_counter)
        self._maybe_switch_strategy()

    def on_read(self, latency: int):
        self.operation_counter += 1
        self.ops_since_switch += 1
        self.workload_stats.record_read_latency(latency, self.operation_counter)
        self._maybe_switch_strategy()

    def _maybe_switch_strategy(self):
        if self.ops_since_switch < self.min_ops_before_switch:
            return

        read_ratio = self.workload_stats.get_read_ratio(self.operation_counter)
        avg_latency = self.workload_stats.get_avg_read_latency()

        old_strategy = self.current_strategy

        if read_ratio > self.switch_threshold:
            if self.current_strategy != 'leveled':
                self.current_strategy = 'leveled'
                self.active_strategy = self.leveled
                self.workload_stats.record_switch(
                    self.operation_counter, 'size_tiered', 'leveled'
                )
                self.ops_since_switch = 0
        elif read_ratio < (1 - self.switch_threshold):
            if self.current_strategy != 'size_tiered':
                self.current_strategy = 'size_tiered'
                self.active_strategy = self.size_tiered
                self.workload_stats.record_switch(
                    self.operation_counter, 'leveled', 'size_tiered'
                )
                self.ops_since_switch = 0

        if avg_latency > 10 and self.current_strategy != 'leveled':
            if self.current_strategy != 'leveled':
                self.current_strategy = 'leveled'
                self.active_strategy = self.leveled
                self.workload_stats.record_switch(
                    self.operation_counter, old_strategy, 'leveled'
                )
                self.ops_since_switch = 0

    def create_sstable(self, keys: Set[int], level: int) -> SSTable:
        return self.active_strategy.create_sstable(keys, level)

    def should_compact(self, levels: List[Level]) -> Optional[int]:
        return self.active_strategy.should_compact(levels)

    def do_compaction(self, levels: List[Level], metrics: Metrics) -> None:
        old_next_id = self.next_sstable_id
        self.active_strategy.next_sstable_id = self.next_sstable_id
        self.active_strategy.do_compaction(levels, metrics)
        self.next_sstable_id = self.active_strategy.next_sstable_id

    def get_level_target_size(self, level: int) -> int:
        return self.active_strategy.get_level_target_size(level)

    @property
    def strategy_name(self) -> str:
        return f"Adaptive(current={self.current_strategy})"


@dataclass
class ComparisonReport:
    strategies: List[str] = field(default_factory=list)
    metrics: Dict[str, Metrics] = field(default_factory=dict)
    level_stats: Dict[str, Dict] = field(default_factory=dict)
    adaptive_stats: Optional[Dict] = None

    def add_result(self, name: str, metrics: Metrics, stats: Dict):
        self.strategies.append(name)
        self.metrics[name] = metrics
        self.level_stats[name] = stats

    def set_adaptive_stats(self, stats: Dict):
        self.adaptive_stats = stats

    def generate_report(self) -> str:
        lines = []
        lines.append("=" * 90)
        lines.append("LSM树SSTable合并策略性能对比报告")
        lines.append("=" * 90)
        lines.append("")

        lines.append(f"{'策略':25s} {'写放大':>10s} {'读放大':>10s} {'读放大(BF)':>12s} "
                     f"{'合并次数':>10s} {'总字节':>15s}")
        lines.append("-" * 90)

        for name in self.strategies:
            m = self.metrics[name]
            ra_str = f"{m.read_amplification:.2f}"
            ra_bf_str = f"{m.read_amplification_with_bf:.2f}" if m.use_bloom_filter else "N/A"
            lines.append(f"{name:25s} {m.write_amplification:10.2f} {ra_str:>10s} {ra_bf_str:>12s} "
                         f"{m.compaction_count:10d} {m.bytes_written:>15,}")

        lines.append("")
        lines.append("=" * 90)
        lines.append("详细分析")
        lines.append("=" * 90)

        best_wa = min(self.strategies, key=lambda s: self.metrics[s].write_amplification)
        best_ra = min(self.strategies, key=lambda s: (
            self.metrics[s].read_amplification_with_bf
            if self.metrics[s].use_bloom_filter
            else self.metrics[s].read_amplification
        ))

        lines.append(f"✓ 最优写放大: {best_wa} ({self.metrics[best_wa].write_amplification:.2f})")
        best_ra_value = (self.metrics[best_ra].read_amplification_with_bf
                         if self.metrics[best_ra].use_bloom_filter
                         else self.metrics[best_ra].read_amplification)
        lines.append(f"✓ 最优读放大: {best_ra} ({best_ra_value:.2f})")
        lines.append("")

        if 'Adaptive' in self.metrics and self.adaptive_stats:
            lines.append("-" * 90)
            lines.append("自适应策略详细统计:")
            lines.append(f"  - 策略切换次数: {self.adaptive_stats.get('switches', 0)}")
            lines.append(f"  - 最终使用策略: {self.adaptive_stats.get('final_strategy', 'N/A')}")
            lines.append(f"  - 切换历史: {self.adaptive_stats.get('switch_history', [])}")
            lines.append("")

        lines.append("=" * 90)
        lines.append("推荐建议")
        lines.append("=" * 90)

        if 'Adaptive' in self.metrics:
            adaptive_ra = (self.metrics['Adaptive'].read_amplification_with_bf
                           if self.metrics['Adaptive'].use_bloom_filter
                           else self.metrics['Adaptive'].read_amplification)
            adaptive_wa = self.metrics['Adaptive'].write_amplification

            leveled_ra = (self.metrics['Leveled'].read_amplification_with_bf
                          if self.metrics['Leveled'].use_bloom_filter
                          else self.metrics['Leveled'].read_amplification)
            size_wa = self.metrics['Size-tiered'].write_amplification

            if adaptive_ra <= leveled_ra * 1.1 and adaptive_wa <= size_wa * 1.1:
                lines.append("🌟 自适应策略表现优异，同时在读写两方面都接近最优！")
                lines.append("   推荐使用自适应策略应对变化的工作负载。")
            elif adaptive_wa > size_wa * 1.3:
                lines.append("📝 自适应策略写放大较高，可能是因为策略切换开销")
                lines.append("   - 写入密集型负载: 推荐 Size-tiered")
                lines.append("   - 读取密集型负载: 推荐 Leveled")
            else:
                lines.append("✨ 自适应策略表现良好，适合混合工作负载")
        else:
            lines.append("   - 写入密集型负载 (读比例 < 50%): 推荐 Size-tiered")
            lines.append("   - 读取密集型负载 (读比例 > 50%): 推荐 Leveled")

        lines.append("")
        lines.append("=" * 90)

        return "\n".join(lines)


class LSMTreeSimulator:
    def __init__(self, strategy: CompactionStrategy, memtable_size: int = 100):
        self.strategy = strategy
        self.memtable_size = memtable_size
        self.memtable: Set[int] = set()
        self.levels: List[Level] = [Level(level_num=0, target_size=strategy.get_level_target_size(0))]
        self.metrics = Metrics()
        self.metrics.use_bloom_filter = strategy.use_bloom_filter
        self.key_space = 100000
        self.all_written_keys: Set[int] = set()

    def write(self, key: Optional[int] = None) -> None:
        if key is None:
            key = random.randint(0, self.key_space - 1)

        self.memtable.add(key)
        self.all_written_keys.add(key)
        self.metrics.total_writes += 1

        if hasattr(self.strategy, 'on_write'):
            self.strategy.on_write()

        if len(self.memtable) >= self.memtable_size:
            self._flush_memtable()

        while self.strategy.should_compact(self.levels) is not None:
            self.strategy.do_compaction(self.levels, self.metrics)

    def _flush_memtable(self) -> None:
        if not self.memtable:
            return

        keys = set(self.memtable)
        sst = self.strategy.create_sstable(keys, 0)

        self.levels[0].sstables.append(sst)
        self.metrics.bytes_written += sst.size
        self.memtable = set()

    def _check_bloom_filter(self, sst: SSTable, key: int) -> Tuple[bool, bool]:
        if not self.strategy.use_bloom_filter or sst.bloom_filter is None:
            return True, False

        self.metrics.bloom_filter_stats.total_checks += 1
        bf_result = sst.bloom_filter.contains(key)
        actual_result = key in sst.keys

        if not bf_result:
            self.metrics.bloom_filter_stats.true_negative += 1
            self.metrics.bloom_filter_stats.saved_lookups += 1
            return False, False
        else:
            if actual_result:
                self.metrics.bloom_filter_stats.true_positive += 1
            else:
                self.metrics.bloom_filter_stats.false_positive += 1
            return True, not actual_result

    def read(self, key: Optional[int] = None) -> Tuple[bool, int]:
        if key is None:
            key = random.randint(0, self.key_space - 1)

        self.metrics.total_reads += 1

        if key in self.memtable:
            self.metrics.read_amplification_samples.append(1)
            if self.strategy.use_bloom_filter:
                self.metrics.read_amplification_with_bf_samples.append(1)
            if hasattr(self.strategy, 'on_read'):
                self.strategy.on_read(1)
            return True, 1

        sstables_read = 0
        sstables_read_with_bf = 0
        result_found = False

        for level in self.levels:
            sorted_ssts = sorted(level.sstables, key=lambda s: s.min_key)

            if level.level_num == 0:
                for sst in sorted_ssts:
                    sstables_read += 1

                    bf_pass, is_false_positive = self._check_bloom_filter(sst, key)
                    if not bf_pass:
                        continue

                    sstables_read_with_bf += 1

                    if sst.min_key <= key <= sst.max_key and key in sst.keys:
                        self.metrics.bytes_read += sst.size
                        self.metrics.read_amplification_samples.append(sstables_read)
                        if self.strategy.use_bloom_filter:
                            self.metrics.read_amplification_with_bf_samples.append(sstables_read_with_bf)
                        result_found = True
                        break
            else:
                left, right = 0, len(sorted_ssts) - 1
                while left <= right:
                    mid = (left + right) // 2
                    sst = sorted_ssts[mid]
                    sstables_read += 1

                    bf_pass, is_false_positive = self._check_bloom_filter(sst, key)
                    if not bf_pass:
                        if key < sst.min_key:
                            right = mid - 1
                        else:
                            left = mid + 1
                        continue

                    sstables_read_with_bf += 1

                    if sst.min_key <= key <= sst.max_key:
                        if key in sst.keys:
                            self.metrics.bytes_read += sst.size
                            self.metrics.read_amplification_samples.append(sstables_read)
                            if self.strategy.use_bloom_filter:
                                self.metrics.read_amplification_with_bf_samples.append(sstables_read_with_bf)
                            result_found = True
                            break
                        else:
                            if key < sst.min_key:
                                right = mid - 1
                            else:
                                left = mid + 1
                    elif key < sst.min_key:
                        right = mid - 1
                    else:
                        left = mid + 1
                if result_found:
                    break

        if not result_found:
            self.metrics.read_amplification_samples.append(sstables_read)
            if self.strategy.use_bloom_filter:
                self.metrics.read_amplification_with_bf_samples.append(sstables_read_with_bf)

        if hasattr(self.strategy, 'on_read'):
            latency = sstables_read_with_bf if self.strategy.use_bloom_filter else sstables_read
            self.strategy.on_read(latency)

        return result_found, sstables_read_with_bf

    def simulate(self, write_rate: float, read_ratio: float,
                 total_operations: int = 10000) -> Metrics:
        num_writes = int(total_operations * (1 - read_ratio))
        num_reads = int(total_operations * read_ratio)

        operations = []
        for _ in range(num_writes):
            operations.append('write')
        for _ in range(num_reads):
            operations.append('read')

        random.shuffle(operations)

        bf_status = "启用" if self.strategy.use_bloom_filter else "禁用"
        fpr = self.strategy.bloom_filter_fpr if self.strategy.use_bloom_filter else 0
        print(f"开始模拟: 写入速率={write_rate}, 读取比例={read_ratio}, 总操作数={total_operations}")
        print(f"预期写入数={num_writes}, 预期读取数={num_reads}, 布隆过滤器={bf_status}, 目标FPR={fpr:.2%}")

        for i, op in enumerate(operations):
            if op == 'write':
                self.write()
            else:
                self.read()

            if i % 5000 == 0 and i > 0:
                if self.strategy.use_bloom_filter:
                    print(f"进度: {i}/{total_operations}, "
                          f"写放大={self.metrics.write_amplification:.2f}, "
                          f"读放大(无BF)={self.metrics.read_amplification:.2f}, "
                          f"读放大(有BF)={self.metrics.read_amplification_with_bf:.2f}, "
                          f"BF节省={self.metrics.bloom_filter_stats.saved_lookups}, "
                          f"合并次数={self.metrics.compaction_count}")
                else:
                    print(f"进度: {i}/{total_operations}, "
                          f"写放大={self.metrics.write_amplification:.2f}, "
                          f"读放大={self.metrics.read_amplification:.2f}, "
                          f"合并次数={self.metrics.compaction_count}")

        if self.strategy.use_bloom_filter:
            print(f"模拟完成: {self.metrics}")
            print(f"布隆过滤器统计: {self.metrics.bloom_filter_stats}")
        else:
            print(f"模拟完成: {self.metrics}")

        return self.metrics

    def get_level_stats(self) -> Dict[int, Dict]:
        stats = {}
        for level in self.levels:
            stats[level.level_num] = {
                'sstables': len(level.sstables),
                'total_size': level.total_size(),
                'target_size': self.strategy.get_level_target_size(level.level_num),
                'total_keys': level.total_keys()
            }
        return stats


def run_simulation(strategy_type: str, write_rate: float, read_ratio: float,
                   total_operations: int = 10000, **kwargs) -> Tuple[Metrics, Dict]:
    use_bloom_filter = kwargs.get('use_bloom_filter', True)
    bloom_filter_fpr = kwargs.get('bloom_filter_fpr', 0.01)

    if strategy_type == 'leveled':
        strategy = LeveledCompaction(
            max_levels=kwargs.get('max_levels', 7),
            sstable_size=kwargs.get('sstable_size', 2000),
            max_bytes_for_level_base=kwargs.get('max_bytes_for_level_base', 8000),
            max_bytes_for_level_multiplier=kwargs.get('max_bytes_for_level_multiplier', 10),
            use_bloom_filter=use_bloom_filter,
            bloom_filter_fpr=bloom_filter_fpr
        )
    elif strategy_type == 'size_tiered':
        strategy = SizeTieredCompaction(
            max_levels=kwargs.get('max_levels', 7),
            sstable_size=kwargs.get('sstable_size', 2000),
            min_threshold=kwargs.get('min_threshold', 4),
            max_threshold=kwargs.get('max_threshold', 32),
            bucket_high=kwargs.get('bucket_high', 1.5),
            bucket_low=kwargs.get('bucket_low', 0.5),
            use_bloom_filter=use_bloom_filter,
            bloom_filter_fpr=bloom_filter_fpr
        )
    else:
        raise ValueError(f"Unknown strategy type: {strategy_type}")

    simulator = LSMTreeSimulator(
        strategy=strategy,
        memtable_size=kwargs.get('memtable_size', 100)
    )

    metrics = simulator.simulate(write_rate, read_ratio, total_operations)
    level_stats = simulator.get_level_stats()

    return metrics, level_stats


def compare_bloom_filter_effect(strategy_type: str, write_rate: float, read_ratio: float,
                                total_operations: int = 10000, **kwargs) -> Dict:
    print("=" * 70)
    print(f"布隆过滤器效果对比 - {strategy_type}")
    print("=" * 70)

    print("\n--- 无布隆过滤器 ---")
    kwargs_no_bf = {**kwargs, 'use_bloom_filter': False}
    metrics_no_bf, stats_no_bf = run_simulation(
        strategy_type, write_rate, read_ratio, total_operations, **kwargs_no_bf
    )

    print("\n--- 有布隆过滤器 (FPR=1%) ---")
    kwargs_bf_1 = {**kwargs, 'use_bloom_filter': True, 'bloom_filter_fpr': 0.01}
    metrics_bf_1, stats_bf_1 = run_simulation(
        strategy_type, write_rate, read_ratio, total_operations, **kwargs_bf_1
    )

    print("\n--- 有布隆过滤器 (FPR=0.1%) ---")
    kwargs_bf_01 = {**kwargs, 'use_bloom_filter': True, 'bloom_filter_fpr': 0.001}
    metrics_bf_01, stats_bf_01 = run_simulation(
        strategy_type, write_rate, read_ratio, total_operations, **kwargs_bf_01
    )

    print("\n" + "=" * 70)
    print("对比结果")
    print("=" * 70)
    print(f"{'配置':30s} {'写放大':>10s} {'读放大':>10s} {'改善率':>10s} {'节省查询':>10s}")
    print("-" * 70)
    print(f"{'无布隆过滤器':30s} {metrics_no_bf.write_amplification:10.2f} "
          f"{metrics_no_bf.read_amplification:10.2f} {'-':>10s} {'-':>10s}")
    print(f"{'BF (FPR=1%)':30s} {metrics_bf_1.write_amplification:10.2f} "
          f"{metrics_bf_1.read_amplification_with_bf:10.2f} "
          f"{metrics_bf_1.read_amplification_improvement:9.1f}% "
          f"{metrics_bf_1.bloom_filter_stats.saved_lookups:>10d}")
    print(f"{'BF (FPR=0.1%)':30s} {metrics_bf_01.write_amplification:10.2f} "
          f"{metrics_bf_01.read_amplification_with_bf:10.2f} "
          f"{metrics_bf_01.read_amplification_improvement:9.1f}% "
          f"{metrics_bf_01.bloom_filter_stats.saved_lookups:>10d}")

    bf_stats_1 = metrics_bf_1.bloom_filter_stats
    bf_stats_01 = metrics_bf_01.bloom_filter_stats
    print("\n布隆过滤器详细统计:")
    print(f"  BF (FPR=1%):  总检查={bf_stats_1.total_checks}, "
          f"真阴性率={bf_stats_1.true_negative_rate:.2%}, "
          f"实际FPR={bf_stats_1.false_positive_rate_actual:.2%}")
    print(f"  BF (FPR=0.1%): 总检查={bf_stats_01.total_checks}, "
          f"真阴性率={bf_stats_01.true_negative_rate:.2%}, "
          f"实际FPR={bf_stats_01.false_positive_rate_actual:.2%}")

    return {
        'no_bf': {'metrics': metrics_no_bf, 'stats': stats_no_bf},
        'bf_1pct': {'metrics': metrics_bf_1, 'stats': stats_bf_1},
        'bf_01pct': {'metrics': metrics_bf_01, 'stats': stats_bf_01}
    }


def recommend_strategy(write_rate: float, read_ratio: float,
                       total_operations: int = 10000,
                       use_bloom_filter: bool = True,
                       bloom_filter_fpr: float = 0.01) -> Dict:
    print("=" * 70)
    print("开始策略比较模拟")
    print("=" * 70)

    base_params = {
        'sstable_size': 2000,
        'max_bytes_for_level_base': 8000,
        'memtable_size': 100,
        'max_levels': 5,
        'use_bloom_filter': use_bloom_filter,
        'bloom_filter_fpr': bloom_filter_fpr
    }

    configs = [
        ('leveled', 'Leveled (default)', {'max_bytes_for_level_multiplier': 10}),
        ('leveled', 'Leveled (aggressive)', {'max_bytes_for_level_multiplier': 20}),
        ('leveled', 'Leveled (conservative)', {'max_bytes_for_level_multiplier': 5}),
        ('size_tiered', 'Size-tiered (default)', {'min_threshold': 4}),
        ('size_tiered', 'Size-tiered (min=2)', {'min_threshold': 2}),
        ('size_tiered', 'Size-tiered (min=8)', {'min_threshold': 8}),
    ]

    results = []
    for strategy_type, name, params in configs:
        print(f"\n--- 测试策略: {name} ---")
        try:
            full_params = {**base_params, **params}
            metrics, level_stats = run_simulation(
                strategy_type, write_rate, read_ratio, total_operations, **full_params
            )

            if use_bloom_filter:
                ra = metrics.read_amplification_with_bf
            else:
                ra = metrics.read_amplification
            wa = metrics.write_amplification

            score = 0
            if read_ratio > 0.5:
                score = 100 - (ra * 15 + wa * 5)
            else:
                score = 100 - (wa * 15 + ra * 5)

            results.append({
                'name': name,
                'strategy_type': strategy_type,
                'params': params,
                'metrics': metrics,
                'level_stats': level_stats,
                'score': score
            })
        except Exception as e:
            print(f"策略 {name} 执行出错: {e}")
            import traceback
            traceback.print_exc()

    if not results:
        return {'error': 'No valid results'}

    results.sort(key=lambda x: x['score'], reverse=True)
    best = results[0]

    print("\n" + "=" * 70)
    print("模拟结果排名")
    print("=" * 70)
    for i, res in enumerate(results):
        m = res['metrics']
        if use_bloom_filter:
            ra_str = f"{m.read_amplification_with_bf:.2f}(BF)"
        else:
            ra_str = f"{m.read_amplification:.2f}"
        print(f"{i + 1}. {res['name']:30s} 写放大={m.write_amplification:.2f}, "
              f"读放大={ra_str}, 合并次数={m.compaction_count}, "
              f"得分={res['score']:.1f}")

    print("\n" + "=" * 70)
    print("推荐策略")
    print("=" * 70)
    print(f"最佳策略: {best['name']}")
    print(f"策略类型: {best['strategy_type']}")
    print(f"参数配置: {best['params']}")
    print(f"写放大系数: {best['metrics'].write_amplification:.2f}")
    if use_bloom_filter:
        print(f"读放大系数(无BF): {best['metrics'].read_amplification:.2f}")
        print(f"读放大系数(有BF): {best['metrics'].read_amplification_with_bf:.2f}")
        print(f"读放大改善: {best['metrics'].read_amplification_improvement:.1f}%")
        print(f"布隆过滤器统计: {best['metrics'].bloom_filter_stats}")
    else:
        print(f"读放大系数: {best['metrics'].read_amplification:.2f}")
    print(f"总合并次数: {best['metrics'].compaction_count}")
    print(f"总写入字节: {best['metrics'].bytes_written:,}")
    print(f"层级统计:")
    for level_num, stats in best['level_stats'].items():
        print(f"  Level {level_num}: {stats['sstables']} 个SSTable, "
              f"大小={stats['total_size']:,}, 目标={stats['target_size']:,}")

    workload_type = "读密集" if read_ratio > 0.5 else "写密集"
    bf_status = "启用" if use_bloom_filter else "禁用"
    print(f"\n工作负载类型: {workload_type} (读取比例={read_ratio:.1%})")
    print(f"布隆过滤器: {bf_status}")
    if use_bloom_filter:
        print(f"布隆过滤器目标FPR: {bloom_filter_fpr:.2%}")

    recommendation = {
        'best_strategy': best['strategy_type'],
        'best_name': best['name'],
        'params': best['params'],
        'write_amplification': best['metrics'].write_amplification,
        'read_amplification': best['metrics'].read_amplification_with_bf if use_bloom_filter else best['metrics'].read_amplification,
        'read_amplification_no_bf': best['metrics'].read_amplification,
        'read_amplification_improvement': best['metrics'].read_amplification_improvement,
        'bloom_filter_stats': best['metrics'].bloom_filter_stats,
        'compaction_count': best['metrics'].compaction_count,
        'all_results': results,
        'workload_type': workload_type,
        'use_bloom_filter': use_bloom_filter
    }

    return recommendation


def run_adaptive_simulation(write_rate: float, read_ratio: float,
                            total_operations: int = 10000, **kwargs) -> Tuple[Metrics, Dict, Dict]:
    use_bloom_filter = kwargs.get('use_bloom_filter', True)
    bloom_filter_fpr = kwargs.get('bloom_filter_fpr', 0.01)

    strategy = AdaptiveCompaction(
        max_levels=kwargs.get('max_levels', 7),
        sstable_size=kwargs.get('sstable_size', 2000),
        use_bloom_filter=use_bloom_filter,
        bloom_filter_fpr=bloom_filter_fpr,
        switch_threshold=kwargs.get('switch_threshold', 0.6),
        min_ops_before_switch=kwargs.get('min_ops_before_switch', 50),
        max_bytes_for_level_base=kwargs.get('max_bytes_for_level_base', 8000),
        max_bytes_for_level_multiplier=kwargs.get('max_bytes_for_level_multiplier', 10),
        min_threshold=kwargs.get('min_threshold', 4),
        max_threshold=kwargs.get('max_threshold', 32)
    )

    simulator = LSMTreeSimulator(
        strategy=strategy,
        memtable_size=kwargs.get('memtable_size', 100)
    )

    print(f"开始自适应策略模拟: 写入速率={write_rate}, 读取比例={read_ratio}, 总操作数={total_operations}")
    print(f"初始策略: {strategy.current_strategy}, 切换阈值={strategy.switch_threshold}")

    metrics = simulator.simulate(write_rate, read_ratio, total_operations)
    level_stats = simulator.get_level_stats()

    adaptive_stats = {
        'switches': strategy.workload_stats.total_switches,
        'switch_history': [(t, f, to) for t, f, to in strategy.workload_stats.strategy_switches],
        'final_strategy': strategy.current_strategy,
        'final_read_ratio': strategy.workload_stats.get_read_ratio(strategy.operation_counter)
    }

    print(f"\n自适应策略统计:")
    print(f"  策略切换次数: {adaptive_stats['switches']}")
    print(f"  最终使用策略: {adaptive_stats['final_strategy']}")
    print(f"  切换历史: {adaptive_stats['switch_history'][:5]}")
    if len(adaptive_stats['switch_history']) > 5:
        print(f"    ... 还有 {len(adaptive_stats['switch_history']) - 5} 次切换")

    return metrics, level_stats, adaptive_stats


def compare_all_strategies(write_rate: float, read_ratio: float,
                           total_operations: int = 10000, **kwargs) -> ComparisonReport:
    report = ComparisonReport()
    use_bf = kwargs.get('use_bloom_filter', True)

    print("=" * 90)
    print("开始全策略对比模拟")
    print("=" * 90)

    base_params = {
        'sstable_size': kwargs.get('sstable_size', 2000),
        'max_bytes_for_level_base': kwargs.get('max_bytes_for_level_base', 8000),
        'memtable_size': kwargs.get('memtable_size', 100),
        'max_levels': kwargs.get('max_levels', 5),
        'use_bloom_filter': use_bf,
        'bloom_filter_fpr': kwargs.get('bloom_filter_fpr', 0.01),
        'min_threshold': kwargs.get('min_threshold', 4),
        'max_bytes_for_level_multiplier': kwargs.get('max_bytes_for_level_multiplier', 10)
    }

    print("\n--- 测试策略: Leveled ---")
    leveled_params = {**base_params}
    metrics_leveled, stats_leveled = run_simulation(
        'leveled', write_rate, read_ratio, total_operations, **leveled_params
    )
    report.add_result('Leveled', metrics_leveled, stats_leveled)

    print("\n--- 测试策略: Size-tiered ---")
    size_params = {**base_params}
    metrics_size, stats_size = run_simulation(
        'size_tiered', write_rate, read_ratio, total_operations, **size_params
    )
    report.add_result('Size-tiered', metrics_size, stats_size)

    print("\n--- 测试策略: Adaptive ---")
    adaptive_params = {**base_params}
    metrics_adaptive, stats_adaptive, adaptive_stats = run_adaptive_simulation(
        write_rate, read_ratio, total_operations, **adaptive_params
    )
    report.add_result('Adaptive', metrics_adaptive, stats_adaptive)
    report.set_adaptive_stats(adaptive_stats)

    print("\n" + report.generate_report())

    return report


if __name__ == "__main__":
    print("LSM树SSTable合并策略模拟器 (含布隆过滤器)")
    print("=" * 70)

    import argparse

    parser = argparse.ArgumentParser(description='LSM Tree SSTable Compaction Simulator with Bloom Filter and Adaptive Strategy')
    parser.add_argument('--write-rate', type=float, default=1000.0,
                        help='写入速率 (ops/sec)')
    parser.add_argument('--read-ratio', type=float, default=0.5,
                        help='读取操作比例 (0.0-1.0)')
    parser.add_argument('--total-ops', type=int, default=5000,
                        help='总模拟操作数')
    parser.add_argument('--strategy', type=str, default='compare_all',
                        choices=['leveled', 'size_tiered', 'adaptive', 'both', 'compare_bf', 'compare_all'],
                        help='要测试的合并策略')
    parser.add_argument('--no-bloom-filter', action='store_true',
                        help='禁用布隆过滤器')
    parser.add_argument('--bloom-fpr', type=float, default=0.01,
                        help='布隆过滤器目标误判率 (default: 0.01)')
    parser.add_argument('--switch-threshold', type=float, default=0.6,
                        help='自适应策略切换阈值 (default: 0.6)')
    parser.add_argument('--min-ops-before-switch', type=int, default=50,
                        help='自适应策略切换前最小操作数 (default: 50)')

    args = parser.parse_args()

    use_bf = not args.no_bloom_filter

    if args.strategy == 'compare_bf':
        print("\n=== 对比Leveled策略的布隆过滤器效果 ===")
        compare_bloom_filter_effect(
            'leveled', args.write_rate, args.read_ratio, args.total_ops,
            sstable_size=2000, max_bytes_for_level_base=8000, memtable_size=100, max_levels=5
        )
        print("\n=== 对比Size-tiered策略的布隆过滤器效果 ===")
        compare_bloom_filter_effect(
            'size_tiered', args.write_rate, args.read_ratio, args.total_ops,
            sstable_size=2000, min_threshold=4, memtable_size=100, max_levels=5
        )
    elif args.strategy == 'compare_all':
        compare_all_strategies(
            write_rate=args.write_rate,
            read_ratio=args.read_ratio,
            total_operations=args.total_ops,
            use_bloom_filter=use_bf,
            bloom_filter_fpr=args.bloom_fpr,
            switch_threshold=args.switch_threshold,
            min_ops_before_switch=args.min_ops_before_switch
        )
    elif args.strategy == 'adaptive':
        metrics, level_stats, adaptive_stats = run_adaptive_simulation(
            args.write_rate, args.read_ratio, args.total_ops,
            use_bloom_filter=use_bf,
            bloom_filter_fpr=args.bloom_fpr,
            switch_threshold=args.switch_threshold,
            min_ops_before_switch=args.min_ops_before_switch
        )
        print(f"\nAdaptive策略结果:")
        print(f"写放大: {metrics.write_amplification:.2f}")
        if use_bf:
            print(f"读放大(无BF): {metrics.read_amplification:.2f}")
            print(f"读放大(有BF): {metrics.read_amplification_with_bf:.2f}")
            print(f"读放大改善: {metrics.read_amplification_improvement:.1f}%")
        else:
            print(f"读放大: {metrics.read_amplification:.2f}")
        print(f"合并次数: {metrics.compaction_count}")
        print(f"策略切换次数: {adaptive_stats['switches']}")
        print(f"最终策略: {adaptive_stats['final_strategy']}")
        for level_num, stats in level_stats.items():
            print(f"Level {level_num}: {stats['sstables']} SSTables, 大小={stats['total_size']:,}")
    elif args.strategy == 'both':
        recommendation = recommend_strategy(
            write_rate=args.write_rate,
            read_ratio=args.read_ratio,
            total_operations=args.total_ops,
            use_bloom_filter=use_bf,
            bloom_filter_fpr=args.bloom_fpr
        )
    else:
        metrics, level_stats = run_simulation(
            args.strategy, args.write_rate, args.read_ratio, args.total_ops,
            use_bloom_filter=use_bf, bloom_filter_fpr=args.bloom_fpr
        )
        print(f"\n{args.strategy} 策略结果:")
        print(f"写放大: {metrics.write_amplification:.2f}")
        if use_bf:
            print(f"读放大(无BF): {metrics.read_amplification:.2f}")
            print(f"读放大(有BF): {metrics.read_amplification_with_bf:.2f}")
            print(f"读放大改善: {metrics.read_amplification_improvement:.1f}%")
            print(f"布隆过滤器统计: {metrics.bloom_filter_stats}")
        else:
            print(f"读放大: {metrics.read_amplification:.2f}")
        print(f"合并次数: {metrics.compaction_count}")
        for level_num, stats in level_stats.items():
            print(f"Level {level_num}: {stats['sstables']} SSTables, 大小={stats['total_size']:,}")
