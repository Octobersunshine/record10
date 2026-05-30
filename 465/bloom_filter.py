import math
import hashlib
import pickle
import os
from typing import List, Tuple, Optional


class _BloomFilterCore:
    def __init__(self, m: int, k: int):
        self._m = m
        self._k = k
        self._bit_array = bytearray(math.ceil(m / 8))
        self._count = 0

    @property
    def bit_count(self) -> int:
        return self._m

    @property
    def hash_count(self) -> int:
        return self._k

    @property
    def count(self) -> int:
        return self._count

    @property
    def memory_bytes(self) -> int:
        return len(self._bit_array)

    @staticmethod
    def _hashes(item: bytes, m: int, k: int):
        h1 = int(hashlib.md5(item).hexdigest(), 16)
        h2 = int(hashlib.sha1(item).hexdigest(), 16)
        for i in range(k):
            yield (h1 + i * h2) % m

    def add(self, item: bytes):
        for pos in self._hashes(item, self._m, self._k):
            byte_idx = pos // 8
            bit_idx = pos % 8
            self._bit_array[byte_idx] |= (1 << bit_idx)
        self._count += 1

    def contains(self, item: bytes) -> bool:
        return all(
            bool(self._bit_array[pos // 8] & (1 << (pos % 8)))
            for pos in self._hashes(item, self._m, self._k)
        )


class _CountingBloomFilterCore:
    def __init__(self, m: int, k: int, counter_bits: int = 4):
        self._m = m
        self._k = k
        self._counter_bits = counter_bits
        self._max_counter = (1 << counter_bits) - 1
        self._counters_per_byte = 8 // counter_bits
        self._counter_array = bytearray(math.ceil(m / self._counters_per_byte))
        self._count = 0

    @property
    def bit_count(self) -> int:
        return self._m

    @property
    def hash_count(self) -> int:
        return self._k

    @property
    def count(self) -> int:
        return self._count

    @property
    def memory_bytes(self) -> int:
        return len(self._counter_array)

    @staticmethod
    def _hashes(item: bytes, m: int, k: int):
        h1 = int(hashlib.md5(item).hexdigest(), 16)
        h2 = int(hashlib.sha1(item).hexdigest(), 16)
        for i in range(k):
            yield (h1 + i * h2) % m

    def _get_counter(self, pos: int) -> int:
        counters_per_byte = self._counters_per_byte
        byte_idx = pos // counters_per_byte
        counter_idx = pos % counters_per_byte
        shift = counter_idx * self._counter_bits
        mask = self._max_counter << shift
        return (self._counter_array[byte_idx] & mask) >> shift

    def _set_counter(self, pos: int, value: int):
        value = max(0, min(value, self._max_counter))
        counters_per_byte = self._counters_per_byte
        byte_idx = pos // counters_per_byte
        counter_idx = pos % counters_per_byte
        shift = counter_idx * self._counter_bits
        mask = ~(self._max_counter << shift) & 0xFF
        self._counter_array[byte_idx] = (self._counter_array[byte_idx] & mask) | (value << shift)

    def add(self, item: bytes):
        for pos in self._hashes(item, self._m, self._k):
            current = self._get_counter(pos)
            self._set_counter(pos, current + 1)
        self._count += 1

    def remove(self, item: bytes) -> bool:
        if not self.contains(item):
            return False
        for pos in self._hashes(item, self._m, self._k):
            current = self._get_counter(pos)
            self._set_counter(pos, max(0, current - 1))
        self._count = max(0, self._count - 1)
        return True

    def contains(self, item: bytes) -> bool:
        return all(
            self._get_counter(pos) > 0
            for pos in self._hashes(item, self._m, self._k)
        )


class _BloomFilterBase:
    @staticmethod
    def _compute_fpr(m: int, k: int, n: int) -> float:
        exponent = -k * n / m
        return (1 - math.exp(exponent)) ** k

    @classmethod
    def _compute_optimal_params(cls, n: int, target_p: float) -> Tuple[int, int, float]:
        m_float = -(n * math.log(target_p)) / (math.log(2) ** 2)
        k_float = (m_float / n) * math.log(2)

        candidates = []
        for m in range(max(1, int(m_float) - 2), int(m_float) + 10):
            for k in range(max(1, int(k_float) - 2), int(k_float) + 5):
                fpr = cls._compute_fpr(m, k, n)
                if fpr <= target_p:
                    candidates.append((m, k, fpr))

        if not candidates:
            m = max(1, math.ceil(m_float))
            k = max(1, round(k_float))
            while True:
                fpr = cls._compute_fpr(m, k, n)
                if fpr <= target_p:
                    break
                m += 1
                k = max(1, round((m / n) * math.log(2)))
            return m, k, fpr

        candidates.sort(key=lambda x: (x[0], -x[2]))
        return candidates[0]

    def save(self, filepath: str):
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, filepath: str):
        with open(filepath, 'rb') as f:
            return pickle.load(f)


class BloomFilter(_BloomFilterBase):
    def __init__(self, expected_count: int, fpr: float, growth_factor: int = 2, fpr_reduction: float = 0.5):
        if expected_count <= 0:
            raise ValueError("expected_count must be positive")
        if not (0 < fpr < 1):
            raise ValueError("fpr must be between 0 and 1 (exclusive)")
        if growth_factor < 2:
            raise ValueError("growth_factor must be at least 2")
        if not (0 < fpr_reduction < 1):
            raise ValueError("fpr_reduction must be between 0 and 1")

        self._target_fpr = fpr
        self._growth_factor = growth_factor
        self._fpr_reduction = fpr_reduction
        self._filters: List[_BloomFilterCore] = []
        self._filter_params: List[Tuple[int, int, int]] = []

        first_filter_fpr = fpr * (1 - fpr_reduction)
        m, k, actual_fpr = self._compute_optimal_params(expected_count, first_filter_fpr)
        self._actual_fpr = actual_fpr
        self._initial_capacity = expected_count
        self._filters.append(_BloomFilterCore(m, k))
        self._filter_params.append((expected_count, m, k))

    @property
    def bit_count(self) -> int:
        return sum(f.bit_count for f in self._filters)

    @property
    def hash_count(self) -> int:
        return self._filters[0].hash_count

    @property
    def actual_fpr(self) -> float:
        return self._actual_fpr

    @property
    def target_fpr(self) -> float:
        return self._target_fpr

    @property
    def current_fpr(self) -> float:
        if len(self._filters) == 1 and self._filters[0].count == 0:
            return 0.0
        total_fpr = 1.0
        for i, f in enumerate(self._filters):
            if f.count == 0:
                continue
            capacity, m, k = self._filter_params[i]
            exponent = -k * min(f.count, capacity) / m
            single_fpr = (1 - math.exp(exponent)) ** k
            total_fpr *= (1 - single_fpr)
        return 1 - total_fpr

    @property
    def capacity(self) -> int:
        return sum(p[0] for p in self._filter_params)

    @property
    def memory_bytes(self) -> int:
        return sum(f.memory_bytes for f in self._filters)

    def _get_active_filter(self) -> _BloomFilterCore:
        last_idx = len(self._filters) - 1
        last_filter = self._filters[last_idx]
        last_capacity = self._filter_params[last_idx][0]

        if last_filter.count >= last_capacity:
            new_capacity = last_capacity * self._growth_factor
            new_filter_idx = len(self._filters)
            new_fpr = self._target_fpr * (1 - self._fpr_reduction) * (self._fpr_reduction ** new_filter_idx)
            m, k, actual_single_fpr = self._compute_optimal_params(new_capacity, new_fpr)
            self._filters.append(_BloomFilterCore(m, k))
            self._filter_params.append((new_capacity, m, k))

            combined_fpr = 1.0
            for cap, mi, ki in self._filter_params:
                exponent = -ki * cap / mi
                single_fpr = (1 - math.exp(exponent)) ** ki
                combined_fpr *= (1 - single_fpr)
            self._actual_fpr = 1 - combined_fpr

            last_idx += 1

        return self._filters[last_idx]

    def add(self, item):
        data = str(item).encode("utf-8")
        active_filter = self._get_active_filter()
        active_filter.add(data)

    def contains(self, item) -> bool:
        data = str(item).encode("utf-8")
        return any(f.contains(data) for f in self._filters)

    def __len__(self):
        return sum(f.count for f in self._filters)

    def __contains__(self, item):
        return self.contains(item)

    def __repr__(self):
        total_count = sum(f.count for f in self._filters)
        total_capacity = self.capacity
        filter_info = ", ".join(
            f"[filter{i}: n={p[0]}, m={p[1]}, k={p[2]}, inserted={f.count}]"
            for i, (f, p) in enumerate(zip(self._filters, self._filter_params))
        )
        return (
            f"BloomFilter(target_fpr={self._target_fpr:.6f}, "
            f"actual_fpr={self._actual_fpr:.6f}, "
            f"current_fpr={self.current_fpr:.6f}, "
            f"filters={len(self._filters)}, "
            f"inserted={total_count}/{total_capacity}, "
            f"{filter_info})"
        )


class CountingBloomFilter(_BloomFilterBase):
    def __init__(self, expected_count: int, fpr: float, counter_bits: int = 4):
        if expected_count <= 0:
            raise ValueError("expected_count must be positive")
        if not (0 < fpr < 1):
            raise ValueError("fpr must be between 0 and 1 (exclusive)")
        if counter_bits not in [1, 2, 4, 8]:
            raise ValueError("counter_bits must be 1, 2, 4, or 8")

        self._target_fpr = fpr
        self._counter_bits = counter_bits
        self._capacity = expected_count

        m, k, actual_fpr = self._compute_optimal_params(expected_count, fpr)
        self._m = m
        self._k = k
        self._actual_fpr = actual_fpr
        self._filter = _CountingBloomFilterCore(m, k, counter_bits)

    @property
    def bit_count(self) -> int:
        return self._m

    @property
    def hash_count(self) -> int:
        return self._k

    @property
    def actual_fpr(self) -> float:
        return self._actual_fpr

    @property
    def target_fpr(self) -> float:
        return self._target_fpr

    @property
    def current_fpr(self) -> float:
        if self._filter.count == 0:
            return 0.0
        exponent = -self._k * self._filter.count / self._m
        return (1 - math.exp(exponent)) ** self._k

    @property
    def capacity(self) -> int:
        return self._capacity

    @property
    def counter_bits(self) -> int:
        return self._counter_bits

    @property
    def memory_bytes(self) -> int:
        return self._filter.memory_bytes

    def add(self, item):
        data = str(item).encode("utf-8")
        self._filter.add(data)

    def remove(self, item) -> bool:
        data = str(item).encode("utf-8")
        return self._filter.remove(data)

    def contains(self, item) -> bool:
        data = str(item).encode("utf-8")
        return self._filter.contains(data)

    def __len__(self):
        return self._filter.count

    def __contains__(self, item):
        return self.contains(item)

    def __repr__(self):
        return (
            f"CountingBloomFilter(target_fpr={self._target_fpr:.6f}, "
            f"actual_fpr={self._actual_fpr:.6f}, "
            f"current_fpr={self.current_fpr:.6f}, "
            f"m={self._m}, k={self._k}, "
            f"counter_bits={self._counter_bits}, "
            f"inserted={self._filter.count}/{self._capacity})"
        )


if __name__ == "__main__":
    n = 10000
    p = 0.01

    print("=" * 70)
    print("测试1: 标准布隆过滤器与计数布隆过滤器对比")
    print("=" * 70)
    bf_standard = BloomFilter(expected_count=n, fpr=p)
    bf_counting = CountingBloomFilter(expected_count=n, fpr=p, counter_bits=4)

    print(f"配置: n={n}, target_fpr={p}")
    print()
    print(f"{'指标':<30} {'标准BF':<15} {'计数BF(4bit)':<15}")
    print("-" * 60)
    print(f"{'位数 m':<30} {bf_standard.bit_count:<15} {bf_counting.bit_count:<15}")
    print(f"{'哈希函数 k':<30} {bf_standard.hash_count:<15} {bf_counting.hash_count:<15}")
    print(f"{'实际满载FPR':<30} {bf_standard.actual_fpr:<15.8f} {bf_counting.actual_fpr:<15.8f}")
    print(f"{'内存占用 (字节)':<30} {bf_standard.memory_bytes:<15} {bf_counting.memory_bytes:<15}")
    print(f"{'内存占用 (KB)':<30} {bf_standard.memory_bytes/1024:<15.2f} {bf_counting.memory_bytes/1024:<15.2f}")
    memory_ratio = bf_counting.memory_bytes / bf_standard.memory_bytes
    print(f"{'计数BF/标准BF 内存比':<30} {'-':<15} {memory_ratio:<15.2f}x")
    print()

    for i in range(n):
        bf_standard.add(i)
        bf_counting.add(i)

    print(f"插入 {n} 个元素后:")
    print(f"  标准BF 当前FPR: {bf_standard.current_fpr:.8f}")
    print(f"  计数BF 当前FPR: {bf_counting.current_fpr:.8f}")
    print()

    all_standard_found = all(bf_standard.contains(i) for i in range(n))
    all_counting_found = all(bf_counting.contains(i) for i in range(n))
    print(f"所有已插入元素可查询: 标准BF={all_standard_found}, 计数BF={all_counting_found}")
    print()

    print("=" * 70)
    print("测试2: 计数布隆过滤器 - 删除操作")
    print("=" * 70)
    delete_count = 3000
    success_deletes = 0
    for i in range(delete_count):
        if bf_counting.remove(i):
            success_deletes += 1

    print(f"尝试删除 {delete_count} 个元素, 成功删除 {success_deletes} 个")
    print(f"剩余元素数量: {len(bf_counting)}")
    print(f"删除后当前FPR: {bf_counting.current_fpr:.8f}")
    print()

    found_after_delete = sum(1 for i in range(delete_count) if bf_counting.contains(i))
    print(f"已删除元素中仍被误判存在的数量: {found_after_delete}/{delete_count}")
    print(f"未删除元素({n-delete_count}个)全部可查询: {all(bf_counting.contains(i) for i in range(delete_count, n))}")
    print()

    print("=" * 70)
    print("测试3: 不同计数器位数的内存对比")
    print("=" * 70)
    print(f"{'计数器位数':<15} {'内存(字节)':<15} {'内存(KB)':<15} {'相对标准BF':<15}")
    print("-" * 60)
    for bits in [1, 2, 4, 8]:
        bf_c = CountingBloomFilter(expected_count=n, fpr=p, counter_bits=bits)
        ratio = bf_c.memory_bytes / bf_standard.memory_bytes
        print(f"{bits:<15} {bf_c.memory_bytes:<15} {bf_c.memory_bytes/1024:<15.2f} {ratio:<15.2f}x")
    print()

    print("=" * 70)
    print("测试4: 序列化功能 (保存/加载)")
    print("=" * 70)
    test_file = "test_bloom_filter.pkl"
    bf_counting.save(test_file)
    file_size = os.path.getsize(test_file)
    print(f"计数布隆过滤器已保存到 {test_file}")
    print(f"文件大小: {file_size} 字节 ({file_size/1024:.2f} KB)")

    bf_loaded = CountingBloomFilter.load(test_file)
    print(f"加载后元素数量: {len(bf_loaded)}")
    print(f"加载后元素可全部查询: {all(bf_loaded.contains(i) for i in range(delete_count, n))}")

    os.remove(test_file)
    print(f"测试文件已删除")
    print()

    print("=" * 70)
    print("测试5: 不同规模下的内存对比")
    print("=" * 70)
    print(f"{'预期元素数':<15} {'标准BF(KB)':<15} {'计数BF(KB)':<15} {'比率':<10}")
    print("-" * 55)
    for test_n in [1000, 10000, 100000, 1000000]:
        bf_s = BloomFilter(expected_count=test_n, fpr=p)
        bf_c = CountingBloomFilter(expected_count=test_n, fpr=p, counter_bits=4)
        ratio = bf_c.memory_bytes / bf_s.memory_bytes
        print(f"{test_n:<15} {bf_s.memory_bytes/1024:<15.2f} {bf_c.memory_bytes/1024:<15.2f} {ratio:<10.2f}x")
