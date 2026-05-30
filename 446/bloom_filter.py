import math
import hashlib
import json
import pickle
import sys
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Any


@dataclass
class _BloomSegment:
    n: int
    p: float
    m: int
    k: int
    count: int
    bit_array: List[int]


@dataclass
class _CountingBloomSegment:
    n: int
    p: float
    m: int
    k: int
    count: int
    counter_array: List[int]


class _BaseBloomFilter:
    @staticmethod
    def _optimal_m(n: int, p: float) -> int:
        return math.ceil(-(n * math.log(p)) / (math.log(2) ** 2))

    @staticmethod
    def _optimal_k(m: int, n: int) -> int:
        return max(1, math.ceil((m / n) * math.log(2)))

    @staticmethod
    def _hashes(item: bytes, m: int, k: int) -> List[int]:
        h1 = int(hashlib.md5(item).hexdigest(), 16)
        h2 = int(hashlib.sha256(item).hexdigest(), 16)
        return [(h1 + i * h2) % m for i in range(k)]


class BloomFilter(_BaseBloomFilter):
    def __init__(
        self,
        n: int,
        p: float,
        auto_scale: bool = True,
        scale_threshold: float = 0.8,
        scale_factor: float = 2.0,
    ):
        if n <= 0:
            raise ValueError("Expected number of elements (n) must be > 0")
        if not (0 < p < 1):
            raise ValueError("False positive rate (p) must be between 0 and 1")
        if not (0 < scale_threshold <= 1):
            raise ValueError("Scale threshold must be between 0 and 1")
        if scale_factor <= 1:
            raise ValueError("Scale factor must be > 1")

        self._initial_n = n
        self._p = p
        self._auto_scale = auto_scale
        self._scale_threshold = scale_threshold
        self._scale_factor = scale_factor
        self._filters: List[_BloomSegment] = []
        self._add_segment(n)

    def _add_segment(self, n: int) -> None:
        m = self._optimal_m(n, self._p)
        k = self._optimal_k(m, n)
        segment = _BloomSegment(
            n=n,
            p=self._p,
            m=m,
            k=k,
            count=0,
            bit_array=[0] * m,
        )
        self._filters.append(segment)

    def _check_and_scale(self) -> None:
        if not self._auto_scale:
            return
        last = self._filters[-1]
        usage = last.count / last.n
        if usage >= self._scale_threshold:
            new_n = math.ceil(last.n * self._scale_factor)
            self._add_segment(new_n)

    def add(self, item) -> None:
        self._check_and_scale()
        segment = self._filters[-1]
        item_bytes = str(item).encode("utf-8")
        indices = self._hashes(item_bytes, segment.m, segment.k)
        for idx in indices:
            segment.bit_array[idx] = 1
        segment.count += 1

    def __contains__(self, item) -> bool:
        item_bytes = str(item).encode("utf-8")
        for segment in self._filters:
            indices = self._hashes(item_bytes, segment.m, segment.k)
            if all(segment.bit_array[idx] for idx in indices):
                return True
        return False

    def might_contain(self, item) -> bool:
        return item in self

    def capacity_status(self) -> Dict:
        total_expected = sum(f.n for f in self._filters)
        total_count = sum(f.count for f in self._filters)
        overall_usage = total_count / total_expected if total_expected > 0 else 0.0

        segments_info = []
        warnings = []

        for i, f in enumerate(self._filters):
            usage = f.count / f.n if f.n > 0 else 0.0
            fill_ratio = sum(f.bit_array) / f.m if f.m > 0 else 0.0
            segments_info.append({
                "segment": i,
                "n": f.n,
                "count": f.count,
                "usage": usage,
                "fill_ratio": fill_ratio,
                "m": f.m,
                "k": f.k,
            })

            if usage >= self._scale_threshold:
                warnings.append(
                    f"Segment {i}: usage {usage:.2%} >= threshold {self._scale_threshold:.0%}"
                )
            if fill_ratio >= 0.5:
                warnings.append(
                    f"Segment {i}: bit array filled {fill_ratio:.2%} - false positive risk elevated"
                )

        status = "normal"
        if overall_usage >= 0.9:
            status = "critical"
        elif overall_usage >= self._scale_threshold:
            status = "warning"

        return {
            "total_expected": total_expected,
            "total_count": total_count,
            "overall_usage": overall_usage,
            "num_segments": len(self._filters),
            "auto_scale": self._auto_scale,
            "scale_threshold": self._scale_threshold,
            "status": status,
            "segments": segments_info,
            "warnings": warnings,
        }

    def memory_usage(self) -> Dict[str, Any]:
        total_bits = sum(f.m for f in self._filters)
        total_bytes = total_bits / 8
        overhead = sys.getsizeof(self)
        for f in self._filters:
            overhead += sys.getsizeof(f)
            overhead += sys.getsizeof(f.bit_array)

        return {
            "type": "Standard Bloom Filter",
            "total_bits": total_bits,
            "total_bytes": total_bytes,
            "total_kb": total_bytes / 1024,
            "total_mb": total_bytes / (1024 * 1024),
            "overhead_bytes": overhead,
            "num_segments": len(self._filters),
        }

    def serialize(self, filepath: str) -> None:
        data = {
            "initial_n": self._initial_n,
            "p": self._p,
            "auto_scale": self._auto_scale,
            "scale_threshold": self._scale_threshold,
            "scale_factor": self._scale_factor,
            "filters": [asdict(f) for f in self._filters],
        }
        with open(filepath, "wb") as f:
            pickle.dump(data, f)

    @classmethod
    def deserialize(cls, filepath: str) -> "BloomFilter":
        with open(filepath, "rb") as f:
            data = pickle.load(f)

        bf = cls(
            n=data["initial_n"],
            p=data["p"],
            auto_scale=data["auto_scale"],
            scale_threshold=data["scale_threshold"],
            scale_factor=data["scale_factor"],
        )
        bf._filters = [_BloomSegment(**f) for f in data["filters"]]
        return bf

    def to_json(self) -> str:
        data = {
            "initial_n": self._initial_n,
            "p": self._p,
            "auto_scale": self._auto_scale,
            "scale_threshold": self._scale_threshold,
            "scale_factor": self._scale_factor,
            "filters": [asdict(f) for f in self._filters],
        }
        return json.dumps(data)

    @classmethod
    def from_json(cls, json_str: str) -> "BloomFilter":
        data = json.loads(json_str)
        bf = cls(
            n=data["initial_n"],
            p=data["p"],
            auto_scale=data["auto_scale"],
            scale_threshold=data["scale_threshold"],
            scale_factor=data["scale_factor"],
        )
        bf._filters = [_BloomSegment(**f) for f in data["filters"]]
        return bf

    def __repr__(self) -> str:
        cs = self.capacity_status()
        return (
            f"BloomFilter(segments={cs['num_segments']}, "
            f"total_count={cs['total_count']}/{cs['total_expected']}, "
            f"usage={cs['overall_usage']:.2%}, status={cs['status']})"
        )


class CountingBloomFilter(_BaseBloomFilter):
    def __init__(
        self,
        n: int,
        p: float,
        auto_scale: bool = True,
        scale_threshold: float = 0.8,
        scale_factor: float = 2.0,
        counter_bits: int = 4,
    ):
        if n <= 0:
            raise ValueError("Expected number of elements (n) must be > 0")
        if not (0 < p < 1):
            raise ValueError("False positive rate (p) must be between 0 and 1")
        if not (0 < scale_threshold <= 1):
            raise ValueError("Scale threshold must be between 0 and 1")
        if scale_factor <= 1:
            raise ValueError("Scale factor must be > 1")
        if counter_bits < 1:
            raise ValueError("Counter bits must be >= 1")

        self._initial_n = n
        self._p = p
        self._auto_scale = auto_scale
        self._scale_threshold = scale_threshold
        self._scale_factor = scale_factor
        self._counter_bits = counter_bits
        self._max_counter = (1 << counter_bits) - 1
        self._filters: List[_CountingBloomSegment] = []
        self._add_segment(n)

    def _add_segment(self, n: int) -> None:
        m = self._optimal_m(n, self._p)
        k = self._optimal_k(m, n)
        segment = _CountingBloomSegment(
            n=n,
            p=self._p,
            m=m,
            k=k,
            count=0,
            counter_array=[0] * m,
        )
        self._filters.append(segment)

    def _check_and_scale(self) -> None:
        if not self._auto_scale:
            return
        last = self._filters[-1]
        usage = last.count / last.n
        if usage >= self._scale_threshold:
            new_n = math.ceil(last.n * self._scale_factor)
            self._add_segment(new_n)

    def add(self, item) -> None:
        self._check_and_scale()
        segment = self._filters[-1]
        item_bytes = str(item).encode("utf-8")
        indices = self._hashes(item_bytes, segment.m, segment.k)
        for idx in indices:
            if segment.counter_array[idx] < self._max_counter:
                segment.counter_array[idx] += 1
        segment.count += 1

    def remove(self, item) -> bool:
        item_bytes = str(item).encode("utf-8")
        for segment in self._filters:
            indices = self._hashes(item_bytes, segment.m, segment.k)
            if all(segment.counter_array[idx] > 0 for idx in indices):
                for idx in indices:
                    segment.counter_array[idx] -= 1
                segment.count -= 1
                return True
        return False

    def __contains__(self, item) -> bool:
        item_bytes = str(item).encode("utf-8")
        for segment in self._filters:
            indices = self._hashes(item_bytes, segment.m, segment.k)
            if all(segment.counter_array[idx] > 0 for idx in indices):
                return True
        return False

    def might_contain(self, item) -> bool:
        return item in self

    def capacity_status(self) -> Dict:
        total_expected = sum(f.n for f in self._filters)
        total_count = sum(f.count for f in self._filters)
        overall_usage = total_count / total_expected if total_expected > 0 else 0.0

        segments_info = []
        warnings = []

        for i, f in enumerate(self._filters):
            usage = f.count / f.n if f.n > 0 else 0.0
            non_zero_counters = sum(1 for c in f.counter_array if c > 0)
            fill_ratio = non_zero_counters / f.m if f.m > 0 else 0.0
            avg_counter = sum(f.counter_array) / f.m if f.m > 0 else 0.0
            max_seen = max(f.counter_array) if f.counter_array else 0

            segments_info.append({
                "segment": i,
                "n": f.n,
                "count": f.count,
                "usage": usage,
                "fill_ratio": fill_ratio,
                "avg_counter": avg_counter,
                "max_counter_seen": max_seen,
                "m": f.m,
                "k": f.k,
            })

            if usage >= self._scale_threshold:
                warnings.append(
                    f"Segment {i}: usage {usage:.2%} >= threshold {self._scale_threshold:.0%}"
                )
            if fill_ratio >= 0.5:
                warnings.append(
                    f"Segment {i}: counters filled {fill_ratio:.2%} - false positive risk elevated"
                )
            if max_seen >= self._max_counter:
                warnings.append(
                    f"Segment {i}: counter overflow detected at max value {self._max_counter}"
                )

        status = "normal"
        if overall_usage >= 0.9:
            status = "critical"
        elif overall_usage >= self._scale_threshold:
            status = "warning"

        return {
            "total_expected": total_expected,
            "total_count": total_count,
            "overall_usage": overall_usage,
            "num_segments": len(self._filters),
            "auto_scale": self._auto_scale,
            "scale_threshold": self._scale_threshold,
            "counter_bits": self._counter_bits,
            "max_counter_value": self._max_counter,
            "status": status,
            "segments": segments_info,
            "warnings": warnings,
        }

    def memory_usage(self) -> Dict[str, Any]:
        total_counters = sum(f.m for f in self._filters)
        total_bits = total_counters * self._counter_bits
        total_bytes = total_bits / 8
        overhead = sys.getsizeof(self)
        for f in self._filters:
            overhead += sys.getsizeof(f)
            overhead += sys.getsizeof(f.counter_array)

        return {
            "type": "Counting Bloom Filter",
            "counter_bits": self._counter_bits,
            "total_counters": total_counters,
            "total_bits": total_bits,
            "total_bytes": total_bytes,
            "total_kb": total_bytes / 1024,
            "total_mb": total_bytes / (1024 * 1024),
            "overhead_bytes": overhead,
            "num_segments": len(self._filters),
        }

    def serialize(self, filepath: str) -> None:
        data = {
            "initial_n": self._initial_n,
            "p": self._p,
            "auto_scale": self._auto_scale,
            "scale_threshold": self._scale_threshold,
            "scale_factor": self._scale_factor,
            "counter_bits": self._counter_bits,
            "filters": [asdict(f) for f in self._filters],
        }
        with open(filepath, "wb") as f:
            pickle.dump(data, f)

    @classmethod
    def deserialize(cls, filepath: str) -> "CountingBloomFilter":
        with open(filepath, "rb") as f:
            data = pickle.load(f)

        cbf = cls(
            n=data["initial_n"],
            p=data["p"],
            auto_scale=data["auto_scale"],
            scale_threshold=data["scale_threshold"],
            scale_factor=data["scale_factor"],
            counter_bits=data["counter_bits"],
        )
        cbf._filters = [_CountingBloomSegment(**f) for f in data["filters"]]
        return cbf

    def to_json(self) -> str:
        data = {
            "initial_n": self._initial_n,
            "p": self._p,
            "auto_scale": self._auto_scale,
            "scale_threshold": self._scale_threshold,
            "scale_factor": self._scale_factor,
            "counter_bits": self._counter_bits,
            "filters": [asdict(f) for f in data["filters"]],
        }
        return json.dumps(data)

    @classmethod
    def from_json(cls, json_str: str) -> "CountingBloomFilter":
        data = json.loads(json_str)
        cbf = cls(
            n=data["initial_n"],
            p=data["p"],
            auto_scale=data["auto_scale"],
            scale_threshold=data["scale_threshold"],
            scale_factor=data["scale_factor"],
            counter_bits=data["counter_bits"],
        )
        cbf._filters = [_CountingBloomSegment(**f) for f in data["filters"]]
        return cbf

    def __repr__(self) -> str:
        cs = self.capacity_status()
        return (
            f"CountingBloomFilter(segments={cs['num_segments']}, "
            f"total_count={cs['total_count']}/{cs['total_expected']}, "
            f"usage={cs['overall_usage']:.2%}, status={cs['status']})"
        )


def compare_memory_usage(n: int, p: float, counter_bits: int = 4) -> Dict:
    bf = BloomFilter(n=n, p=p, auto_scale=False)
    cbf = CountingBloomFilter(n=n, p=p, auto_scale=False, counter_bits=counter_bits)

    bf_mem = bf.memory_usage()
    cbf_mem = cbf.memory_usage()

    ratio = cbf_mem["total_bits"] / bf_mem["total_bits"]

    return {
        "params": {"n": n, "p": p, "counter_bits": counter_bits},
        "standard": bf_mem,
        "counting": cbf_mem,
        "memory_ratio": ratio,
        "comparison": f"Counting uses {ratio:.1f}x more memory than standard",
    }


if __name__ == "__main__":
    print("=" * 60)
    print("Testing Standard Bloom Filter")
    print("=" * 60)
    bf = BloomFilter(n=100, p=0.01, scale_threshold=0.8, scale_factor=2.0)
    print("Initial:", bf)

    for i in range(500):
        bf.add(f"item_{i}")

    print(f"After adding 500 items: {bf}")
    print(f"item_0 in bf: {'item_0' in bf}")
    print(f"item_999 in bf: {'item_999' in bf}")

    bf_mem = bf.memory_usage()
    print(f"\nStandard BF Memory: {bf_mem['total_kb']:.2f} KB ({bf_mem['total_bits']} bits)")

    print("\n" + "=" * 60)
    print("Testing Serialization/Deserialization")
    print("=" * 60)
    bf.serialize("test_bf.pkl")
    bf_restored = BloomFilter.deserialize("test_bf.pkl")
    print(f"Restored: {bf_restored}")
    print(f"item_0 in restored: {'item_0' in bf_restored}")

    print("\n" + "=" * 60)
    print("Testing Counting Bloom Filter with Delete")
    print("=" * 60)
    cbf = CountingBloomFilter(n=100, p=0.01, auto_scale=False, counter_bits=4)
    print("Initial:", cbf)

    for i in range(100):
        cbf.add(f"item_{i}")

    print(f"After adding 100 items: {cbf}")
    print(f"item_42 in cbf: {'item_42' in cbf}")

    removed = cbf.remove("item_42")
    print(f"Removed item_42: {removed}")
    print(f"item_42 in cbf after remove: {'item_42' in cbf}")

    removed_nonexistent = cbf.remove("nonexistent")
    print(f"Removed nonexistent: {removed_nonexistent}")

    cbf_mem = cbf.memory_usage()
    print(f"\nCounting BF Memory: {cbf_mem['total_kb']:.2f} KB ({cbf_mem['total_bits']} bits)")

    print("\n" + "=" * 60)
    print("Memory Usage Comparison")
    print("=" * 60)
    comparison = compare_memory_usage(n=10000, p=0.01, counter_bits=4)
    print(f"Parameters: n={comparison['params']['n']}, p={comparison['params']['p']}")
    print(f"Standard BF: {comparison['standard']['total_kb']:.2f} KB")
    print(f"Counting BF ({comparison['params']['counter_bits']} bits/counter): {comparison['counting']['total_kb']:.2f} KB")
    print(f"Ratio: {comparison['comparison']}")

    comparison2 = compare_memory_usage(n=10000, p=0.01, counter_bits=8)
    print(f"\nWith 8 bits/counter: {comparison2['counting']['total_kb']:.2f} KB")
    print(f"Ratio: {comparison2['comparison']}")
