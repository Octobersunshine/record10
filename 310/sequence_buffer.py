import random
import time
import heapq
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple


@dataclass(order=True)
class Message:
    seq: int = field(compare=True)
    data: str = field(compare=False)
    timestamp: float = field(compare=False, default_factory=time.time)
    lamport_ts: int = field(compare=False, default=0)


class MessageGenerator:
    def __init__(self, total_messages: int = 20, shuffle_range: int = 5, seed: int = 42):
        self.total_messages = total_messages
        self.shuffle_range = shuffle_range
        random.seed(seed)

    def generate_ordered(self) -> List[Message]:
        messages = []
        for i in range(self.total_messages):
            messages.append(Message(seq=i, data=f"Message_{i}", timestamp=time.time(), lamport_ts=i))
            time.sleep(0.001)
        return messages

    def generate_out_of_order(self) -> List[Message]:
        ordered = self.generate_ordered()
        result = []
        buffer = []

        for msg in ordered:
            buffer.append(msg)
            if len(buffer) >= self.shuffle_range:
                random.shuffle(buffer)
                result.extend(buffer)
                buffer = []

        if buffer:
            random.shuffle(buffer)
            result.extend(buffer)

        return result

    def generate_causal_ordered(self, causality_depth: int = 3) -> List[Message]:
        messages = []
        lamport_clock = 0

        for i in range(self.total_messages):
            if i > 0 and random.random() < 0.3:
                deps = random.sample(range(max(0, i - causality_depth), i),
                                   min(random.randint(1, 2), max(1, i - max(0, i - causality_depth))))
                max_dep_ts = max(messages[d].lamport_ts for d in deps) if deps else lamport_clock
                lamport_clock = max(lamport_clock, max_dep_ts) + 1
            else:
                lamport_clock += 1

            messages.append(Message(
                seq=i,
                data=f"Message_{i}",
                timestamp=time.time(),
                lamport_ts=lamport_clock
            ))
            time.sleep(0.001)

        return messages

    def generate_out_of_order_causal(self, causality_depth: int = 3) -> List[Message]:
        ordered = self.generate_causal_ordered(causality_depth)
        result = []
        buffer = []

        for msg in ordered:
            buffer.append(msg)
            if len(buffer) >= self.shuffle_range:
                random.shuffle(buffer)
                result.extend(buffer)
                buffer = []

        if buffer:
            random.shuffle(buffer)
            result.extend(buffer)

        return result


class SequenceBuffer:
    def __init__(
        self,
        expected_next_seq: int = 0,
        timeout: float = 5.0,
        max_buffer_size: Optional[int] = None,
        overflow_mode: str = 'dynamic',
        dynamic_grow_factor: float = 1.5,
        min_buffer_size: int = 5
    ):
        assert overflow_mode in ['drop', 'slide', 'dynamic'], \
            "overflow_mode must be one of: 'drop', 'slide', 'dynamic'"
        assert dynamic_grow_factor > 1.0, "dynamic_grow_factor must be > 1.0"

        self.expected_next_seq = expected_next_seq
        self.timeout = timeout
        self.max_buffer_size = max_buffer_size
        self.overflow_mode = overflow_mode
        self.dynamic_grow_factor = dynamic_grow_factor
        self.min_buffer_size = min_buffer_size
        self.current_max_buffer_size = max_buffer_size if max_buffer_size else min_buffer_size

        self.buffer: List[Message] = []
        self.delivered: List[Message] = []
        self.latencies: List[float] = []
        self.stats = defaultdict(int)
        self.adjustment_history: List[Tuple[int, int, str]] = []

        self.max_observed_seq = expected_next_seq - 1
        self.max_seq_gap = 0

    def _update_seq_gap(self, msg: Message):
        if msg.seq > self.max_observed_seq:
            self.max_observed_seq = msg.seq
            gap = self.max_observed_seq - self.expected_next_seq + 1
            if gap > self.max_seq_gap:
                self.max_seq_gap = gap

    def _dynamic_adjust_buffer(self, msg: Message) -> bool:
        if self.max_buffer_size is not None:
            return False

        required_size = self.max_seq_gap + 1
        if required_size > self.current_max_buffer_size:
            new_size = max(
                int(self.current_max_buffer_size * self.dynamic_grow_factor),
                required_size,
                self.min_buffer_size
            )
            self.adjustment_history.append(
                (self.current_max_buffer_size, new_size, f"gap={self.max_seq_gap}")
            )
            self.current_max_buffer_size = new_size
            self.stats['buffer_expanded'] += 1
            return True
        return False

    def _slide_window(self) -> List[Message]:
        delivered_batch = []
        while len(self.buffer) > 0 and (
            self.current_max_buffer_size is not None and
            len(self.buffer) >= self.current_max_buffer_size
        ):
            if self.buffer[0].seq == self.expected_next_seq:
                msg = heapq.heappop(self.buffer)
                delivered_batch.extend(self._deliver_no_flush(msg))
                delivered_batch.extend(self._flush_available())
            else:
                missing_seq = self.expected_next_seq
                self.stats['skipped_missing'] += 1
                next_msg = heapq.heappop(self.buffer)
                skipped_count = next_msg.seq - self.expected_next_seq
                self.stats['skipped_count'] += skipped_count
                delivered_batch.extend(self._deliver_no_flush(next_msg))
                delivered_batch.extend(self._flush_available())
        return delivered_batch

    def _deliver_no_flush(self, msg: Message) -> List[Message]:
        latency = time.time() - msg.timestamp
        self.latencies.append(latency)
        self.delivered.append(msg)
        self.expected_next_seq = msg.seq + 1
        self.stats['delivered'] += 1
        return [msg]

    def add(self, msg: Message) -> List[Message]:
        self.stats['received'] += 1
        self._update_seq_gap(msg)

        if msg.seq == self.expected_next_seq:
            return self._deliver(msg)

        if msg.seq < self.expected_next_seq:
            self.stats['duplicate'] += 1
            return []

        if any(m.seq == msg.seq for m in self.buffer):
            self.stats['duplicate'] += 1
            return []

        will_buffer = True
        delivered_before = []

        if self.current_max_buffer_size is not None and len(self.buffer) >= self.current_max_buffer_size:
            if self.overflow_mode == 'drop':
                self.stats['dropped_overflow'] += 1
                will_buffer = False
            elif self.overflow_mode == 'slide':
                self.stats['slide_triggered'] += 1
                delivered_before = self._slide_window()
                if len(self.buffer) >= self.current_max_buffer_size:
                    self.stats['dropped_after_slide'] += 1
                    will_buffer = False
                else:
                    will_buffer = True
                if msg.seq < self.expected_next_seq:
                    self.stats['duplicate'] += 1
                    will_buffer = False
            elif self.overflow_mode == 'dynamic':
                if self.max_buffer_size is None:
                    self._dynamic_adjust_buffer(msg)
                    will_buffer = True
                else:
                    self.stats['dropped_overflow'] += 1
                    will_buffer = False

        if will_buffer:
            heapq.heappush(self.buffer, msg)
            self.stats['buffered'] += 1
            delivered_after = self._flush_available()
            return delivered_before + delivered_after

        return delivered_before

    def _deliver(self, msg: Message) -> List[Message]:
        delivered_batch = self._deliver_no_flush(msg)
        delivered_batch.extend(self._flush_available())
        return delivered_batch

    def _flush_available(self) -> List[Message]:
        delivered_batch = []
        while self.buffer and self.buffer[0].seq == self.expected_next_seq:
            msg = heapq.heappop(self.buffer)
            delivered_batch.extend(self._deliver_no_flush(msg))
        return delivered_batch

    def flush_remaining(self) -> List[Message]:
        delivered_batch = []
        while self.buffer:
            msg = heapq.heappop(self.buffer)
            delivered_batch.extend(self._deliver_no_flush(msg))
        return delivered_batch

    def is_empty(self) -> bool:
        return len(self.buffer) == 0

    def get_buffer_size(self) -> int:
        return len(self.buffer)

    def get_max_buffer_size(self) -> int:
        return self.current_max_buffer_size

    def get_adjustment_history(self) -> List[Tuple[int, int, str]]:
        return self.adjustment_history


class BaseSorter(ABC):
    def __init__(self):
        self.stats = defaultdict(int)
        self.latencies: List[float] = []
        self.delivered: List[Message] = []
        self.processing_times: List[float] = []

    @abstractmethod
    def add(self, msg: Message) -> List[Message]:
        pass

    @abstractmethod
    def flush(self) -> List[Message]:
        pass

    @abstractmethod
    def is_empty(self) -> bool:
        pass

    def get_metrics(self) -> Dict:
        return {
            'stats': dict(self.stats),
            'latencies': self.latencies,
            'delivered': self.delivered,
            'processing_times': self.processing_times,
        }


class TimestampWindowSorter(BaseSorter):
    def __init__(
        self,
        window_seconds: float = 0.05,
        max_buffer_size: Optional[int] = None,
        use_watermark: bool = True
    ):
        super().__init__()
        self.window_seconds = window_seconds
        self.max_buffer_size = max_buffer_size
        self.use_watermark = use_watermark

        self.buffer: List[Tuple[float, int, Message]] = []
        self.low_watermark: float = 0.0
        self.high_watermark: float = 0.0
        self.msg_counter = 0

    def _update_watermarks(self, msg_timestamp: float):
        if msg_timestamp > self.high_watermark:
            self.high_watermark = msg_timestamp
        if self.use_watermark:
            self.low_watermark = self.high_watermark - self.window_seconds

    def add(self, msg: Message) -> List[Message]:
        start_time = time.time()
        self.stats['received'] += 1
        self.msg_counter += 1

        self._update_watermarks(msg.timestamp)

        heapq.heappush(self.buffer, (msg.timestamp, self.msg_counter, msg))
        self.stats['buffered'] += 1

        delivered = []
        while self.buffer and self.buffer[0][0] <= self.low_watermark:
            ts, counter, buffered_msg = heapq.heappop(self.buffer)
            latency = time.time() - buffered_msg.timestamp
            self.latencies.append(latency)
            self.delivered.append(buffered_msg)
            self.stats['delivered'] += 1
            delivered.append(buffered_msg)

        if self.max_buffer_size is not None and len(self.buffer) > self.max_buffer_size:
            while len(self.buffer) > self.max_buffer_size:
                ts, counter, buffered_msg = heapq.heappop(self.buffer)
                latency = time.time() - buffered_msg.timestamp
                self.latencies.append(latency)
                self.delivered.append(buffered_msg)
                self.stats['forced_flush'] += 1
                self.stats['delivered'] += 1
                delivered.append(buffered_msg)

        self.processing_times.append(time.time() - start_time)
        return delivered

    def flush(self) -> List[Message]:
        delivered = []
        while self.buffer:
            ts, counter, buffered_msg = heapq.heappop(self.buffer)
            latency = time.time() - buffered_msg.timestamp
            self.latencies.append(latency)
            self.delivered.append(buffered_msg)
            self.stats['delivered'] += 1
            delivered.append(buffered_msg)
        return delivered

    def is_empty(self) -> bool:
        return len(self.buffer) == 0

    def get_buffer_size(self) -> int:
        return len(self.buffer)


class LamportClock:
    def __init__(self):
        self.time = 0

    def increment(self) -> int:
        self.time += 1
        return self.time

    def update(self, observed_time: int) -> int:
        self.time = max(self.time, observed_time) + 1
        return self.time

    def get_time(self) -> int:
        return self.time


class LamportSorter(BaseSorter):
    def __init__(
        self,
        expected_next_seq: int = 0,
        max_buffer_size: Optional[int] = None,
        allow_concurrent: bool = True
    ):
        super().__init__()
        self.expected_next_seq = expected_next_seq
        self.max_buffer_size = max_buffer_size
        self.allow_concurrent = allow_concurrent

        self.local_clock = LamportClock()
        self.seq_buffer: List[Message] = []
        self.lamport_buffer: List[Tuple[int, int, Message]] = []
        self.lamport_counter = 0

        self.max_observed_lamport = 0
        self.concurrent_events = 0

    def add(self, msg: Message) -> List[Message]:
        start_time = time.time()
        self.stats['received'] += 1

        self.local_clock.update(msg.lamport_ts)
        if msg.lamport_ts > self.max_observed_lamport:
            self.max_observed_lamport = msg.lamport_ts

        if msg.seq == self.expected_next_seq:
            self.local_clock.increment()
            latency = time.time() - msg.timestamp
            self.latencies.append(latency)
            self.delivered.append(msg)
            self.expected_next_seq += 1
            self.stats['delivered'] += 1
            result = [msg]
            result.extend(self._flush_seq_buffer())
            self.processing_times.append(time.time() - start_time)
            return result

        if msg.seq < self.expected_next_seq:
            self.stats['duplicate'] += 1
            self.processing_times.append(time.time() - start_time)
            return []

        if any(m.seq == msg.seq for m in self.seq_buffer):
            self.stats['duplicate'] += 1
            self.processing_times.append(time.time() - start_time)
            return []

        if self.allow_concurrent and msg.lamport_ts == self.local_clock.get_time():
            self.concurrent_events += 1
            self.stats['concurrent'] += 1

        heapq.heappush(self.seq_buffer, msg)
        self.stats['buffered'] += 1

        if self.max_buffer_size is not None and len(self.seq_buffer) > self.max_buffer_size:
            result = self._force_deliver_oldest()
            self.processing_times.append(time.time() - start_time)
            return result

        result = self._flush_seq_buffer()
        self.processing_times.append(time.time() - start_time)
        return result

    def _flush_seq_buffer(self) -> List[Message]:
        delivered_batch = []
        while self.seq_buffer and self.seq_buffer[0].seq == self.expected_next_seq:
            msg = heapq.heappop(self.seq_buffer)
            self.local_clock.update(msg.lamport_ts)
            latency = time.time() - msg.timestamp
            self.latencies.append(latency)
            self.delivered.append(msg)
            self.expected_next_seq += 1
            self.stats['delivered'] += 1
            delivered_batch.append(msg)
        return delivered_batch

    def _force_deliver_oldest(self) -> List[Message]:
        delivered_batch = []
        while len(self.seq_buffer) > (self.max_buffer_size or 0) // 2 and self.seq_buffer:
            msg = heapq.heappop(self.seq_buffer)
            skipped = msg.seq - self.expected_next_seq
            if skipped > 0:
                self.stats['skipped'] += skipped
                self.expected_next_seq = msg.seq
            self.local_clock.update(msg.lamport_ts)
            latency = time.time() - msg.timestamp
            self.latencies.append(latency)
            self.delivered.append(msg)
            self.expected_next_seq += 1
            self.stats['delivered'] += 1
            self.stats['force_delivered'] += 1
            delivered_batch.append(msg)
            delivered_batch.extend(self._flush_seq_buffer())
        return delivered_batch

    def flush(self) -> List[Message]:
        delivered_batch = []
        while self.seq_buffer:
            msg = heapq.heappop(self.seq_buffer)
            self.local_clock.update(msg.lamport_ts)
            latency = time.time() - msg.timestamp
            self.latencies.append(latency)
            self.delivered.append(msg)
            self.expected_next_seq = msg.seq + 1
            self.stats['delivered'] += 1
            delivered_batch.append(msg)
        return delivered_batch

    def is_empty(self) -> bool:
        return len(self.seq_buffer) == 0

    def get_buffer_size(self) -> int:
        return len(self.seq_buffer)

    def get_local_time(self) -> int:
        return self.local_clock.get_time()


class LatencyAnalyzer:
    def __init__(self, latencies: List[float]):
        self.latencies = latencies

    def get_statistics(self) -> Dict:
        if not self.latencies:
            return {}

        sorted_latencies = sorted(self.latencies)
        n = len(sorted_latencies)

        def percentile(p: float) -> float:
            k = (n - 1) * p
            f = int(k)
            c = min(f + 1, n - 1)
            return sorted_latencies[f] + (k - f) * (sorted_latencies[c] - sorted_latencies[f])

        return {
            'count': n,
            'min': min(self.latencies),
            'max': max(self.latencies),
            'mean': sum(self.latencies) / n,
            'p50': percentile(0.5),
            'p90': percentile(0.9),
            'p95': percentile(0.95),
            'p99': percentile(0.99),
        }

    def get_distribution(self, bins: int = 10) -> List[Tuple[float, float, int]]:
        if not self.latencies:
            return []

        min_lat = min(self.latencies)
        max_lat = max(self.latencies)
        bin_width = (max_lat - min_lat) / bins if bins > 0 else 0

        distribution = []
        for i in range(bins):
            bin_start = min_lat + i * bin_width
            bin_end = min_lat + (i + 1) * bin_width
            count = sum(1 for l in self.latencies if bin_start <= l < bin_end)
            if i == bins - 1:
                count += sum(1 for l in self.latencies if l == max_lat)
            distribution.append((bin_start, bin_end, count))

        return distribution

    def print_histogram(self, bins: int = 10, max_width: int = 50):
        distribution = self.get_distribution(bins)
        if not distribution:
            print("No latency data available.")
            return

        max_count = max(count for _, _, count in distribution)

        print("\n=== 延迟分布直方图 ===")
        for bin_start, bin_end, count in distribution:
            bar_width = int(count / max_count * max_width) if max_count > 0 else 0
            bar = '█' * bar_width
            print(f"{bin_start * 1000:7.2f}ms - {bin_end * 1000:7.2f}ms | {count:4d} | {bar}")

    def print_statistics(self):
        stats = self.get_statistics()
        if not stats:
            print("No latency data available.")
            return

        print("\n=== 延迟统计信息 ===")
        print(f"消息总数: {stats['count']}")
        print(f"最小延迟: {stats['min'] * 1000:.4f} ms")
        print(f"最大延迟: {stats['max'] * 1000:.4f} ms")
        print(f"平均延迟: {stats['mean'] * 1000:.4f} ms")
        print(f"中位数 (P50): {stats['p50'] * 1000:.4f} ms")
        print(f"P90: {stats['p90'] * 1000:.4f} ms")
        print(f"P95: {stats['p95'] * 1000:.4f} ms")
        print(f"P99: {stats['p99'] * 1000:.4f} ms")


def simulate_out_of_order_scenario(
    total_messages: int = 20,
    shuffle_range: int = 5,
    seed: int = 42,
    max_buffer_size: Optional[int] = None,
    overflow_mode: str = 'dynamic',
    show_details: bool = True
):
    mode_names = {
        'drop': '固定缓冲区+丢包模式',
        'slide': '滑动窗口模式',
        'dynamic': '动态缓冲区调整模式'
    }

    print("=" * 60)
    print(f"消息乱序场景模拟 - {mode_names.get(overflow_mode, overflow_mode)}")
    print("=" * 60)

    generator = MessageGenerator(
        total_messages=total_messages,
        shuffle_range=shuffle_range,
        seed=seed
    )

    print(f"\n[配置] 消息总数: {total_messages}, 乱序窗口: {shuffle_range}")
    print(f"[配置] 最大缓冲区: {max_buffer_size if max_buffer_size else '动态'}, 模式: {overflow_mode}")

    out_of_order_messages = generator.generate_out_of_order()

    if show_details:
        print(f"\n[1] 乱序消息流 (接收顺序):")
        for i, msg in enumerate(out_of_order_messages):
            print(f"  [{i:2d}] Seq={msg.seq:2d} | {msg.data}")

    buffer = SequenceBuffer(
        expected_next_seq=0,
        max_buffer_size=max_buffer_size,
        overflow_mode=overflow_mode
    )
    ordered_messages = []

    if show_details:
        print(f"\n[2] 重排序过程:")
    for i, msg in enumerate(out_of_order_messages):
        delivered = buffer.add(msg)
        if show_details:
            buffer_info = f"大小={buffer.get_buffer_size()}/{buffer.get_max_buffer_size()}"
            if delivered:
                seqs = [m.seq for m in delivered]
                print(f"  接收 Seq={msg.seq:2d} -> 交付 {seqs} | {buffer_info}")
            else:
                if buffer.stats.get('dropped_overflow', 0) > 0:
                    last_dropped = buffer.stats['dropped_overflow']
                    print(f"  接收 Seq={msg.seq:2d} -> 丢弃(溢出) | {buffer_info}, 已丢={last_dropped}")
                else:
                    print(f"  接收 Seq={msg.seq:2d} -> 缓存等待 | {buffer_info}")
        ordered_messages.extend(delivered)

    remaining = buffer.flush_remaining()
    if remaining and show_details:
        print(f"  刷新缓冲区 -> 交付 {[m.seq for m in remaining]}")
    ordered_messages.extend(remaining)

    if show_details:
        print(f"\n[3] 重排序后的有序消息:")
        for i, msg in enumerate(ordered_messages):
            latency = buffer.latencies[i] * 1000
            print(f"  [{i:2d}] Seq={msg.seq:2d} | {msg.data} | 延迟={latency:.4f}ms")

    expected_seqs = list(range(total_messages))
    actual_seqs = [m.seq for m in ordered_messages]
    is_ordered = actual_seqs == expected_seqs
    has_gaps = any(actual_seqs[i] != actual_seqs[i-1] + 1 for i in range(1, len(actual_seqs)))

    print(f"\n[4] 排序验证: {'✓ 完全有序且无丢失' if is_ordered else ('✓ 有序但有跳号' if not has_gaps and len(actual_seqs) == len(expected_seqs) else '✗ 存在问题')}")
    print(f"    期望序列: {expected_seqs}")
    print(f"    实际序列: {actual_seqs}")
    if len(actual_seqs) < len(expected_seqs):
        missing = set(expected_seqs) - set(actual_seqs)
        print(f"    丢失序列号: {sorted(missing)}")

    print(f"\n[5] 缓冲区统计:")
    for key, value in buffer.stats.items():
        if value > 0:
            print(f"  {key}: {value}")

    if overflow_mode == 'dynamic' and buffer.get_adjustment_history():
        print(f"\n[6] 缓冲区动态调整历史:")
        for old_size, new_size, reason in buffer.get_adjustment_history():
            print(f"  {old_size} -> {new_size} ({reason})")

    if buffer.latencies:
        analyzer = LatencyAnalyzer(buffer.latencies)
        analyzer.print_statistics()
        if show_details:
            analyzer.print_histogram(bins=8)

    return {
        'out_of_order': out_of_order_messages,
        'ordered': ordered_messages,
        'latencies': buffer.latencies,
        'stats': buffer.stats,
        'is_ordered': is_ordered,
        'buffer': buffer
    }


def simulate_with_packet_loss(
    total_messages: int = 30,
    shuffle_range: int = 6,
    loss_rate: float = 0.1,
    seed: int = 123
):
    print("\n" + "=" * 60)
    print("扩展场景: 包含丢包的消息乱序模拟")
    print("=" * 60)

    random.seed(seed)
    generator = MessageGenerator(
        total_messages=total_messages,
        shuffle_range=shuffle_range,
        seed=seed
    )

    all_messages = generator.generate_out_of_order()

    received_messages = []
    lost_seqs = []
    for msg in all_messages:
        if random.random() > loss_rate:
            received_messages.append(msg)
        else:
            lost_seqs.append(msg.seq)

    print(f"\n[配置] 消息总数: {total_messages}, 丢包率: {loss_rate * 100:.0f}%")
    print(f"丢失的序列号: {sorted(lost_seqs)}")

    buffer = SequenceBuffer(
        expected_next_seq=0,
        max_buffer_size=10,
        overflow_mode='slide'
    )
    ordered_messages = []

    for msg in received_messages:
        delivered = buffer.add(msg)
        ordered_messages.extend(delivered)

    ordered_messages.extend(buffer.flush_remaining())

    print(f"\n[结果] 收到 {len(received_messages)} 条, 丢失 {len(lost_seqs)} 条")
    print(f"交付序列: {[m.seq for m in ordered_messages]}")

    print(f"\n缓冲区统计:")
    for key, value in buffer.stats.items():
        if value > 0:
            print(f"  {key}: {value}")

    if buffer.latencies:
        analyzer = LatencyAnalyzer(buffer.latencies)
        analyzer.print_statistics()
        analyzer.print_histogram(bins=6)


def compare_overflow_modes(
    total_messages: int = 25,
    shuffle_range: int = 8,
    fixed_buffer_size: int = 4,
    seed: int = 42
):
    print("\n" + "=" * 60)
    print("三种缓冲区溢出处理模式对比测试")
    print("=" * 60)
    print(f"\n测试配置: 消息数={total_messages}, 乱序窗口={shuffle_range}, 固定缓冲={fixed_buffer_size}")

    modes = [
        ('drop', fixed_buffer_size, "模式1: 固定缓冲区+丢包 (有丢消息问题)"),
        ('dynamic', None, "模式2: 动态缓冲区调整 (修复方案)"),
        ('slide', fixed_buffer_size, "模式3: 滑动窗口 (修复方案)"),
    ]

    results = {}
    for mode, buf_size, description in modes:
        print(f"\n{'-' * 60}")
        print(description)
        print('-' * 60)

        result = simulate_out_of_order_scenario(
            total_messages=total_messages,
            shuffle_range=shuffle_range,
            seed=seed,
            max_buffer_size=buf_size,
            overflow_mode=mode,
            show_details=False
        )
        results[mode] = result

    print("\n" + "=" * 60)
    print("对比总结")
    print("=" * 60)
    print(f"{'模式':<25} {'交付数':<8} {'丢失数':<8} {'平均延迟(ms)':<15} {'有序':<6}")
    print("-" * 70)

    for mode, buf_size, description in modes:
        r = results[mode]
        delivered = len(r['ordered'])
        dropped = r['stats'].get('dropped_overflow', 0)
        skipped = r['stats'].get('skipped_count', 0)
        avg_lat = (sum(r['latencies']) / len(r['latencies']) * 1000) if r['latencies'] else 0
        mode_label = f"{mode} (buf={buf_size})" if buf_size else f"{mode} (动态)"
        status = "✓" if r['is_ordered'] else ("△" if skipped > 0 else "✗")
        print(f"{mode_label:<25} {delivered:<8} {dropped + skipped:<8} {avg_lat:<15.2f} {status:<6}")

    print("\n图例: ✓=完全有序, △=滑动跳过缺失, ✗=有丢消息")
    return results


def simulate_extreme_out_of_order(
    total_messages: int = 30,
    seed: int = 99
):
    print("\n" + "=" * 60)
    print("极端乱序场景: 动态缓冲区自适应调整")
    print("=" * 60)

    random.seed(seed)
    messages = []
    for i in range(total_messages):
        messages.append(Message(seq=i, data=f"Message_{i}", timestamp=time.time()))
        time.sleep(0.001)

    random.shuffle(messages)

    print(f"\n[配置] 消息总数: {total_messages}, 完全随机打乱")

    buffer = SequenceBuffer(
        expected_next_seq=0,
        overflow_mode='dynamic',
        min_buffer_size=3,
        dynamic_grow_factor=2.0
    )
    ordered_messages = []

    print(f"\n[1] 处理过程:")
    for i, msg in enumerate(messages):
        delivered = buffer.add(msg)
        buf_size = buffer.get_buffer_size()
        max_buf = buffer.get_max_buffer_size()
        if delivered:
            print(f"  接收 Seq={msg.seq:2d} -> 交付 {[m.seq for m in delivered]} | 缓存 {buf_size}/{max_buf}")
        else:
            print(f"  接收 Seq={msg.seq:2d} -> 缓存 | 缓存 {buf_size}/{max_buf}")
        ordered_messages.extend(delivered)

    ordered_messages.extend(buffer.flush_remaining())

    print(f"\n[2] 动态调整历史:")
    history = buffer.get_adjustment_history()
    if history:
        for old, new, reason in history:
            print(f"  {old:2d} -> {new:2d} ({reason})")
    else:
        print("  无调整")

    is_ordered = all(ordered_messages[i].seq == i for i in range(len(ordered_messages)))
    print(f"\n[3] 结果: {'✓ 完全有序' if is_ordered else '✗ 有问题'}")
    print(f"    最大观测序列号间隔: {buffer.max_seq_gap}")
    print(f"    最终缓冲区容量: {buffer.get_max_buffer_size()}")

    if buffer.latencies:
        analyzer = LatencyAnalyzer(buffer.latencies)
        analyzer.print_statistics()


class OrderingAccuracy:
    @staticmethod
    def calculate_accuracy(expected: List[int], actual: List[int]) -> Dict:
        if not actual:
            return {'accuracy': 0.0, 'inversions': 0, 'missing': len(expected), 'extra': 0}

        expected_set = set(expected)
        actual_set = set(actual)

        missing = len(expected_set - actual_set)
        extra = len(actual_set - expected_set)

        inversions = 0
        for i in range(len(actual)):
            for j in range(i + 1, len(actual)):
                if actual[i] > actual[j]:
                    inversions += 1

        max_inversions = len(actual) * (len(actual) - 1) // 2
        accuracy = 1.0 - (inversions / max_inversions) if max_inversions > 0 else 1.0

        delivery_rate = len(actual) / len(expected) if expected else 0
        correct_order = sum(1 for i in range(min(len(expected), len(actual)))
                           if actual[i] == expected[i])
        perfect_match_ratio = correct_order / len(expected) if expected else 0

        return {
            'accuracy': accuracy,
            'inversions': inversions,
            'missing_count': missing,
            'extra_count': extra,
            'delivery_rate': delivery_rate,
            'perfect_match_ratio': perfect_match_ratio,
            'correct_order_count': correct_order,
        }

    @staticmethod
    def print_accuracy_report(name: str, accuracy: Dict):
        print(f"\n  {name}:")
        print(f"    排序准确率: {accuracy['accuracy'] * 100:.2f}%")
        print(f"    逆序对数量: {accuracy['inversions']}")
        print(f"    交付率: {accuracy['delivery_rate'] * 100:.2f}%")
        print(f"    丢失消息: {accuracy['missing_count']}")
        print(f"    完全匹配率: {accuracy['perfect_match_ratio'] * 100:.2f}%")


class ThroughputCalculator:
    @staticmethod
    def calculate(processing_times: List[float], total_messages: int) -> Dict:
        if not processing_times:
            return {'throughput_msg_per_sec': 0, 'avg_processing_time_ms': 0}

        total_time = sum(processing_times)
        avg_time = total_time / len(processing_times)
        throughput = total_messages / total_time if total_time > 0 else 0

        return {
            'throughput_msg_per_sec': throughput,
            'throughput_msg_per_ms': throughput / 1000,
            'avg_processing_time_ms': avg_time * 1000,
            'total_processing_time_ms': total_time * 1000,
            'total_messages': total_messages,
        }

    @staticmethod
    def print_throughput_report(name: str, throughput: Dict):
        print(f"\n  {name}:")
        print(f"    吞吐量: {throughput['throughput_msg_per_sec']:.2f} msg/s")
        print(f"    平均处理时间: {throughput['avg_processing_time_ms']:.4f} ms")
        print(f"    总处理时间: {throughput['total_processing_time_ms']:.4f} ms")


def simulate_timestamp_window_scenario(
    total_messages: int = 25,
    shuffle_range: int = 8,
    window_seconds: float = 0.02,
    seed: int = 42,
    show_details: bool = True
):
    print("\n" + "=" * 60)
    print("基于物理时间戳的时间窗口重排序")
    print("=" * 60)

    generator = MessageGenerator(
        total_messages=total_messages,
        shuffle_range=shuffle_range,
        seed=seed
    )

    print(f"\n[配置] 消息数={total_messages}, 乱序窗口={shuffle_range}, 时间窗口={window_seconds*1000:.1f}ms")

    messages = generator.generate_out_of_order()

    if show_details:
        print(f"\n[1] 乱序消息流 (带时间戳):")
        for i, msg in enumerate(messages):
            print(f"  [{i:2d}] Seq={msg.seq:2d} | ts={msg.timestamp*1000:.1f}ms | {msg.data}")

    sorter = TimestampWindowSorter(window_seconds=window_seconds)
    ordered = []

    if show_details:
        print(f"\n[2] 重排序过程:")
    for i, msg in enumerate(messages):
        delivered = sorter.add(msg)
        if show_details:
            info = f"水位={sorter.low_watermark*1000:.1f}ms, 缓存={sorter.get_buffer_size()}"
            if delivered:
                print(f"  接收 Seq={msg.seq:2d} -> 交付 {[m.seq for m in delivered]} | {info}")
            else:
                print(f"  接收 Seq={msg.seq:2d} -> 缓存等待 | {info}")
        ordered.extend(delivered)

    ordered.extend(sorter.flush())

    if show_details:
        print(f"\n[3] 最终有序序列:")
        for i, msg in enumerate(ordered):
            latency = sorter.latencies[i] * 1000
            print(f"  [{i:2d}] Seq={msg.seq:2d} | 延迟={latency:.2f}ms")

    expected = list(range(total_messages))
    actual = [m.seq for m in ordered]

    accuracy = OrderingAccuracy.calculate_accuracy(expected, actual)
    OrderingAccuracy.print_accuracy_report("排序准确性", accuracy)

    throughput = ThroughputCalculator.calculate(sorter.processing_times, total_messages)
    ThroughputCalculator.print_throughput_report("性能指标", throughput)

    return {
        'ordered': ordered,
        'accuracy': accuracy,
        'throughput': throughput,
        'sorter': sorter,
    }


def simulate_lamport_causal_scenario(
    total_messages: int = 25,
    shuffle_range: int = 8,
    causality_depth: int = 3,
    seed: int = 42,
    show_details: bool = True
):
    print("\n" + "=" * 60)
    print("基于Lamport时间戳的因果序重排序")
    print("=" * 60)

    generator = MessageGenerator(
        total_messages=total_messages,
        shuffle_range=shuffle_range,
        seed=seed
    )

    print(f"\n[配置] 消息数={total_messages}, 乱序窗口={shuffle_range}, 因果深度={causality_depth}")

    messages = generator.generate_out_of_order_causal(causality_depth=causality_depth)

    if show_details:
        print(f"\n[1] 乱序消息流 (带Lamport时间戳):")
        for i, msg in enumerate(messages):
            print(f"  [{i:2d}] Seq={msg.seq:2d} | Lamport={msg.lamport_ts:3d} | {msg.data}")

    sorter = LamportSorter(expected_next_seq=0)
    ordered = []

    if show_details:
        print(f"\n[2] 重排序过程:")
    for i, msg in enumerate(messages):
        delivered = sorter.add(msg)
        if show_details:
            info = f"本地时钟={sorter.get_local_time():3d}, 缓存={sorter.get_buffer_size()}"
            if delivered:
                print(f"  接收 Seq={msg.seq:2d} (L={msg.lamport_ts:3d}) -> 交付 {[m.seq for m in delivered]} | {info}")
            else:
                print(f"  接收 Seq={msg.seq:2d} (L={msg.lamport_ts:3d}) -> 缓存等待 | {info}")
        ordered.extend(delivered)

    ordered.extend(sorter.flush())

    if show_details:
        print(f"\n[3] 最终有序序列:")
        for i, msg in enumerate(ordered):
            latency = sorter.latencies[i] * 1000
            print(f"  [{i:2d}] Seq={msg.seq:2d} | Lamport={msg.lamport_ts:3d} | 延迟={latency:.2f}ms")

    expected = list(range(total_messages))
    actual = [m.seq for m in ordered]

    accuracy = OrderingAccuracy.calculate_accuracy(expected, actual)
    OrderingAccuracy.print_accuracy_report("排序准确性", accuracy)

    throughput = ThroughputCalculator.calculate(sorter.processing_times, total_messages)
    ThroughputCalculator.print_throughput_report("性能指标", throughput)

    print(f"\n[4] Lamport时钟统计:")
    print(f"    最大观测Lamport时间戳: {sorter.max_observed_lamport}")
    print(f"    最终本地时钟: {sorter.get_local_time()}")
    print(f"    并发事件数: {sorter.concurrent_events}")

    return {
        'ordered': ordered,
        'accuracy': accuracy,
        'throughput': throughput,
        'sorter': sorter,
    }


def compare_all_sorters(
    total_messages: int = 100,
    shuffle_range: int = 15,
    seed: int = 42,
    runs: int = 3
):
    print("\n" + "=" * 70)
    print("多方案综合对比测试: 吞吐量 vs 排序准确性")
    print("=" * 70)
    print(f"\n测试配置: 消息数={total_messages}, 乱序窗口={shuffle_range}, 运行次数={runs}")

    sorter_configs = [
        ('SequenceBuffer(dynamic)', lambda: SequenceBuffer(
            expected_next_seq=0, overflow_mode='dynamic'), 'seq'),
        ('TimestampWindow(10ms)', lambda: TimestampWindowSorter(
            window_seconds=0.01), 'ts'),
        ('TimestampWindow(50ms)', lambda: TimestampWindowSorter(
            window_seconds=0.05), 'ts'),
        ('LamportSorter', lambda: LamportSorter(
            expected_next_seq=0), 'lamport'),
    ]

    all_results = defaultdict(lambda: defaultdict(list))

    for run in range(runs):
        print(f"\n[Run {run + 1}/{runs}] 生成测试数据...")
        current_seed = seed + run * 1000
        generator = MessageGenerator(
            total_messages=total_messages,
            shuffle_range=shuffle_range,
            seed=current_seed
        )
        messages = generator.generate_out_of_order_causal(causality_depth=5)

        for name, sorter_factory, _ in sorter_configs:
            sorter = sorter_factory()
            ordered = []

            start_time = time.time()
            for msg in messages:
                if hasattr(sorter, 'flush_remaining'):
                    delivered = sorter.add(msg)
                else:
                    delivered = sorter.add(msg)
                ordered.extend(delivered)

            if hasattr(sorter, 'flush_remaining'):
                ordered.extend(sorter.flush_remaining())
            else:
                ordered.extend(sorter.flush())

            elapsed = time.time() - start_time

            expected = list(range(total_messages))
            actual = [m.seq for m in ordered]

            accuracy = OrderingAccuracy.calculate_accuracy(expected, actual)
            throughput = len(ordered) / elapsed if elapsed > 0 else 0
            avg_latency = (sum(sorter.latencies) / len(sorter.latencies) * 1000
                          if sorter.latencies else 0)

            all_results[name]['accuracy'].append(accuracy['accuracy'])
            all_results[name]['perfect_match'].append(accuracy['perfect_match_ratio'])
            all_results[name]['throughput'].append(throughput)
            all_results[name]['avg_latency_ms'].append(avg_latency)
            all_results[name]['delivery_rate'].append(accuracy['delivery_rate'])

    print("\n" + "=" * 70)
    print("综合对比结果 (平均值)")
    print("=" * 70)
    print(f"{'方案':<25} {'吞吐量(msg/s)':<18} {'准确率(%)':<12} {'完全匹配(%)':<14} {'平均延迟(ms)':<14} {'交付率(%)':<12}")
    print("-" * 95)

    for name, _, _ in sorter_configs:
        r = all_results[name]
        avg_throughput = sum(r['throughput']) / len(r['throughput'])
        avg_accuracy = sum(r['accuracy']) / len(r['accuracy']) * 100
        avg_perfect = sum(r['perfect_match']) / len(r['perfect_match']) * 100
        avg_latency = sum(r['avg_latency_ms']) / len(r['avg_latency_ms'])
        avg_delivery = sum(r['delivery_rate']) / len(r['delivery_rate']) * 100

        print(f"{name:<25} {avg_throughput:<18.2f} {avg_accuracy:<12.2f} "
              f"{avg_perfect:<14.2f} {avg_latency:<14.2f} {avg_delivery:<12.2f}")

    print("\n" + "=" * 70)
    print("方案特性总结")
    print("=" * 70)
    print("""
  SequenceBuffer (dynamic):
    ✓ 完全有序 (基于序列号)
    ✓ 不丢消息
    ✓ 动态扩容适应乱序
    ? 延迟取决于乱序程度

  TimestampWindow (时间窗口):
    ✓ 基于物理时间的自然排序
    ✓ 可配置延迟-准确性权衡
    ? 窗口太小会丢排序准确性
    ? 窗口太大会增加延迟

  LamportSorter (因果序):
    ✓ 维护因果关系 (happens-before)
    ✓ 检测并发事件
    ✓ 适合分布式系统
    ? 逻辑时钟与物理时间无直接关系
    """)

    return all_results


if __name__ == "__main__":
    result1 = simulate_out_of_order_scenario(
        total_messages=20,
        shuffle_range=5,
        seed=42,
        max_buffer_size=None,
        overflow_mode='dynamic'
    )

    simulate_timestamp_window_scenario(
        total_messages=25,
        shuffle_range=8,
        window_seconds=0.02,
        seed=42
    )

    simulate_lamport_causal_scenario(
        total_messages=25,
        shuffle_range=8,
        causality_depth=3,
        seed=42
    )

    compare_overflow_modes(
        total_messages=25,
        shuffle_range=8,
        fixed_buffer_size=4,
        seed=42
    )

    simulate_extreme_out_of_order(
        total_messages=30,
        seed=99
    )

    compare_all_sorters(
        total_messages=100,
        shuffle_range=15,
        seed=42,
        runs=3
    )
