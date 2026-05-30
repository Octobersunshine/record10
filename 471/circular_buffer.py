from __future__ import annotations
import threading
import time
from typing import Generic, TypeVar, Optional

T = TypeVar("T")


class CircularBuffer(Generic[T]):
    """计数器法：用 _count 区分空/满，head==tail 时靠 _count 判断状态。"""

    def __init__(self, capacity: int, overwrite: bool = True):
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self._buffer: list[Optional[T]] = [None] * capacity
        self._capacity = capacity
        self._head = 0
        self._tail = 0
        self._count = 0
        self._overwrite = overwrite

    @property
    def capacity(self) -> int:
        return self._capacity

    @property
    def used(self) -> int:
        return self._count

    @property
    def remaining(self) -> int:
        return self._capacity - self._count

    @property
    def is_empty(self) -> bool:
        return self._count == 0

    @property
    def is_full(self) -> bool:
        return self._count == self._capacity

    def write(self, item: T) -> None:
        if self.is_full:
            if not self._overwrite:
                raise OverflowError("buffer is full and overwrite is disabled")
            self._head = (self._head + 1) % self._capacity
            self._count -= 1
        self._buffer[self._tail] = item
        self._tail = (self._tail + 1) % self._capacity
        self._count += 1

    def read(self) -> T:
        if self.is_empty:
            raise IndexError("read from empty buffer")
        item = self._buffer[self._head]
        self._buffer[self._head] = None
        self._head = (self._head + 1) % self._capacity
        self._count -= 1
        return item

    def peek(self) -> T:
        if self.is_empty:
            raise IndexError("peek from empty buffer")
        return self._buffer[self._head]

    def clear(self) -> None:
        self._buffer = [None] * self._capacity
        self._head = 0
        self._tail = 0
        self._count = 0

    def status(self) -> dict:
        return {
            "capacity": self._capacity,
            "used": self.used,
            "remaining": self.remaining,
            "is_empty": self.is_empty,
            "is_full": self.is_full,
            "overwrite_mode": self._overwrite,
            "strategy": "counter",
        }

    def __len__(self) -> int:
        return self._count

    def __repr__(self) -> str:
        items = []
        for i in range(self._count):
            idx = (self._head + i) % self._capacity
            items.append(repr(self._buffer[idx]))
        return f"CircularBuffer([{', '.join(items)}], capacity={self._capacity})"

    def __iter__(self):
        for i in range(self._count):
            idx = (self._head + i) % self._capacity
            yield self._buffer[idx]


class CircularBufferSacrifice(Generic[T]):
    """牺牲一个存储单元法：不维护计数器，通过 (tail+1)%capacity==head 判断满，
    head==tail 判断空。实际可用容量为 capacity-1。"""

    def __init__(self, capacity: int, overwrite: bool = True):
        if capacity <= 1:
            raise ValueError("capacity must be >= 2 for sacrifice strategy")
        self._buffer: list[Optional[T]] = [None] * capacity
        self._storage_size = capacity
        self._head = 0
        self._tail = 0
        self._overwrite = overwrite

    @property
    def capacity(self) -> int:
        return self._storage_size - 1

    @property
    def used(self) -> int:
        diff = self._tail - self._head
        if diff < 0:
            diff += self._storage_size
        return diff

    @property
    def remaining(self) -> int:
        return self.capacity - self.used

    @property
    def is_empty(self) -> bool:
        return self._head == self._tail

    @property
    def is_full(self) -> bool:
        return (self._tail + 1) % self._storage_size == self._head

    def write(self, item: T) -> None:
        if self.is_full:
            if not self._overwrite:
                raise OverflowError("buffer is full and overwrite is disabled")
            self._head = (self._head + 1) % self._storage_size
        self._buffer[self._tail] = item
        self._tail = (self._tail + 1) % self._storage_size

    def read(self) -> T:
        if self.is_empty:
            raise IndexError("read from empty buffer")
        item = self._buffer[self._head]
        self._buffer[self._head] = None
        self._head = (self._head + 1) % self._storage_size
        return item

    def peek(self) -> T:
        if self.is_empty:
            raise IndexError("peek from empty buffer")
        return self._buffer[self._head]

    def clear(self) -> None:
        self._buffer = [None] * self._storage_size
        self._head = 0
        self._tail = 0

    def status(self) -> dict:
        return {
            "storage_size": self._storage_size,
            "usable_capacity": self.capacity,
            "used": self.used,
            "remaining": self.remaining,
            "is_empty": self.is_empty,
            "is_full": self.is_full,
            "overwrite_mode": self._overwrite,
            "strategy": "sacrifice_one_slot",
            "head": self._head,
            "tail": self._tail,
        }

    def __len__(self) -> int:
        return self.used

    def __repr__(self) -> str:
        items = []
        count = self.used
        for i in range(count):
            idx = (self._head + i) % self._storage_size
            items.append(repr(self._buffer[idx]))
        return (
            f"CircularBufferSacrifice([{', '.join(items)}], "
            f"usable={self.capacity}, storage={self._storage_size})"
        )

    def __iter__(self):
        count = self.used
        for i in range(count):
            idx = (self._head + i) % self._storage_size
            yield self._buffer[idx]


class ThreadSafeCircularBuffer(Generic[T]):
    """加锁线程安全环形缓冲区，支持批量读写。

    使用 threading.Lock 保护所有操作，提供 write_batch / read_batch
    批量接口，批量操作在单次锁内完成，避免逐元素加锁的开销。
    """

    def __init__(self, capacity: int, overwrite: bool = True):
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self._buffer: list[Optional[T]] = [None] * capacity
        self._capacity = capacity
        self._head = 0
        self._tail = 0
        self._count = 0
        self._overwrite = overwrite
        self._lock = threading.Lock()

    @property
    def capacity(self) -> int:
        return self._capacity

    @property
    def used(self) -> int:
        with self._lock:
            return self._count

    @property
    def remaining(self) -> int:
        with self._lock:
            return self._capacity - self._count

    @property
    def is_empty(self) -> bool:
        with self._lock:
            return self._count == 0

    @property
    def is_full(self) -> bool:
        with self._lock:
            return self._count == self._capacity

    def write(self, item: T) -> None:
        with self._lock:
            self._write_unsafe(item)

    def write_batch(self, items: list[T]) -> int:
        """批量写入，返回实际写入数量。overwrite=False 时满即停。"""
        written = 0
        with self._lock:
            for item in items:
                if self._count == self._capacity:
                    if not self._overwrite:
                        break
                    self._head = (self._head + 1) % self._capacity
                    self._count -= 1
                self._buffer[self._tail] = item
                self._tail = (self._tail + 1) % self._capacity
                self._count += 1
                written += 1
        return written

    def read(self) -> T:
        with self._lock:
            return self._read_unsafe()

    def read_batch(self, n: int) -> list[T]:
        """批量读取最多 n 个元素，返回实际读到的列表。"""
        result: list[T] = []
        with self._lock:
            for _ in range(n):
                if self._count == 0:
                    break
                result.append(self._read_unsafe())
        return result

    def peek(self) -> T:
        with self._lock:
            if self._count == 0:
                raise IndexError("peek from empty buffer")
            return self._buffer[self._head]

    def clear(self) -> None:
        with self._lock:
            self._buffer = [None] * self._capacity
            self._head = 0
            self._tail = 0
            self._count = 0

    def status(self) -> dict:
        with self._lock:
            return {
                "capacity": self._capacity,
                "used": self._count,
                "remaining": self._capacity - self._count,
                "is_empty": self._count == 0,
                "is_full": self._count == self._capacity,
                "overwrite_mode": self._overwrite,
                "strategy": "thread_safe_locked",
            }

    def _write_unsafe(self, item: T) -> None:
        if self._count == self._capacity:
            if not self._overwrite:
                raise OverflowError("buffer is full and overwrite is disabled")
            self._head = (self._head + 1) % self._capacity
            self._count -= 1
        self._buffer[self._tail] = item
        self._tail = (self._tail + 1) % self._capacity
        self._count += 1

    def _read_unsafe(self) -> T:
        if self._count == 0:
            raise IndexError("read from empty buffer")
        item = self._buffer[self._head]
        self._buffer[self._head] = None
        self._head = (self._head + 1) % self._capacity
        self._count -= 1
        return item

    def __len__(self) -> int:
        with self._lock:
            return self._count

    def __repr__(self) -> str:
        with self._lock:
            items = []
            for i in range(self._count):
                idx = (self._head + i) % self._capacity
                items.append(repr(self._buffer[idx]))
            return (
                f"ThreadSafeCircularBuffer([{', '.join(items)}], "
                f"capacity={self._capacity})"
            )


class LockFreeCircularBuffer(Generic[T]):
    """无锁环形缓冲区（SeqLock 模式）。

    原理：
      写者获取写锁（同一时刻只有一个写者），写入完成后递增 _write_seq。
      读者通过读取 _write_seq 的前后值，检测是否在读取期间发生了写入；
      若发生写入则重试（乐观读）。单写多读场景下读者完全无锁。

    注意：Python 的 GIL 保证了单条字节码的原子性，但复合操作仍需同步。
    SeqLock 模式在此的实现侧重于演示原理，生产环境建议使用加锁方案。
    """

    def __init__(self, capacity: int, overwrite: bool = True):
        if capacity <= 1:
            raise ValueError("capacity must be >= 2 for lock-free buffer")
        self._buffer: list[Optional[T]] = [None] * capacity
        self._storage_size = capacity
        self._head = 0
        self._tail = 0
        self._overwrite = overwrite
        self._write_seq = 0
        self._write_lock = threading.Lock()

    @property
    def capacity(self) -> int:
        return self._storage_size - 1

    @property
    def used(self) -> int:
        seq1, seq2 = 0, -1
        while seq1 != seq2:
            seq1 = self._write_seq
            head = self._head
            tail = self._tail
            seq2 = self._write_seq
        diff = tail - head
        if diff < 0:
            diff += self._storage_size
        return diff

    @property
    def remaining(self) -> int:
        return self.capacity - self.used

    @property
    def is_empty(self) -> bool:
        seq1, seq2 = 0, -1
        while seq1 != seq2:
            seq1 = self._write_seq
            result = self._head == self._tail
            seq2 = self._write_seq
        return result

    @property
    def is_full(self) -> bool:
        seq1, seq2 = 0, -1
        while seq1 != seq2:
            seq1 = self._write_seq
            result = (self._tail + 1) % self._storage_size == self._head
            seq2 = self._write_seq
        return result

    def write(self, item: T) -> None:
        with self._write_lock:
            if (self._tail + 1) % self._storage_size == self._head:
                if not self._overwrite:
                    raise OverflowError("buffer is full and overwrite is disabled")
                self._head = (self._head + 1) % self._storage_size
            self._buffer[self._tail] = item
            self._tail = (self._tail + 1) % self._storage_size
            self._write_seq += 1

    def write_batch(self, items: list[T]) -> int:
        written = 0
        with self._write_lock:
            for item in items:
                if (self._tail + 1) % self._storage_size == self._head:
                    if not self._overwrite:
                        break
                    self._head = (self._head + 1) % self._storage_size
                self._buffer[self._tail] = item
                self._tail = (self._tail + 1) % self._storage_size
                written += 1
            self._write_seq += 1
        return written

    def read(self) -> T:
        with self._write_lock:
            if self._head == self._tail:
                raise IndexError("read from empty buffer")
            item = self._buffer[self._head]
            self._buffer[self._head] = None
            self._head = (self._head + 1) % self._storage_size
            self._write_seq += 1
            return item

    def read_batch(self, n: int) -> list[T]:
        result: list[T] = []
        with self._write_lock:
            for _ in range(n):
                if self._head == self._tail:
                    break
                result.append(self._buffer[self._head])
                self._buffer[self._head] = None
                self._head = (self._head + 1) % self._storage_size
            self._write_seq += 1
        return result

    def peek(self) -> T:
        seq1, seq2 = 0, -1
        while seq1 != seq2:
            seq1 = self._write_seq
            if self._head == self._tail:
                raise IndexError("peek from empty buffer")
            item = self._buffer[self._head]
            seq2 = self._write_seq
        return item

    def clear(self) -> None:
        with self._write_lock:
            self._buffer = [None] * self._storage_size
            self._head = 0
            self._tail = 0
            self._write_seq += 1

    def status(self) -> dict:
        return {
            "storage_size": self._storage_size,
            "usable_capacity": self.capacity,
            "used": self.used,
            "remaining": self.remaining,
            "is_empty": self.is_empty,
            "is_full": self.is_full,
            "overwrite_mode": self._overwrite,
            "strategy": "lock_free_seqlock",
            "write_seq": self._write_seq,
        }

    def __len__(self) -> int:
        return self.used


class CircularBufferDynamic(Generic[T]):
    """动态扩容环形缓冲区。

    当 overwrite=False 且缓冲区满时，自动按 growth_factor 扩大容量。
    扩容时分配新数组，将旧数据按逻辑顺序复制到新数组起始位置，
    重置 head=0, tail=count，保证语义不变。
    支持 shrink_threshold 自动缩容：当 used/capacity < shrink_threshold
    且容量大于初始容量时，缩减到 max(initial_capacity, used * 2)。
    """

    def __init__(
        self,
        capacity: int,
        overwrite: bool = False,
        growth_factor: float = 2.0,
        max_capacity: Optional[int] = None,
        shrink_threshold: float = 0.25,
    ):
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        if growth_factor <= 1.0:
            raise ValueError("growth_factor must be > 1.0")
        self._initial_capacity = capacity
        self._buffer: list[Optional[T]] = [None] * capacity
        self._capacity = capacity
        self._head = 0
        self._tail = 0
        self._count = 0
        self._overwrite = overwrite
        self._growth_factor = growth_factor
        self._max_capacity = max_capacity
        self._shrink_threshold = shrink_threshold
        self._resize_count = 0

    @property
    def capacity(self) -> int:
        return self._capacity

    @property
    def used(self) -> int:
        return self._count

    @property
    def remaining(self) -> int:
        return self._capacity - self._count

    @property
    def is_empty(self) -> bool:
        return self._count == 0

    @property
    def is_full(self) -> bool:
        return self._count == self._capacity

    def write(self, item: T) -> None:
        if self._count == self._capacity:
            if self._overwrite:
                self._buffer[self._head] = None
                self._head = (self._head + 1) % self._capacity
                self._count -= 1
            else:
                self._grow()
        self._buffer[self._tail] = item
        self._tail = (self._tail + 1) % self._capacity
        self._count += 1

    def write_batch(self, items: list[T]) -> int:
        written = 0
        for item in items:
            if self._count == self._capacity:
                if self._overwrite:
                    self._buffer[self._head] = None
                    self._head = (self._head + 1) % self._capacity
                    self._count -= 1
                else:
                    self._grow()
            self._buffer[self._tail] = item
            self._tail = (self._tail + 1) % self._capacity
            self._count += 1
            written += 1
        return written

    def read(self) -> T:
        if self._count == 0:
            raise IndexError("read from empty buffer")
        item = self._buffer[self._head]
        self._buffer[self._head] = None
        self._head = (self._head + 1) % self._capacity
        self._count -= 1
        self._try_shrink()
        return item

    def read_batch(self, n: int) -> list[T]:
        result: list[T] = []
        for _ in range(n):
            if self._count == 0:
                break
            result.append(self._buffer[self._head])
            self._buffer[self._head] = None
            self._head = (self._head + 1) % self._capacity
            self._count -= 1
        self._try_shrink()
        return result

    def peek(self) -> T:
        if self._count == 0:
            raise IndexError("peek from empty buffer")
        return self._buffer[self._head]

    def clear(self) -> None:
        self._buffer = [None] * self._initial_capacity
        self._capacity = self._initial_capacity
        self._head = 0
        self._tail = 0
        self._count = 0

    def _grow(self) -> None:
        new_cap = int(self._capacity * self._growth_factor)
        if self._max_capacity is not None:
            new_cap = min(new_cap, self._max_capacity)
        if new_cap <= self._capacity:
            raise OverflowError(
                f"cannot grow beyond capacity {self._capacity}"
            )
        self._reallocate(new_cap)

    def _try_shrink(self) -> None:
        if self._capacity <= self._initial_capacity:
            return
        if self._count == 0:
            ratio = 0.0
        else:
            ratio = self._count / self._capacity
        if ratio < self._shrink_threshold:
            new_cap = max(self._initial_capacity, self._count * 2)
            if new_cap < self._capacity:
                self._reallocate(new_cap)

    def _reallocate(self, new_cap: int) -> None:
        old_items = []
        for i in range(self._count):
            idx = (self._head + i) % self._capacity
            old_items.append(self._buffer[idx])
        self._buffer = [None] * new_cap
        for i, item in enumerate(old_items):
            self._buffer[i] = item
        self._head = 0
        self._tail = self._count
        self._capacity = new_cap
        self._resize_count += 1

    def status(self) -> dict:
        return {
            "capacity": self._capacity,
            "initial_capacity": self._initial_capacity,
            "used": self._count,
            "remaining": self._capacity - self._count,
            "is_empty": self._count == 0,
            "is_full": self._count == self._capacity,
            "overwrite_mode": self._overwrite,
            "growth_factor": self._growth_factor,
            "max_capacity": self._max_capacity,
            "shrink_threshold": self._shrink_threshold,
            "resize_count": self._resize_count,
            "strategy": "dynamic",
        }

    def __len__(self) -> int:
        return self._count

    def __repr__(self) -> str:
        items = []
        for i in range(self._count):
            idx = (self._head + i) % self._capacity
            items.append(repr(self._buffer[idx]))
        return (
            f"CircularBufferDynamic([{', '.join(items)}], "
            f"capacity={self._capacity})"
        )

    def __iter__(self):
        for i in range(self._count):
            idx = (self._head + i) % self._capacity
            yield self._buffer[idx]


class BufferMonitor(Generic[T]):
    """缓冲区使用率监控装饰器。

    包装任意缓冲区实例，透明代理所有操作，同时记录：
      - total_writes / total_reads：累计写入/读取次数
      - total_write_items / total_read_items：累计写入/读取元素数
      - peak_used：历史最高使用量
      - utilization()：当前使用率 (used / capacity)
      - snapshot()：返回完整监控快照
    """

    def __init__(self, buffer: CircularBuffer | CircularBufferSacrifice
                 | ThreadSafeCircularBuffer | LockFreeCircularBuffer
                 | CircularBufferDynamic):
        self._buf = buffer
        self._monitor_lock = threading.Lock()
        self._total_writes = 0
        self._total_reads = 0
        self._total_write_items = 0
        self._total_read_items = 0
        self._peak_used = 0
        self._start_time = time.monotonic()

    def _record_write(self, count: int = 1) -> None:
        with self._monitor_lock:
            self._total_writes += 1
            self._total_write_items += count
            current = self._buf.used
            if current > self._peak_used:
                self._peak_used = current

    def _record_read(self, count: int = 1) -> None:
        with self._monitor_lock:
            self._total_reads += 1
            self._total_read_items += count

    def write(self, item: T) -> None:
        self._buf.write(item)
        self._record_write()

    def write_batch(self, items: list[T]) -> int:
        written = self._buf.write_batch(items)
        if written > 0:
            self._record_write(written)
        return written

    def read(self) -> T:
        item = self._buf.read()
        self._record_read()
        return item

    def read_batch(self, n: int) -> list[T]:
        items = self._buf.read_batch(n)
        if items:
            self._record_read(len(items))
        return items

    def peek(self) -> T:
        return self._buf.peek()

    def clear(self) -> None:
        self._buf.clear()

    @property
    def capacity(self) -> int:
        return self._buf.capacity

    @property
    def used(self) -> int:
        return self._buf.used

    @property
    def remaining(self) -> int:
        return self._buf.remaining

    @property
    def is_empty(self) -> bool:
        return self._buf.is_empty

    @property
    def is_full(self) -> bool:
        return self._buf.is_full

    def utilization(self) -> float:
        """返回当前使用率 0.0 ~ 1.0。"""
        cap = self._buf.capacity
        if cap == 0:
            return 0.0
        return self._buf.used / cap

    def snapshot(self) -> dict:
        elapsed = time.monotonic() - self._start_time
        with self._monitor_lock:
            tw = self._total_writes
            tr = self._total_reads
            twi = self._total_write_items
            tri = self._total_read_items
            peak = self._peak_used
        write_rate = twi / elapsed if elapsed > 0 else 0.0
        read_rate = tri / elapsed if elapsed > 0 else 0.0
        return {
            "capacity": self._buf.capacity,
            "used": self._buf.used,
            "utilization": self.utilization(),
            "peak_used": peak,
            "total_write_ops": tw,
            "total_read_ops": tr,
            "total_write_items": twi,
            "total_read_items": tri,
            "write_rate_per_sec": round(write_rate, 2),
            "read_rate_per_sec": round(read_rate, 2),
            "elapsed_sec": round(elapsed, 3),
        }

    def status(self) -> dict:
        return self._buf.status()

    def __len__(self) -> int:
        return len(self._buf)

    def __repr__(self) -> str:
        return f"BufferMonitor({self._buf!r})"


def _demo_counter():
    print("=" * 60)
    print("策略一：计数器法 (CircularBuffer)")
    print("  空/满判断：head==tail 时靠 _count 区分")
    print("  实际容量 == capacity")
    print("=" * 60)

    buf = CircularBuffer(capacity=4, overwrite=True)
    print(f"初始状态: {buf.status()}\n")

    for i in range(7):
        buf.write(i)
        print(f"write({i}) -> head={buf._head}, tail={buf._tail}, "
              f"count={buf._count} | {buf}")

    print()
    print("--- 逐个读取 ---")
    while not buf.is_empty:
        val = buf.read()
        print(f"read() -> {val}, head={buf._head}, tail={buf._tail}, "
              f"count={buf._count}")

    print(f"\n读空后: {buf.status()}")

    print("\n--- 覆盖关闭模式 ---")
    buf2 = CircularBuffer(capacity=3, overwrite=False)
    for i in range(3):
        buf2.write(i)
    print(f"写入3个: {buf2}")
    try:
        buf2.write(99)
    except OverflowError as e:
        print(f"write(99) -> OverflowError: {e}")


def _demo_sacrifice():
    print("\n" + "=" * 60)
    print("策略二：牺牲一个存储单元法 (CircularBufferSacrifice)")
    print("  空判断：head == tail")
    print("  满判断：(tail + 1) % storage_size == head")
    print("  实际可用容量 == storage_size - 1")
    print("=" * 60)

    buf = CircularBufferSacrifice(capacity=5, overwrite=True)
    print(f"初始状态: {buf.status()}")
    print(f"  底层数组长度={buf._storage_size}, 可用容量={buf.capacity}\n")

    for i in range(7):
        buf.write(i)
        print(f"write({i}) -> head={buf._head}, tail={buf._tail}, "
              f"used={buf.used}/{buf.capacity} | {buf}")

    print()
    print("--- 逐个读取 ---")
    while not buf.is_empty:
        val = buf.read()
        print(f"read() -> {val}, head={buf._head}, tail={buf._tail}, "
              f"used={buf.used}/{buf.capacity}")

    print(f"\n读空后: {buf.status()}")


def _demo_thread_safe():
    print("\n" + "=" * 60)
    print("线程安全缓冲区 + 批量读写")
    print("=" * 60)

    buf = ThreadSafeCircularBuffer(capacity=10, overwrite=True)

    written = buf.write_batch(list(range(15)))
    print(f"write_batch([0..14]) -> 写入 {written} 个")
    print(f"  状态: {buf.status()}")

    batch = buf.read_batch(5)
    print(f"\nread_batch(5) -> {batch}")
    print(f"  剩余: {buf.used}, 状态: {buf.status()}")

    batch = buf.read_batch(20)
    print(f"\nread_batch(20) -> {batch} (请求20，实际5)")
    print(f"  剩余: {buf.used}")

    print("\n--- 多线程并发写入/读取 ---")
    buf2 = ThreadSafeCircularBuffer(capacity=1000, overwrite=False)
    errors: list[str] = []

    def writer(start: int, count: int):
        try:
            for i in range(count):
                buf2.write(start + i)
        except Exception as e:
            errors.append(f"writer-{start}: {e}")

    def reader(count: int):
        try:
            read_items = buf2.read_batch(count)
        except Exception as e:
            errors.append(f"reader: {e}")

    wt1 = threading.Thread(target=writer, args=(0, 500))
    wt2 = threading.Thread(target=writer, args=(500, 500))
    wt1.start()
    wt2.start()
    wt1.join()
    wt2.join()

    print(f"  两个写者各写500 -> used={buf2.used}/{buf2.capacity}")
    print(f"  状态: {buf2.status()}")

    rt1 = threading.Thread(target=reader, args=(300,))
    rt2 = threading.Thread(target=reader, args=(300,))
    rt1.start()
    rt2.start()
    rt1.join()
    rt2.join()

    print(f"  两个读者各读300 -> used={buf2.used}/{buf2.capacity}")
    if errors:
        print(f"  错误: {errors}")
    else:
        print(f"  无并发错误")


def _demo_lock_free():
    print("\n" + "=" * 60)
    print("无锁缓冲区 (SeqLock 模式)")
    print("=" * 60)

    buf = LockFreeCircularBuffer(capacity=6, overwrite=True)
    print(f"初始: {buf.status()}")

    buf.write_batch([10, 20, 30])
    print(f"write_batch([10,20,30]) -> used={buf.used}")

    items = buf.read_batch(2)
    print(f"read_batch(2) -> {items}, used={buf.used}")

    buf.write_batch([40, 50, 60, 70])
    print(f"write_batch([40,50,60,70]) -> used={buf.used}")

    items = buf.read_batch(10)
    print(f"read_batch(10) -> {items}")

    print(f"最终: {buf.status()}")


def _demo_dynamic():
    print("\n" + "=" * 60)
    print("动态扩容缓冲区")
    print("=" * 60)

    buf = CircularBufferDynamic(
        capacity=4,
        overwrite=False,
        growth_factor=2.0,
        shrink_threshold=0.25,
    )
    print(f"初始: capacity={buf.capacity}, {buf.status()}\n")

    for i in range(10):
        buf.write(i)
        print(f"write({i}) -> capacity={buf.capacity}, used={buf.used}, "
              f"resizes={buf._resize_count}")

    print(f"\n写入10个后: {buf.status()}")

    print("\n--- 批量读取触发缩容 ---")
    batch = buf.read_batch(8)
    print(f"read_batch(8) -> {batch}")
    print(f"读取后: capacity={buf.capacity}, used={buf.used}, "
          f"resizes={buf._resize_count}")

    batch = buf.read_batch(10)
    print(f"read_batch(10) -> {batch}")
    print(f"全部读出: capacity={buf.capacity}, used={buf.used}, "
          f"resizes={buf._resize_count}")

    print("\n--- 重新写入触发扩容 ---")
    buf.write_batch(list(range(20)))
    print(f"write_batch([0..19]) -> capacity={buf.capacity}, used={buf.used}")
    print(f"最终: {buf.status()}")


def _demo_monitor():
    print("\n" + "=" * 60)
    print("缓冲区使用率监控 (BufferMonitor)")
    print("=" * 60)

    inner = CircularBufferDynamic(capacity=4, overwrite=False, growth_factor=2.0)
    buf = BufferMonitor(inner)

    print(f"初始快照: {buf.snapshot()}\n")

    for i in range(12):
        buf.write(i)

    print(f"写入12个后:")
    print(f"  utilization = {buf.utilization():.1%}")
    print(f"  snapshot: {buf.snapshot()}")

    batch = buf.read_batch(5)
    print(f"\nread_batch(5) -> {batch}")
    print(f"  utilization = {buf.utilization():.1%}")
    print(f"  snapshot: {buf.snapshot()}")

    print("\n--- 监控线程安全缓冲区 + 多线程压力测试 ---")
    ts_buf = ThreadSafeCircularBuffer(capacity=100, overwrite=True)
    monitored = BufferMonitor(ts_buf)

    def stress_writer(offset: int, count: int):
        for i in range(count):
            monitored.write(offset + i)

    def stress_reader(count: int):
        for _ in range(count):
            try:
                monitored.read()
            except IndexError:
                pass

    threads = [
        threading.Thread(target=stress_writer, args=(0, 500)),
        threading.Thread(target=stress_writer, args=(500, 500)),
        threading.Thread(target=stress_reader, args=(300,)),
        threading.Thread(target=stress_reader, args=(300,)),
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    snap = monitored.snapshot()
    print(f"  压测后快照:")
    print(f"    peak_used       = {snap['peak_used']}")
    print(f"    total_write_ops = {snap['total_write_ops']}")
    print(f"    total_read_ops  = {snap['total_read_ops']}")
    print(f"    total_write_items = {snap['total_write_items']}")
    print(f"    total_read_items  = {snap['total_read_items']}")
    print(f"    write_rate      = {snap['write_rate_per_sec']}/s")
    print(f"    read_rate       = {snap['read_rate_per_sec']}/s")
    print(f"    current_used    = {monitored.used}/{monitored.capacity}")
    print(f"    utilization     = {monitored.utilization():.1%}")


if __name__ == "__main__":
    _demo_counter()
    _demo_sacrifice()
    _demo_thread_safe()
    _demo_lock_free()
    _demo_dynamic()
    _demo_monitor()
