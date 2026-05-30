import time
from threading import Lock
from typing import Optional, Tuple


class CircularArray:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.array = [0.0] * capacity
        self._write_ptr = 0
        self._window_start_ptr = 0
        self._count = 0
        self._lock = Lock()

    def append(self, value: float) -> None:
        with self._lock:
            self.array[self._write_ptr] = value
            self._write_ptr = (self._write_ptr + 1) % self.capacity
            self._count = min(self._count + 1, self.capacity)

    def clear(self) -> None:
        with self._lock:
            self._write_ptr = 0
            self._window_start_ptr = 0
            self._count = 0

    def _advance_window(self, threshold: float) -> int:
        if self._count == 0:
            return 0

        effective_start = self._write_ptr - self._count
        if effective_start < 0:
            effective_start += self.capacity

        while self._count > 0:
            idx = effective_start % self.capacity
            if self.array[idx] <= threshold:
                effective_start = (effective_start + 1) % self.capacity
                self._count -= 1
            else:
                break

        self._window_start_ptr = effective_start
        return self._count

    def count_greater_than(self, threshold: float) -> int:
        with self._lock:
            return self._advance_window(threshold)

    def get_oldest(self) -> Optional[float]:
        with self._lock:
            if self._count == 0:
                return None
            effective_start = self._write_ptr - self._count
            if effective_start < 0:
                effective_start += self.capacity
            return self.array[effective_start % self.capacity]

    def append_and_count(self, value: float, threshold: float) -> Tuple[int, int, Optional[float]]:
        with self._lock:
            self.array[self._write_ptr] = value
            self._write_ptr = (self._write_ptr + 1) % self.capacity
            self._count = min(self._count + 1, self.capacity)

            count = self._advance_window(threshold)

            if self._count > 0:
                effective_start = self._write_ptr - self._count
                if effective_start < 0:
                    effective_start += self.capacity
                oldest = self.array[effective_start % self.capacity]
            else:
                oldest = None

            return count, self._count, oldest

    def __len__(self) -> int:
        return self._count

    @property
    def size(self) -> int:
        return self._count


class SlidingWindowCircularLimiter:
    ALGORITHM_NAME = 'sliding_window_circular'

    def __init__(self, default_qps: int = 10, window_size: int = 1):
        self.default_qps = default_qps
        self.window_size = window_size
        self._arrays: dict[str, CircularArray] = {}
        self._limits: dict[str, int] = {}
        self._default_capacity = default_qps * window_size * 2
        self._lock = Lock()

    def _get_or_create_array(self, api_path: str, limit: int) -> CircularArray:
        if api_path not in self._arrays:
            capacity = max(limit * self.window_size * 2, self._default_capacity, 100)
            self._arrays[api_path] = CircularArray(capacity)
        return self._arrays[api_path]

    def set_limit(self, api_path: str, qps: int) -> None:
        with self._lock:
            self._limits[api_path] = qps
            if api_path in self._arrays:
                new_capacity = max(qps * self.window_size * 2, 100)
                old_array = self._arrays[api_path]
                if new_capacity > old_array.capacity:
                    new_array = CircularArray(new_capacity)
                    with old_array._lock:
                        count = old_array._count
                        if count > 0:
                            effective_start = old_array._write_ptr - count
                            if effective_start < 0:
                                effective_start += old_array.capacity
                            for _ in range(count):
                                idx = effective_start % old_array.capacity
                                new_array.append(old_array.array[idx])
                                effective_start = (effective_start + 1) % old_array.capacity
                    self._arrays[api_path] = new_array

    def get_limit(self, api_path: str) -> int:
        return self._limits.get(api_path, self.default_qps)

    def get_all_limits(self) -> dict[str, int]:
        return dict(self._limits)

    def remove_limit(self, api_path: str) -> None:
        with self._lock:
            if api_path in self._limits:
                del self._limits[api_path]
                if api_path in self._arrays:
                    del self._arrays[api_path]

    def allow_request(self, api_path: str) -> tuple[bool, dict]:
        now = time.time()
        limit = self.get_limit(api_path)
        window_start = now - self.window_size

        with self._lock:
            array = self._get_or_create_array(api_path, limit)

        current_count = array.count_greater_than(window_start)
        qps = current_count / self.window_size if self.window_size > 0 else 0

        if current_count >= limit:
            oldest = array.get_oldest()
            retry_after = oldest + self.window_size - now if oldest else self.window_size
            return False, {
                'allowed': False,
                'current_qps': qps,
                'limit_qps': limit,
                'remaining': 0,
                'retry_after': max(0, retry_after),
                'algorithm': self.ALGORITHM_NAME,
                'memory_fixed': True,
                'array_capacity': array.capacity,
                'array_size': len(array)
            }

        array.append(now)
        remaining = limit - current_count - 1

        return True, {
            'allowed': True,
            'current_qps': qps,
            'limit_qps': limit,
            'remaining': remaining,
            'retry_after': 0,
            'algorithm': self.ALGORITHM_NAME,
            'memory_fixed': True,
            'array_capacity': array.capacity,
            'array_size': len(array)
        }

    def get_stats(self, api_path: str) -> dict:
        now = time.time()
        window_start = now - self.window_size
        limit = self.get_limit(api_path)

        with self._lock:
            if api_path not in self._arrays:
                array = self._get_or_create_array(api_path, limit)
            else:
                array = self._arrays[api_path]

        count = array.count_greater_than(window_start)
        remaining = max(0, limit - count)

        return {
            'api_path': api_path,
            'current_qps': count / self.window_size if self.window_size > 0 else 0,
            'limit_qps': limit,
            'remaining': remaining,
            'window_size': self.window_size,
            'algorithm': self.ALGORITHM_NAME,
            'memory_fixed': True,
            'array_capacity': array.capacity,
            'array_size': len(array)
        }

    def get_memory_info(self) -> dict:
        total_capacity = 0
        total_size = 0
        for key, array in self._arrays.items():
            total_capacity += array.capacity
            total_size += array.size
        return {
            'total_arrays': len(self._arrays),
            'total_capacity': total_capacity,
            'total_size': total_size,
            'memory_usage_bytes': total_capacity * 8,
            'memory_fixed': True
        }

    def reset(self, api_path: str = None) -> None:
        with self._lock:
            if api_path:
                if api_path in self._arrays:
                    self._arrays[api_path].clear()
            else:
                for key in self._arrays:
                    self._arrays[key].clear()
