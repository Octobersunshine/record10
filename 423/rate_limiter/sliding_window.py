import time
from collections import defaultdict, deque
from threading import Lock


class SlidingWindowLimiter:
    def __init__(self, default_qps: int = 10, window_size: int = 1):
        self.default_qps = default_qps
        self.window_size = window_size
        self._timestamps: dict[str, deque] = defaultdict(deque)
        self._limits: dict[str, int] = {}
        self._lock = Lock()

    def set_limit(self, api_path: str, qps: int) -> None:
        with self._lock:
            self._limits[api_path] = qps

    def get_limit(self, api_path: str) -> int:
        return self._limits.get(api_path, self.default_qps)

    def get_all_limits(self) -> dict[str, int]:
        return dict(self._limits)

    def remove_limit(self, api_path: str) -> None:
        with self._lock:
            if api_path in self._limits:
                del self._limits[api_path]
                if api_path in self._timestamps:
                    del self._timestamps[api_path]

    def allow_request(self, api_path: str) -> tuple[bool, dict]:
        now = time.time()
        limit = self.get_limit(api_path)
        window_start = now - self.window_size

        with self._lock:
            timestamps = self._timestamps[api_path]

            while timestamps and timestamps[0] <= window_start:
                timestamps.popleft()

            current_count = len(timestamps)
            qps = current_count / self.window_size if self.window_size > 0 else 0

            if current_count >= limit:
                retry_after = timestamps[0] + self.window_size - now if timestamps else self.window_size
                return False, {
                    'allowed': False,
                    'current_qps': qps,
                    'limit_qps': limit,
                    'remaining': 0,
                    'retry_after': max(0, retry_after),
                    'algorithm': 'sliding_window'
                }

            timestamps.append(now)
            remaining = limit - len(timestamps)

            return True, {
                'allowed': True,
                'current_qps': qps,
                'limit_qps': limit,
                'remaining': remaining,
                'retry_after': 0,
                'algorithm': 'sliding_window'
            }

    def get_stats(self, api_path: str) -> dict:
        now = time.time()
        window_start = now - self.window_size

        with self._lock:
            timestamps = self._timestamps.get(api_path, deque())
            count = sum(1 for t in timestamps if t > window_start)

        return {
            'api_path': api_path,
            'current_qps': count / self.window_size if self.window_size > 0 else 0,
            'limit_qps': self.get_limit(api_path),
            'window_size': self.window_size,
            'algorithm': 'sliding_window'
        }

    def reset(self, api_path: str = None) -> None:
        with self._lock:
            if api_path:
                if api_path in self._timestamps:
                    self._timestamps[api_path].clear()
            else:
                for key in self._timestamps:
                    self._timestamps[key].clear()
