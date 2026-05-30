import time
from collections import defaultdict
from threading import Lock


class TokenBucketLimiter:
    def __init__(self, default_qps: int = 10, burst: int = None):
        self.default_qps = default_qps
        self.default_burst = burst or default_qps
        self._buckets: dict[str, dict] = defaultdict(self._create_bucket)
        self._limits: dict[str, dict] = {}
        self._lock = Lock()

    def _create_bucket(self) -> dict:
        return {
            'tokens': self.default_burst,
            'last_refill': time.time(),
            'rate': self.default_qps,
            'capacity': self.default_burst
        }

    def set_limit(self, api_path: str, qps: int, burst: int = None) -> None:
        with self._lock:
            burst = burst or qps
            self._limits[api_path] = {'qps': qps, 'burst': burst}

            if api_path in self._buckets:
                bucket = self._buckets[api_path]
                bucket['rate'] = qps
                bucket['capacity'] = burst
                if bucket['tokens'] > burst:
                    bucket['tokens'] = burst

    def get_limit(self, api_path: str) -> dict:
        if api_path in self._limits:
            return self._limits[api_path]
        return {'qps': self.default_qps, 'burst': self.default_burst}

    def get_all_limits(self) -> dict[str, dict]:
        return dict(self._limits)

    def remove_limit(self, api_path: str) -> None:
        with self._lock:
            if api_path in self._limits:
                del self._limits[api_path]
                if api_path in self._buckets:
                    del self._buckets[api_path]

    def _refill_tokens(self, bucket: dict) -> None:
        now = time.time()
        elapsed = now - bucket['last_refill']
        tokens_to_add = elapsed * bucket['rate']
        bucket['tokens'] = min(bucket['capacity'], bucket['tokens'] + tokens_to_add)
        bucket['last_refill'] = now

    def allow_request(self, api_path: str) -> tuple[bool, dict]:
        now = time.time()
        limit = self.get_limit(api_path)

        with self._lock:
            bucket = self._buckets[api_path]
            self._refill_tokens(bucket)

            qps = bucket['rate']

            if bucket['tokens'] < 1:
                tokens_needed = 1 - bucket['tokens']
                retry_after = tokens_needed / bucket['rate'] if bucket['rate'] > 0 else float('inf')
                return False, {
                    'allowed': False,
                    'current_qps': qps,
                    'limit_qps': limit['qps'],
                    'remaining': int(bucket['tokens']),
                    'retry_after': max(0, retry_after),
                    'algorithm': 'token_bucket'
                }

            bucket['tokens'] -= 1

            return True, {
                'allowed': True,
                'current_qps': qps,
                'limit_qps': limit['qps'],
                'remaining': int(bucket['tokens']),
                'retry_after': 0,
                'algorithm': 'token_bucket'
            }

    def get_stats(self, api_path: str) -> dict:
        limit = self.get_limit(api_path)

        with self._lock:
            if api_path not in self._buckets:
                return {
                    'api_path': api_path,
                    'tokens': limit['burst'],
                    'rate': limit['qps'],
                    'capacity': limit['burst'],
                    'limit_qps': limit['qps'],
                    'algorithm': 'token_bucket'
                }

            bucket = self._buckets[api_path]
            self._refill_tokens(bucket)

            return {
                'api_path': api_path,
                'tokens': bucket['tokens'],
                'rate': bucket['rate'],
                'capacity': bucket['capacity'],
                'limit_qps': limit['qps'],
                'algorithm': 'token_bucket'
            }

    def reset(self, api_path: str = None) -> None:
        with self._lock:
            if api_path:
                if api_path in self._buckets:
                    limit = self.get_limit(api_path)
                    bucket = self._buckets[api_path]
                    bucket['tokens'] = limit['burst']
                    bucket['last_refill'] = time.time()
            else:
                for key in self._buckets:
                    limit = self.get_limit(key)
                    bucket = self._buckets[key]
                    bucket['tokens'] = limit['burst']
                    bucket['last_refill'] = time.time()
