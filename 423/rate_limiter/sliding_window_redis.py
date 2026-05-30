import time
from threading import Lock
from typing import Optional

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False


class SlidingWindowRedisLimiter:
    ALGORITHM_NAME = 'sliding_window_redis'

    def __init__(self, default_qps: int = 10, window_size: int = 1,
                 redis_host: str = 'localhost', redis_port: int = 6379,
                 redis_db: int = 0, redis_password: Optional[str] = None,
                 redis_prefix: str = 'rate_limit:', redis_timeout: float = 1.0):
        if not REDIS_AVAILABLE:
            raise ImportError(
                "Redis library not installed. "
                "Install with: pip install redis"
            )

        self.default_qps = default_qps
        self.window_size = window_size
        self.redis_prefix = redis_prefix
        self._limits: dict[str, int] = {}
        self._limits_lock = Lock()

        self._redis_client = redis.Redis(
            host=redis_host,
            port=redis_port,
            db=redis_db,
            password=redis_password,
            decode_responses=True,
            socket_timeout=redis_timeout,
            socket_connect_timeout=redis_timeout
        )

        self._test_connection()

    def _test_connection(self) -> None:
        try:
            self._redis_client.ping()
        except redis.ConnectionError as e:
            raise ConnectionError(
                f"Failed to connect to Redis: {e}. "
                f"Please ensure Redis is running at the specified address."
            )

    def _get_key(self, api_path: str) -> str:
        return f"{self.redis_prefix}{api_path}"

    def set_limit(self, api_path: str, qps: int) -> None:
        with self._limits_lock:
            self._limits[api_path] = qps

    def get_limit(self, api_path: str) -> int:
        with self._limits_lock:
            return self._limits.get(api_path, self.default_qps)

    def get_all_limits(self) -> dict[str, int]:
        with self._limits_lock:
            return dict(self._limits)

    def remove_limit(self, api_path: str) -> None:
        with self._limits_lock:
            if api_path in self._limits:
                del self._limits[api_path]
                key = self._get_key(api_path)
                self._redis_client.delete(key)

    def allow_request(self, api_path: str) -> tuple[bool, dict]:
        now = time.time()
        limit = self.get_limit(api_path)
        window_start = now - self.window_size
        key = self._get_key(api_path)

        member = f"{now}-{id(now)}"

        pipe = self._redis_client.pipeline()
        pipe.zadd(key, {member: now})
        pipe.zremrangebyscore(key, '-inf', window_start)
        pipe.zcard(key)
        pipe.expire(key, int(self.window_size * 2) + 1)

        try:
            results = pipe.execute()
            current_count = results[2]
        except Exception as e:
            return True, {
                'allowed': True,
                'current_qps': 0,
                'limit_qps': limit,
                'remaining': limit,
                'retry_after': 0,
                'algorithm': self.ALGORITHM_NAME,
                'distributed': True,
                'error': str(e)
            }

        qps = current_count / self.window_size if self.window_size > 0 else 0

        if current_count > limit:
            try:
                oldest = self._redis_client.zrange(key, 0, 0, withscores=True)
                if oldest:
                    oldest_score = oldest[0][1]
                    retry_after = oldest_score + self.window_size - now
                else:
                    retry_after = self.window_size
            except Exception:
                retry_after = self.window_size

            self._redis_client.zrem(key, member)

            return False, {
                'allowed': False,
                'current_qps': qps,
                'limit_qps': limit,
                'remaining': 0,
                'retry_after': max(0, retry_after),
                'algorithm': self.ALGORITHM_NAME,
                'distributed': True
            }

        remaining = limit - current_count

        return True, {
            'allowed': True,
            'current_qps': qps,
            'limit_qps': limit,
            'remaining': remaining,
            'retry_after': 0,
            'algorithm': self.ALGORITHM_NAME,
            'distributed': True,
            'window_count': current_count
        }

    def get_stats(self, api_path: str) -> dict:
        now = time.time()
        window_start = now - self.window_size
        limit = self.get_limit(api_path)
        key = self._get_key(api_path)

        try:
            count = self._redis_client.zcount(key, window_start, '+inf')
            zcard = self._redis_client.zcard(key)
        except Exception as e:
            return {
                'api_path': api_path,
                'current_qps': 0,
                'limit_qps': limit,
                'window_size': self.window_size,
                'algorithm': self.ALGORITHM_NAME,
                'distributed': True,
                'error': str(e)
            }

        return {
            'api_path': api_path,
            'current_qps': count / self.window_size if self.window_size > 0 else 0,
            'limit_qps': limit,
            'window_size': self.window_size,
            'algorithm': self.ALGORITHM_NAME,
            'distributed': True,
            'total_stored': zcard,
            'window_count': count
        }

    def get_memory_info(self) -> dict:
        try:
            info = self._redis_client.info('memory')
            keys = self._redis_client.keys(f"{self.redis_prefix}*")
            return {
                'distributed': True,
                'total_keys': len(keys),
                'redis_used_memory_human': info.get('used_memory_human', 'N/A'),
                'redis_used_memory_peak_human': info.get('used_memory_peak_human', 'N/A'),
                'redis_keys': keys
            }
        except Exception as e:
            return {
                'distributed': True,
                'error': str(e)
            }

    def reset(self, api_path: str = None) -> None:
        if api_path:
            key = self._get_key(api_path)
            self._redis_client.delete(key)
        else:
            keys = self._redis_client.keys(f"{self.redis_prefix}*")
            if keys:
                self._redis_client.delete(*keys)

    def close(self) -> None:
        try:
            self._redis_client.close()
        except Exception:
            pass

    def __del__(self):
        self.close()
