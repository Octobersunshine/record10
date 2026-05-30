import time
import uuid
from threading import Lock
from typing import Optional

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

LUA_SLIDING_WINDOW_SCRIPT = """
local key = KEYS[1]
local now = tonumber(ARGV[1])
local window_start = tonumber(ARGV[2])
local limit = tonumber(ARGV[3])
local ttl = tonumber(ARGV[4])
local member = ARGV[5]

redis.call('ZADD', key, now, member)
redis.call('ZREMRANGEBYSCORE', key, '-inf', window_start)
local count = redis.call('ZCARD', key)
redis.call('EXPIRE', key, ttl)

if count > limit then
    redis.call('ZREM', key, member)
    local oldest = redis.call('ZRANGE', key, 0, 0, 'WITHSCORES')
    local retry_after = 0
    if #oldest >= 2 then
        retry_after = tonumber(oldest[2]) + (now - window_start) - now
        if retry_after < 0 then retry_after = 0 end
    else
        retry_after = now - window_start
    end
    return {0, count, retry_after}
else
    return {1, count, 0}
end
"""

LUA_TOKEN_BUCKET_SCRIPT = """
local key = KEYS[1]
local now = tonumber(ARGV[1])
local rate = tonumber(ARGV[2])
local capacity = tonumber(ARGV[3])
local requested = tonumber(ARGV[4])

local bucket = redis.call('HMGET', key, 'tokens', 'last_refill')
local tokens = tonumber(bucket[1])
local last_refill = tonumber(bucket[2])

if tokens == nil then
    tokens = capacity
    last_refill = now
end

local elapsed = now - last_refill
tokens = math.min(capacity, tokens + elapsed * rate)
last_refill = now

if tokens < requested then
    local tokens_needed = requested - tokens
    local retry_after = 0
    if rate > 0 then
        retry_after = tokens_needed / rate
    else
        retry_after = 999999
    end
    redis.call('HMSET', key, 'tokens', tokens, 'last_refill', last_refill)
    redis.call('EXPIRE', key, math.ceil(capacity / rate) + 10)
    return {0, math.floor(tokens), math.ceil(retry_after)}
else
    tokens = tokens - requested
    redis.call('HMSET', key, 'tokens', tokens, 'last_refill', last_refill)
    redis.call('EXPIRE', key, math.ceil(capacity / rate) + 10)
    return {1, math.floor(tokens), 0}
end
"""


class DistributedRateLimiter:
    ALGORITHM_NAME = 'distributed_redis'

    DIMENSION_API = 'api'
    DIMENSION_USER = 'user'
    DIMENSION_IP = 'ip'
    DIMENSION_API_USER = 'api:user'
    DIMENSION_API_IP = 'api:ip'
    DIMENSION_API_USER_IP = 'api:user:ip'

    ALL_DIMENSIONS = [
        DIMENSION_API, DIMENSION_USER, DIMENSION_IP,
        DIMENSION_API_USER, DIMENSION_API_IP,
        DIMENSION_API_USER_IP
    ]

    def __init__(self, default_qps: int = 10, window_size: int = 1,
                 redis_host: str = 'localhost', redis_port: int = 6379,
                 redis_db: int = 0, redis_password: Optional[str] = None,
                 redis_prefix: str = 'rl:', redis_timeout: float = 1.0):
        if not REDIS_AVAILABLE:
            raise ImportError("Redis library not installed. pip install redis")

        self.default_qps = default_qps
        self.window_size = window_size
        self.redis_prefix = redis_prefix
        self._dimension_rules: dict[str, dict] = {}
        self._rules_lock = Lock()

        self._redis = redis.Redis(
            host=redis_host, port=redis_port, db=redis_db,
            password=redis_password, decode_responses=False,
            socket_timeout=redis_timeout,
            socket_connect_timeout=redis_timeout
        )

        self._sw_script = self._redis.register_script(LUA_SLIDING_WINDOW_SCRIPT)
        self._tb_script = self._redis.register_script(LUA_TOKEN_BUCKET_SCRIPT)

        self._test_connection()

    def _test_connection(self) -> None:
        try:
            self._redis.ping()
        except redis.ConnectionError as e:
            raise ConnectionError(f"Redis connection failed: {e}")

    def _make_key(self, *parts) -> str:
        return f"{self.redis_prefix}{'|'.join(str(p) for p in parts)}"

    def set_dimension_rule(self, dimension: str, qps: int,
                           api_pattern: str = None,
                           burst: int = None,
                           strategy: str = 'sliding_window') -> None:
        if dimension not in self.ALL_DIMENSIONS:
            raise ValueError(f"Invalid dimension: {dimension}")

        with self._rules_lock:
            rule_key = f"{dimension}:{api_pattern or '*'}"
            self._dimension_rules[rule_key] = {
                'dimension': dimension,
                'qps': qps,
                'burst': burst or qps,
                'api_pattern': api_pattern,
                'strategy': strategy
            }

    def remove_dimension_rule(self, dimension: str,
                              api_pattern: str = None) -> None:
        with self._rules_lock:
            rule_key = f"{dimension}:{api_pattern or '*'}"
            if rule_key in self._dimension_rules:
                del self._dimension_rules[rule_key]

    def get_all_rules(self) -> dict:
        return dict(self._dimension_rules)

    def _get_rules_for_dimension(self, dimension: str) -> list[dict]:
        rules = []
        with self._rules_lock:
            for key, rule in self._dimension_rules.items():
                if rule['dimension'] == dimension:
                    rules.append(rule)
        return rules

    def _match_api_pattern(self, pattern: Optional[str], api_path: str) -> bool:
        if pattern is None or pattern == '*':
            return True
        if pattern == api_path:
            return True
        if pattern.endswith('*') and api_path.startswith(pattern[:-1]):
            return True
        return False

    def _get_qps_for(self, dimension: str, api_path: str) -> dict:
        rules = self._get_rules_for_dimension(dimension)
        specific_match = None
        wildcard_match = None

        for rule in rules:
            if rule['api_pattern'] == api_path:
                specific_match = rule
            elif rule['api_pattern'] is None or rule['api_pattern'] == '*':
                wildcard_match = rule

        match = specific_match or wildcard_match
        if match:
            return {
                'qps': match['qps'],
                'burst': match.get('burst', match['qps']),
                'strategy': match.get('strategy', 'sliding_window')
            }
        return {
            'qps': self.default_qps,
            'burst': self.default_qps,
            'strategy': 'sliding_window'
        }

    def _check_sliding_window(self, key: str, now: float,
                              limit: int) -> tuple[bool, int, float]:
        window_start = now - self.window_size
        ttl = int(self.window_size * 2) + 1
        member = f"{now}-{uuid.uuid4().hex[:8]}"

        try:
            result = self._sw_script(
                keys=[key],
                args=[now, window_start, limit, ttl, member]
            )
            allowed = bool(result[0])
            count = int(result[1])
            retry_after = float(result[2])
            return allowed, count, retry_after
        except Exception as e:
            return True, 0, 0

    def _check_token_bucket(self, key: str, now: float,
                            rate: int, burst: int) -> tuple[bool, int, float]:
        try:
            result = self._tb_script(
                keys=[key],
                args=[now, rate, burst, 1]
            )
            allowed = bool(result[0])
            remaining = int(result[1])
            retry_after = float(result[2])
            return allowed, remaining, retry_after
        except Exception:
            return True, burst, 0

    def allow_request(self, api_path: str = '', user_id: str = '',
                      ip: str = '') -> tuple[bool, dict]:
        now = time.time()
        all_results = {}
        overall_allowed = True
        min_retry = float('inf')
        blocked_by = None

        active_dimensions = set()
        for rule in self._dimension_rules.values():
            active_dimensions.add(rule['dimension'])

        if not active_dimensions:
            active_dimensions = {self.DIMENSION_API}

        for dimension in active_dimensions:
            cfg = self._get_qps_for(dimension, api_path)

            key_parts = [dimension]
            if dimension in (self.DIMENSION_API, self.DIMENSION_API_USER,
                             self.DIMENSION_API_IP, self.DIMENSION_API_USER_IP):
                key_parts.append(f"api:{api_path}")
            if dimension in (self.DIMENSION_USER, self.DIMENSION_API_USER,
                             self.DIMENSION_API_USER_IP):
                key_parts.append(f"u:{user_id}")
            if dimension in (self.DIMENSION_IP, self.DIMENSION_API_IP,
                             self.DIMENSION_API_USER_IP):
                key_parts.append(f"ip:{ip}")

            redis_key = self._make_key(*key_parts)

            if cfg['strategy'] == 'token_bucket':
                allowed, remaining, retry = self._check_token_bucket(
                    redis_key, now, cfg['qps'], cfg['burst']
                )
            else:
                allowed, count, retry = self._check_sliding_window(
                    redis_key, now, cfg['qps']
                )
                remaining = cfg['qps'] - count if allowed else 0

            all_results[dimension] = {
                'allowed': allowed,
                'key': ':'.join(key_parts),
                'limit_qps': cfg['qps'],
                'remaining': remaining,
                'retry_after': retry,
                'strategy': cfg['strategy']
            }

            if not allowed:
                overall_allowed = False
                if retry < min_retry:
                    min_retry = retry
                    blocked_by = dimension

        result = {
            'allowed': overall_allowed,
            'algorithm': self.ALGORITHM_NAME,
            'distributed': True,
            'dimensions_checked': len(all_results),
            'dimensions': all_results
        }

        if not overall_allowed:
            result['retry_after'] = max(0, min_retry)
            result['blocked_by'] = blocked_by
            result['limit_qps'] = all_results[blocked_by]['limit_qps']
            result['remaining'] = 0
        else:
            result['retry_after'] = 0
            min_rem = min((d['remaining'] for d in all_results.values()), default=0)
            result['remaining'] = min_rem
            if all_results:
                first = next(iter(all_results.values()))
                result['limit_qps'] = first['limit_qps']

        return overall_allowed, result

    def get_stats(self, api_path: str = '', user_id: str = '',
                  ip: str = '') -> dict:
        return {
            'algorithm': self.ALGORITHM_NAME,
            'distributed': True,
            'active_rules': len(self._dimension_rules),
            'dimensions': list(set(
                r['dimension'] for r in self._dimension_rules.values()
            ))
        }

    def get_memory_info(self) -> dict:
        try:
            info = self._redis.info('memory')
            return {
                'distributed': True,
                'redis_used_memory_human': info.get('used_memory_human', 'N/A'),
                'active_rules': len(self._dimension_rules),
                'lua_scripts': 'registered'
            }
        except Exception as e:
            return {'distributed': True, 'error': str(e)}

    def reset(self, dimension: str = None) -> None:
        pattern = f"{self.redis_prefix}{dimension}:*" if dimension else f"{self.redis_prefix}*"
        try:
            keys = self._redis.keys(pattern)
            if keys:
                self._redis.delete(*keys)
        except Exception:
            pass

    def close(self) -> None:
        try:
            self._redis.close()
        except Exception:
            pass

    def __del__(self):
        self.close()
