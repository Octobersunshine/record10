from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Dict, Tuple

import redis


LUA_TOKEN_BUCKET_ACQUIRE = """
redis.replicate_commands()

local key        = KEYS[1]
local stats_key  = KEYS[2]
local rate       = tonumber(ARGV[1])
local capacity   = tonumber(ARGV[2])
local use_server_time = tonumber(ARGV[3])
local requested  = tonumber(ARGV[4])
local client_now = tonumber(ARGV[5])
local enable_stats = tonumber(ARGV[6])

local now
if use_server_time == 1 then
    local t = redis.call('TIME')
    now = tonumber(t[1]) + tonumber(t[2]) / 1000000
else
    now = client_now
end

local data = redis.call('HMGET', key, 'tokens', 'last_time')
local tokens    = data[1]
local last_time = data[2]

if tokens == false then
    tokens    = capacity
    last_time = now
else
    tokens    = tonumber(tokens)
    last_time = tonumber(last_time)
end

local elapsed = math.max(0, now - last_time)
local refill  = elapsed * rate
tokens        = math.min(capacity, tokens + refill)

local allowed   = 0
local wait_time = 0
local remaining = 0

if tokens >= requested then
    tokens    = tokens - requested
    allowed   = 1
    remaining = tokens
    wait_time = 0
else
    allowed   = 0
    remaining = tokens
    wait_time = (requested - tokens) / rate
end

redis.call('HSET', key, 'tokens', tokens, 'last_time', now)

local ttl = math.ceil(capacity / rate * 2)
if ttl > 0 then
    redis.call('EXPIRE', key, ttl)
end

if enable_stats == 1 then
    if allowed == 1 then
        redis.call('HINCRBY', stats_key, 'allowed', 1)
    else
        redis.call('HINCRBY', stats_key, 'blocked', 1)
    end
    redis.call('EXPIRE', stats_key, 86400)
end

return {allowed, remaining, wait_time, now}
"""


LUA_TOKEN_BUCKET_STATUS = """
redis.replicate_commands()

local key        = KEYS[1]
local rate       = tonumber(ARGV[1])
local capacity   = tonumber(ARGV[2])
local use_server_time = tonumber(ARGV[3])
local client_now = tonumber(ARGV[4])

local now
if use_server_time == 1 then
    local t = redis.call('TIME')
    now = tonumber(t[1]) + tonumber(t[2]) / 1000000
else
    now = client_now
end

local data = redis.call('HMGET', key, 'tokens', 'last_time')
local tokens    = data[1]
local last_time = data[2]

if tokens == false then
    return {capacity, '', capacity, rate, capacity, now}
else
    tokens    = tonumber(tokens)
    last_time = tonumber(last_time)
end

local elapsed = math.max(0, now - last_time)
local refill  = elapsed * rate
tokens        = math.min(capacity, tokens + refill)

return {tokens, last_time, tokens, rate, capacity, now}
"""


LUA_TOKEN_BUCKET_STATS = """
local stats_key = KEYS[1]

local data = redis.call('HMGET', stats_key, 'allowed', 'blocked')
local allowed = data[1]
local blocked = data[2]

allowed = (allowed ~= false) and tonumber(allowed) or 0
blocked = (blocked ~= false) and tonumber(blocked) or 0

local total = allowed + blocked
local hit_ratio = 0
if total > 0 then
    hit_ratio = blocked / total
end

return {allowed, blocked, total, hit_ratio}
"""


@dataclass
class RateLimitResult:
    allowed: bool
    remaining_tokens: float
    wait_time: float
    effective_rate: float = 0
    effective_capacity: int = 0


@dataclass
class LoadMetrics:
    cpu_usage: Optional[float] = None
    memory_usage: Optional[float] = None
    queue_depth: Optional[int] = None
    error_rate: Optional[float] = None
    response_time_p95: Optional[float] = None


@dataclass
class RateLimitStats:
    allowed: int = 0
    blocked: int = 0
    total: int = 0
    hit_ratio: float = 0.0
    scope: str = "global"
    key: Optional[str] = None


class LoadAdaptor(ABC):

    @abstractmethod
    def get_current_load(self) -> LoadMetrics:
        pass

    @abstractmethod
    def adjust_rate(
        self,
        current_rate: float,
        current_capacity: int,
        metrics: LoadMetrics,
    ) -> Tuple[float, int]:
        pass


class DefaultLoadAdaptor(LoadAdaptor):

    def __init__(
        self,
        min_rate: float = 1.0,
        max_rate: float = 1000.0,
        high_load_threshold: float = 80.0,
        low_load_threshold: float = 30.0,
        adjustment_factor: float = 0.2,
    ):
        self._min_rate = min_rate
        self._max_rate = max_rate
        self._high_load_threshold = high_load_threshold
        self._low_load_threshold = low_load_threshold
        self._adjustment_factor = adjustment_factor

    def get_current_load(self) -> LoadMetrics:
        metrics = LoadMetrics()
        try:
            import psutil
            metrics.cpu_usage = psutil.cpu_percent(interval=None)
            metrics.memory_usage = psutil.virtual_memory().percent
        except ImportError:
            pass
        return metrics

    def adjust_rate(
        self,
        current_rate: float,
        current_capacity: int,
        metrics: LoadMetrics,
    ) -> Tuple[float, int]:
        load_indicators = []
        if metrics.cpu_usage is not None:
            load_indicators.append(metrics.cpu_usage)
        if metrics.memory_usage is not None:
            load_indicators.append(metrics.memory_usage)
        if metrics.error_rate is not None:
            load_indicators.append(metrics.error_rate * 100)

        if not load_indicators:
            return current_rate, current_capacity

        avg_load = sum(load_indicators) / len(load_indicators)
        new_rate = current_rate
        new_capacity = current_capacity

        if avg_load > self._high_load_threshold:
            new_rate = max(
                self._min_rate,
                current_rate * (1 - self._adjustment_factor),
            )
        elif avg_load < self._low_load_threshold:
            new_rate = min(
                self._max_rate,
                current_rate * (1 + self._adjustment_factor),
            )

        if new_rate != current_rate:
            new_capacity = max(1, int(new_rate))

        return round(new_rate, 2), new_capacity


@dataclass
class RateLimitConfig:
    rate: float
    capacity: int
    dynamic_adjustment: bool = False
    adaptor: Optional[LoadAdaptor] = None
    min_rate: Optional[float] = None
    max_rate: Optional[float] = None
    last_adjusted: float = field(default_factory=time.time)
    adjustment_interval: float = 60.0


class TokenBucketRateLimiter:

    def __init__(
        self,
        redis_client: redis.Redis,
        rate: float,
        capacity: int,
        key_prefix: str = "ratelimit:token_bucket",
        use_server_time: bool = True,
        enable_stats: bool = True,
        dynamic_adjustment: bool = False,
        load_adaptor: Optional[LoadAdaptor] = None,
        min_rate: Optional[float] = None,
        max_rate: Optional[float] = None,
    ):
        if rate <= 0:
            raise ValueError("rate must be positive")
        if capacity <= 0:
            raise ValueError("capacity must be positive")

        self._redis = redis_client
        self._key_prefix = key_prefix
        self._use_server_time = use_server_time
        self._enable_stats = enable_stats
        self._acquire_script = self._redis.register_script(LUA_TOKEN_BUCKET_ACQUIRE)
        self._status_script = self._redis.register_script(LUA_TOKEN_BUCKET_STATUS)
        self._stats_script = self._redis.register_script(LUA_TOKEN_BUCKET_STATS)

        self._global_config = RateLimitConfig(
            rate=rate,
            capacity=capacity,
            dynamic_adjustment=dynamic_adjustment,
            adaptor=load_adaptor or DefaultLoadAdaptor(),
            min_rate=min_rate or rate * 0.1,
            max_rate=max_rate or rate * 10,
        )

        self._key_configs: Dict[str, RateLimitConfig] = {}

    def _build_key(self, key: str) -> str:
        return f"{self._key_prefix}:{key}"

    def _build_stats_key(self, key: str) -> str:
        return f"{self._key_prefix}:stats:{key}"

    def _build_global_stats_key(self) -> str:
        return f"{self._key_prefix}:stats:__global__"

    def _get_effective_config(self, key: Optional[str]) -> RateLimitConfig:
        if key is not None and key in self._key_configs:
            return self._key_configs[key]
        return self._global_config

    def _maybe_adjust_rate(self, config: RateLimitConfig) -> None:
        if not config.dynamic_adjustment or config.adaptor is None:
            return

        now = time.time()
        if now - config.last_adjusted < config.adjustment_interval:
            return

        metrics = config.adaptor.get_current_load()
        new_rate, new_capacity = config.adaptor.adjust_rate(
            config.rate,
            config.capacity,
            metrics,
        )

        if config.min_rate is not None:
            new_rate = max(config.min_rate, new_rate)
        if config.max_rate is not None:
            new_rate = min(config.max_rate, new_rate)

        config.rate = new_rate
        config.capacity = new_capacity
        config.last_adjusted = now

    def set_key_config(
        self,
        key: str,
        rate: Optional[float] = None,
        capacity: Optional[int] = None,
        dynamic_adjustment: Optional[bool] = None,
        load_adaptor: Optional[LoadAdaptor] = None,
        min_rate: Optional[float] = None,
        max_rate: Optional[float] = None,
    ) -> None:
        current = self._get_effective_config(key)
        if key not in self._key_configs:
            self._key_configs[key] = RateLimitConfig(
                rate=current.rate,
                capacity=current.capacity,
                dynamic_adjustment=current.dynamic_adjustment,
                adaptor=current.adaptor,
                min_rate=current.min_rate,
                max_rate=current.max_rate,
            )

        cfg = self._key_configs[key]
        if rate is not None:
            cfg.rate = rate
        if capacity is not None:
            cfg.capacity = capacity
        if dynamic_adjustment is not None:
            cfg.dynamic_adjustment = dynamic_adjustment
        if load_adaptor is not None:
            cfg.adaptor = load_adaptor
        if min_rate is not None:
            cfg.min_rate = min_rate
        if max_rate is not None:
            cfg.max_rate = max_rate

    def remove_key_config(self, key: str) -> bool:
        if key in self._key_configs:
            del self._key_configs[key]
            return True
        return False

    def acquire(
        self,
        key: str,
        tokens: int = 1,
        rate: Optional[float] = None,
        capacity: Optional[int] = None,
    ) -> RateLimitResult:
        config = self._get_effective_config(key)
        self._maybe_adjust_rate(config)

        effective_rate = rate if rate is not None else config.rate
        effective_capacity = capacity if capacity is not None else config.capacity

        if tokens > effective_capacity:
            return RateLimitResult(
                allowed=False,
                remaining_tokens=0,
                wait_time=(tokens - effective_capacity) / effective_rate,
                effective_rate=effective_rate,
                effective_capacity=effective_capacity,
            )

        now = time.time()
        redis_key = self._build_key(key)
        stats_key = self._build_stats_key(key)
        global_stats_key = self._build_global_stats_key()
        use_server_time = 1 if self._use_server_time else 0
        enable_stats = 1 if self._enable_stats else 0

        result = self._acquire_script(
            keys=[redis_key, stats_key],
            args=[
                effective_rate,
                effective_capacity,
                use_server_time,
                tokens,
                now,
                enable_stats,
            ],
        )

        allowed = bool(result[0])
        remaining = float(result[1])
        wait_time = float(result[2])

        if self._enable_stats:
            pipeline = self._redis.pipeline()
            if allowed:
                pipeline.hincrby(global_stats_key, "allowed", 1)
            else:
                pipeline.hincrby(global_stats_key, "blocked", 1)
            pipeline.expire(global_stats_key, 86400)
            pipeline.execute()

        return RateLimitResult(
            allowed=allowed,
            remaining_tokens=remaining,
            wait_time=wait_time,
            effective_rate=effective_rate,
            effective_capacity=effective_capacity,
        )

    def acquire_by_user_id(
        self,
        user_id: str,
        tokens: int = 1,
        rate: Optional[float] = None,
        capacity: Optional[int] = None,
    ) -> RateLimitResult:
        return self.acquire(f"user:{user_id}", tokens, rate, capacity)

    def acquire_by_ip(
        self,
        ip: str,
        tokens: int = 1,
        rate: Optional[float] = None,
        capacity: Optional[int] = None,
    ) -> RateLimitResult:
        return self.acquire(f"ip:{ip}", tokens, rate, capacity)

    def acquire_by_api_endpoint(
        self,
        endpoint: str,
        identifier: str,
        tokens: int = 1,
        rate: Optional[float] = None,
        capacity: Optional[int] = None,
    ) -> RateLimitResult:
        return self.acquire(f"api:{endpoint}:{identifier}", tokens, rate, capacity)

    def try_acquire(
        self,
        key: str,
        tokens: int = 1,
        rate: Optional[float] = None,
        capacity: Optional[int] = None,
    ) -> RateLimitResult:
        return self.acquire(key, tokens, rate, capacity)

    def get_status(
        self,
        key: str,
        rate: Optional[float] = None,
        capacity: Optional[int] = None,
    ) -> dict:
        config = self._get_effective_config(key)
        effective_rate = rate if rate is not None else config.rate
        effective_capacity = capacity if capacity is not None else config.capacity

        now = time.time()
        redis_key = self._build_key(key)
        use_server_time = 1 if self._use_server_time else 0

        result = self._status_script(
            keys=[redis_key],
            args=[effective_rate, effective_capacity, use_server_time, now],
        )

        tokens = float(result[0])
        last_time = result[1]
        last_time = float(last_time) if last_time != "" else None
        remaining = float(result[2])
        actual_rate = float(result[3])
        actual_capacity = float(result[4])
        server_now = float(result[5])

        return {
            "tokens": tokens,
            "last_time": last_time,
            "remaining_tokens": remaining,
            "capacity": actual_capacity,
            "rate": actual_rate,
            "server_now": server_now,
            "effective_rate": effective_rate,
            "effective_capacity": effective_capacity,
        }

    def get_stats(self, key: Optional[str] = None) -> RateLimitStats:
        if key is not None:
            stats_key = self._build_stats_key(key)
            scope = "key"
        else:
            stats_key = self._build_global_stats_key()
            scope = "global"

        result = self._stats_script(keys=[stats_key])

        return RateLimitStats(
            allowed=int(result[0]),
            blocked=int(result[1]),
            total=int(result[2]),
            hit_ratio=float(result[3]),
            scope=scope,
            key=key,
        )

    def reset_stats(self, key: Optional[str] = None) -> None:
        if key is not None:
            stats_key = self._build_stats_key(key)
        else:
            stats_key = self._build_global_stats_key()
        self._redis.delete(stats_key)

    def reset(self, key: str) -> bool:
        redis_key = self._build_key(key)
        return bool(self._redis.delete(redis_key))

    def reset_all(self, pattern: str = "*") -> int:
        match_pattern = self._build_key(pattern)
        keys = self._redis.keys(match_pattern)
        if keys:
            return self._redis.delete(*keys)
        return 0
