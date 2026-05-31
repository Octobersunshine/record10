import time
import redis


class CounterUnderflowError(Exception):
    pass


class AtomicCounter:

    _DECR_SCRIPT = """
    local current = redis.call('GET', KEYS[1])
    if current == false then
        current = 0
    else
        current = tonumber(current)
    end
    local amount = tonumber(ARGV[1])
    local allow_negative = tonumber(ARGV[2])
    local ttl = tonumber(ARGV[3])
    local new_val = current - amount
    if allow_negative == 0 and new_val < 0 then
        return redis.error_reply('underflow')
    end
    if ttl > 0 then
        redis.call('SET', KEYS[1], new_val, 'EX', ttl)
    else
        redis.call('SET', KEYS[1], new_val)
    end
    return new_val
    """

    _INCR_SCRIPT = """
    local current = redis.call('GET', KEYS[1])
    if current == false then
        current = 0
    else
        current = tonumber(current)
    end
    local amount = tonumber(ARGV[1])
    local ttl = tonumber(ARGV[2])
    local new_val = current + amount
    if ttl > 0 then
        redis.call('SET', KEYS[1], new_val, 'EX', ttl)
    else
        redis.call('SET', KEYS[1], new_val)
    end
    return new_val
    """

    _MULTI_INCR_SCRIPT = """
    local ttl = tonumber(ARGV[#ARGV])
    local results = {}
    for i = 1, #KEYS do
        local key = KEYS[i]
        local amount = tonumber(ARGV[i])
        local current = redis.call('GET', key)
        if current == false then
            current = 0
        else
            current = tonumber(current)
        end
        local new_val = current + amount
        if ttl > 0 then
            redis.call('SET', key, new_val, 'EX', ttl)
        else
            redis.call('SET', key, new_val)
        end
        results[i] = new_val
    end
    return results
    """

    _MULTI_DECR_SCRIPT = """
    local allow_negative = tonumber(ARGV[#ARGV - 1])
    local ttl = tonumber(ARGV[#ARGV])
    local n = (#ARGV - 2) / 2
    local results = {}
    for i = 1, #KEYS do
        local key = KEYS[i]
        local amount = tonumber(ARGV[i])
        local current = redis.call('GET', key)
        if current == false then
            current = 0
        else
            current = tonumber(current)
        end
        local new_val = current - amount
        if allow_negative == 0 and new_val < 0 then
            return redis.error_reply('underflow:' .. i)
        end
        if ttl > 0 then
            redis.call('SET', key, new_val, 'EX', ttl)
        else
            redis.call('SET', key, new_val)
        end
        results[i] = new_val
    end
    return results
    """

    def __init__(
        self,
        redis_client: redis.Redis,
        key: str,
        initial: int = 0,
        namespace: str = "",
        allow_negative: bool = False,
        ttl: int = 0,
        track_stats: bool = True,
    ):
        if initial < 0 and not allow_negative:
            raise ValueError("initial value cannot be negative when allow_negative is False")
        self._client = redis_client
        self._namespace = namespace
        self._base_key = key
        self._allow_negative = allow_negative
        self._ttl = ttl
        self._track_stats = track_stats
        self._key = f"{namespace}:{key}" if namespace else key
        self._stats_key = f"{self._key}:stats" if track_stats else None
        self._decr_script = self._client.register_script(self._DECR_SCRIPT)
        self._incr_script = self._client.register_script(self._INCR_SCRIPT)
        self._multi_incr_script = self._client.register_script(self._MULTI_INCR_SCRIPT)
        self._multi_decr_script = self._client.register_script(self._MULTI_DECR_SCRIPT)
        self._init_counter(initial)

    def _init_counter(self, initial: int):
        if self._ttl > 0:
            self._client.set(self._key, initial, ex=self._ttl, nx=True)
        else:
            self._client.setnx(self._key, initial)

    def _record_access(self, op: str):
        if not self._track_stats:
            return
        now = int(time.time())
        pipe = self._client.pipeline()
        pipe.hincrby(self._stats_key, "total_ops", 1)
        pipe.hincrby(self._stats_key, f"{op}_count", 1)
        pipe.zincrby(f"{self._stats_key}:minute", 1, now // 60)
        pipe.zincrby(f"{self._stats_key}:hour", 1, now // 3600)
        pipe.execute()

    @property
    def key(self) -> str:
        return self._key

    @property
    def namespace(self) -> str:
        return self._namespace

    @property
    def allow_negative(self) -> bool:
        return self._allow_negative

    @property
    def ttl(self) -> int:
        return self._ttl

    def incr(self, amount: int = 1) -> int:
        if amount < 0:
            raise ValueError("amount must be non-negative, use decr() instead")
        self._record_access("incr")
        result = self._incr_script(keys=[self._key], args=[amount, self._ttl])
        return int(result)

    def decr(self, amount: int = 1) -> int:
        if amount < 0:
            raise ValueError("amount must be non-negative, use incr() instead")
        self._record_access("decr")
        try:
            result = self._decr_script(
                keys=[self._key],
                args=[amount, 1 if self._allow_negative else 0, self._ttl],
            )
            return int(result)
        except redis.exceptions.ResponseError as e:
            if "underflow" in str(e):
                current = self.get()
                raise CounterUnderflowError(
                    f"Cannot decrement by {amount}: current value is {current}, "
                    f"which would go below 0 (allow_negative=False)"
                ) from e
            raise

    def get(self) -> int:
        self._record_access("get")
        val = self._client.get(self._key)
        if val is None:
            return 0
        return int(val)

    def reset(self, value: int = 0) -> int:
        if value < 0 and not self._allow_negative:
            raise ValueError("reset value cannot be negative when allow_negative is False")
        self._record_access("reset")
        if self._ttl > 0:
            self._client.set(self._key, value, ex=self._ttl)
        else:
            self._client.set(self._key, value)
        return value

    def remaining_ttl(self) -> int:
        ttl = self._client.ttl(self._key)
        return max(0, ttl) if ttl > 0 else 0

    def get_qps(self, window_seconds: int = 60) -> float:
        if not self._track_stats:
            return 0.0
        now = int(time.time())
        minute_key = now // 60
        total = 0
        for i in range(min(window_seconds // 60 + 1, 5)):
            ts = minute_key - i
            count = self._client.zscore(f"{self._stats_key}:minute", ts)
            if count:
                total += int(count)
        return total / min(window_seconds, 300)

    def get_stats(self) -> dict:
        if not self._track_stats:
            return {}
        stats = self._client.hgetall(self._stats_key) or {}
        return {k: int(v) for k, v in stats.items()}

    def reset_stats(self):
        if self._track_stats:
            pipe = self._client.pipeline()
            pipe.delete(self._stats_key)
            pipe.delete(f"{self._stats_key}:minute")
            pipe.delete(f"{self._stats_key}:hour")
            pipe.execute()


class CounterManager:

    def __init__(self, redis_client: redis.Redis, namespace: str = ""):
        self._client = redis_client
        self._namespace = namespace
        self._counters = {}

    def get_counter(
        self,
        key: str,
        initial: int = 0,
        allow_negative: bool = False,
        ttl: int = 0,
        track_stats: bool = True,
    ) -> AtomicCounter:
        if key not in self._counters:
            self._counters[key] = AtomicCounter(
                self._client,
                key,
                initial=initial,
                namespace=self._namespace,
                allow_negative=allow_negative,
                ttl=ttl,
                track_stats=track_stats,
            )
        return self._counters[key]

    def _make_key(self, key: str) -> str:
        return f"{self._namespace}:{key}" if self._namespace else key

    def multi_incr(
        self,
        increments: dict,
        ttl: int = 0,
    ) -> dict:
        if not increments:
            return {}
        keys = list(increments.keys())
        amounts = list(increments.values())
        full_keys = [self._make_key(k) for k in keys]
        script = AtomicCounter._MULTI_INCR_SCRIPT
        registered = self._client.register_script(script)
        results = registered(keys=full_keys, args=amounts + [ttl])
        return {k: int(v) for k, v in zip(keys, results)}

    def multi_decr(
        self,
        decrements: dict,
        allow_negative: bool = False,
        ttl: int = 0,
    ) -> dict:
        if not decrements:
            return {}
        keys = list(decrements.keys())
        amounts = list(decrements.values())
        full_keys = [self._make_key(k) for k in keys]
        script = AtomicCounter._MULTI_DECR_SCRIPT
        registered = self._client.register_script(script)
        try:
            results = registered(
                keys=full_keys,
                args=amounts + [1 if allow_negative else 0, ttl],
            )
            return {k: int(v) for k, v in zip(keys, results)}
        except redis.exceptions.ResponseError as e:
            msg = str(e)
            if "underflow:" in msg:
                idx = int(msg.split(":")[1]) - 1
                raise CounterUnderflowError(
                    f"Cannot decrement '{keys[idx]}' by {amounts[idx]}: would go below 0"
                ) from e
            raise

    def multi_get(self, keys: list) -> dict:
        if not keys:
            return {}
        full_keys = [self._make_key(k) for k in keys]
        values = self._client.mget(full_keys)
        return {k: (int(v) if v else 0) for k, v in zip(keys, values)}

    def get_hot_counters(self, top_n: int = 10, namespace: str = None) -> list:
        ns = namespace if namespace is not None else self._namespace
        pattern = f"{ns}:*:stats" if ns else "*:stats"
        hot = []
        for key in self._client.scan_iter(match=pattern, count=100):
            key_str = key if isinstance(key, str) else key.decode("utf-8")
            total = self._client.hget(key_str, "total_ops")
            if total:
                counter_key = key_str.replace(":stats", "")
                if ns:
                    counter_key = counter_key[len(ns) + 1 :]
                hot.append((counter_key, int(total)))
        hot.sort(key=lambda x: x[1], reverse=True)
        return hot[:top_n]


if __name__ == "__main__":
    r = redis.Redis(host="localhost", port=6379, decode_responses=True)

    print("=== TTL Expiration ===")
    ttl_counter = AtomicCounter(r, "session", initial=100, ttl=3600)
    print(f"key: {ttl_counter.key}")
    print(f"ttl: {ttl_counter.ttl}")
    print(f"remaining_ttl: {ttl_counter.remaining_ttl()}s")
    print(f"incr: {ttl_counter.incr()}")
    print()

    print("=== Batch Operations ===")
    manager = CounterManager(r, namespace="shop")
    results = manager.multi_incr({"apple": 5, "banana": 3, "orange": 10})
    print(f"multi_incr result: {results}")
    values = manager.multi_get(["apple", "banana", "orange"])
    print(f"multi_get result: {values}")
    try:
        decr_results = manager.multi_decr({"apple": 2, "banana": 1}, allow_negative=False)
        print(f"multi_decr result: {decr_results}")
    except CounterUnderflowError as e:
        print(f"multi_decr error: {e}")
    print()

    print("=== QPS & Stats ===")
    stats_counter = manager.get_counter("page_view", track_stats=True)
    for _ in range(50):
        stats_counter.incr()
    time.sleep(0.1)
    print(f"total_ops: {stats_counter.get_stats().get('total_ops', 0)}")
    print(f"qps (1min window): {stats_counter.get_qps(60):.2f}")
    print()

    print("=== Hot Counters ===")
    hot = manager.get_hot_counters(top_n=5)
    print(f"top hot counters: {hot}")
