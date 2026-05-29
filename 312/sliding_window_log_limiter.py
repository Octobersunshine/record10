import time
import uuid
import json
import threading
import queue
from typing import Optional, Dict, List, Tuple, Callable
from dataclasses import dataclass, field

try:
    import redis
    from redis.client import Redis
    from redis.exceptions import RedisError, ConnectionError
except ImportError:
    redis = None


class SlidingWindowCounterRateLimiter:
    def __init__(self, window_seconds: int, limit: int, slot_seconds: int = 1):
        self.window_seconds = window_seconds
        self.limit = limit
        self.slot_seconds = slot_seconds
        self.num_slots = window_seconds // slot_seconds

        self._slots: list[int] = [0] * self.num_slots
        self._slot_timestamps: list[int] = [0] * self.num_slots
        self._head: int = 0
        self._total: int = 0

    def _get_slot_index(self, now_ts: int) -> int:
        return (now_ts // self.slot_seconds) % self.num_slots

    def _advance(self, now_ts: int) -> None:
        current_slot_ts = now_ts // self.slot_seconds
        head_slot_ts = self._slot_timestamps[self._head]

        if head_slot_ts == 0:
            self._slot_timestamps[self._head] = current_slot_ts
            return

        slots_diff = current_slot_ts - head_slot_ts
        if slots_diff >= self.num_slots:
            for i in range(self.num_slots):
                self._slots[i] = 0
                self._slot_timestamps[i] = current_slot_ts - self.num_slots + i + 1
            self._head = self._get_slot_index(now_ts)
            self._total = 0
            return

        for _ in range(slots_diff):
            self._head = (self._head + 1) % self.num_slots
            self._total -= self._slots[self._head]
            self._slots[self._head] = 0
            self._slot_timestamps[self._head] = head_slot_ts + _ + 1

    def allow_request(self) -> bool:
        now = time.time()
        now_ts = int(now)

        self._advance(now_ts)

        if self._total >= self.limit:
            return False

        slot_idx = self._get_slot_index(now_ts)
        self._slots[slot_idx] += 1
        self._total += 1
        return True

    @property
    def current_count(self) -> int:
        self._advance(int(time.time()))
        return self._total


# ============================================================
# 分布式滑动窗口限流 - 多机房容灾版
# ============================================================

@dataclass
class RedisNodeConfig:
    host: str
    port: int
    password: Optional[str] = None
    db: int = 0
    is_primary: bool = False
    dc_name: str = "default"


@dataclass
class RateLimitSyncMessage:
    key: str
    timestamp: float
    member: str
    source_dc: str
    message_id: str = field(default_factory=lambda: uuid.uuid4().hex)


class RedisMultiDCClient:
    def __init__(
        self,
        primary_config: RedisNodeConfig,
        replica_configs: Optional[List[RedisNodeConfig]] = None,
        connect_timeout_ms: int = 200,
        retry_attempts: int = 2,
    ):
        if redis is None:
            raise ImportError("redis-py is required. Install with: pip install redis")

        self._primary_config = primary_config
        self._replica_configs = replica_configs or []
        self._connect_timeout_ms = connect_timeout_ms
        self._retry_attempts = retry_attempts
        self._lock = threading.RLock()

        self._primary_client: Optional[Redis] = None
        self._replica_clients: Dict[str, Redis] = {}
        self._current_primary_config = primary_config
        self._health_check_thread: Optional[threading.Thread] = None
        self._stop_health_check = threading.Event()

        self._connect_clients()
        self._start_health_check()

    def _create_client(self, config: RedisNodeConfig) -> Redis:
        return redis.Redis(
            host=config.host,
            port=config.port,
            password=config.password,
            db=config.db,
            socket_connect_timeout=self._connect_timeout_ms / 1000,
            socket_timeout=self._connect_timeout_ms / 1000,
            decode_responses=True,
        )

    def _connect_clients(self) -> None:
        with self._lock:
            try:
                self._primary_client = self._create_client(self._current_primary_config)
                self._primary_client.ping()
            except (RedisError, ConnectionError):
                self._primary_client = None
                self._try_failover()

            self._replica_clients = {}
            for cfg in self._replica_configs:
                try:
                    client = self._create_client(cfg)
                    client.ping()
                    self._replica_clients[cfg.dc_name] = client
                except (RedisError, ConnectionError):
                    continue

    def _try_failover(self) -> bool:
        for cfg in self._replica_configs:
            try:
                client = self._create_client(cfg)
                client.ping()
                self._current_primary_config = cfg
                self._primary_client = client
                self._replica_configs = [
                    c for c in (self._replica_configs + [self._primary_config])
                    if c.dc_name != cfg.dc_name
                ]
                self._primary_config = cfg
                return True
            except (RedisError, ConnectionError):
                continue
        return False

    def _start_health_check(self) -> None:
        def check_loop():
            while not self._stop_health_check.is_set():
                try:
                    with self._lock:
                        if self._primary_client is None:
                            self._try_failover()
                        else:
                            try:
                                self._primary_client.ping()
                            except (RedisError, ConnectionError):
                                self._primary_client = None
                                self._try_failover()

                        for cfg in list(self._replica_configs):
                            if cfg.dc_name not in self._replica_clients:
                                try:
                                    client = self._create_client(cfg)
                                    client.ping()
                                    self._replica_clients[cfg.dc_name] = client
                                except (RedisError, ConnectionError):
                                    continue
                except Exception:
                    pass
                self._stop_health_check.wait(2)

        self._health_check_thread = threading.Thread(target=check_loop, daemon=True)
        self._health_check_thread.start()

    def get_primary(self) -> Optional[Redis]:
        with self._lock:
            return self._primary_client

    def get_replica(self, dc_name: str) -> Optional[Redis]:
        return self._replica_clients.get(dc_name)

    def get_all_replicas(self) -> Dict[str, Redis]:
        return self._replica_clients.copy()

    def execute_on_primary(self, func: Callable[[Redis], any]) -> Tuple[bool, any]:
        for attempt in range(self._retry_attempts):
            client = self.get_primary()
            if client is None:
                with self._lock:
                    self._try_failover()
                continue
            try:
                return True, func(client)
            except (RedisError, ConnectionError):
                with self._lock:
                    self._primary_client = None
                    self._try_failover()
        return False, None

    def close(self) -> None:
        self._stop_health_check.set()
        with self._lock:
            if self._primary_client:
                self._primary_client.close()
            for client in self._replica_clients.values():
                client.close()
            self._replica_clients.clear()
            self._primary_client = None


class AsyncRateSyncManager:
    def __init__(
        self,
        redis_client: RedisMultiDCClient,
        local_dc: str,
        sync_channel: str = "rate_limit_sync",
        batch_size: int = 50,
        sync_interval_ms: int = 50,
    ):
        self._redis_client = redis_client
        self._local_dc = local_dc
        self._sync_channel = sync_channel
        self._batch_size = batch_size
        self._sync_interval_ms = sync_interval_ms

        self._sync_queue: queue.Queue[RateLimitSyncMessage] = queue.Queue()
        self._publisher_thread: Optional[threading.Thread] = None
        self._subscriber_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._local_sync_ids: set[str] = set()
        self._sync_lock = threading.Lock()

        self._on_sync_callback: Optional[Callable[[str, float, str], None]] = None

    def set_sync_callback(self, callback: Callable[[str, float, str], None]) -> None:
        self._on_sync_callback = callback

    def enqueue_sync(self, key: str, timestamp: float, member: str) -> None:
        msg = RateLimitSyncMessage(
            key=key,
            timestamp=timestamp,
            member=member,
            source_dc=self._local_dc,
        )
        self._sync_queue.put(msg)
        with self._sync_lock:
            self._local_sync_ids.add(msg.message_id)

    def _publisher_loop(self) -> None:
        batch: List[RateLimitSyncMessage] = []
        last_sync = time.time()

        while not self._stop_event.is_set():
            try:
                try:
                    msg = self._sync_queue.get(timeout=0.01)
                    batch.append(msg)
                except queue.Empty:
                    pass

                should_sync = (
                    len(batch) >= self._batch_size
                    or (time.time() - last_sync) * 1000 >= self._sync_interval_ms
                )

                if should_sync and batch:
                    messages = [
                        json.dumps({
                            "message_id": m.message_id,
                            "key": m.key,
                            "timestamp": m.timestamp,
                            "member": m.member,
                            "source_dc": m.source_dc,
                        })
                        for m in batch
                    ]

                    success, _ = self._redis_client.execute_on_primary(
                        lambda r: r.publish(self._sync_channel, json.dumps(messages))
                    )

                    if success:
                        for m in batch:
                            with self._sync_lock:
                                if m.message_id in self._local_sync_ids:
                                    self._local_sync_ids.remove(m.message_id)
                        batch.clear()
                        last_sync = time.time()

            except Exception:
                time.sleep(0.01)

    def _subscriber_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                success, pubsub = self._redis_client.execute_on_primary(
                    lambda r: r.pubsub()
                )
                if not success or pubsub is None:
                    time.sleep(1)
                    continue

                pubsub.subscribe(self._sync_channel)

                for message in pubsub.listen():
                    if self._stop_event.is_set():
                        break
                    if message["type"] != "message":
                        continue

                    try:
                        data = json.loads(message["data"])
                        if isinstance(data, list):
                            for msg_data in data:
                                self._handle_incoming_sync(msg_data)
                        else:
                            self._handle_incoming_sync(data)
                    except (json.JSONDecodeError, KeyError):
                        continue

                pubsub.close()
            except Exception:
                time.sleep(1)

    def _handle_incoming_sync(self, msg_data: Dict) -> None:
        msg_id = msg_data.get("message_id")
        source_dc = msg_data.get("source_dc")
        key = msg_data.get("key")
        timestamp = msg_data.get("timestamp")
        member = msg_data.get("member")

        if source_dc == self._local_dc:
            with self._sync_lock:
                if msg_id in self._local_sync_ids:
                    self._local_sync_ids.remove(msg_id)
            return

        if self._on_sync_callback:
            self._on_sync_callback(key, timestamp, member)

    def start(self) -> None:
        self._publisher_thread = threading.Thread(target=self._publisher_loop, daemon=True)
        self._publisher_thread.start()

        self._subscriber_thread = threading.Thread(target=self._subscriber_loop, daemon=True)
        self._subscriber_thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._publisher_thread:
            self._publisher_thread.join(timeout=1)
        if self._subscriber_thread:
            self._subscriber_thread.join(timeout=1)

    def get_queue_size(self) -> int:
        return self._sync_queue.qsize()


class DistributedSlidingWindowRateLimiter:
    def __init__(
        self,
        window_seconds: int,
        limit: int,
        redis_client: RedisMultiDCClient,
        key_prefix: str = "rate_limit",
        enable_async_sync: bool = True,
        local_dc: str = "default",
        sync_batch_size: int = 50,
        sync_interval_ms: int = 50,
    ):
        if redis is None:
            raise ImportError("redis-py is required. Install with: pip install redis")

        self.window_seconds = window_seconds
        self.limit = limit
        self._redis_client = redis_client
        self._key_prefix = key_prefix
        self._enable_async_sync = enable_async_sync
        self._local_dc = local_dc

        self._sync_manager: Optional[AsyncRateSyncManager] = None
        if enable_async_sync:
            self._sync_manager = AsyncRateSyncManager(
                redis_client=redis_client,
                local_dc=local_dc,
                batch_size=sync_batch_size,
                sync_interval_ms=sync_interval_ms,
            )
            self._sync_manager.set_sync_callback(self._handle_remote_sync)
            self._sync_manager.start()

        self._allow_script = self._register_lua_script()

    def _register_lua_script(self) -> Optional[str]:
        script = """
        local key = KEYS[1]
        local now = tonumber(ARGV[1])
        local window = tonumber(ARGV[2])
        local limit = tonumber(ARGV[3])
        local member = ARGV[4]
        local score = tonumber(ARGV[5])

        local cutoff = now - window

        redis.call('ZREMRANGEBYSCORE', key, '-inf', cutoff)

        local count = tonumber(redis.call('ZCARD', key))

        if count >= limit then
            return 0
        end

        redis.call('ZADD', key, score, member)
        redis.call('EXPIRE', key, math.ceil(window * 2))

        return 1
        """
        try:
            success, sha = self._redis_client.execute_on_primary(
                lambda r: r.script_load(script)
            )
            return sha if success else None
        except Exception:
            return None

    def _get_key(self, identity: str) -> str:
        return f"{self._key_prefix}:{identity}"

    def allow_request(self, identity: str = "default") -> bool:
        now = time.time()
        member = f"{now}:{uuid.uuid4().hex}"
        key = self._get_key(identity)

        success, result = self._redis_client.execute_on_primary(
            lambda r: self._execute_allow_request(r, key, now, member)
        )

        if not success:
            success, result = self._redis_client.execute_on_primary(
                lambda r: self._fallback_allow_request(r, key, now, member)
            )
            if not success:
                return self._local_fallback(identity)

        if result == 1 and self._enable_async_sync and self._sync_manager:
            self._sync_manager.enqueue_sync(key, now, member)

        return result == 1

    def _execute_allow_request(self, r: Redis, key: str, now: float, member: str) -> int:
        if self._allow_script:
            return r.evalsha(
                self._allow_script,
                1,
                key,
                str(now),
                str(self.window_seconds),
                str(self.limit),
                member,
                str(now),
            )
        return self._fallback_allow_request(r, key, now, member)

    def _fallback_allow_request(self, r: Redis, key: str, now: float, member: str) -> int:
        cutoff = now - self.window_seconds
        r.zremrangebyscore(key, "-inf", cutoff)
        count = r.zcard(key)
        if count >= self.limit:
            return 0
        r.zadd(key, {member: now})
        r.expire(key, int(self.window_seconds * 2))
        return 1

    def _local_fallback(self, identity: str) -> bool:
        now_ts = int(time.time())
        slot = now_ts % 60
        attr_name = f"_fallback_{identity.replace(':', '_')}"
        if not hasattr(self, attr_name):
            setattr(self, attr_name, {"slots": [0] * 60, "last_ts": 0, "total": 0})
        state = getattr(self, attr_name)

        if now_ts - state["last_ts"] >= 60:
            state["slots"] = [0] * 60
            state["total"] = 0

        if state["total"] >= self.limit:
            return False

        if now_ts != state["last_ts"]:
            state["slots"][slot] = 0
        state["slots"][slot] += 1
        state["total"] += 1
        state["last_ts"] = now_ts
        return True

    def _handle_remote_sync(self, key: str, timestamp: float, member: str) -> None:
        try:
            self._redis_client.execute_on_primary(
                lambda r: r.zadd(key, {member: timestamp}, nx=True)
            )
        except Exception:
            pass

    def current_count(self, identity: str = "default") -> int:
        now = time.time()
        key = self._get_key(identity)
        cutoff = now - self.window_seconds

        success, result = self._redis_client.execute_on_primary(
            lambda r: (r.zremrangebyscore(key, "-inf", cutoff), r.zcard(key))
        )
        return result[1] if success else 0

    def reset(self, identity: str = "default") -> None:
        key = self._get_key(identity)
        self._redis_client.execute_on_primary(lambda r: r.delete(key))

    def close(self) -> None:
        if self._sync_manager:
            self._sync_manager.stop()


# ============================================================
# 测试代码
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("测试1: 本地滑动窗口计数限流")
    print("=" * 60)
    limiter = SlidingWindowCounterRateLimiter(window_seconds=60, limit=5, slot_seconds=1)

    print("=== 快速连续请求 ===")
    for i in range(8):
        allowed = limiter.allow_request()
        status = "通过" if allowed else "拒绝"
        print(f"请求 {i + 1}: {status} (窗口内请求数: {limiter.current_count})")

    print("\n" + "=" * 60)
    print("测试2: 模拟时间推进的窗口滑动")
    print("=" * 60)

    class TimeControlledLimiter(SlidingWindowCounterRateLimiter):
        def __init__(self):
            super().__init__(window_seconds=10, limit=5, slot_seconds=1)
            self._virtual_time = 1000000

        def _get_now(self) -> float:
            return float(self._virtual_time)

        def set_time(self, ts: int) -> None:
            self._virtual_time = ts

        def allow_request(self) -> bool:
            now = self._get_now()
            now_ts = int(now)
            self._advance(now_ts)
            if self._total >= self.limit:
                return False
            slot_idx = self._get_slot_index(now_ts)
            self._slots[slot_idx] += 1
            self._total += 1
            return True

        @property
        def current_count(self) -> int:
            self._advance(int(self._get_now()))
            return self._total

    limiter2 = TimeControlledLimiter()

    t = 1000000
    print(f"时间 {t}: 发送 3 个请求")
    for i in range(3):
        allowed = limiter2.allow_request()
        print(f"  请求 {i + 1}: {'通过' if allowed else '拒绝'} (计数: {limiter2.current_count})")

    t += 5
    limiter2.set_time(t)
    print(f"\n时间 {t} (+5秒): 再发送 3 个请求")
    for i in range(3):
        allowed = limiter2.allow_request()
        status = "通过" if allowed else "拒绝"
        print(f"  请求 {i + 1}: {status} (计数: {limiter2.current_count})")

    t += 6
    limiter2.set_time(t)
    print(f"\n时间 {t} (+6秒): 最早的 3 个请求已滑出窗口，再发送 3 个")
    for i in range(3):
        allowed = limiter2.allow_request()
        status = "通过" if allowed else "拒绝"
        print(f"  请求 {i + 1}: {status} (计数: {limiter2.current_count})")

    t += 10
    limiter2.set_time(t)
    print(f"\n时间 {t} (+10秒): 整个窗口已过期，清零重置")
    for i in range(6):
        allowed = limiter2.allow_request()
        status = "通过" if allowed else "拒绝"
        print(f"  请求 {i + 1}: {status} (计数: {limiter2.current_count})")

    print(f"\n=== 内存对比 ===")
    print(f"滑动窗口日志（10000 请求）: 约 {10000 * 8} 字节（每个时间戳8字节）")
    print(f"滑动窗口计数（60秒窗口，1秒槽）: {60 * (4 + 4) + 16} 字节（固定大小）")

    print("\n" + "=" * 60)
    print("测试3: 分布式滑动窗口限流架构说明")
    print("=" * 60)
    print("\n架构组件:")
    print("  1. RedisMultiDCClient - 多机房Redis客户端")
    print("     - 支持主备配置与自动故障转移")
    print("     - 后台健康检查（每2秒）")
    print("     - 连接超时保护（默认200ms）")
    print("")
    print("  2. AsyncRateSyncManager - 异步同步管理器")
    print("     - 基于Redis Pub/Sub的跨机房同步")
    print("     - 批量发送 + 定时刷出（50条/50ms）")
    print("     - 本地消息去重，避免循环同步")
    print("")
    print("  3. DistributedSlidingWindowRateLimiter - 分布式限流器")
    print("     - Redis ZSET存储时间戳")
    print("     - Lua脚本原子执行ZREMRANGEBYSCORE + ZCARD + ZADD")
    print("     - 本地环形数组降级，Redis不可用时继续服务")
    print("")
    print("Redis ZSET 操作流程:")
    print("  1. ZREMRANGEBYSCORE key -inf (now-window)  # 清理过期")
    print("  2. ZCARD key                               # 统计窗口内请求数")
    print("  3. ZADD key score member                   # 记录当前请求")
    print("  4. EXPIRE key window*2                     # 设置过期时间")
    print("")
    print("使用示例:")
    print("  primary = RedisNodeConfig('redis-primary', 6379, is_primary=True, dc_name='beijing')")
    print("  replica = RedisNodeConfig('redis-replica', 6379, dc_name='shanghai')")
    print("  client = RedisMultiDCClient(primary, [replica])")
    print("  limiter = DistributedSlidingWindowRateLimiter(")
    print("      window_seconds=60, limit=100, redis_client=client,")
    print("      enable_async_sync=True, local_dc='beijing'")
    print("  )")
    print("  limiter.allow_request('user_123')")

