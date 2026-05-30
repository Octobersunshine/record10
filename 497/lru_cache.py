import threading
import time


class _Node:
    __slots__ = ('key', 'value', 'prev', 'next', 'expires_at')

    def __init__(self, key=0, value=0, expires_at=None):
        self.key = key
        self.value = value
        self.prev = None
        self.next = None
        self.expires_at = expires_at


class LRUCache:

    def __init__(self, capacity: int, cleanup_interval: float = 1.0):
        if capacity < 0:
            raise ValueError("capacity must be a non-negative integer")
        if cleanup_interval <= 0:
            raise ValueError("cleanup_interval must be a positive number")
        self._capacity = capacity
        self._cache = {}
        self._head = _Node()
        self._tail = _Node()
        self._head.next = self._tail
        self._tail.prev = self._head
        self._hits = 0
        self._misses = 0
        self._expired_evictions = 0
        self._lru_evictions = 0
        self._lock = threading.RLock()
        self._cleanup_interval = cleanup_interval
        self._shutdown_event = threading.Event()
        self._cleanup_thread = threading.Thread(target=self._cleanup_loop, daemon=True)
        self._cleanup_thread.start()

    def _remove(self, node: _Node):
        node.prev.next = node.next
        node.next.prev = node.prev
        node.prev = None
        node.next = None

    def _add_to_front(self, node: _Node):
        node.next = self._head.next
        node.prev = self._head
        self._head.next.prev = node
        self._head.next = node

    def _move_to_front(self, node: _Node):
        self._remove(node)
        self._add_to_front(node)

    def _evict_node(self, node: _Node):
        self._remove(node)
        if node.key in self._cache:
            del self._cache[node.key]
        node.key = None
        node.value = None
        node.expires_at = None

    def _is_expired(self, node: _Node) -> bool:
        if node.expires_at is None:
            return False
        return time.time() >= node.expires_at

    def get(self, key: int) -> int:
        with self._lock:
            if self._capacity == 0:
                self._misses += 1
                return -1
            if key in self._cache:
                node = self._cache[key]
                if self._is_expired(node):
                    self._evict_node(node)
                    self._expired_evictions += 1
                    self._misses += 1
                    return -1
                self._move_to_front(node)
                self._hits += 1
                return node.value
            self._misses += 1
            return -1

    def put(self, key: int, value: int, ttl: float = None):
        with self._lock:
            if self._capacity == 0:
                return
            expires_at = None
            if ttl is not None:
                if ttl <= 0:
                    return
                expires_at = time.time() + ttl
            if key in self._cache:
                node = self._cache[key]
                node.value = value
                node.expires_at = expires_at
                self._move_to_front(node)
            else:
                node = _Node(key, value, expires_at)
                self._cache[key] = node
                self._add_to_front(node)
                if len(self._cache) > self._capacity:
                    lru = self._tail.prev
                    self._evict_node(lru)
                    self._lru_evictions += 1

    def _cleanup_expired(self):
        with self._lock:
            node = self._tail.prev
            while node is not self._head:
                prev = node.prev
                if self._is_expired(node):
                    self._evict_node(node)
                    self._expired_evictions += 1
                node = prev

    def _cleanup_loop(self):
        while not self._shutdown_event.is_set():
            self._shutdown_event.wait(self._cleanup_interval)
            if self._shutdown_event.is_set():
                break
            self._cleanup_expired()

    def shutdown(self):
        self._shutdown_event.set()
        self._cleanup_thread.join(timeout=5.0)

    def _get_state_ordered_unlocked(self) -> list:
        items = []
        node = self._head.next
        while node is not self._tail:
            ttl_remaining = None
            if node.expires_at is not None:
                ttl_remaining = round(node.expires_at - time.time(), 2)
            items.append((node.key, node.value, ttl_remaining))
            node = node.next
        return items

    @property
    def state(self) -> dict:
        with self._lock:
            result = {}
            node = self._head.next
            while node is not self._tail:
                if not self._is_expired(node):
                    result[node.key] = node.value
                node = node.next
            return result

    @property
    def state_ordered(self) -> list:
        with self._lock:
            return self._get_state_ordered_unlocked()

    @property
    def hit_rate(self) -> float:
        total = self._hits + self._misses
        if total == 0:
            return 0.0
        return self._hits / total

    @property
    def stats(self) -> dict:
        with self._lock:
            return {
                "capacity": self._capacity,
                "size": len(self._cache),
                "hits": self._hits,
                "misses": self._misses,
                "expired_evictions": self._expired_evictions,
                "lru_evictions": self._lru_evictions,
                "hit_rate": f"{self.hit_rate:.2%}",
                "state": self._get_state_ordered_unlocked(),
            }

    def reset_stats(self):
        with self._lock:
            self._hits = 0
            self._misses = 0
            self._expired_evictions = 0
            self._lru_evictions = 0

    def __len__(self):
        with self._lock:
            return len(self._cache)

    def __contains__(self, key):
        with self._lock:
            return key in self._cache

    def __repr__(self):
        with self._lock:
            items = ", ".join(f"{k}={v}" for k, v, _ in self._get_state_ordered_unlocked())
            return f"LRUCache(capacity={self._capacity}, [{items}])"

    def __del__(self):
        try:
            self.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    print("=" * 60)
    print("LRU Cache 演示 (双向链表 + 哈希表 + TTL, O(1) 操作)")
    print("=" * 60)

    lru = LRUCache(3, cleanup_interval=0.5)
    print(f"\n创建容量为 3 的 LRU 缓存 (清理间隔 0.5s): {lru}")

    lru.put(1, 10)
    lru.put(2, 20)
    lru.put(3, 30)
    print(f"put(1,10), put(2,20), put(3,30)")
    print(f"  缓存状态 (MRU->LRU): {lru.state_ordered}")

    print(f"\nget(1) = {lru.get(1)}")
    print(f"  缓存状态 (MRU->LRU): {lru.state_ordered}  <- key=1 被移到最前")

    print(f"get(4) = {lru.get(4)}  <- key=4 不存在")

    lru.put(4, 40)
    print(f"\nput(4,40)")
    print(f"  缓存状态 (MRU->LRU): {lru.state_ordered}  <- key=2 (LRU) 被淘汰")

    lru.put(3, 99)
    print(f"put(3,99)  <- 更新已存在的 key")
    print(f"  缓存状态 (MRU->LRU): {lru.state_ordered}  <- key=3 更新并移到最前")

    print(f"\nget(2) = {lru.get(2)}  <- key=2 已被淘汰")
    print(f"get(3) = {lru.get(3)}")
    print(f"get(4) = {lru.get(4)}")

    print("\n" + "=" * 60)
    print("命中率统计")
    print("=" * 60)
    print(lru.stats)

    print("\n" + "=" * 60)
    print("TTL 过期测试")
    print("=" * 60)
    lru_ttl = LRUCache(5, cleanup_interval=0.3)
    lru_ttl.put(1, 100, ttl=2.0)
    lru_ttl.put(2, 200, ttl=2.0)
    lru_ttl.put(3, 300, ttl=0.5)
    lru_ttl.put(4, 400)
    print(f"put(1,100,ttl=2.0), put(2,200,ttl=2.0), put(3,300,ttl=0.5), put(4,400)")
    print(f"  缓存状态 (key, value, ttl_remaining): {lru_ttl.state_ordered}")

    print(f"\nget(3) = {lru_ttl.get(3)}  <- 立即访问，尚未过期")
    print(f"  缓存状态: {lru_ttl.state_ordered}")

    print("\n等待 0.6 秒 (key=3 的 ttl=0.5 已过期)...")
    time.sleep(0.6)

    print(f"get(3) = {lru_ttl.get(3)}  <- key=3 已过期，惰性淘汰")
    print(f"  缓存状态: {lru_ttl.state_ordered}")

    print("\n等待 1.5 秒 (key=1, key=2 的 ttl=2.0 将过期，后台线程清理)...")
    time.sleep(1.5)

    print(f"  缓存状态: {lru_ttl.state_ordered}")
    print(f"get(1) = {lru_ttl.get(1)}  <- key=1 可能已过期")
    print(f"get(2) = {lru_ttl.get(2)}  <- key=2 可能已过期")
    print(f"get(4) = {lru_ttl.get(4)}  <- key=4 无 TTL，永不过期")
    print(f"\nTTL 缓存统计: {lru_ttl.stats}")

    print("\n" + "=" * 60)
    print("TTL 更新测试: 重新 put 可刷新过期时间")
    print("=" * 60)
    lru_refresh = LRUCache(3, cleanup_interval=0.3)
    lru_refresh.put(1, 10, ttl=1.0)
    print(f"put(1,10,ttl=1.0) -> {lru_refresh.state_ordered}")
    time.sleep(0.5)
    lru_refresh.put(1, 10, ttl=1.0)
    print(f"0.5s 后 put(1,10,ttl=1.0) 刷新 TTL -> {lru_refresh.state_ordered}")
    time.sleep(0.6)
    print(f"再过 0.6s (距刷新 0.6s < 1.0s): get(1) = {lru_refresh.get(1)}  <- 仍有效")
    time.sleep(0.5)
    print(f"再过 0.5s (距刷新 1.1s > 1.0s): get(1) = {lru_refresh.get(1)}  <- 已过期")

    print("\n" + "=" * 60)
    print("边界测试: capacity = 0")
    print("=" * 60)
    lru0 = LRUCache(0, cleanup_interval=1.0)
    lru0.put(1, 100)
    print(f"put(1,100) -> 缓存状态: {lru0.state_ordered}")
    print(f"get(1) = {lru0.get(1)}")
    print(f"容量 0 缓存统计: {lru0.stats}")

    print("\n" + "=" * 60)
    print("边界测试: ttl <= 0 拒绝写入")
    print("=" * 60)
    lru_neg = LRUCache(3, cleanup_interval=1.0)
    lru_neg.put(1, 10, ttl=-1.0)
    lru_neg.put(2, 20, ttl=0)
    print(f"put(1,10,ttl=-1), put(2,20,ttl=0) -> {lru_neg.state_ordered}  <- 均被拒绝")
    lru_neg.put(3, 30, ttl=1.0)
    print(f"put(3,30,ttl=1.0) -> {lru_neg.state_ordered}  <- 正常写入")

    print("\n" + "=" * 60)
    print("线程安全测试: 多线程并发读写")
    print("=" * 60)
    lru_mt = LRUCache(100, cleanup_interval=0.5)
    errors = []

    def worker(thread_id):
        try:
            for i in range(200):
                key = (thread_id * 1000 + i) % 150
                lru_mt.put(key, i, ttl=2.0)
                lru_mt.get(key)
        except Exception as e:
            errors.append((thread_id, str(e)))

    threads = [threading.Thread(target=worker, args=(t,)) for t in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    if errors:
        print(f"✗ 线程安全测试失败: {errors}")
    else:
        print("✓ 5 线程各 200 次读写完成，无异常")
        print(f"  最终统计: size={len(lru_mt)}, hits={lru_mt._hits}, misses={lru_mt._misses}")

    lru.shutdown()
    lru_ttl.shutdown()
    lru_refresh.shutdown()
    lru0.shutdown()
    lru_neg.shutdown()
    lru_mt.shutdown()
    print("\n✓ 所有缓存实例已安全关闭")
