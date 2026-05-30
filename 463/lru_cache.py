from collections import OrderedDict, defaultdict


class LRUCache:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.cache = OrderedDict()

    def get(self, key: int) -> int:
        if key not in self.cache:
            return -1
        self.cache.move_to_end(key)
        return self.cache[key]

    def put(self, key: int, value: int) -> None:
        if self.capacity <= 0:
            return
        if key in self.cache:
            del self.cache[key]
        self.cache[key] = value
        if len(self.cache) > self.capacity:
            self.cache.popitem(last=False)


class LFUCache:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.size = 0
        self.min_freq = 0
        self.key_to_val_freq = {}
        self.freq_to_keys = defaultdict(OrderedDict)

    def get(self, key: int) -> int:
        if key not in self.key_to_val_freq:
            return -1
        val, freq = self.key_to_val_freq[key]
        del self.freq_to_keys[freq][key]
        if not self.freq_to_keys[freq]:
            del self.freq_to_keys[freq]
            if self.min_freq == freq:
                self.min_freq += 1
        self.freq_to_keys[freq + 1][key] = True
        self.key_to_val_freq[key] = (val, freq + 1)
        return val

    def put(self, key: int, value: int) -> None:
        if self.capacity <= 0:
            return
        if key in self.key_to_val_freq:
            _, freq = self.key_to_val_freq[key]
            del self.freq_to_keys[freq][key]
            if not self.freq_to_keys[freq]:
                del self.freq_to_keys[freq]
                if self.min_freq == freq:
                    self.min_freq += 1
            self.freq_to_keys[freq + 1][key] = True
            self.key_to_val_freq[key] = (value, freq + 1)
            return
        if self.size == self.capacity:
            evict_key, _ = self.freq_to_keys[self.min_freq].popitem(last=False)
            del self.key_to_val_freq[evict_key]
            if not self.freq_to_keys[self.min_freq]:
                del self.freq_to_keys[self.min_freq]
            self.size -= 1
        self.key_to_val_freq[key] = (value, 1)
        self.freq_to_keys[1][key] = True
        self.min_freq = 1
        self.size += 1


def simulate(cache_class, capacity: int, accesses: list) -> float:
    cache = cache_class(capacity)
    hits = 0
    for key in accesses:
        if cache.get(key) != -1:
            hits += 1
        else:
            cache.put(key, key)
    return hits / len(accesses) if accesses else 0.0


def run_comparison():
    capacity = 3

    sequential = list(range(6)) * 3

    hotspot = [0, 1] * 8 + list(range(6))

    loop = [0, 1, 2, 3, 0, 1, 4, 5]

    patterns = [
        ("顺序重复访问", sequential),
        ("热点集中访问", hotspot),
        ("循环+偶发访问", loop),
    ]

    print(f"缓存容量: {capacity}")
    print(f"{'访问模式':<12} {'LRU命中率':<12} {'LFU命中率':<12} {'胜出'}")
    print("-" * 48)
    for name, accesses in patterns:
        lru_rate = simulate(LRUCache, capacity, accesses)
        lfu_rate = simulate(LFUCache, capacity, accesses)
        winner = "LRU" if lru_rate > lfu_rate else ("LFU" if lfu_rate > lru_rate else "平局")
        print(f"{name:<12} {lru_rate:<12.2%} {lfu_rate:<12.2%} {winner}")


if __name__ == "__main__":
    print("=== LRU 基本测试 ===")
    lru = LRUCache(2)
    lru.put(1, 1)
    lru.put(2, 2)
    print(lru.get(1))
    lru.put(3, 3)
    print(lru.get(2))
    lru.put(4, 4)
    print(lru.get(1))
    print(lru.get(3))
    print(lru.get(4))

    print("\n=== LFU 基本测试 ===")
    lfu = LFUCache(2)
    lfu.put(1, 1)
    lfu.put(2, 2)
    print(lfu.get(1))
    lfu.put(3, 3)
    print(lfu.get(2))
    lfu.put(4, 4)
    print(lfu.get(1))
    print(lfu.get(3))
    print(lfu.get(4))

    print("\n=== LRU vs LFU 命中率对比 ===")
    run_comparison()
