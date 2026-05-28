from collections import OrderedDict, defaultdict, deque


class FIFOCache:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.cache = {}
        self.queue = deque()
        self.hits = 0
        self.misses = 0
        self.total_accesses = 0

    def get(self, key: int) -> int:
        self.total_accesses += 1
        if key in self.cache:
            self.hits += 1
            return self.cache[key]
        self.misses += 1
        return -1

    def put(self, key: int, value: int) -> None:
        if key in self.cache:
            self.cache[key] = value
            return

        if len(self.cache) >= self.capacity:
            oldest_key = self.queue.popleft()
            del self.cache[oldest_key]

        self.cache[key] = value
        self.queue.append(key)

    def get_hit_rate(self) -> float:
        if self.total_accesses == 0:
            return 0.0
        return self.hits / self.total_accesses

    def get_stats(self) -> dict:
        return {
            'strategy': 'FIFO',
            'capacity': self.capacity,
            'total_accesses': self.total_accesses,
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': self.get_hit_rate(),
            'current_cache': list(self.cache.keys())
        }


class LRUCache:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.cache = OrderedDict()
        self.hits = 0
        self.misses = 0
        self.total_accesses = 0

    def get(self, key: int) -> int:
        self.total_accesses += 1
        if key in self.cache:
            self.hits += 1
            self.cache.move_to_end(key)
            return self.cache[key]
        self.misses += 1
        return -1

    def put(self, key: int, value: int) -> None:
        if key in self.cache:
            self.cache[key] = value
            self.cache.move_to_end(key)
            return

        if len(self.cache) >= self.capacity:
            self.cache.popitem(last=False)

        self.cache[key] = value

    def get_hit_rate(self) -> float:
        if self.total_accesses == 0:
            return 0.0
        return self.hits / self.total_accesses

    def get_stats(self) -> dict:
        return {
            'strategy': 'LRU',
            'capacity': self.capacity,
            'total_accesses': self.total_accesses,
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': self.get_hit_rate(),
            'current_cache': list(self.cache.keys())
        }


class LFUCache:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.key_to_value = {}
        self.key_to_freq = {}
        self.freq_to_keys = defaultdict(OrderedDict)
        self.min_freq = 0
        self.hits = 0
        self.misses = 0
        self.total_accesses = 0

    def _update_freq(self, key: int) -> None:
        freq = self.key_to_freq[key]
        del self.freq_to_keys[freq][key]
        if not self.freq_to_keys[freq]:
            del self.freq_to_keys[freq]
            if self.min_freq == freq:
                self.min_freq += 1

        self.key_to_freq[key] = freq + 1
        self.freq_to_keys[freq + 1][key] = None

    def get(self, key: int) -> int:
        self.total_accesses += 1
        if key not in self.key_to_value:
            self.misses += 1
            return -1

        self.hits += 1
        self._update_freq(key)
        return self.key_to_value[key]

    def put(self, key: int, value: int) -> None:
        if self.capacity == 0:
            return

        if key in self.key_to_value:
            self.key_to_value[key] = value
            self._update_freq(key)
            return

        if len(self.key_to_value) >= self.capacity:
            evict_key, _ = self.freq_to_keys[self.min_freq].popitem(last=False)
            del self.key_to_value[evict_key]
            del self.key_to_freq[evict_key]
            if not self.freq_to_keys[self.min_freq]:
                del self.freq_to_keys[self.min_freq]

        self.key_to_value[key] = value
        self.key_to_freq[key] = 1
        self.freq_to_keys[1][key] = None
        self.min_freq = 1

    def get_hit_rate(self) -> float:
        if self.total_accesses == 0:
            return 0.0
        return self.hits / self.total_accesses

    def get_stats(self) -> dict:
        return {
            'strategy': 'LFU',
            'capacity': self.capacity,
            'total_accesses': self.total_accesses,
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': self.get_hit_rate(),
            'current_cache': list(self.key_to_value.keys()),
            'key_frequencies': dict(self.key_to_freq)
        }


class LFUAgingCache:
    def __init__(self, capacity: int, decay_factor: float = 0.5, decay_interval: int = 10):
        self.capacity = capacity
        self.decay_factor = decay_factor
        self.decay_interval = decay_interval
        self.key_to_value = {}
        self.key_to_freq = {}
        self.key_to_last_access = {}
        self.freq_to_keys = defaultdict(OrderedDict)
        self.min_freq = 0
        self.access_count = 0
        self.hits = 0
        self.misses = 0
        self.total_accesses = 0

    def _decay_frequencies(self) -> None:
        new_key_to_freq = {}
        new_freq_to_keys = defaultdict(OrderedDict)
        new_min_freq = float('inf')

        for key in self.key_to_freq:
            old_freq = self.key_to_freq[key]
            new_freq = max(1, int(old_freq * self.decay_factor))
            new_key_to_freq[key] = new_freq
            new_freq_to_keys[new_freq][key] = None
            if new_freq < new_min_freq:
                new_min_freq = new_freq

        self.key_to_freq = new_key_to_freq
        self.freq_to_keys = new_freq_to_keys
        self.min_freq = new_min_freq if new_min_freq != float('inf') else 0

    def _update_freq(self, key: int) -> None:
        freq = self.key_to_freq[key]
        del self.freq_to_keys[freq][key]
        if not self.freq_to_keys[freq]:
            del self.freq_to_keys[freq]
            if self.min_freq == freq:
                self.min_freq += 1

        self.key_to_freq[key] = freq + 1
        self.freq_to_keys[freq + 1][key] = None
        self.key_to_last_access[key] = self.access_count

    def get(self, key: int) -> int:
        self.total_accesses += 1
        self.access_count += 1

        if self.access_count % self.decay_interval == 0:
            self._decay_frequencies()

        if key not in self.key_to_value:
            self.misses += 1
            return -1

        self.hits += 1
        self._update_freq(key)
        return self.key_to_value[key]

    def put(self, key: int, value: int) -> None:
        if self.capacity == 0:
            return

        if key in self.key_to_value:
            self.key_to_value[key] = value
            self._update_freq(key)
            return

        if len(self.key_to_value) >= self.capacity:
            evict_key, _ = self.freq_to_keys[self.min_freq].popitem(last=False)
            del self.key_to_value[evict_key]
            del self.key_to_freq[evict_key]
            del self.key_to_last_access[evict_key]
            if not self.freq_to_keys[self.min_freq]:
                del self.freq_to_keys[self.min_freq]
                if self.freq_to_keys:
                    self.min_freq = min(self.freq_to_keys.keys())
                else:
                    self.min_freq = 0

        self.key_to_value[key] = value
        self.key_to_freq[key] = 1
        self.key_to_last_access[key] = self.access_count
        self.freq_to_keys[1][key] = None
        self.min_freq = 1

    def get_hit_rate(self) -> float:
        if self.total_accesses == 0:
            return 0.0
        return self.hits / self.total_accesses

    def get_stats(self) -> dict:
        return {
            'strategy': 'LFU-Aging',
            'capacity': self.capacity,
            'total_accesses': self.total_accesses,
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': self.get_hit_rate(),
            'current_cache': list(self.key_to_value.keys()),
            'key_frequencies': dict(self.key_to_freq),
            'decay_factor': self.decay_factor,
            'decay_interval': self.decay_interval
        }


class WindowLFUCache:
    def __init__(self, capacity: int, window_size: int = 10):
        self.capacity = capacity
        self.window_size = window_size
        self.key_to_value = {}
        self.key_to_window_freq = {}
        self.access_history = deque(maxlen=window_size)
        self.insert_order = OrderedDict()
        self.hits = 0
        self.misses = 0
        self.total_accesses = 0

    def _update_window_freq(self, key: int, is_hit: bool) -> None:
        if len(self.access_history) == self.window_size:
            evicted_key = self.access_history[0]
            if evicted_key in self.key_to_window_freq:
                self.key_to_window_freq[evicted_key] -= 1
                if self.key_to_window_freq[evicted_key] <= 0:
                    del self.key_to_window_freq[evicted_key]

        self.access_history.append(key)

        if is_hit:
            self.key_to_window_freq[key] = self.key_to_window_freq.get(key, 0) + 1

    def _get_min_freq_key(self) -> int:
        min_freq = float('inf')
        min_key = None

        for key in self.insert_order:
            freq = self.key_to_window_freq.get(key, 0)
            if freq < min_freq:
                min_freq = freq
                min_key = key

        return min_key

    def get(self, key: int) -> int:
        self.total_accesses += 1

        if key not in self.key_to_value:
            self.misses += 1
            self._update_window_freq(key, False)
            return -1

        self.hits += 1
        self._update_window_freq(key, True)
        return self.key_to_value[key]

    def put(self, key: int, value: int) -> None:
        if self.capacity == 0:
            return

        if key in self.key_to_value:
            self.key_to_value[key] = value
            return

        if len(self.key_to_value) >= self.capacity:
            evict_key = self._get_min_freq_key()
            del self.key_to_value[evict_key]
            del self.insert_order[evict_key]

        self.key_to_value[key] = value
        self.insert_order[key] = None

    def get_hit_rate(self) -> float:
        if self.total_accesses == 0:
            return 0.0
        return self.hits / self.total_accesses

    def get_stats(self) -> dict:
        return {
            'strategy': 'Window-LFU',
            'capacity': self.capacity,
            'total_accesses': self.total_accesses,
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': self.get_hit_rate(),
            'current_cache': list(self.key_to_value.keys()),
            'window_frequencies': {k: self.key_to_window_freq.get(k, 0) for k in self.key_to_value},
            'window_size': self.window_size
        }


class ARCCache:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.p = 0
        self.t1 = OrderedDict()
        self.t2 = OrderedDict()
        self.b1 = OrderedDict()
        self.b2 = OrderedDict()
        self.hits = 0
        self.misses = 0
        self.total_accesses = 0

    def _replace(self, key: int, in_b1: bool) -> None:
        if len(self.t1) > 0 and (len(self.t1) > self.p or (len(self.t1) == self.p and not in_b1)):
            old_key, _ = self.t1.popitem(last=False)
            self.b1[old_key] = None
        elif len(self.t2) > 0:
            old_key, _ = self.t2.popitem(last=False)
            self.b2[old_key] = None

    def get(self, key: int) -> int:
        self.total_accesses += 1

        if key in self.t1:
            self.hits += 1
            del self.t1[key]
            self.t2[key] = None
            return key

        if key in self.t2:
            self.hits += 1
            self.t2.move_to_end(key)
            return key

        self.misses += 1

        if key in self.b1:
            delta = max(1, len(self.b2) // max(1, len(self.b1)))
            self.p = min(self.capacity, self.p + delta)

            while len(self.t1) + len(self.t2) >= self.capacity:
                self._replace(key, True)

            del self.b1[key]
            self.t2[key] = None
            return -1

        if key in self.b2:
            delta = max(1, len(self.b1) // max(1, len(self.b2)))
            self.p = max(0, self.p - delta)

            while len(self.t1) + len(self.t2) >= self.capacity:
                self._replace(key, False)

            del self.b2[key]
            self.t2[key] = None
            return -1

        total = len(self.t1) + len(self.t2) + len(self.b1) + len(self.b2)
        if total >= self.capacity * 2:
            if len(self.b1) > 0:
                self.b1.popitem(last=False)
            elif len(self.b2) > 0:
                self.b2.popitem(last=False)

        while len(self.t1) + len(self.t2) >= self.capacity:
            self._replace(key, False)

        self.t1[key] = None
        return -1

    def put(self, key: int, value: int) -> None:
        pass

    def get_hit_rate(self) -> float:
        if self.total_accesses == 0:
            return 0.0
        return self.hits / self.total_accesses

    def get_stats(self) -> dict:
        return {
            'strategy': 'ARC',
            'capacity': self.capacity,
            'total_accesses': self.total_accesses,
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': self.get_hit_rate(),
            'current_cache': list(self.t1.keys()) + list(self.t2.keys()),
            't1_size': len(self.t1),
            't2_size': len(self.t2),
            'b1_size': len(self.b1),
            'b2_size': len(self.b2),
            'p': self.p
        }


class TwoQCache:
    def __init__(self, capacity: int, kin_ratio: float = 0.25, kout_ratio: float = 4.0):
        self.capacity = capacity
        self.kin = max(1, int(capacity * kin_ratio))
        self.kout = max(capacity, int(capacity * kout_ratio))
        self.a1in = OrderedDict()
        self.a1out = OrderedDict()
        self.am = OrderedDict()
        self.hits = 0
        self.misses = 0
        self.total_accesses = 0

    def get(self, key: int) -> int:
        self.total_accesses += 1

        if key in self.a1in:
            self.hits += 1
            return self.a1in[key]

        if key in self.am:
            self.hits += 1
            self.am.move_to_end(key)
            return self.am[key]

        if key in self.a1out:
            self.misses += 1
            del self.a1out[key]

            if len(self.am) >= self.capacity - len(self.a1in):
                self.am.popitem(last=False)

            self.am[key] = key
            return -1

        self.misses += 1

        if len(self.a1in) >= self.kin:
            old_key, _ = self.a1in.popitem(last=False)
            self.a1out[old_key] = None
            if len(self.a1out) > self.kout:
                self.a1out.popitem(last=False)

        self.a1in[key] = key
        return -1

    def put(self, key: int, value: int) -> None:
        pass

    def get_hit_rate(self) -> float:
        if self.total_accesses == 0:
            return 0.0
        return self.hits / self.total_accesses

    def get_stats(self) -> dict:
        return {
            'strategy': '2Q',
            'capacity': self.capacity,
            'total_accesses': self.total_accesses,
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': self.get_hit_rate(),
            'current_cache': list(self.a1in.keys()) + list(self.am.keys()),
            'a1in_size': len(self.a1in),
            'a1out_size': len(self.a1out),
            'am_size': len(self.am),
            'kin': self.kin,
            'kout': self.kout
        }


def simulate_cache(access_sequence: list, capacity: int, verbose: bool = False) -> dict:
    fifo_cache = FIFOCache(capacity)
    lru_cache = LRUCache(capacity)
    lfu_cache = LFUCache(capacity)
    lfu_aging_cache = LFUAgingCache(capacity, decay_factor=0.5, decay_interval=10)
    window_lfu_cache = WindowLFUCache(capacity, window_size=10)
    arc_cache = ARCCache(capacity)
    twoq_cache = TwoQCache(capacity)

    caches = [fifo_cache, lru_cache, lfu_cache, lfu_aging_cache,
              window_lfu_cache, arc_cache, twoq_cache]

    for i, key in enumerate(access_sequence):
        if verbose:
            print(f"\n访问 {i + 1}: 键 = {key}")

        for cache in caches:
            result = cache.get(key)
            if result == -1:
                cache.put(key, key)
            if verbose:
                stats = cache.get_stats()
                print(f"  {stats['strategy']}: {'命中' if result != -1 else '未命中'}, "
                      f"缓存 = {stats['current_cache']}")

    return {
        'FIFO': fifo_cache.get_stats(),
        'LRU': lru_cache.get_stats(),
        'LFU': lfu_cache.get_stats(),
        'LFU-Aging': lfu_aging_cache.get_stats(),
        'Window-LFU': window_lfu_cache.get_stats(),
        'ARC': arc_cache.get_stats(),
        '2Q': twoq_cache.get_stats()
    }


def print_comparison(results: dict) -> None:
    print("\n" + "=" * 90)
    print("缓存淘汰策略命中率对比".center(90))
    print("=" * 90)

    strategies = ['FIFO', 'LRU', 'LFU', 'LFU-Aging', 'Window-LFU', 'ARC', '2Q']
    headers = ['策略', '容量', '总访问', '命中', '未命中', '命中率']
    print(f"{headers[0]:<12} {headers[1]:<6} {headers[2]:<8} {headers[3]:<6} {headers[4]:<8} {headers[5]:<10}")
    print("-" * 90)

    for strategy in strategies:
        stats = results[strategy]
        print(f"{stats['strategy']:<12} {stats['capacity']:<6} {stats['total_accesses']:<8} "
              f"{stats['hits']:<6} {stats['misses']:<8} {stats['hit_rate']:<10.2%}")

    print("=" * 90)

    best = max(strategies, key=lambda s: results[s]['hit_rate'])
    print(f"\n最优策略: {best} (命中率: {results[best]['hit_rate']:.2%})")

    print("\n各策略最终缓存内容:")
    for strategy in strategies:
        stats = results[strategy]
        print(f"  {strategy}: {stats['current_cache']}")
        if 'key_frequencies' in stats:
            print(f"       访问频率: {stats['key_frequencies']}")
        if 'window_frequencies' in stats:
            print(f"       窗口频率: {stats['window_frequencies']}")
        if 't1_size' in stats:
            print(f"       T1(近期)={stats['t1_size']} T2(频繁)={stats['t2_size']} "
                  f"B1(近期幽灵)={stats['b1_size']} B2(频繁幽灵)={stats['b2_size']} "
                  f"p={stats['p']}")
        if 'a1in_size' in stats:
            print(f"       A1in(准入)={stats['a1in_size']} A1out(候选)={stats['a1out_size']} "
                  f"Am(主存)={stats['am_size']}")


def recommend_strategy(all_results: list) -> None:
    strategies = ['FIFO', 'LRU', 'LFU', 'LFU-Aging', 'Window-LFU', 'ARC', '2Q']

    print("\n\n" + "=" * 90)
    print("多工作负载策略推荐".center(90))
    print("=" * 90)

    avg_hit_rates = {}
    win_counts = {s: 0 for s in strategies}
    worst_hit_rates = {s: 1.0 for s in strategies}

    for results in all_results:
        best_strategy = max(strategies, key=lambda s: results[s]['hit_rate'])
        win_counts[best_strategy] += 1
        for s in strategies:
            worst_hit_rates[s] = min(worst_hit_rates[s], results[s]['hit_rate'])

    for s in strategies:
        rates = [r[s]['hit_rate'] for r in all_results]
        avg_hit_rates[s] = sum(rates) / len(rates)

    print(f"\n{'策略':<12} {'平均命中率':<12} {'最低命中率':<12} {'获胜次数':<10} {'推荐指数'}")
    print("-" * 90)

    for s in sorted(strategies, key=lambda x: avg_hit_rates[x], reverse=True):
        stability = "稳定" if worst_hit_rates[s] > 0.3 else ("中等" if worst_hit_rates[s] > 0.15 else "不稳定")
        score = avg_hit_rates[s] * 0.6 + worst_hit_rates[s] * 0.25 + (win_counts[s] / len(all_results)) * 0.15
        print(f"{s:<12} {avg_hit_rates[s]:<12.2%} {worst_hit_rates[s]:<12.2%} {win_counts[s]:<10} "
              f"{score:.2%} ({stability})")

    recommended = max(strategies, key=lambda s: avg_hit_rates[s] * 0.6 + worst_hit_rates[s] * 0.25 +
                      (win_counts[s] / len(all_results)) * 0.15)
    print(f"\n综合推荐策略: {recommended}")
    print(f"  平均命中率: {avg_hit_rates[recommended]:.2%}")
    print(f"  最低命中率: {worst_hit_rates[recommended]:.2%}")

    print("\n策略选择建议:")
    if avg_hit_rates['ARC'] >= avg_hit_rates['LRU'] and avg_hit_rates['ARC'] >= avg_hit_rates['LFU']:
        print("  ARC自适应能力强，适合工作负载多变且难以预测的场景")
    if avg_hit_rates['2Q'] >= avg_hit_rates['LRU']:
        print("  2Q适合存在明显冷热区分、需要过滤一次性访问的场景")
    if avg_hit_rates['LFU-Aging'] > avg_hit_rates['LFU']:
        print("  LFU-Aging适合访问模式会随时间变化的场景(优于原始LFU)")
    if avg_hit_rates['LRU'] >= avg_hit_rates['LFU']:
        print("  LRU适合局部性强的场景(近期访问的数据大概率再次访问)")
    else:
        print("  LFU适合访问分布稳定的场景(热点数据长期不变)")


def run_test_cases():
    import random
    random.seed(42)

    phase1_12 = [1, 1, 1, 2, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2]
    phase2_34 = [3, 3, 4, 4, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 3]
    phase3_56 = [5, 5, 6, 6, 5, 6, 5, 6, 5, 6, 5, 6, 5, 6, 5]
    shift_sequence = phase1_12 + phase2_34 + phase3_56

    zipf_keys = list(range(1, 21))
    zipf_weights = [1.0 / (i ** 0.8) for i in range(1, 21)]
    zipf_total = sum(zipf_weights)
    zipf_probs = [w / zipf_total for w in zipf_weights]
    zipf_sequence = random.choices(zipf_keys, weights=zipf_probs, k=100)

    all_results = []

    test_cases = [
        {
            'name': '测试用例 1: 顺序重复访问',
            'sequence': [1, 2, 3, 4, 1, 2, 5, 1, 2, 3, 4, 5],
            'capacity': 3
        },
        {
            'name': '测试用例 2: 循环访问模式',
            'sequence': [1, 2, 3, 1, 2, 3, 1, 2, 3, 4, 4, 4],
            'capacity': 3
        },
        {
            'name': '测试用例 3: 热点数据',
            'sequence': [1, 1, 1, 2, 2, 3, 1, 1, 2, 2, 3, 3, 4, 5],
            'capacity': 3
        },
        {
            'name': '测试用例 4: Belady异常验证',
            'sequence': [1, 2, 3, 4, 1, 2, 5, 1, 2, 3, 4, 5],
            'capacity': 4
        },
        {
            'name': '测试用例 5: 访问模式变化 (经典LFU问题场景)',
            'sequence': shift_sequence,
            'capacity': 2,
            'description': '前15次访问1/2，中间15次访问3/4，最后15次访问5/6'
        },
        {
            'name': '测试用例 6: 工作集迁移场景',
            'sequence': [1, 2, 3, 1, 2, 3, 1, 2, 3, 4, 5, 6, 4, 5, 6, 4, 5, 6, 7, 8, 9, 7, 8, 9, 7, 8, 9],
            'capacity': 3
        },
        {
            'name': '测试用例 7: Zipf分布 (模拟真实Web缓存)',
            'sequence': zipf_sequence,
            'capacity': 5,
            'description': '100次访问，20个key，访问频率服从Zipf分布(少数热点)'
        },
        {
            'name': '测试用例 8: 顺序扫描+热点回访 (2Q优势场景)',
            'sequence': (list(range(1, 51)) +
                        [1, 1, 2, 2, 3, 3, 1, 2, 3] +
                        list(range(51, 71)) +
                        [1, 2, 3, 1, 2, 3]),
            'capacity': 5,
            'description': '全量扫描(污染缓存) + 热点回访 + 再扫描 + 再回访，2Q应能过滤一次性扫描'
        }
    ]

    for test in test_cases:
        print(f"\n\n{test['name']}")
        print("-" * 90)
        if 'description' in test:
            print(f"说明: {test['description']}")
        seq_display = test['sequence'][:30]
        if len(test['sequence']) > 30:
            seq_display = list(seq_display) + ['...']
        print(f"访问序列: {seq_display} (共{len(test['sequence'])}次)")
        print(f"缓存容量: {test['capacity']}")

        results = simulate_cache(test['sequence'], test['capacity'], verbose=False)
        print_comparison(results)
        all_results.append(results)

    recommend_strategy(all_results)


if __name__ == '__main__':
    print("缓存淘汰策略模拟器")
    print("支持策略: FIFO, LRU, LFU, LFU-Aging, Window-LFU, ARC, 2Q")
    print("\n策略说明:")
    print("  - FIFO:     先进先出，最简单的淘汰策略")
    print("  - LRU:      最近最少使用，基于时间局部性")
    print("  - LFU:      最不经常使用，基于频率统计(存在缓存污染问题)")
    print("  - LFU-Aging: 定期衰减频率(每10次衰减50%)，解决LFU缓存污染")
    print("  - Window-LFU: 滑动窗口计数(窗口10)，只统计近期频率")
    print("  - ARC:      自适应替换缓存，同时学习LRU和LFU，自动调节比例")
    print("  - 2Q:       双队列策略，FIFO准入过滤+LRU主存，过滤一次性访问")

    custom = input("\n是否使用自定义访问序列? (y/n): ").strip().lower()

    if custom == 'y':
        try:
            sequence = list(map(int, input("请输入访问序列 (空格分隔数字): ").split()))
            capacity = int(input("请输入缓存容量: "))

            verbose = input("是否显示详细访问过程? (y/n): ").strip().lower() == 'y'

            print(f"\n访问序列: {sequence}")
            print(f"缓存容量: {capacity}")

            results = simulate_cache(sequence, capacity, verbose=verbose)
            print_comparison(results)
        except ValueError:
            print("输入错误，请输入有效的数字序列。")
    else:
        run_test_cases()
