import time
import sys
import gc
from concurrent.futures import ThreadPoolExecutor, as_completed

from rate_limiter import RateLimiter


def get_obj_size(obj):
    """估算对象内存占用"""
    seen = set()

    def sizeof(o):
        if id(o) in seen:
            return 0
        seen.add(id(o))
        s = sys.getsizeof(o)
        if isinstance(o, dict):
            for k, v in o.items():
                s += sizeof(k) + sizeof(v)
        elif isinstance(o, (list, tuple, set, frozenset)):
            for item in o:
                s += sizeof(item)
        elif hasattr(o, '__dict__'):
            s += sizeof(o.__dict__)
        elif hasattr(o, '__slots__'):
            for slot in o.__slots__:
                if hasattr(o, slot):
                    s += sizeof(getattr(o, slot))
        return s

    return sizeof(obj)


def print_separator(title=""):
    print("\n" + "=" * 80)
    if title:
        print(f"  {title}")
        print("=" * 80)


def test_algorithm_memory(algorithm_name, algorithm, num_requests=10000, qps_limit=100):
    """测试单个算法的内存占用"""
    print(f"\n  Testing: {algorithm_name}")
    print(f"  Request count: {num_requests:,}")
    print(f"  QPS limit: {qps_limit}")

    try:
        limiter = RateLimiter(algorithm=algorithm, default_qps=qps_limit, window_size=1)
    except ImportError as e:
        print(f"  ⚠ Skipped: {e}")
        return None

    gc.collect()
    time.sleep(0.1)

    initial_memory = get_obj_size(limiter._limiter)

    print(f"\n  Phase 1: Sending {num_requests:,} requests...")
    start_time = time.time()

    for i in range(num_requests):
        limiter.allow_request('/api/test')

    elapsed = time.time() - start_time
    gc.collect()
    time.sleep(0.1)

    peak_memory = get_obj_size(limiter._limiter)
    throughput = num_requests / elapsed

    print(f"    Time: {elapsed:.2f}s")
    print(f"    Throughput: {throughput:,.0f} req/s")

    print(f"\n  Phase 2: Sending another {num_requests:,} requests...")
    start_time = time.time()

    for i in range(num_requests):
        limiter.allow_request('/api/test')

    elapsed = time.time() - start_time
    gc.collect()
    time.sleep(0.1)

    final_memory = get_obj_size(limiter._limiter)
    total_requests = num_requests * 2
    total_elapsed = elapsed + (time.time() - start_time - elapsed) + (start_time - start_time + elapsed)

    memory_growth = final_memory - initial_memory
    memory_growth_ratio = (memory_growth / initial_memory * 100) if initial_memory > 0 else 0

    stats = limiter.get_stats('/api/test')
    memory_info = limiter.get_memory_info() if hasattr(limiter, 'get_memory_info') else {}

    print(f"    Time: {elapsed:.2f}s")
    print(f"    Total requests: {total_requests:,}")

    print(f"\n  Memory Analysis:")
    print(f"    Initial: {initial_memory:,} bytes")
    print(f"    After {num_requests:,} reqs: {peak_memory:,} bytes")
    print(f"    After {total_requests:,} reqs: {final_memory:,} bytes")
    print(f"    Growth: {memory_growth:+,} bytes ({memory_growth_ratio:+.1f}%)")
    print(f"    Memory fixed: {memory_info.get('memory_fixed', 'N/A')}")

    if hasattr(limiter._limiter, '_timestamps'):
        deque_size = len(limiter._limiter._timestamps.get('/api/test', []))
        print(f"    Deque size: {deque_size:,} entries")

    if hasattr(limiter._limiter, '_arrays'):
        array = limiter._limiter._arrays.get('/api/test')
        if array:
            print(f"    Array capacity: {array.capacity:,}")
            print(f"    Array size: {len(array):,}")

    del limiter
    gc.collect()

    return {
        'algorithm': algorithm_name,
        'initial_memory': initial_memory,
        'final_memory': final_memory,
        'memory_growth': memory_growth,
        'memory_growth_pct': memory_growth_ratio,
        'total_requests': total_requests,
        'memory_fixed': memory_info.get('memory_fixed', False)
    }


def test_concurrent_memory(algorithm_name, algorithm, num_workers=20, requests_per_worker=1000):
    """测试高并发下的内存表现"""
    print(f"\n  Concurrent Test: {algorithm_name}")
    print(f"  Workers: {num_workers}, Requests/worker: {requests_per_worker}")
    print(f"  Total requests: {num_workers * requests_per_worker:,}")

    try:
        limiter = RateLimiter(algorithm=algorithm, default_qps=1000, window_size=1)
    except ImportError as e:
        print(f"  ⚠ Skipped: {e}")
        return None

    gc.collect()
    time.sleep(0.1)
    initial_memory = get_obj_size(limiter._limiter)

    def worker():
        for _ in range(requests_per_worker):
            limiter.allow_request('/api/concurrent')

    print(f"\n  Running concurrent requests...")
    start_time = time.time()

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(worker) for _ in range(num_workers)]
        for future in as_completed(futures):
            future.result()

    elapsed = time.time() - start_time
    gc.collect()
    time.sleep(0.1)

    final_memory = get_obj_size(limiter._limiter)
    total_requests = num_workers * requests_per_worker
    throughput = total_requests / elapsed

    print(f"    Time: {elapsed:.2f}s")
    print(f"    Throughput: {throughput:,.0f} req/s")
    print(f"    Initial memory: {initial_memory:,} bytes")
    print(f"    Final memory: {final_memory:,} bytes")
    print(f"    Growth: {final_memory - initial_memory:+,} bytes")

    memory_info = limiter.get_memory_info() if hasattr(limiter, 'get_memory_info') else {}
    print(f"    Memory fixed: {memory_info.get('memory_fixed', 'N/A')}")

    del limiter
    gc.collect()

    return {
        'initial_memory': initial_memory,
        'final_memory': final_memory,
        'memory_growth': final_memory - initial_memory,
        'throughput': throughput
    }


def main():
    print("\n" + "=" * 80)
    print("  MEMORY OPTIMIZATION TEST SUITE")
    print("=" * 80)
    print("  Comparing memory usage between algorithms")
    print("=" * 80)

    results = []

    algorithms = [
        ('Original Sliding Window (deque)',
         RateLimiter.ALGORITHM_SLIDING_WINDOW,
         'Memory grows with each request'),
        ('Optimized Sliding Window (circular array)',
         RateLimiter.ALGORITHM_SLIDING_WINDOW_CIRCULAR,
         'Fixed O(1) memory'),
        ('Token Bucket',
         RateLimiter.ALGORITHM_TOKEN_BUCKET,
         'Fixed O(1) memory per API'),
    ]

    print_separator("PHASE 1: Sequential Memory Test")
    print("\n  Testing sequential requests to observe memory growth patterns")
    print("  WARNING: Original deque version will use significant memory!")

    for name, algo, note in algorithms:
        print(f"\n  Algorithm: {name}")
        print(f"  Note: {note}")
        result = test_algorithm_memory(name, algo, num_requests=20000, qps_limit=100)
        if result:
            results.append(result)
        print(f"\n  {'-' * 60}")

    print_separator("PHASE 2: Memory Growth Comparison Summary")

    print(f"\n  {'Algorithm':<45} {'Initial':>12} {'Final':>12} {'Growth':>12} {'Fixed':>8}")
    print("  " + "-" * 90)

    for r in results:
        if r is None:
            continue
        fixed_mark = "✓" if r['memory_fixed'] else "✗"
        print(f"  {r['algorithm']:<45} {r['initial_memory']:>10,} B {r['final_memory']:>10,} B {r['memory_growth']:>+10,} B {fixed_mark:>8}")

    print_separator("PHASE 3: High Concurrency Memory Test")
    print("\n  Testing memory under concurrent load (20 workers x 1000 requests)")

    concurrent_results = {}
    for name, algo, note in algorithms:
        result = test_concurrent_memory(name, algo, num_workers=20, requests_per_worker=1000)
        if result:
            concurrent_results[name] = result
        print(f"\n  {'-' * 60}")

    print_separator("PHASE 4: Concurrent Test Summary")

    print(f"\n  {'Algorithm':<45} {'Initial':>12} {'Final':>12} {'Growth':>12} {'Throughput':>15}")
    print("  " + "-" * 100)

    for name, r in concurrent_results.items():
        print(f"  {name:<45} {r['initial_memory']:>10,} B {r['final_memory']:>10,} B {r['memory_growth']:>+10,} B {r['throughput']:>12,.0f} req/s")

    print_separator("PHASE 5: Memory Optimization Analysis")

    print("\n  Key Findings:")
    print("  " + "-" * 80)

    deque_result = next((r for r in results if r and 'Original' in r['algorithm']), None)
    circular_result = next((r for r in results if r and 'circular' in r['algorithm']), None)

    if deque_result and circular_result:
        deque_growth = deque_result['memory_growth']
        circular_growth = circular_result['memory_growth']

        print(f"\n  1. Original deque version:")
        print(f"     - Memory growth: {deque_growth:+,} bytes")
        print(f"     - Problem: Each request adds a timestamp to the deque")
        print(f"     - Memory complexity: O(N) where N = number of requests")

        print(f"\n  2. Optimized circular array version:")
        print(f"     - Memory growth: {circular_growth:+,} bytes")
        print(f"     - Solution: Fixed-size array pre-allocated at initialization")
        print(f"     - Memory complexity: O(1) regardless of request count")
        print(f"     - Array is reused in circular fashion, oldest entries are overwritten")

        if deque_growth > 0 and circular_growth <= 0:
            savings_pct = 100
            print(f"\n  3. Memory Savings: {savings_pct}%!")
            print(f"     - The optimized version uses FIXED memory")
            print(f"     - No matter how many requests are processed")
            print(f"     - Critical for long-running high-concurrency services")

    print(f"\n  4. When to use which algorithm:")
    print(f"     ✓ sliding_window_circular: Default, best for most cases")
    print(f"       - Fixed memory, precise QPS control")
    print(f"       - O(1) memory, no growth")
    print(f"     ✓ token_bucket: When burst traffic support is needed")
    print(f"       - Allows temporary traffic spikes")
    print(f"     ✓ sliding_window_redis: Distributed deployments")
    print(f"       - Uses Redis sorted set (ZSET)")
    print(f"       - Rate limit shared across multiple server instances")
    print(f"     ✗ sliding_window (original): NOT RECOMMENDED")
    print(f"       - Memory grows linearly with requests")
    print(f"       - Use only for low-traffic scenarios")

    print_separator("TEST COMPLETE")

    print("\n  Recommendations:")
    print(f"  - Use sliding_window_circular as the default algorithm")
    print(f"  - Monitor memory via GET /api/memory endpoint")
    print(f"  - For distributed systems, use sliding_window_redis with Redis")
    print(f"  - Avoid original sliding_window for high-concurrency scenarios")
    print("\n" + "=" * 80 + "\n")


if __name__ == "__main__":
    main()
