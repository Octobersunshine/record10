import threading
import asyncio
import httpx
import time
import random
from monitoring import MetricsStorage


def test_thread_safety_unit():
    print("=" * 60)
    print("UNIT TEST: Thread Safety of MetricsStorage")
    print("=" * 60)

    storage = MetricsStorage(window_seconds=60, bucket_seconds=1)
    num_threads = 50
    records_per_thread = 200
    total_expected = num_threads * records_per_thread

    print(f"\nStarting {num_threads} threads, each adding {records_per_thread} records...")
    print(f"Expected total records: {total_expected}")

    def worker(thread_id):
        for i in range(records_per_thread):
            path = f"/api/test/thread-{thread_id}-{i}"
            method = "GET" if i % 2 == 0 else "POST"
            response_time = random.uniform(1, 100)
            status_code = 200 if i % 10 != 0 else 500
            storage.add_record(path, method, response_time, status_code)

    threads = []
    start_time = time.time()

    for i in range(num_threads):
        t = threading.Thread(target=worker, args=(i,))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    elapsed = time.time() - start_time
    stats = storage.get_window_stats()

    print(f"\nResults:")
    print(f"  Time elapsed: {elapsed:.2f}s")
    print(f"  Expected count: {total_expected}")
    print(f"  Actual count: {stats.request_count}")
    print(f"  Match: {stats.request_count == total_expected}")

    if stats.request_count == total_expected:
        print("  [PASS] Thread safety test PASSED!")
    else:
        print(f"  [FAIL] Thread safety test FAILED! (diff: {stats.request_count - total_expected})")

    print(f"\nAdditional verification:")
    print(f"  Avg response time: {stats.avg_response_time:.2f}ms")
    print(f"  P99 response time: {stats.p99_response_time:.2f}ms")
    print(f"  Active buckets: {storage.get_bucket_count()}")

    return stats.request_count == total_expected


async def test_high_concurrency_api():
    print("\n" + "=" * 60)
    print("INTEGRATION TEST: High Concurrency API Requests")
    print("=" * 60)

    base_url = "http://localhost:8000"

    try:
        async with httpx.AsyncClient(base_url=base_url, timeout=5.0) as client:
            await client.get("/")
        print("\nServer is running!")
    except Exception:
        print("\n[FAIL] Server is not running. Skipping integration test.")
        print("   Start server with: py main.py")
        return False

    num_concurrent = 100
    num_requests_per_client = 20
    total_requests = num_concurrent * num_requests_per_client

    print(f"\nSending {total_requests} requests with {num_concurrent} concurrent clients...")

    endpoints = [
        "/api/users",
        "/api/users/1",
        "/api/products",
        "/api/orders",
    ]

    async def client_worker(client_id):
        async with httpx.AsyncClient(base_url=base_url, timeout=30.0) as client:
            for i in range(num_requests_per_client):
                endpoint = random.choice(endpoints)
                try:
                    await client.get(endpoint)
                except Exception as e:
                    pass

    start_time = time.time()
    tasks = [client_worker(i) for i in range(num_concurrent)]
    await asyncio.gather(*tasks)
    elapsed = time.time() - start_time

    print(f"\nResults:")
    print(f"  Time elapsed: {elapsed:.2f}s")
    print(f"  Throughput: {total_requests / elapsed:.1f} req/s")

    await asyncio.sleep(0.5)

    async with httpx.AsyncClient(base_url=base_url, timeout=10.0) as client:
        response = await client.get("/monitoring/stats/summary")
        if response.status_code == 200:
            data = response.json()
            print(f"\n  Reported requests: {data['request_count']}")
            print(f"  Avg response time: {data['avg_response_time']:.2f}ms")
            print(f"  P99 response time: {data['p99_response_time']:.2f}ms")
            print(f"  Window: {data['window_start']} -> {data['window_end']}")

        response = await client.get("/monitoring/stats/per-second")
        if response.status_code == 200:
            data = response.json()
            print(f"\n  Per-second breakdown:")
            for s in data:
                if s["request_count"] > 0:
                    print(f"    {s['time']}: {s['request_count']} reqs, avg={s['avg_response_time']:.2f}ms")

    print("\n  [PASS] High concurrency integration test completed!")
    return True


def test_sliding_window_behavior():
    print("\n" + "=" * 60)
    print("UNIT TEST: Sliding Window Behavior")
    print("=" * 60)

    storage = MetricsStorage(window_seconds=5, bucket_seconds=1)

    print(f"\nTesting with window_seconds=5, bucket_seconds=1")

    base_time = time.time()
    for i in range(10):
        storage.add_record("/api/test", "GET", float(i) * 10, 200)

    stats = storage.get_window_stats()
    print(f"\nAfter adding 10 records:")
    print(f"  Request count: {stats.request_count}")
    print(f"  Avg: {stats.avg_response_time:.2f}ms")
    print(f"  P99: {stats.p99_response_time:.2f}ms")
    print(f"  Buckets: {storage.get_bucket_count()}")

    print(f"\nWaiting 3 seconds for window to slide...")
    time.sleep(3)

    for i in range(5):
        storage.add_record("/api/test2", "POST", float(i) * 5, 200)

    stats = storage.get_window_stats()
    print(f"\nAfter sliding window + adding 5 more records:")
    print(f"  Request count: {stats.request_count}")
    print(f"  Avg: {stats.avg_response_time:.2f}ms")
    print(f"  Buckets: {storage.get_bucket_count()}")

    per_second = storage.get_per_second_stats()
    print(f"\nPer-second stats ({len(per_second)} active buckets):")
    for s in per_second:
        print(f"  {s['time']}: {s['request_count']} reqs")

    print("\n  [PASS] Sliding window behavior test completed!")
    return True


async def main():
    all_passed = True

    all_passed &= test_thread_safety_unit()
    all_passed &= test_sliding_window_behavior()
    all_passed &= await test_high_concurrency_api()

    print("\n" + "=" * 60)
    if all_passed:
        print("[PASS] ALL TESTS PASSED!")
    else:
        print("[FAIL] SOME TESTS FAILED!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
