import time
import threading
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

BASE_URL = "http://127.0.0.1:5000"


def print_separator(title=""):
    print("\n" + "=" * 70)
    if title:
        print(f"  {title}")
        print("=" * 70)


def test_health_check():
    print_separator("1. Health Check")
    response = requests.get(f"{BASE_URL}/api/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")
    return response.status_code == 200


def test_set_limit(api_path, method, qps, burst=None):
    print(f"\n  Setting limit for {method}:{api_path} to {qps} QPS...")
    data = {"api_path": api_path, "method": method, "qps": qps}
    if burst:
        data["burst"] = burst
    response = requests.post(f"{BASE_URL}/api/limits", json=data)
    print(f"  Status: {response.status_code}")
    print(f"  Response: {response.json()}")
    return response.status_code in [200, 201]


def test_get_limits():
    print_separator("2. Get All Limits")
    response = requests.get(f"{BASE_URL}/api/limits")
    print(f"Status: {response.status_code}")
    data = response.json()
    print(f"Algorithm: {data['data']['algorithm']}")
    print(f"Default QPS: {data['data']['default_qps']}")
    print(f"Limits: {data['data']['limits']}")
    return response.status_code == 200


def test_rate_limit_trigger(api_path, method, qps, duration=2):
    print_separator(f"3. Rate Limit Test - {method} {api_path} (limit: {qps} QPS)")

    url = f"{BASE_URL}{api_path}"
    success_count = 0
    rate_limited_count = 0
    start_time = time.time()
    end_time = start_time + duration

    print(f"\n  Sending requests for {duration} seconds...")

    while time.time() < end_time:
        if method == "GET":
            response = requests.get(url)
        else:
            response = requests.post(url, json={"name": "Test"})

        if response.status_code == 200 or response.status_code == 201:
            success_count += 1
        elif response.status_code == 429:
            rate_limited_count += 1
        else:
            print(f"  Unexpected status: {response.status_code}")

        time.sleep(0.01)

    actual_qps = success_count / duration
    print(f"\n  Results:")
    print(f"    Total requests: {success_count + rate_limited_count}")
    print(f"    Success (2xx): {success_count}")
    print(f"    Rate Limited (429): {rate_limited_count}")
    print(f"    Actual QPS: {actual_qps:.2f}")
    print(f"    Expected QPS limit: {qps}")

    if rate_limited_count > 0:
        print(f"    ✓ Rate limiting is working!")
        if response.status_code == 429:
            data = response.json()
            print(f"    429 Response: {data}")
            print(f"    Headers: X-RateLimit-Limit={response.headers.get('X-RateLimit-Limit')}, "
                  f"Retry-After={response.headers.get('Retry-After')}")
    else:
        print(f"    ⚠ No rate limiting triggered, consider increasing request rate")

    return rate_limited_count > 0


def test_concurrent_requests(api_path, method, qps, num_requests=50):
    print_separator(f"4. Concurrent Request Test - {method} {api_path} ({num_requests} requests)")

    url = f"{BASE_URL}{api_path}"
    success_count = 0
    rate_limited_count = 0

    def make_request():
        if method == "GET":
            return requests.get(url)
        return requests.post(url, json={"name": "Concurrent"})

    print(f"\n  Sending {num_requests} concurrent requests...")
    start_time = time.time()

    with ThreadPoolExecutor(max_workers=20) as executor:
        futures = [executor.submit(make_request) for _ in range(num_requests)]
        for future in as_completed(futures):
            response = future.result()
            if response.status_code in [200, 201]:
                success_count += 1
            elif response.status_code == 429:
                rate_limited_count += 1

    elapsed = time.time() - start_time
    print(f"\n  Results:")
    print(f"    Time elapsed: {elapsed:.2f}s")
    print(f"    Success (2xx): {success_count}")
    print(f"    Rate Limited (429): {rate_limited_count}")
    print(f"    Throughput: {num_requests / elapsed:.2f} req/s")
    print(f"    Success rate: {(success_count / num_requests) * 100:.1f}%")

    return rate_limited_count > 0


def test_switch_algorithm(algorithm, **kwargs):
    print_separator(f"5. Switch Algorithm to: {algorithm}")
    data = {"algorithm": algorithm}
    data.update(kwargs)
    response = requests.put(f"{BASE_URL}/api/algorithm", json=data)
    print(f"Status: {response.status_code}")
    result = response.json()
    print(f"Response: {result}")

    if response.status_code == 200:
        print(f"\n  Verifying algorithm change...")
        response2 = requests.get(f"{BASE_URL}/api/algorithm")
        data2 = response2.json()
        print(f"  Current algorithm: {data2['data']['current_algorithm']}")
        print(f"  ✓ Algorithm switched successfully!")

    return response.status_code == 200


def test_get_stats(api_path, method):
    print(f"\n  Getting stats for {method}:{api_path}...")
    response = requests.get(f"{BASE_URL}/api/stats{api_path}", params={"method": method})
    print(f"  Status: {response.status_code}")
    print(f"  Stats: {response.json()}")
    return response.status_code == 200


def test_reset_stats():
    print_separator("6. Reset Stats")
    response = requests.post(f"{BASE_URL}/api/reset")
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")
    return response.status_code == 200


def test_rate_limit_headers():
    print_separator("7. Rate Limit Headers Test")
    url = f"{BASE_URL}/api/users"

    print("\n  Making request to check headers...")
    response = requests.get(url)

    if response.status_code == 200:
        headers = response.headers
        print(f"\n  Response Headers:")
        print(f"    X-RateLimit-Limit: {headers.get('X-RateLimit-Limit')}")
        print(f"    X-RateLimit-Remaining: {headers.get('X-RateLimit-Remaining')}")
        print(f"    X-RateLimit-Used: {headers.get('X-RateLimit-Used')}")
        print(f"    X-RateLimit-Algorithm: {headers.get('X-RateLimit-Algorithm')}")

        required_headers = ['X-RateLimit-Limit', 'X-RateLimit-Remaining',
                           'X-RateLimit-Used', 'X-RateLimit-Algorithm']
        all_present = all(h in headers for h in required_headers)
        if all_present:
            print(f"\n  ✓ All rate limit headers present!")
        else:
            print(f"\n  ✗ Missing some headers")

        return all_present
    else:
        print(f"  Unexpected status: {response.status_code}")
        return False


def test_different_methods():
    print_separator("8. Different HTTP Methods Test")
    print("\n  Testing GET /api/users limit...")
    test_set_limit("/api/users", "GET", 3)
    test_get_stats("/api/users", "GET")

    print("\n  Testing POST /api/users limit...")
    test_set_limit("/api/users", "POST", 2)
    test_get_stats("/api/users", "POST")

    print("\n  Making GET requests...")
    success = 0
    limited = 0
    for i in range(5):
        response = requests.get(f"{BASE_URL}/api/users")
        if response.status_code == 200:
            success += 1
        elif response.status_code == 429:
            limited += 1
    print(f"  GET - Success: {success}, Limited: {limited}")

    print("\n  Making POST requests...")
    success = 0
    limited = 0
    for i in range(5):
        response = requests.post(f"{BASE_URL}/api/users", json={"name": "Test"})
        if response.status_code == 201:
            success += 1
        elif response.status_code == 429:
            limited += 1
    print(f"  POST - Success: {success}, Limited: {limited}")

    return True


def test_remove_limit():
    print_separator("9. Remove Limit Test")

    print("\n  Setting limit first...")
    test_set_limit("/api/test", "GET", 5)

    print("\n  Removing limit...")
    response = requests.delete(f"{BASE_URL}/api/limits/api/test", params={"method": "GET"})
    print(f"  Status: {response.status_code}")
    print(f"  Response: {response.json()}")

    print("\n  Verifying removal...")
    response = requests.get(f"{BASE_URL}/api/limits")
    data = response.json()
    if "GET:/api/test" not in data["data"]["limits"]:
        print("  ✓ Limit removed successfully!")
        return True
    else:
        print("  ✗ Limit still exists")
        return False


def main():
    print("\n" + "=" * 70)
    print("  API RATE LIMITER TEST SUITE")
    print("=" * 70)
    print(f"  Target Server: {BASE_URL}")
    print("=" * 70)

    results = []

    try:
        print("\n" + "#" * 70)
        print("#  PHASE 1: INITIAL SETUP")
        print("#" * 70)

        results.append(("Health Check", test_health_check()))
        results.append(("Get Limits", test_get_limits()))

        print("\n" + "#" * 70)
        print("#  PHASE 2: SLIDING WINDOW ALGORITHM TESTS")
        print("#" * 70)

        test_set_limit("/api/users", "GET", 5)
        test_set_limit("/api/orders", "GET", 3)

        results.append(("Headers Test", test_rate_limit_headers()))
        results.append(("Sliding Window Rate Limit",
                       test_rate_limit_trigger("/api/users", "GET", 5, duration=2)))
        results.append(("Concurrent Requests",
                       test_concurrent_requests("/api/users", "GET", 5, num_requests=30)))

        results.append(("Different Methods", test_different_methods()))

        test_get_stats("/api/users", "GET")
        test_get_stats("/api/orders", "GET")

        results.append(("Remove Limit", test_remove_limit()))
        results.append(("Reset Stats", test_reset_stats()))

        print("\n" + "#" * 70)
        print("#  PHASE 3: SWITCH TO TOKEN BUCKET ALGORITHM")
        print("#" * 70)

        results.append(("Switch to Token Bucket",
                       test_switch_algorithm("token_bucket", burst=10)))

        test_set_limit("/api/users", "GET", 5, burst=10)
        test_set_limit("/api/orders", "GET", 3, burst=5)

        results.append(("Token Bucket Rate Limit",
                       test_rate_limit_trigger("/api/users", "GET", 5, duration=2)))
        results.append(("Token Bucket Concurrent",
                       test_concurrent_requests("/api/users", "GET", 5, num_requests=30)))

        test_get_stats("/api/users", "GET")

        print("\n" + "#" * 70)
        print("#  PHASE 4: SWITCH BACK TO SLIDING WINDOW")
        print("#" * 70)

        results.append(("Switch to Sliding Window",
                       test_switch_algorithm("sliding_window", window_size=1)))

        test_reset_stats()

    except requests.exceptions.ConnectionError:
        print("\n" + "!" * 70)
        print("  ERROR: Cannot connect to server!")
        print("  Please start the server first: python app.py")
        print("!" * 70)
        return

    print("\n" + "=" * 70)
    print("  TEST SUMMARY")
    print("=" * 70)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    print(f"\n  {'Test Name':<40} {'Result':<10}")
    print("  " + "-" * 50)
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {name:<40} {status:<10}")

    print("\n" + "=" * 70)
    print(f"  Total: {passed}/{total} tests passed")
    if passed == total:
        print("  All tests passed! ✓")
    else:
        print(f"  {total - passed} test(s) failed")
    print("=" * 70)

    print("\n" + "#" * 70)
    print("#  ADDITIONAL MANUAL TEST IDEAS")
    print("#" * 70)
    print("""
  1. Test burst capacity with token bucket algorithm
  2. Test window size > 1 second for sliding window
  3. Test QPS = 0 (block all requests)
  4. Test limit updates take effect immediately
  5. Test long-running requests under load
  6. Test memory usage over extended periods
  """)
    print("#" * 70 + "\n")


if __name__ == "__main__":
    main()
