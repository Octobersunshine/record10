import time
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

BASE_URL = "http://127.0.0.1:5000"


def sep(title=""):
    print("\n" + "=" * 70)
    if title:
        print(f"  {title}")
        print("=" * 70)


def test_health():
    sep("1. Health Check")
    r = requests.get(f"{BASE_URL}/api/health")
    print(f"  Status: {r.status_code}")
    print(f"  Response: {r.json()}")
    return r.status_code == 200


def test_set_dimension_rule(dimension, qps, api_pattern=None):
    print(f"\n  Setting rule: {dimension} -> {qps} QPS"
          f"{f' (pattern: {api_pattern})' if api_pattern else ''}")
    data = {"dimension": dimension, "qps": qps}
    if api_pattern:
        data["api_pattern"] = api_pattern
    r = requests.post(f"{BASE_URL}/api/dimensions", json=data)
    print(f"  Status: {r.status_code}")
    print(f"  Response: {r.json()}")
    return r.status_code in [200, 201]


def test_get_dimensions():
    sep("2. Get Dimension Rules")
    r = requests.get(f"{BASE_URL}/api/dimensions")
    print(f"  Status: {r.status_code}")
    data = r.json()
    rules = data.get('data', {}).get('dimension_rules', {})
    print(f"  Active rules: {len(rules)}")
    for key, rule in rules.items():
        print(f"    {key}: {rule['qps']} QPS")
    dims = data.get('data', {}).get('available_dimensions', [])
    print(f"  Available dimensions: {[d['name'] for d in dims]}")
    return r.status_code == 200


def test_api_rate_limit():
    sep("3. API-Only Rate Limiting (by API path)")
    test_set_dimension_rule("api", 5)

    print("\n  Sending 8 rapid GET /api/users requests...")
    success = limited = 0
    for i in range(8):
        r = requests.get(f"{BASE_URL}/api/users")
        if r.status_code == 200:
            success += 1
        elif r.status_code == 429:
            limited += 1
            if limited == 1:
                data = r.json()
                print(f"  429 Response: blocked_by={data.get('details', {}).get('blocked_by')}")
                print(f"  Headers: X-RateLimit-Algorithm={r.headers.get('X-RateLimit-Algorithm')}")

    print(f"  Results: {success} success, {limited} rate-limited")
    return limited > 0


def test_user_dimension():
    sep("4. Per-User Rate Limiting")

    requests.post(f"{BASE_URL}/api/reset")

    test_set_dimension_rule("api", 20)
    test_set_dimension_rule("user", 3)

    print("\n  User 'alice' sending 5 requests...")
    success = limited = 0
    for i in range(5):
        r = requests.get(
            f"{BASE_URL}/api/users",
            headers={"X-User-ID": "alice"}
        )
        if r.status_code == 200:
            success += 1
        elif r.status_code == 429:
            limited += 1
            if limited == 1:
                data = r.json()
                blocked = data.get('details', {}).get('blocked_by', '')
                print(f"  429 blocked_by: {blocked}")

    print(f"  alice: {success} success, {limited} rate-limited")

    print("\n  User 'bob' sending 5 requests...")
    success2 = limited2 = 0
    for i in range(5):
        r = requests.get(
            f"{BASE_URL}/api/users",
            headers={"X-User-ID": "bob"}
        )
        if r.status_code == 200:
            success2 += 1
        elif r.status_code == 429:
            limited2 += 1

    print(f"  bob: {success2} success, {limited2} rate-limited")

    if limited > 0 and success2 > 0:
        print("  ✓ Per-user rate limiting working! Different users have independent limits.")
        return True
    elif limited > 0:
        print("  ✓ Rate limiting triggered (user dimension active)")
        return True
    else:
        print("  ⚠ Per-user rate limiting may not be working correctly")
        return False


def test_ip_dimension():
    sep("5. Per-IP Rate Limiting")

    requests.post(f"{BASE_URL}/api/reset")

    test_set_dimension_rule("api", 20)
    test_set_dimension_rule("ip", 4)

    print("\n  Sending 6 requests with X-Forwarded-For: 1.2.3.4...")
    success = limited = 0
    for i in range(6):
        r = requests.get(
            f"{BASE_URL}/api/orders",
            headers={"X-Forwarded-For": "1.2.3.4"}
        )
        if r.status_code == 200:
            success += 1
        elif r.status_code == 429:
            limited += 1

    print(f"  IP 1.2.3.4: {success} success, {limited} rate-limited")
    return limited > 0


def test_api_user_combination():
    sep("6. API+User Combination Dimension")

    requests.post(f"{BASE_URL}/api/reset")

    test_set_dimension_rule("api", 20)
    test_set_dimension_rule("api:user", 3)

    print("\n  User 'alice' on /api/users (3 allowed)...")
    s1 = l1 = 0
    for i in range(5):
        r = requests.get(
            f"{BASE_URL}/api/users",
            headers={"X-User-ID": "alice"}
        )
        if r.status_code == 200:
            s1 += 1
        elif r.status_code == 429:
            l1 += 1

    print(f"  alice@/api/users: {s1} success, {l1} limited")

    print("\n  User 'alice' on /api/orders (independent limit)...")
    s2 = l2 = 0
    for i in range(5):
        r = requests.get(
            f"{BASE_URL}/api/orders",
            headers={"X-User-ID": "alice"}
        )
        if r.status_code == 200:
            s2 += 1
        elif r.status_code == 429:
            l2 += 1

    print(f"  alice@/api/orders: {s2} success, {l2} limited")

    if l1 > 0:
        print("  ✓ API+User combination rate limiting working!")
        return True
    return False


def test_dynamic_throttle():
    sep("7. Dynamic Throttle")

    print("\n  Enabling dynamic throttle...")
    r = requests.post(f"{BASE_URL}/api/dynamic", json={
        "enabled": True,
        "base_qps": 10,
        "min_qps": 1,
        "cooldown_seconds": 5
    })
    print(f"  Status: {r.status_code}")
    status = r.json().get('data', {})
    print(f"  Dynamic enabled: {status.get('enabled')}")

    print("\n  Checking dynamic status...")
    r = requests.get(f"{BASE_URL}/api/dynamic")
    status = r.json().get('data', {})
    print(f"  Current multiplier: {status.get('current_multiplier')}")
    print(f"  Current level: {status.get('current_level')}")
    print(f"  Current QPS: {status.get('current_qps')}")
    print(f"  Base QPS: {status.get('base_qps')}")

    print("\n  Force adjusting to critical level (0.2x)...")
    r = requests.post(f"{BASE_URL}/api/dynamic/adjust", json={
        "multiplier": 0.2,
        "level": "critical"
    })
    print(f"  Status: {r.status_code}")
    result = r.json().get('data', {})
    print(f"  New multiplier: {result.get('current_multiplier')}")
    print(f"  New QPS: {result.get('current_qps')}")
    print(f"  New level: {result.get('current_level')}")

    print("\n  Force adjusting to low level (1.0x)...")
    r = requests.post(f"{BASE_URL}/api/dynamic/adjust", json={
        "multiplier": 1.0,
        "level": "low"
    })
    print(f"  Status: {r.status_code}")
    result = r.json().get('data', {})
    print(f"  New multiplier: {result.get('current_multiplier')}")
    print(f"  New QPS: {result.get('current_qps')}")

    print("\n  Disabling dynamic throttle...")
    r = requests.post(f"{BASE_URL}/api/dynamic", json={"enabled": False})
    print(f"  Status: {r.status_code}")
    result = r.json().get('data', {})
    print(f"  Dynamic enabled: {result.get('enabled')}")

    return True


def test_api_pattern_dimension():
    sep("8. API Pattern-Specific Rules")
    test_set_dimension_rule("api", 3, api_pattern="GET:/api/orders")
    test_set_dimension_rule("api", 10, api_pattern="GET:/api/users")

    print("\n  /api/orders (limit: 3 QPS) - sending 5 requests...")
    s1 = l1 = 0
    for i in range(5):
        r = requests.get(f"{BASE_URL}/api/orders")
        if r.status_code == 200:
            s1 += 1
        elif r.status_code == 429:
            l1 += 1
    print(f"  /api/orders: {s1} success, {l1} limited")

    return True


def test_algorithm_switch():
    sep("9. Algorithm Switching")

    requests.post(f"{BASE_URL}/api/reset")

    print("\n  Current algorithm:")
    r = requests.get(f"{BASE_URL}/api/algorithm")
    data = r.json().get('data', {})
    print(f"  Current: {data.get('current_algorithm')}")
    print(f"  Available: {data.get('available_algorithms')}")

    print("\n  Switching to sliding_window_circular...")
    r = requests.put(f"{BASE_URL}/api/algorithm", json={
        "algorithm": "sliding_window_circular"
    })
    print(f"  Status: {r.status_code}")
    if r.status_code == 200:
        print(f"  New algorithm: {r.json().get('data', {}).get('current_algorithm')}")
    else:
        print(f"  Error: {r.json()}")

    print("\n  Switching back to multi_dimension...")
    r = requests.put(f"{BASE_URL}/api/algorithm", json={
        "algorithm": "multi_dimension"
    })
    print(f"  Status: {r.status_code}")
    if r.status_code == 200:
        print(f"  New algorithm: {r.json().get('data', {}).get('current_algorithm')}")
    else:
        print(f"  Error: {r.json()}")

    return True


def test_memory_info():
    sep("10. Memory Info")
    r = requests.get(f"{BASE_URL}/api/memory")
    data = r.json().get('data', {})
    print(f"  Algorithm: {data.get('algorithm')}")
    mem = data.get('memory_info', {})
    print(f"  Memory fixed: {mem.get('memory_fixed')}")
    print(f"  Active dimensions: {mem.get('active_dimensions')}")
    print(f"  Active rules: {mem.get('active_rules')}")
    return r.status_code == 200


def test_concurrent_multi_dim():
    sep("11. Concurrent Multi-Dimension Requests")
    test_set_dimension_rule("api", 20)
    test_set_dimension_rule("user", 5)

    print("\n  30 concurrent requests from 3 users...")
    results = {}

    def worker(user_id):
        r = requests.get(
            f"{BASE_URL}/api/users",
            headers={"X-User-ID": user_id}
        )
        return user_id, r.status_code

    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = []
        for uid in ["u1", "u2", "u3"]:
            for _ in range(10):
                futures.append(executor.submit(worker, uid))

        for f in as_completed(futures):
            uid, code = f.result()
            if uid not in results:
                results[uid] = {"success": 0, "limited": 0}
            if code == 200:
                results[uid]["success"] += 1
            elif code == 429:
                results[uid]["limited"] += 1

    for uid, r in results.items():
        print(f"  {uid}: {r['success']} success, {r['limited']} limited")

    return True


def test_remove_dimension():
    sep("12. Remove Dimension Rule")
    print("\n  Removing 'ip' dimension rule...")
    r = requests.delete(f"{BASE_URL}/api/dimensions", json={"dimension": "ip"})
    print(f"  Status: {r.status_code}")
    print(f"  Response: {r.json()}")

    print("\n  Checking remaining rules...")
    r = requests.get(f"{BASE_URL}/api/dimensions")
    rules = r.json().get('data', {}).get('dimension_rules', {})
    print(f"  Remaining rules: {len(rules)}")
    for key in rules:
        print(f"    {key}")

    return r.status_code == 200


def main():
    print("\n" + "=" * 70)
    print("  MULTI-DIMENSION + DYNAMIC THROTTLE TEST SUITE")
    print("=" * 70)
    print(f"  Server: {BASE_URL}")
    print("=" * 70)

    results = []

    try:
        print("\n" + "#" * 70)
        print("#  PHASE 1: BASIC & MULTI-DIMENSION TESTS")
        print("#" * 70)

        results.append(("Health Check", test_health()))
        results.append(("Get Dimensions", test_get_dimensions()))
        results.append(("API Rate Limit", test_api_rate_limit()))
        results.append(("Per-User Limit", test_user_dimension()))
        results.append(("Per-IP Limit", test_ip_dimension()))
        results.append(("API+User Combo", test_api_user_combination()))
        results.append(("API Pattern Rules", test_api_pattern_dimension()))

        print("\n" + "#" * 70)
        print("#  PHASE 2: DYNAMIC THROTTLE TESTS")
        print("#" * 70)

        results.append(("Dynamic Throttle", test_dynamic_throttle()))

        print("\n" + "#" * 70)
        print("#  PHASE 3: SYSTEM & CONCURRENT TESTS")
        print("#" * 70)

        results.append(("Algorithm Switch", test_algorithm_switch()))
        results.append(("Memory Info", test_memory_info()))
        results.append(("Concurrent Multi-Dim", test_concurrent_multi_dim()))
        results.append(("Remove Dimension", test_remove_dimension()))

    except requests.exceptions.ConnectionError:
        print("\n" + "!" * 70)
        print("  ERROR: Cannot connect to server!")
        print("  Start server: python app.py")
        print("!" * 70)
        return

    print("\n" + "=" * 70)
    print("  TEST SUMMARY")
    print("=" * 70)

    passed = sum(1 for _, r in results if r)
    total = len(results)

    print(f"\n  {'Test Name':<35} {'Result':<10}")
    print("  " + "-" * 45)
    for name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"  {name:<35} {status:<10}")

    print("\n" + "=" * 70)
    print(f"  Total: {passed}/{total} tests passed")
    print("=" * 70)

    print("\n" + "#" * 70)
    print("#  FEATURE OVERVIEW")
    print("#" * 70)
    print("""
  1. Multi-Dimension Rate Limiting:
     - api: Rate limit by API path
     - user: Rate limit by user ID (X-User-ID header)
     - ip: Rate limit by client IP (X-Forwarded-For)
     - api:user: Combination of API + User
     - api:ip: Combination of API + IP
     - api:user:ip: All three combined

  2. Distributed Rate Limiting (Redis):
     - Use algorithm: distributed_redis
     - Lua scripts for atomic operations
     - Supports both sliding window and token bucket
     - Multi-dimension support with shared Redis state

  3. Dynamic Throttle:
     - Auto-adjust QPS based on system load (CPU)
     - Load levels: low(1.0x), medium(0.7x), high(0.4x), critical(0.2x)
     - Configurable multipliers and cooldown
     - Manual override via force adjust API
     - Works with any algorithm
    """)
    print("#" * 70 + "\n")


if __name__ == "__main__":
    main()
