import asyncio
import httpx
import random
import time
import json


async def send_requests(base_url: str, num_requests: int = 50):
    endpoints = [
        ("/api/users", "GET"),
        ("/api/users/1", "GET"),
        ("/api/products", "GET"),
        ("/api/orders", "GET"),
        ("/api/slow", "GET"),
        ("/api/random-status", "GET"),
        ("/api/echo?message=test", "GET"),
        ("/api/users", "POST"),
    ]

    async with httpx.AsyncClient(base_url=base_url, timeout=30.0) as client:
        tasks = []
        for _ in range(num_requests):
            endpoint, method = random.choice(endpoints)
            if method == "GET":
                tasks.append(client.get(endpoint))
            elif method == "POST":
                tasks.append(client.post(endpoint, json={"name": f"TestUser{random.randint(1,100)}", "email": "test@example.com"}))

        responses = await asyncio.gather(*tasks, return_exceptions=True)

        success = sum(1 for r in responses if not isinstance(r, Exception) and r.status_code < 400)
        errors = sum(1 for r in responses if isinstance(r, Exception) or (hasattr(r, 'status_code') and r.status_code >= 400))

        print(f"Sent {num_requests} requests:")
        print(f"  - Successful: {success}")
        print(f"  - Errors: {errors}")


async def get_stats(base_url: str):
    async with httpx.AsyncClient(base_url=base_url, timeout=10.0) as client:
        print("\n" + "=" * 70)
        print("ADVANCED MONITORING STATISTICS (v3)")
        print("=" * 70)

        print("\n[1/7] Current Window Stats (last 60s) ---")
        try:
            response = await client.get("/monitoring/stats/current")
            if response.status_code == 200:
                data = response.json()
                print(f"  Window: {data['window_start']} -> {data['window_end']}")
                print(f"  Request Count: {data['request_count']}")
                print(f"  Error Count: {data['error_count']}")
                print(f"  Error Rate: {data['error_rate']:.2f}%")
                print(f"  Avg Response Time: {data['avg_response_time']:.2f} ms")
                print(f"  P99 Response Time: {data['p99_response_time']:.2f} ms")
                print(f"  Sample Requests: {len(data['requests'])}")
            else:
                print(f"  Status: {response.status_code} - {response.text}")
        except Exception as e:
            print(f"  Error: {e}")

        print("\n[2/7] Per-Path Statistics ---")
        try:
            response = await client.get("/monitoring/stats/paths")
            if response.status_code == 200:
                data = response.json()
                print(f"  Total paths tracked: {len(data)}")
                for p in data[:5]:
                    print(f"    {p['method']} {p['path']}: "
                          f"{p['request_count']} reqs, "
                          f"avg={p['avg_response_time']:.2f}ms, "
                          f"p99={p['p99_response_time']:.2f}ms, "
                          f"errors={p['error_count']} ({p['error_rate']:.1f}%)")
                if len(data) > 5:
                    print(f"    ... and {len(data) - 5} more paths")
            else:
                print(f"  Status: {response.status_code} - {response.text}")
        except Exception as e:
            print(f"  Error: {e}")

        print("\n[3/7] Slow Requests ---")
        try:
            response = await client.get("/monitoring/slow-requests", params={"limit": 5})
            if response.status_code == 200:
                data = response.json()
                print(f"  Total slow requests: {len(data)}")
                for req in data[:3]:
                    print(f"    [{req['method']}] {req['path']} - "
                          f"Time: {req['response_time']:.2f}ms "
                          f"(threshold: {req['threshold']}ms), "
                          f"Status: {req['status_code']}")
                    if req.get('request_body'):
                        body_preview = req['request_body'][:50]
                        print(f"      Request Body: {body_preview}...")
        except Exception as e:
            print(f"  Error: {e}")

        print("\n[4/7] Alert Rules ---")
        try:
            response = await client.get("/monitoring/alerts/rules")
            if response.status_code == 200:
                data = response.json()
                print(f"  Total rules: {len(data)}")
                for rule in data:
                    status = "[ON]" if rule['enabled'] else "[OFF]"
                    print(f"    {status} {rule['name']} ({rule['severity']}): "
                          f"{rule['alert_type']} {rule['operator']} {rule['threshold']}")
            else:
                print(f"  Status: {response.status_code} - {response.text}")
        except Exception as e:
            print(f"  Error: {e}")

        print("\n[5/7] Alert History ---")
        try:
            response = await client.get("/monitoring/alerts/history", params={"limit": 10})
            if response.status_code == 200:
                data = response.json()
                print(f"  Total alerts: {len(data)}")
                for alert in data[:5]:
                    print(f"    [{alert['time']}] [{alert['severity'].upper()}] "
                          f"{alert['rule_name']}: {alert['message']}")
            else:
                print(f"  Status: {response.status_code} - {response.text}")
        except Exception as e:
            print(f"  Error: {e}")

        print("\n[6/7] Per-Second Stats ---")
        try:
            response = await client.get("/monitoring/stats/per-second")
            if response.status_code == 200:
                data = response.json()
                for stats in data[-10:]:
                    print(f"    {stats['time']}: "
                          f"{stats['request_count']} reqs, "
                          f"avg={stats['avg_response_time']:.2f}ms, "
                          f"errors={stats['error_count']}")
                if len(data) > 10:
                    print(f"    ... and {len(data) - 10} more seconds")
            else:
                print(f"  Status: {response.status_code} - {response.text}")
        except Exception as e:
            print(f"  Error: {e}")

        print("\n[7/7] Health Check ---")
        try:
            response = await client.get("/monitoring/health")
            if response.status_code == 200:
                data = response.json()
                print(f"  Status: {data['status']}")
                print(f"  Instance ID: {data['instance_id']}")
                print(f"  Hostname: {data['hostname']}")
                print(f"  Window: {data['window_seconds']}s, Bucket: {data['bucket_seconds']}s")
                print(f"  Total Requests: {data['total_requests']}")
                print(f"  Slow Requests: {data['slow_requests_count']} (threshold: {data['slow_request_threshold_ms']}ms)")
                if 'alert_rules_count' in data:
                    print(f"  Alert Rules: {data['alert_rules_count']}")
            else:
                print(f"  Status: {response.status_code} - {response.text}")
        except Exception as e:
            print(f"  Error: {e}")


async def test_alert_rule_creation(base_url: str):
    print("\n" + "=" * 70)
    print("TEST: Alert Rule CRUD Operations")
    print("=" * 70)

    async with httpx.AsyncClient(base_url=base_url, timeout=10.0) as client:
        rule_name = f"test_rule_{int(time.time())}"
        print(f"\nCreating alert rule: {rule_name}")
        response = await client.post("/monitoring/alerts/rules", json={
            "name": rule_name,
            "alert_type": "slow_request",
            "severity": "warning",
            "threshold": 500.0,
            "operator": "gt",
            "description": "Test rule for demo"
        })
        if response.status_code == 200:
            print(f"  [PASS] Rule created: {response.json()['name']}")
        else:
            print(f"  [FAIL] Create failed: {response.status_code} - {response.text}")
            return

        print(f"\nGetting rule: {rule_name}")
        response = await client.get(f"/monitoring/alerts/rules/{rule_name}")
        if response.status_code == 200:
            print(f"  [PASS] Rule retrieved")
        else:
            print(f"  [FAIL] Get failed: {response.status_code}")

        print(f"\nUpdating rule: {rule_name}")
        response = await client.put(f"/monitoring/alerts/rules/{rule_name}", json={
            "threshold": 600.0,
            "enabled": False
        })
        if response.status_code == 200:
            data = response.json()
            print(f"  [PASS] Rule updated - threshold={data['rule']['threshold']}, enabled={data['rule']['enabled']}")
        else:
            print(f"  [FAIL] Update failed: {response.status_code}")

        print(f"\nDeleting rule: {rule_name}")
        response = await client.delete(f"/monitoring/alerts/rules/{rule_name}")
        if response.status_code == 200:
            print(f"  [PASS] Rule deleted")
        else:
            print(f"  [FAIL] Delete failed: {response.status_code}")


async def test_slow_request_threshold(base_url: str):
    print("\n" + "=" * 70)
    print("TEST: Slow Request Threshold Update")
    print("=" * 70)

    async with httpx.AsyncClient(base_url=base_url, timeout=10.0) as client:
        print("\nGetting current threshold...")
        response = await client.get("/monitoring/slow-requests/count")
        if response.status_code == 200:
            data = response.json()
            print(f"  Current threshold: {data['threshold_ms']}ms")
            print(f"  Slow request count: {data['count']}")

        print("\nUpdating threshold to 300ms...")
        response = await client.put("/monitoring/slow-requests/threshold", json={"threshold_ms": 300.0})
        if response.status_code == 200:
            print(f"  [PASS] Threshold updated to 300ms")
        else:
            print(f"  [FAIL] Update failed: {response.status_code} - {response.text}")

        print("\nResetting threshold to 500ms...")
        response = await client.put("/monitoring/slow-requests/threshold", json={"threshold_ms": 500.0})
        if response.status_code == 200:
            print(f"  [PASS] Threshold reset to 500ms")
        else:
            print(f"  [FAIL] Reset failed: {response.status_code}")


async def main():
    base_url = "http://localhost:8000"

    print("Checking if server is running...")
    try:
        async with httpx.AsyncClient(base_url=base_url, timeout=5.0) as client:
            resp = await client.get("/")
            root_data = resp.json()
            print(f"Server is running!")
            print(f"Version: {root_data.get('message', 'N/A')}")
            print(f"Instance ID: {root_data.get('instance_id', 'N/A')}")
    except Exception as e:
        print(f"Error: Server is not running. {e}")
        print("Please start the server first:")
        print("  py main.py")
        return

    print(f"\nSending requests to {base_url}...")
    await send_requests(base_url, num_requests=40)

    time.sleep(2)
    await get_stats(base_url)

    await test_alert_rule_creation(base_url)
    await test_slow_request_threshold(base_url)

    print("\n" + "=" * 70)
    print("[PASS] ALL TESTS COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
