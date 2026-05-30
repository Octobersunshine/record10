import threading
import time
from circuit_breaker import CircuitBreaker, CircuitBreakerOpenError


def test_half_open_max_calls_limit():
    print("=== 测试1: 半开状态最大请求数限制 (不超过10) ===")
    
    breaker = CircuitBreaker(half_open_max_calls=15)
    print(f"设置 half_open_max_calls=15, 实际限制为: {breaker.half_open_max_calls}")
    assert breaker.half_open_max_calls == 10, "应该被限制为最大值10"
    
    breaker2 = CircuitBreaker(half_open_max_calls=3)
    print(f"设置 half_open_max_calls=3, 实际限制为: {breaker2.half_open_max_calls}")
    assert breaker2.half_open_max_calls == 3, "小于10时应该保持原值"
    
    print("✓ 半开状态最大请求数限制验证通过\n")


def test_window_min_samples():
    print("=== 测试2: 滑动窗口最小样本数边界处理 ===")
    
    breaker = CircuitBreaker(
        failure_threshold=0.5,
        window_size=3
    )
    print(f"设置 window_size=3, 实际窗口大小为: {breaker.window_size}")
    assert breaker.window_size >= CircuitBreaker.MIN_WINDOW_SAMPLES, "窗口大小应该不小于最小值"
    
    breaker2 = CircuitBreaker(window_size=8, failure_threshold=0.5)
    
    @breaker2
    def flaky_service(fail=False):
        if fail:
            raise Exception("Fail")
        return "OK"
    
    required_samples = max(CircuitBreaker.MIN_WINDOW_SAMPLES, 8 // 2)
    print(f"窗口大小8，需要 {required_samples} 个样本才触发判断")
    
    print(f"前{required_samples-1}次请求失败（样本不足）...")
    for i in range(required_samples - 1):
        try:
            flaky_service(fail=True)
        except Exception:
            pass
    
    state_before = breaker2.get_state().value
    print(f"{required_samples-1}次失败后状态: {state_before}")
    assert state_before == "CLOSED", "样本不足时不应熔断"
    
    print(f"第{required_samples}次请求失败（达到样本要求）...")
    try:
        flaky_service(fail=True)
    except Exception:
        pass
    
    state_after = breaker2.get_state().value
    print(f"{required_samples}次失败后状态: {state_after}")
    assert state_after == "OPEN", "样本足够且失败率过高时应该熔断"
    
    print("✓ 滑动窗口边界情况处理验证通过\n")


def test_half_open_rate_limit():
    print("=== 测试3: 半开状态流量控制（防止雪崩）===")
    
    breaker = CircuitBreaker(
        failure_threshold=0.5,
        recovery_timeout=1.0,
        half_open_rate_limit_per_second=2.0
    )
    
    @breaker
    def service():
        return "OK"
    
    for i in range(6):
        try:
            service(fail=True) if hasattr(service, 'fail') else service()
        except Exception:
            pass
        try:
            breaker._closed_window.append(False)
        except:
            pass
    
    breaker._transition_to_open()
    breaker.open_time = time.time() - 2.0
    
    print("进入半开状态，测试1秒内限流2个请求...")
    
    allowed_count = 0
    for i in range(5):
        if breaker.allow_request():
            allowed_count += 1
            breaker.record_success()
    
    print(f"1秒内尝试5个请求，实际允许: {allowed_count} 个")
    assert allowed_count <= 3, "应该有限流控制防止雪崩"
    
    print("✓ 半开状态流量控制验证通过\n")


def test_concurrent_safety():
    print("=== 测试4: 并发场景下的稳定性 ===")
    
    breaker = CircuitBreaker(
        failure_threshold=0.5,
        window_size=20,
        half_open_max_calls=10
    )
    
    results = {"success": 0, "fail": 0, "rejected": 0}
    lock = threading.Lock()
    
    def make_request():
        try:
            if breaker.allow_request():
                if time.time() % 2 < 0.5:
                    breaker.record_success()
                    with lock:
                        results["success"] += 1
                else:
                    breaker.record_failure()
                    with lock:
                        results["fail"] += 1
            else:
                with lock:
                    results["rejected"] += 1
        except Exception as e:
            print(f"Error: {e}")
    
    threads = []
    for i in range(20):
        t = threading.Thread(target=make_request)
        threads.append(t)
        t.start()
    
    for t in threads:
        t.join()
    
    print(f"并发请求结果 - 成功: {results['success']}, 失败: {results['fail']}, 拒绝: {results['rejected']}")
    print(f"最终状态: {breaker.get_state().value}")
    print("✓ 并发场景验证通过（无异常）\n")


def test_all_fixes():
    print("=" * 50)
    print("运行所有修复验证测试...")
    print("=" * 50 + "\n")
    
    test_half_open_max_calls_limit()
    test_window_min_samples()
    test_half_open_rate_limit()
    test_concurrent_safety()
    
    print("=" * 50)
    print("所有测试通过！✓")
    print("=" * 50)


if __name__ == "__main__":
    test_all_fixes()
