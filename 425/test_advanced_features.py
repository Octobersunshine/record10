import time
from circuit_breaker import (
    CircuitBreaker, CircuitBreakerOpenError,
    TimeoutError, ServerError, FallbackValue
)


def test_fallback_default_value():
    print("=== 测试1: 降级策略 - 默认值（无缓存时）===")
    
    breaker = CircuitBreaker(
        failure_threshold=0.5,
        window_size=8,
        fallback_value=FallbackValue({"status": "degraded", "data": None}),
        cache_ttl=0.0
    )
    
    @breaker
    def service(fail=False):
        if fail:
            raise Exception("Error")
        return {"status": "ok", "data": "live"}
    
    for i in range(10):
        try:
            service(fail=True)
        except (CircuitBreakerOpenError, Exception):
            pass
    
    assert breaker.get_state().value == "OPEN"
    
    result = service(fail=False)
    print(f"  熔断期间（返回默认降级值）: {result}")
    assert result == {"status": "degraded", "data": None}
    
    print("✓ 默认值降级策略验证通过\n")


def test_fallback_cached_result():
    print("=== 测试2: 降级策略 - 缓存旧数据 ===")
    
    breaker = CircuitBreaker(
        failure_threshold=0.5,
        window_size=8,
        fallback_value=FallbackValue("default_fallback"),
        cache_ttl=60.0
    )
    
    @breaker
    def service(fail=False):
        if fail:
            raise Exception("Error")
        return "cached_live_data"
    
    result = service(fail=False)
    print(f"  正常请求: {result}")
    assert result == "cached_live_data"
    
    metrics = breaker.get_metrics()
    print(f"  缓存状态: has_cache={metrics['has_cache']}, cache_age={metrics['cache_age']:.3f}s")
    assert metrics["has_cache"] is True
    
    for i in range(10):
        try:
            service(fail=True)
        except (CircuitBreakerOpenError, Exception):
            pass
    
    assert breaker.get_state().value == "OPEN"
    
    result = service(fail=False)
    print(f"  熔断期间（返回缓存数据）: {result}")
    assert result == "cached_live_data", "应该返回缓存的成功结果，不是默认降级值"
    
    print("✓ 缓存降级策略验证通过\n")


def test_fallback_factory():
    print("=== 测试3: 降级策略 - 工厂函数 ===")
    
    call_count = {"count": 0}
    
    def fallback_factory():
        call_count["count"] += 1
        return f"fallback_response_{call_count['count']}"
    
    breaker = CircuitBreaker(
        failure_threshold=0.5,
        window_size=8,
        fallback_factory=fallback_factory,
        cache_ttl=0.0
    )
    
    @breaker
    def service(fail=False):
        if fail:
            raise Exception("Error")
        return "live"
    
    for i in range(10):
        try:
            service(fail=True)
        except (CircuitBreakerOpenError, Exception):
            pass
    
    assert breaker.get_state().value == "OPEN"
    
    result1 = service(fail=False)
    result2 = service(fail=False)
    print(f"  熔断期间工厂调用1: {result1}")
    print(f"  熔断期间工厂调用2: {result2}")
    assert "fallback_response" in result1
    assert call_count["count"] >= 2, "工厂函数应该被调用"
    
    print("✓ 工厂函数降级策略验证通过\n")


def test_cache_ttl_expiry():
    print("=== 测试4: 缓存TTL过期 ===")
    
    breaker = CircuitBreaker(
        failure_threshold=0.5,
        window_size=8,
        fallback_value=FallbackValue("expired_default"),
        cache_ttl=0.5
    )
    
    @breaker
    def service(fail=False):
        if fail:
            raise Exception("Error")
        return "fresh_data"
    
    service(fail=False)
    
    for i in range(10):
        try:
            service(fail=True)
        except (CircuitBreakerOpenError, Exception):
            pass
    
    assert breaker.get_state().value == "OPEN"
    
    result = service(fail=False)
    print(f"  缓存未过期时: {result}")
    assert result == "fresh_data"
    
    time.sleep(0.6)
    
    result = service(fail=False)
    print(f"  缓存过期后（回退到默认值）: {result}")
    assert result == "expired_default"
    
    print("✓ 缓存TTL过期验证通过\n")


def test_exception_type_threshold_timeout():
    print("=== 测试5: 按异常类型阈值 - 超时触发熔断 ===")
    
    breaker = CircuitBreaker(
        failure_threshold=0.8,
        window_size=8,
        exception_thresholds={
            TimeoutError: 0.3,
            ServerError: 0.5
        }
    )
    
    @breaker
    def service(error_type=None):
        if error_type == "timeout":
            raise TimeoutError("Timeout")
        if error_type == "server":
            raise ServerError("500")
        return "OK"
    
    for i in range(4):
        try:
            service(error_type="timeout")
        except TimeoutError:
            pass
    
    print(f"  4次超时后状态: {breaker.get_state().value}")
    assert breaker.get_state().value == "CLOSED", "样本不足不应熔断"
    
    try:
        service(error_type="timeout")
    except TimeoutError:
        pass
    
    state = breaker.get_state().value
    print(f"  5次超时后状态: {state}")
    assert state == "OPEN", "超时率达到阈值应该熔断"
    
    metrics = breaker.get_metrics()
    print(f"  熔断原因: {metrics['open_reason']}")
    assert "TimeoutError" in metrics["open_reason"]
    
    print("✓ 超时异常类型阈值验证通过\n")


def test_exception_type_threshold_server():
    print("=== 测试6: 按异常类型阈值 - 5xx错误不触发（未达阈值）===")
    
    breaker = CircuitBreaker(
        failure_threshold=0.8,
        window_size=10,
        exception_thresholds={
            TimeoutError: 0.3,
            ServerError: 0.6
        }
    )
    
    @breaker
    def service(error_type=None, succeed=False):
        if error_type == "timeout":
            raise TimeoutError("Timeout")
        if error_type == "server":
            raise ServerError("500")
        if succeed:
            return "OK"
        raise Exception("Other error")
    
    for i in range(10):
        try:
            if i == 0 or i == 3 or i == 7:
                service(error_type="server")
            else:
                service(succeed=True)
        except (ServerError, Exception):
            pass
    
    state = breaker.get_state().value
    print(f"  3次5xx + 7次成功后状态: {state}")
    assert state == "CLOSED", "5xx率30%低于60%阈值，总失败率30%低于80%阈值，不应熔断"
    
    print("✓ 5xx异常未达阈值不熔断验证通过\n")


def test_exception_type_priority():
    print("=== 测试7: 异常类型阈值优先于总失败率 ===")
    
    breaker = CircuitBreaker(
        failure_threshold=0.6,
        window_size=8,
        exception_thresholds={
            TimeoutError: 0.3
        }
    )
    
    @breaker
    def service(error_type=None):
        if error_type == "timeout":
            raise TimeoutError("Timeout")
        return "OK"
    
    for i in range(4):
        try:
            service(error_type="timeout")
        except TimeoutError:
            pass
    for i in range(3):
        service()
    
    try:
        service(error_type="timeout")
    except TimeoutError:
        pass
    
    state = breaker.get_state().value
    print(f"  5次超时+3次成功后状态: {state}")
    assert state == "OPEN", "超时率超过30%阈值应熔断，即使总失败率未超过60%"
    
    metrics = breaker.get_metrics()
    print(f"  熔断原因: {metrics['open_reason']}")
    
    print("✓ 异常类型阈值优先级验证通过\n")


def test_manual_trip():
    print("=== 测试8: 手动触发熔断 ===")
    
    breaker = CircuitBreaker(failure_threshold=0.5, window_size=10)
    
    @breaker
    def service():
        return "OK"
    
    breaker.trip(reason="运维手动熔断")
    print(f"  手动熔断后状态: {breaker.get_state().value}")
    assert breaker.get_state().value == "OPEN"
    
    metrics = breaker.get_metrics()
    print(f"  熔断原因: {metrics['open_reason']}")
    assert metrics["open_reason"] == "运维手动熔断"
    
    try:
        service()
        assert False, "应该抛出异常"
    except CircuitBreakerOpenError as e:
        print(f"  请求被拒: {e}")
        assert "运维手动熔断" in str(e)
    
    print("✓ 手动触发熔断验证通过\n")


def test_manual_reset():
    print("=== 测试9: 手动重置熔断 ===")
    
    breaker = CircuitBreaker(failure_threshold=0.5, window_size=8)
    
    @breaker
    def service(fail=False):
        if fail:
            raise Exception("Error")
        return "OK"
    
    for i in range(10):
        try:
            service(fail=True)
        except (CircuitBreakerOpenError, Exception):
            pass
    
    assert breaker.get_state().value == "OPEN"
    
    breaker.reset()
    print(f"  重置后状态: {breaker.get_state().value}")
    assert breaker.get_state().value == "CLOSED"
    
    metrics = breaker.get_metrics()
    assert metrics["open_time"] is None
    assert metrics["open_reason"] is None
    assert metrics["closed_window_size"] == 0
    
    result = service(fail=False)
    print(f"  重置后请求成功: {result}")
    assert result == "OK"
    
    print("✓ 手动重置熔断验证通过\n")


def test_manual_trip_then_reset_flow():
    print("=== 测试10: 手动熔断->重置->正常流程 ===")
    
    breaker = CircuitBreaker(
        failure_threshold=0.5,
        window_size=8,
        fallback_value=FallbackValue("fallback_data"),
        cache_ttl=0.0
    )
    
    @breaker
    def service(fail=False):
        if fail:
            raise Exception("Error")
        return "live_data"
    
    breaker.trip(reason="紧急维护")
    
    result = service(fail=False)
    print(f"  熔断期间（降级返回）: {result}")
    assert result == "fallback_data"
    
    breaker.reset()
    
    result = service(fail=False)
    print(f"  重置后请求: {result}")
    assert result == "live_data"
    
    print("✓ 手动熔断->重置完整流程验证通过\n")


def test_no_fallback_raises_error():
    print("=== 测试11: 无降级策略时熔断抛异常 ===")
    
    breaker = CircuitBreaker(failure_threshold=0.5, window_size=8)
    
    @breaker
    def service(fail=False):
        if fail:
            raise Exception("Error")
        return "OK"
    
    for i in range(10):
        try:
            service(fail=True)
        except (CircuitBreakerOpenError, Exception):
            pass
    
    assert breaker.get_state().value == "OPEN"
    
    try:
        service(fail=False)
        assert False, "应该抛出CircuitBreakerOpenError"
    except CircuitBreakerOpenError as e:
        print(f"  无降级策略时正确抛出: {type(e).__name__}")
    
    print("✓ 无降级策略时异常抛出验证通过\n")


def run_all_tests():
    print("=" * 60)
    print("  熔断器高级功能综合测试")
    print("=" * 60 + "\n")
    
    test_fallback_default_value()
    test_fallback_cached_result()
    test_fallback_factory()
    test_cache_ttl_expiry()
    test_exception_type_threshold_timeout()
    test_exception_type_threshold_server()
    test_exception_type_priority()
    test_manual_trip()
    test_manual_reset()
    test_manual_trip_then_reset_flow()
    test_no_fallback_raises_error()
    
    print("=" * 60)
    print("  所有测试通过！✓")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()
