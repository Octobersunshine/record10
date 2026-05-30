from circuit_breaker import CircuitBreaker, CircuitBreakerOpenError
import time


def test_successful_recovery():
    print("=== 熔断器成功恢复测试 ===")
    
    breaker = CircuitBreaker(
        failure_threshold=0.5,
        recovery_timeout=2.0,
        window_size=10,
        half_open_success_threshold=0.6,
        half_open_max_calls=5
    )

    @breaker
    def service_call(fail=False):
        if fail:
            raise Exception("Error")
        return "OK"

    print("\n阶段1: 连续失败触发熔断")
    for i in range(10):
        try:
            service_call(fail=True)
        except CircuitBreakerOpenError:
            print(f"  第{i+1}次请求: 熔断触发")
            break
        except Exception:
            print(f"  第{i+1}次请求: 失败")
    
    print(f"  当前状态: {breaker.get_state().value}")

    print(f"\n阶段2: 等待恢复超时 (2秒)...")
    time.sleep(2.5)
    print(f"  当前状态: {breaker.get_state().value}")

    print("\n阶段3: 半开状态 - 大部分请求成功")
    for i in range(5):
        try:
            service_call(fail=False)
            print(f"  请求 {i+1}: 成功")
        except CircuitBreakerOpenError:
            print(f"  请求 {i+1}: 熔断")
            break
        print(f"    状态: {breaker.get_state().value}")

    print(f"\n最终状态: {breaker.get_state().value}")
    print("成功恢复到关闭状态！" if breaker.get_state().value == "CLOSED" else "未恢复")


if __name__ == "__main__":
    test_successful_recovery()
