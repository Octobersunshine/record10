import time
import threading
import copy
from enum import Enum
from collections import deque


class CircuitState(Enum):
    CLOSED = "CLOSED"
    OPEN = "OPEN"
    HALF_OPEN = "HALF_OPEN"


class FallbackValue:
    def __init__(self, value):
        self.value = value


class CachedResult:
    def __init__(self, value, timestamp: float):
        self.value = value
        self.timestamp = timestamp


class ExceptionThresholdRule:
    def __init__(self, exception_type: type, threshold: float):
        self.exception_type = exception_type
        self.threshold = max(0.0, min(1.0, threshold))


class CircuitBreaker:
    MAX_HALF_OPEN_CALLS = 10
    MIN_WINDOW_SAMPLES = 5

    def __init__(
        self,
        failure_threshold: float = 0.5,
        recovery_timeout: float = 10.0,
        window_size: int = 20,
        half_open_success_threshold: float = 0.8,
        half_open_max_calls: int = 5,
        half_open_rate_limit_per_second: float = 2.0,
        fallback_value=None,
        fallback_factory=None,
        cache_ttl: float = 300.0,
        exception_thresholds: dict = None
    ):
        self.failure_threshold = max(0.0, min(1.0, failure_threshold))
        self.recovery_timeout = max(1.0, recovery_timeout)
        self.window_size = max(self.MIN_WINDOW_SAMPLES, window_size)
        self.half_open_success_threshold = max(0.0, min(1.0, half_open_success_threshold))
        self.half_open_max_calls = max(1, min(self.MAX_HALF_OPEN_CALLS, half_open_max_calls))
        self.half_open_rate_limit_per_second = max(0.5, half_open_rate_limit_per_second)
        self.cache_ttl = max(0.0, cache_ttl)

        self._fallback_value = fallback_value
        self._fallback_factory = fallback_factory
        self._cached_result = None

        self._exception_threshold_rules = []
        if exception_thresholds:
            for exc_type, threshold in exception_thresholds.items():
                if isinstance(exc_type, type) and issubclass(exc_type, BaseException):
                    self._exception_threshold_rules.append(
                        ExceptionThresholdRule(exc_type, threshold)
                    )
                    if threshold < self.failure_threshold:
                        self._exception_threshold_rules.sort(key=lambda r: r.threshold)

        self.state = CircuitState.CLOSED
        self.open_time = None
        self.open_reason = None

        self._closed_window = deque(maxlen=self.window_size)
        self._exception_windows = {}
        for rule in self._exception_threshold_rules:
            self._exception_windows[rule.exception_type] = deque(maxlen=self.window_size)
        self._half_open_results = []
        self._half_open_request_times = []

        self._lock = threading.RLock()

    def _get_failure_rate(self) -> float:
        if not self._closed_window:
            return 0.0
        failures = sum(1 for r in self._closed_window if not r)
        return failures / len(self._closed_window)

    def _get_exception_rate(self, exception_type: type) -> float:
        window = self._exception_windows.get(exception_type)
        if not window:
            return 0.0
        occurrences = sum(1 for r in window if r)
        total = len(self._closed_window) if self._closed_window else 0
        if total == 0:
            return 0.0
        return occurrences / total

    def _get_half_open_success_rate(self) -> float:
        if not self._half_open_results:
            return 0.0
        successes = sum(1 for r in self._half_open_results if r)
        return successes / len(self._half_open_results)

    def _transition_to_open(self, reason: str = None):
        self.state = CircuitState.OPEN
        self.open_time = time.time()
        self.open_reason = reason
        self._closed_window.clear()
        for w in self._exception_windows.values():
            w.clear()

    def _transition_to_half_open(self):
        self.state = CircuitState.HALF_OPEN
        self._half_open_results.clear()
        self._half_open_request_times.clear()
        self.open_reason = None

    def _transition_to_closed(self):
        self.state = CircuitState.CLOSED
        self._closed_window.clear()
        self._half_open_results.clear()
        self._half_open_request_times.clear()
        self.open_reason = None
        for w in self._exception_windows.values():
            w.clear()

    def _cleanup_old_request_times(self, current_time: float):
        cutoff = current_time - 1.0
        while self._half_open_request_times and self._half_open_request_times[0] < cutoff:
            self._half_open_request_times.pop(0)

    def _check_rate_limit(self, current_time: float) -> bool:
        self._cleanup_old_request_times(current_time)
        return len(self._half_open_request_times) < self.half_open_rate_limit_per_second

    def _check_exception_thresholds(self) -> str:
        if len(self._closed_window) < max(self.MIN_WINDOW_SAMPLES, self.window_size // 2):
            return None
        for rule in self._exception_threshold_rules:
            rate = self._get_exception_rate(rule.exception_type)
            if rate >= rule.threshold:
                return f"{rule.exception_type.__name__} rate {rate:.2%} >= threshold {rule.threshold:.2%}"
        return None

    def _check_and_transition_state(self):
        current_time = time.time()
        if self.state == CircuitState.OPEN:
            if current_time - self.open_time >= self.recovery_timeout:
                self._transition_to_half_open()
        elif self.state == CircuitState.CLOSED:
            min_samples = max(self.MIN_WINDOW_SAMPLES, self.window_size // 2)
            if len(self._closed_window) >= min_samples:
                exception_reason = self._check_exception_thresholds()
                if exception_reason:
                    self._transition_to_open(reason=exception_reason)
                elif self._get_failure_rate() >= self.failure_threshold:
                    self._transition_to_open(reason="overall failure rate exceeded")
        elif self.state == CircuitState.HALF_OPEN:
            if len(self._half_open_results) >= self.half_open_max_calls:
                success_rate = self._get_half_open_success_rate()
                if success_rate >= self.half_open_success_threshold:
                    self._transition_to_closed()
                else:
                    self._transition_to_open(reason="half-open success rate too low")

    def allow_request(self) -> bool:
        with self._lock:
            current_time = time.time()
            self._check_and_transition_state()
            if self.state == CircuitState.OPEN:
                return False
            if self.state == CircuitState.HALF_OPEN:
                if len(self._half_open_results) >= self.half_open_max_calls:
                    return False
                if not self._check_rate_limit(current_time):
                    return False
                self._half_open_request_times.append(current_time)
                return True
            return True

    def _get_fallback(self):
        if self._cached_result is not None:
            if time.time() - self._cached_result.timestamp <= self.cache_ttl:
                return self._cached_result.value
        if self._fallback_factory is not None:
            return self._fallback_factory()
        if self._fallback_value is not None:
            if isinstance(self._fallback_value, FallbackValue):
                return self._fallback_value.value
            return self._fallback_value
        return None

    def record_success(self, result=None):
        with self._lock:
            if result is not None:
                try:
                    cached = copy.deepcopy(result)
                except Exception:
                    cached = result
                self._cached_result = CachedResult(cached, time.time())
            if self.state == CircuitState.CLOSED:
                self._closed_window.append(True)
            elif self.state == CircuitState.HALF_OPEN:
                self._half_open_results.append(True)
            self._check_and_transition_state()

    def record_failure(self, exception: Exception = None):
        with self._lock:
            if self.state == CircuitState.CLOSED:
                self._closed_window.append(False)
                if exception is not None:
                    for rule in self._exception_threshold_rules:
                        if isinstance(exception, rule.exception_type):
                            self._exception_windows[rule.exception_type].append(True)
            elif self.state == CircuitState.HALF_OPEN:
                self._half_open_results.append(False)
            self._check_and_transition_state()

    def reset(self):
        with self._lock:
            self._transition_to_closed()
            self._cached_result = None
            self.open_time = None
            self.open_reason = None

    def trip(self, reason: str = "manual trip"):
        with self._lock:
            self._transition_to_open(reason=reason)

    def get_state(self) -> CircuitState:
        with self._lock:
            self._check_and_transition_state()
            return self.state

    def get_metrics(self) -> dict:
        with self._lock:
            exception_rates = {}
            for rule in self._exception_threshold_rules:
                exception_rates[rule.exception_type.__name__] = {
                    "rate": self._get_exception_rate(rule.exception_type),
                    "threshold": rule.threshold
                }
            return {
                "state": self.state.value,
                "failure_rate": self._get_failure_rate(),
                "half_open_success_rate": self._get_half_open_success_rate(),
                "closed_window_size": len(self._closed_window),
                "half_open_calls": len(self._half_open_results),
                "half_open_pending_requests": len(self._half_open_request_times),
                "open_time": self.open_time,
                "open_reason": self.open_reason,
                "max_half_open_calls_limit": self.half_open_max_calls,
                "rate_limit_per_second": self.half_open_rate_limit_per_second,
                "has_cache": self._cached_result is not None,
                "cache_age": time.time() - self._cached_result.timestamp if self._cached_result else None,
                "cache_expired": (time.time() - self._cached_result.timestamp > self.cache_ttl) if self._cached_result else None,
                "exception_rates": exception_rates
            }

    def __call__(self, func):
        def wrapper(*args, **kwargs):
            if not self.allow_request():
                fallback = self._get_fallback()
                if fallback is not None:
                    return fallback
                raise CircuitBreakerOpenError(
                    f"Circuit breaker is OPEN. Reason: {self.open_reason}. "
                    f"Waiting {self.recovery_timeout}s for recovery."
                )
            try:
                result = func(*args, **kwargs)
                self.record_success(result=result)
                return result
            except Exception as e:
                self.record_failure(exception=e)
                raise
        wrapper._circuit_breaker = self
        return wrapper


class CircuitBreakerOpenError(Exception):
    pass


class TimeoutError(Exception):
    pass


class ServerError(Exception):
    pass


def example_usage():
    print("=== 熔断器高级功能示例 ===\n")

    print("--- 1. 降级策略示例 ---")
    breaker_with_fallback = CircuitBreaker(
        failure_threshold=0.5,
        recovery_timeout=3.0,
        window_size=10,
        fallback_value=FallbackValue({"data": "默认降级数据", "source": "fallback"}),
        cache_ttl=60.0
    )

    @breaker_with_fallback
    def api_call(should_fail=False):
        if should_fail:
            raise Exception("API Error")
        return {"data": "实时数据", "source": "live"}

    print("正常请求...")
    result = api_call(should_fail=False)
    print(f"  结果: {result}")

    print("触发熔断...")
    for i in range(10):
        try:
            api_call(should_fail=True)
        except CircuitBreakerOpenError:
            break
        except Exception:
            pass

    print(f"  状态: {breaker_with_fallback.get_state().value}")

    print("熔断期间请求（返回降级数据）...")
    result = api_call(should_fail=False)
    print(f"  结果: {result}")

    time.sleep(3.5)
    result = api_call(should_fail=False)
    print(f"  恢复后结果: {result}\n")

    print("--- 2. 按异常类型配置阈值示例 ---")
    breaker_by_exception = CircuitBreaker(
        failure_threshold=0.8,
        window_size=10,
        exception_thresholds={
            TimeoutError: 0.3,
            ServerError: 0.4
        }
    )

    @breaker_by_exception
    def service_call(error_type=None):
        if error_type == "timeout":
            raise TimeoutError("Request timeout")
        if error_type == "server":
            raise ServerError("500 Internal Server Error")
        return "OK"

    print(f"超时阈值: 30%, 5xx阈值: 40%, 总失败阈值: 80%")
    print("模拟超时错误...")
    for i in range(6):
        try:
            service_call(error_type="timeout")
        except CircuitBreakerOpenError:
            print(f"  第{i+1}次: 熔断触发（超时率过高）")
            break
        except TimeoutError:
            print(f"  第{i+1}次: 超时")
    print(f"  状态: {breaker_by_exception.get_state().value}")
    print(f"  指标: {breaker_by_exception.get_metrics()}\n")

    print("--- 3. 手动重置示例 ---")
    breaker_manual = CircuitBreaker(failure_threshold=0.5, window_size=10)

    @breaker_manual
    def simple_call(should_fail=False):
        if should_fail:
            raise Exception("Fail")
        return "OK"

    print("手动触发熔断...")
    breaker_manual.trip(reason="运维手动熔断")
    print(f"  状态: {breaker_manual.get_state().value}, 原因: {breaker_manual.get_metrics()['open_reason']}")

    try:
        simple_call()
    except CircuitBreakerOpenError as e:
        print(f"  请求被拒绝: {e}")

    print("手动重置熔断...")
    breaker_manual.reset()
    print(f"  状态: {breaker_manual.get_state().value}")

    result = simple_call()
    print(f"  请求成功: {result}")


if __name__ == "__main__":
    example_usage()
