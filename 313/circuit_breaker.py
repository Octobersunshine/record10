import random
from enum import Enum
from collections import deque
from dataclasses import dataclass, field


class CircuitState(Enum):
    CLOSED = "CLOSED"
    OPEN = "OPEN"
    HALF_OPEN = "HALF_OPEN"


class ExceptionStrategy(Enum):
    COUNT_AS_FAILURE = "COUNT_AS_FAILURE"
    IGNORE = "IGNORE"
    COUNT_AS_SLOW_CALL = "COUNT_AS_SLOW_CALL"


@dataclass
class CallRecord:
    timestamp: float
    is_failure: bool
    is_slow: bool
    duration: float = 0.0
    exception_type: type | None = None


@dataclass
class CircuitBreakerStats:
    total_requests: int = 0
    passed: int = 0
    rejected: int = 0
    success_count: int = 0
    failure_count: int = 0
    slow_call_count: int = 0
    ignored_count: int = 0
    state_transitions: list = field(default_factory=list)


class CircuitBreaker:
    def __init__(
        self,
        failure_threshold: float = 0.5,
        time_window: float = 10.0,
        half_open_max_calls: int = 10,
        half_open_success_threshold: int = 5,
        open_duration: float = 5.0,
        slow_call_duration: float = float("inf"),
        slow_call_rate_threshold: float = 1.0,
        minimum_number_of_calls: int = 5,
        exception_strategies: dict | None = None,
        clock=None,
    ):
        self.failure_threshold = failure_threshold
        self.time_window = time_window
        self.half_open_max_calls = half_open_max_calls
        self.half_open_success_threshold = half_open_success_threshold
        self.open_duration = open_duration
        self.slow_call_duration = slow_call_duration
        self.slow_call_rate_threshold = slow_call_rate_threshold
        self.minimum_number_of_calls = minimum_number_of_calls
        self.exception_strategies = exception_strategies or {}
        self._clock = clock if clock else _RealClock()

        self.state = CircuitState.CLOSED
        self.call_records: deque = deque()
        self.opened_at: float = 0.0
        self.half_open_calls: int = 0
        self.half_open_consecutive_successes: int = 0
        self.stats = CircuitBreakerStats()

    def _now(self) -> float:
        return self._clock.now()

    def _prune_expired_records(self):
        cutoff = self._now() - self.time_window
        while self.call_records and self.call_records[0].timestamp < cutoff:
            self.call_records.popleft()

    def _failure_rate(self) -> float:
        self._prune_expired_records()
        if not self.call_records:
            return 0.0
        failures = sum(1 for r in self.call_records if r.is_failure)
        return failures / self.time_window

    def _slow_call_rate(self) -> float:
        self._prune_expired_records()
        total = len(self.call_records)
        if total < self.minimum_number_of_calls:
            return 0.0
        slow = sum(1 for r in self.call_records if r.is_slow)
        return slow / total

    def _should_open(self) -> bool:
        if self._failure_rate() >= self.failure_threshold:
            return True
        if self._slow_call_rate() >= self.slow_call_rate_threshold:
            return True
        return False

    def _transition_to(self, new_state: CircuitState):
        if self.state != new_state:
            old = self.state
            self.state = new_state
            self.stats.state_transitions.append(
                (self._now(), old.value, new_state.value)
            )
            if new_state == CircuitState.OPEN:
                self.opened_at = self._now()

    def _resolve_exception_strategy(
        self, exception: Exception | None
    ) -> ExceptionStrategy:
        if exception is None:
            return ExceptionStrategy.COUNT_AS_FAILURE
        exc_type = type(exception)
        for registered_type, strategy in self.exception_strategies.items():
            if issubclass(exc_type, registered_type):
                return strategy
        return ExceptionStrategy.COUNT_AS_FAILURE

    def allow_request(self) -> bool:
        self.stats.total_requests += 1

        if self.state == CircuitState.CLOSED:
            return True

        if self.state == CircuitState.OPEN:
            if self._now() - self.opened_at >= self.open_duration:
                self._transition_to(CircuitState.HALF_OPEN)
                self.half_open_calls = 0
                self.half_open_consecutive_successes = 0
                return True
            self.stats.rejected += 1
            return False

        if self.state == CircuitState.HALF_OPEN:
            if self.half_open_calls < self.half_open_max_calls:
                return True
            self.stats.rejected += 1
            return False

        return False

    def record_result(
        self,
        success: bool,
        duration: float = 0.0,
        exception: Exception | None = None,
    ):
        is_slow = duration >= self.slow_call_duration
        counts_as_failure = False
        exc_type = type(exception) if exception else None

        if success:
            self.stats.success_count += 1
            if is_slow:
                self.stats.slow_call_count += 1
        else:
            strategy = self._resolve_exception_strategy(exception)

            if strategy == ExceptionStrategy.IGNORE:
                self.stats.ignored_count += 1
                return

            if strategy == ExceptionStrategy.COUNT_AS_SLOW_CALL:
                is_slow = True
                self.stats.slow_call_count += 1
            else:
                counts_as_failure = True
                self.stats.failure_count += 1

        self.call_records.append(
            CallRecord(
                timestamp=self._now(),
                is_failure=counts_as_failure,
                is_slow=is_slow,
                duration=duration,
                exception_type=exc_type,
            )
        )

        if self.state == CircuitState.CLOSED:
            if self._should_open():
                self._transition_to(CircuitState.OPEN)
        elif self.state == CircuitState.HALF_OPEN:
            self.half_open_calls += 1
            if counts_as_failure:
                self.half_open_consecutive_successes = 0
                self._transition_to(CircuitState.OPEN)
            elif is_slow:
                self.half_open_consecutive_successes = 0
            else:
                self.half_open_consecutive_successes += 1

            if self.state == CircuitState.HALF_OPEN:
                if (
                    self.half_open_consecutive_successes
                    >= self.half_open_success_threshold
                ):
                    self._transition_to(CircuitState.CLOSED)
                    self.call_records.clear()
                elif self.half_open_calls >= self.half_open_max_calls:
                    self._transition_to(CircuitState.OPEN)

    def call(self, func, *args, **kwargs):
        if not self.allow_request():
            return None
        start = self._now()
        try:
            result = func(*args, **kwargs)
            elapsed = self._now() - start
            self.record_result(True, duration=elapsed)
            return result
        except Exception as e:
            elapsed = self._now() - start
            self.record_result(False, duration=elapsed, exception=e)
            raise

    def report(self) -> str:
        lines = [
            "=" * 65,
            "          熔断器状态报告",
            "=" * 65,
            f"  当前状态:             {self.state.value}",
            f"  失败率阈值:           {self.failure_threshold} 次/秒",
            f"  慢调用时长阈值:       {self.slow_call_duration:.2f} 秒",
            f"  慢调用比例阈值:       {self.slow_call_rate_threshold * 100:.0f}%",
            f"  最小调用数:           {self.minimum_number_of_calls}",
            f"  统计时间窗口:         {self.time_window} 秒",
            f"  开启持续时间:         {self.open_duration} 秒",
            f"  半开最大探测数:       {self.half_open_max_calls}",
            f"  半开成功阈值:         {self.half_open_success_threshold} 次连续成功",
            "-" * 65,
            f"  总请求数:             {self.stats.total_requests}",
            f"  通过请求数:           {self.stats.passed}",
            f"  拒绝请求数:           {self.stats.rejected}",
            f"  成功数:               {self.stats.success_count}",
            f"  失败数:               {self.stats.failure_count}",
            f"  慢调用数:             {self.stats.slow_call_count}",
            f"  忽略异常数:           {self.stats.ignored_count}",
            f"  当前失败率:           {self._failure_rate():.4f} 次/秒",
            f"  当前慢调用比例:       {self._slow_call_rate() * 100:.1f}%",
        ]

        if self.exception_strategies:
            lines.append("-" * 65)
            lines.append("  异常策略配置:")
            for exc_type, strategy in self.exception_strategies.items():
                lines.append(
                    f"    {exc_type.__name__:20s} -> {strategy.value}"
                )

        lines.append("-" * 65)
        lines.append("  状态转换历史:")
        if self.stats.state_transitions:
            for ts, old_s, new_s in self.stats.state_transitions:
                lines.append(f"    {old_s:9s} -> {new_s:9s}  (t={ts:.2f}s)")
        else:
            lines.append("    (无状态转换)")
        lines.append("=" * 65)
        return "\n".join(lines)


class _RealClock:
    def now(self) -> float:
        import time

        return time.time()


class VirtualClock:
    def __init__(self, start: float = 0.0):
        self._time = start

    def now(self) -> float:
        return self._time

    def advance(self, delta: float):
        self._time += delta


def simulate(
    failure_threshold: float = 0.5,
    time_window: float = 10.0,
    open_duration: float = 5.0,
    half_open_max_calls: int = 10,
    half_open_success_threshold: int = 5,
    slow_call_duration: float = 1.0,
    slow_call_rate_threshold: float = 0.6,
    minimum_number_of_calls: int = 5,
    total_requests: int = 60,
    failure_prob: float = 0.3,
    slow_prob: float = 0.5,
    request_interval: float = 0.5,
    seed: int = 42,
):
    random.seed(seed)
    clock = VirtualClock(start=0.0)
    cb = CircuitBreaker(
        failure_threshold=failure_threshold,
        time_window=time_window,
        open_duration=open_duration,
        half_open_max_calls=half_open_max_calls,
        half_open_success_threshold=half_open_success_threshold,
        slow_call_duration=slow_call_duration,
        slow_call_rate_threshold=slow_call_rate_threshold,
        minimum_number_of_calls=minimum_number_of_calls,
        clock=clock,
    )

    print("=" * 80)
    print("  熔断器模拟 — 慢调用比例熔断")
    print(f"  失败率阈值={failure_threshold}/s  慢调用阈值={slow_call_duration}s")
    print(f"  慢调用比例阈值={slow_call_rate_threshold*100:.0f}%  最小调用数={minimum_number_of_calls}")
    print(f"  故障概率={failure_prob}  慢调用概率={slow_prob}  请求总数={total_requests}")
    print("=" * 80)
    print()

    phase_labels = {
        CircuitState.CLOSED: "🟢关闭",
        CircuitState.OPEN: "🔴开启",
        CircuitState.HALF_OPEN: "🟡半开",
    }

    for i in range(total_requests):
        clock.advance(request_interval)

        allowed = cb.allow_request()
        if allowed:
            cb.stats.passed += 1
            success = random.random() > failure_prob
            is_slow = success and random.random() < slow_prob
            duration = random.uniform(1.5, 3.0) if is_slow else random.uniform(0.1, 0.8)
            cb.record_result(success, duration=duration)
            if not success:
                result = "✗失败"
            elif is_slow:
                result = f"🐢慢 {duration:.1f}s"
            else:
                result = f"✓快 {duration:.1f}s"
        else:
            result = "⊘拒绝"

        phase = phase_labels[cb.state]
        extra = ""
        if cb.state == CircuitState.HALF_OPEN:
            extra = (
                f" | 连续成功={cb.half_open_consecutive_successes}"
                f"/{cb.half_open_success_threshold}"
            )
        print(
            f"  #{i+1:3d} | {phase} | {result:12s} | "
            f"失败率={cb._failure_rate():.3f}/s | "
            f"慢率={cb._slow_call_rate()*100:5.1f}%{extra}"
        )

    print()
    print(cb.report())
    passed = cb.stats.passed
    rejected = cb.stats.rejected
    total = cb.stats.total_requests
    print(f"\n  通过率: {passed}/{total} = {passed/total*100:.1f}%")
    print(f"  拒绝率: {rejected}/{total} = {rejected/total*100:.1f}%")


class TimeoutError(Exception):
    pass


class RetryableError(Exception):
    pass


class BusinessValidationError(Exception):
    pass


def simulate_exception_strategy():
    clock = VirtualClock(start=0.0)
    cb = CircuitBreaker(
        failure_threshold=0.3,
        time_window=10.0,
        open_duration=5.0,
        half_open_max_calls=10,
        half_open_success_threshold=3,
        slow_call_duration=2.0,
        slow_call_rate_threshold=0.6,
        minimum_number_of_calls=5,
        exception_strategies={
            BusinessValidationError: ExceptionStrategy.IGNORE,
            RetryableError: ExceptionStrategy.COUNT_AS_SLOW_CALL,
            TimeoutError: ExceptionStrategy.COUNT_AS_FAILURE,
        },
        clock=clock,
    )

    print()
    print("=" * 80)
    print("  熔断器模拟 — 异常类型策略")
    print("  BusinessValidationError -> IGNORE (不影响熔断)")
    print("  RetryableError          -> COUNT_AS_SLOW_CALL (记为慢调用)")
    print("  TimeoutError            -> COUNT_AS_FAILURE (记为失败)")
    print("  其他异常                -> COUNT_AS_FAILURE (默认)")
    print("=" * 80)
    print()

    scenarios = [
        ("业务校验错误", BusinessValidationError("invalid")),
        ("业务校验错误", BusinessValidationError("invalid")),
        ("业务校验错误", BusinessValidationError("invalid")),
        ("可重试错误", RetryableError("retry")),
        ("可重试错误", RetryableError("retry")),
        ("超时错误", TimeoutError("timeout")),
        ("可重试错误", RetryableError("retry")),
        ("超时错误", TimeoutError("timeout")),
        ("业务校验错误", BusinessValidationError("invalid")),
        ("可重试错误", RetryableError("retry")),
    ]

    for i, (desc, exc) in enumerate(scenarios):
        clock.advance(0.5)
        allowed = cb.allow_request()
        if allowed:
            cb.stats.passed += 1
            cb.record_result(False, duration=0.5, exception=exc)
            strategy = cb._resolve_exception_strategy(exc)
            result = f"✗{desc} [{strategy.value}]"
        else:
            result = "⊘拒绝"

        state_str = cb.state.value
        print(
            f"  #{i+1:2d} | {state_str:9s} | {result:40s} | "
            f"失败率={cb._failure_rate():.3f}/s | 慢率={cb._slow_call_rate()*100:.1f}%"
        )

    print()
    print(cb.report())
    print(f"\n  说明: {cb.stats.ignored_count} 次业务校验错误被忽略，不影响熔断状态")
    print(f"        {cb.stats.slow_call_count} 次可重试错误被记为慢调用，推动慢调用比例")
    print(f"        {cb.stats.failure_count} 次超时错误被记为失败，推动失败率")


if __name__ == "__main__":
    simulate()
    simulate_exception_strategy()
