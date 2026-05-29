from enum import Enum
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Deque
from datetime import datetime, timedelta
from collections import deque
import random


class ErrorType(Enum):
    NETWORK_TIMEOUT = "网络超时"
    BUSINESS_EXCEPTION = "业务异常"
    DATA_FORMAT_ERROR = "数据格式错误"


class RetryStrategy(Enum):
    IMMEDIATE = "立即重试"
    DELAYED = "延迟重试"
    SKIP = "跳过"
    MANUAL = "人工处理"
    CIRCUIT_BREAK = "熔断暂停"


class CircuitState(Enum):
    CLOSED = "正常"
    OPEN = "熔断"
    HALF_OPEN = "半开"


class AlertLevel(Enum):
    INFO = "信息"
    WARNING = "警告"
    CRITICAL = "严重"


@dataclass
class FailedMessage:
    message_id: str
    content: str
    error_type: ErrorType
    error_detail: str
    failed_count: int = 0
    first_failed_at: datetime = field(default_factory=datetime.now)
    last_failed_at: datetime = field(default_factory=datetime.now)


@dataclass
class ProcessSuggestion:
    message_id: str
    strategy: RetryStrategy
    delay_seconds: Optional[int] = None
    reason: str = ""
    max_retries: int = 0
    manual_queue: bool = False
    next_retry_at: Optional[datetime] = None
    circuit_break: bool = False
    alert_level: Optional[AlertLevel] = None


@dataclass
class AlertEvent:
    timestamp: datetime
    level: AlertLevel
    title: str
    message: str
    error_type: Optional[ErrorType] = None
    metrics: Dict = field(default_factory=dict)


@dataclass
class CircuitStatus:
    state: CircuitState
    error_rate: float
    total_requests: int
    failed_requests: int
    opened_at: Optional[datetime] = None
    half_open_at: Optional[datetime] = None


class DeadLetterAnalyzer:
    GLOBAL_MAX_RETRIES = 10
    GLOBAL_MAX_DELAY_SECONDS = 3600
    DEFAULT_CIRCUIT_ERROR_THRESHOLD = 0.5
    DEFAULT_CIRCUIT_WINDOW_SECONDS = 60
    DEFAULT_CIRCUIT_MIN_REQUESTS = 10
    DEFAULT_CIRCUIT_OPEN_DURATION = 60
    DEFAULT_CIRCUIT_HALF_OPEN_LIMIT = 3
    DEFAULT_STORM_WINDOW_SECONDS = 30
    DEFAULT_STORM_THRESHOLD = 50

    def __init__(self, enable_jitter: bool = True, jitter_range: float = 0.3,
                 circuit_error_threshold: float = DEFAULT_CIRCUIT_ERROR_THRESHOLD,
                 circuit_window_seconds: int = DEFAULT_CIRCUIT_WINDOW_SECONDS,
                 circuit_min_requests: int = DEFAULT_CIRCUIT_MIN_REQUESTS,
                 circuit_open_duration: int = DEFAULT_CIRCUIT_OPEN_DURATION,
                 circuit_half_open_limit: int = DEFAULT_CIRCUIT_HALF_OPEN_LIMIT,
                 storm_window_seconds: int = DEFAULT_STORM_WINDOW_SECONDS,
                 storm_threshold: int = DEFAULT_STORM_THRESHOLD):
        self._enable_jitter = enable_jitter
        self._jitter_range = jitter_range
        self._manual_queue: List[FailedMessage] = []
        self._alerts: List[AlertEvent] = []

        self._circuit_config = {
            "error_threshold": circuit_error_threshold,
            "window_seconds": circuit_window_seconds,
            "min_requests": circuit_min_requests,
            "open_duration": circuit_open_duration,
            "half_open_limit": circuit_half_open_limit
        }

        self._storm_config = {
            "window_seconds": storm_window_seconds,
            "threshold": storm_threshold
        }

        self._circuit_states: Dict[ErrorType, Dict] = {
            error_type: {
                "state": CircuitState.CLOSED,
                "opened_at": None,
                "half_open_at": None,
                "half_open_count": 0
            } for error_type in ErrorType
        }

        self._error_windows: Dict[ErrorType, Deque[Tuple[datetime, bool]]] = {
            error_type: deque() for error_type in ErrorType
        }

        self._storm_windows: Dict[ErrorType, Deque[datetime]] = {
            error_type: deque() for error_type in ErrorType
        }

        self._storm_active: Dict[ErrorType, bool] = {
            error_type: False for error_type in ErrorType
        }

        self._strategy_config: Dict[ErrorType, Dict] = {
            ErrorType.NETWORK_TIMEOUT: {
                "strategy": RetryStrategy.DELAYED,
                "base_delay": 30,
                "max_retries": 5,
                "backoff_factor": 2,
                "jitter_enabled": True,
                "reason": "网络超时通常由临时网络波动引起，采用指数退避+随机抖动延迟重试"
            },
            ErrorType.BUSINESS_EXCEPTION: {
                "strategy": RetryStrategy.IMMEDIATE,
                "base_delay": 10,
                "max_retries": 3,
                "backoff_factor": 2,
                "jitter_enabled": True,
                "reason": "业务异常可能由瞬时状态不一致导致，立即重试有限次数后延迟重试"
            },
            ErrorType.DATA_FORMAT_ERROR: {
                "strategy": RetryStrategy.SKIP,
                "max_retries": 0,
                "reason": "数据格式错误无法通过重试解决，直接转入人工处理队列"
            }
        }

    def analyze(self, message: FailedMessage) -> ProcessSuggestion:
        error_type = message.error_type
        now = datetime.now()

        self._record_error_occurrence(error_type, now)
        self._check_retry_storm(error_type, now)

        circuit_suggestion = self._check_circuit_breaker(error_type, message, now)
        if circuit_suggestion:
            self._record_retry_result(error_type, False, now)
            return circuit_suggestion

        storm_suggestion = self._check_retry_storm_protection(error_type, message)
        if storm_suggestion:
            self._record_retry_result(error_type, False, now)
            return storm_suggestion

        config = self._strategy_config[message.error_type]
        base_strategy = config["strategy"]
        max_retries = min(config["max_retries"], self.GLOBAL_MAX_RETRIES)

        if message.failed_count >= max_retries:
            self._record_retry_result(error_type, False, now)
            return self._send_to_manual_queue(message, max_retries)

        if base_strategy == RetryStrategy.SKIP:
            self._record_retry_result(error_type, False, now)
            return self._send_to_manual_queue(message, max_retries)

        if base_strategy == RetryStrategy.DELAYED:
            delay_seconds = self._calculate_backoff_delay(
                message.failed_count,
                config["base_delay"],
                config.get("backoff_factor", 2),
                config.get("jitter_enabled", self._enable_jitter)
            )
            next_retry_at = now + timedelta(seconds=delay_seconds)
            self._record_retry_result(error_type, False, now)
            return ProcessSuggestion(
                message_id=message.message_id,
                strategy=RetryStrategy.DELAYED,
                delay_seconds=delay_seconds,
                reason=config["reason"],
                max_retries=max_retries,
                next_retry_at=next_retry_at
            )

        if base_strategy == RetryStrategy.IMMEDIATE and message.failed_count >= 1:
            delay_seconds = self._calculate_backoff_delay(
                message.failed_count - 1,
                config["base_delay"],
                config.get("backoff_factor", 2),
                config.get("jitter_enabled", self._enable_jitter)
            )
            next_retry_at = now + timedelta(seconds=delay_seconds)
            self._record_retry_result(error_type, False, now)
            return ProcessSuggestion(
                message_id=message.message_id,
                strategy=RetryStrategy.DELAYED,
                delay_seconds=delay_seconds,
                reason=f"首次立即重试失败，转为延迟重试（失败次数: {message.failed_count}）",
                max_retries=max_retries,
                next_retry_at=next_retry_at
            )

        self._record_retry_result(error_type, False, now)
        return ProcessSuggestion(
            message_id=message.message_id,
            strategy=base_strategy,
            delay_seconds=None,
            reason=config["reason"],
            max_retries=max_retries
        )

    def record_retry_success(self, error_type: ErrorType) -> None:
        self._record_retry_result(error_type, True, datetime.now())

    def _calculate_backoff_delay(self, failed_count: int, base_delay: int,
                                 backoff_factor: int, jitter_enabled: bool) -> int:
        exponential_delay = base_delay * (backoff_factor ** failed_count)
        capped_delay = min(exponential_delay, self.GLOBAL_MAX_DELAY_SECONDS)

        if jitter_enabled and self._enable_jitter:
            jitter = capped_delay * self._jitter_range
            min_delay = max(0, capped_delay - jitter)
            max_delay = capped_delay + jitter
            final_delay = random.uniform(min_delay, max_delay)
            return int(final_delay)

        return capped_delay

    def _record_error_occurrence(self, error_type: ErrorType, now: datetime) -> None:
        window = self._storm_windows[error_type]
        window.append(now)
        self._cleanup_old_entries(window, now, self._storm_config["window_seconds"])

    def _record_retry_result(self, error_type: ErrorType, success: bool, now: datetime) -> None:
        window = self._error_windows[error_type]
        window.append((now, success))
        self._cleanup_old_entries(window, now, self._circuit_config["window_seconds"])

    def _cleanup_old_entries(self, window: Deque, now: datetime, window_seconds: int) -> None:
        cutoff = now - timedelta(seconds=window_seconds)
        while window and window[0][0] < cutoff if isinstance(window[0], tuple) else window[0] < cutoff:
            window.popleft()

    def _calculate_error_rate(self, error_type: ErrorType) -> Tuple[float, int, int]:
        window = self._error_windows[error_type]
        total = len(window)
        if total == 0:
            return 0.0, 0, 0
        failed = sum(1 for _, success in window if not success)
        error_rate = failed / total
        return error_rate, total, failed

    def _check_circuit_breaker(self, error_type: ErrorType, message: FailedMessage,
                                now: datetime) -> Optional[ProcessSuggestion]:
        circuit = self._circuit_states[error_type]
        config = self._circuit_config

        self._update_circuit_state(error_type, now)

        current_state = circuit["state"]

        if current_state == CircuitState.OPEN:
            return ProcessSuggestion(
                message_id=message.message_id,
                strategy=RetryStrategy.CIRCUIT_BREAK,
                delay_seconds=config["open_duration"],
                reason=f"熔断器已打开（{error_type.value}），暂停所有重试，请检查服务状态",
                max_retries=0,
                circuit_break=True,
                alert_level=AlertLevel.CRITICAL
            )

        if current_state == CircuitState.HALF_OPEN:
            if circuit["half_open_count"] >= config["half_open_limit"]:
                return ProcessSuggestion(
                    message_id=message.message_id,
                    strategy=RetryStrategy.CIRCUIT_BREAK,
                    delay_seconds=10,
                    reason=f"熔断器半开状态（{error_type.value}），已达探测请求上限，稍后再试",
                    max_retries=0,
                    circuit_break=True,
                    alert_level=AlertLevel.WARNING
                )
            circuit["half_open_count"] += 1

        return None

    def _update_circuit_state(self, error_type: ErrorType, now: datetime) -> None:
        circuit = self._circuit_states[error_type]
        config = self._circuit_config
        current_state = circuit["state"]

        if current_state == CircuitState.OPEN:
            open_duration = now - circuit["opened_at"]
            if open_duration >= timedelta(seconds=config["open_duration"]):
                circuit["state"] = CircuitState.HALF_OPEN
                circuit["half_open_at"] = now
                circuit["half_open_count"] = 0
                self._add_alert(AlertLevel.INFO,
                               f"熔断器进入半开状态",
                               f"{error_type.value}熔断器冷却结束，进入半开探测状态",
                               error_type)
            return

        if current_state == CircuitState.HALF_OPEN:
            error_rate, total, failed = self._calculate_error_rate(error_type)
            if total >= config["min_requests"] // 2:
                if error_rate >= config["error_threshold"]:
                    self._open_circuit(error_type, now, error_rate, total, failed)
                else:
                    circuit["state"] = CircuitState.CLOSED
                    circuit["opened_at"] = None
                    circuit["half_open_at"] = None
                    circuit["half_open_count"] = 0
                    self._add_alert(AlertLevel.INFO,
                                   f"熔断器恢复正常",
                                   f"{error_type.value}熔断器已关闭，恢复正常处理",
                                   error_type)
            return

        error_rate, total, failed = self._calculate_error_rate(error_type)
        if (total >= config["min_requests"] and
                error_rate >= config["error_threshold"]):
            self._open_circuit(error_type, now, error_rate, total, failed)

    def _open_circuit(self, error_type: ErrorType, now: datetime,
                      error_rate: float, total: int, failed: int) -> None:
        circuit = self._circuit_states[error_type]
        circuit["state"] = CircuitState.OPEN
        circuit["opened_at"] = now
        circuit["half_open_at"] = None
        circuit["half_open_count"] = 0

        self._add_alert(AlertLevel.CRITICAL,
                       f"熔断器已打开",
                       f"{error_type.value}错误率达{error_rate:.1%}（{failed}/{total}），触发熔断",
                       error_type,
                       {"error_rate": error_rate, "total": total, "failed": failed})

    def _check_retry_storm(self, error_type: ErrorType, now: datetime) -> None:
        window = self._storm_windows[error_type]
        config = self._storm_config
        count = len(window)

        if count >= config["threshold"] and not self._storm_active[error_type]:
            self._storm_active[error_type] = True
            self._add_alert(AlertLevel.WARNING,
                           f"检测到重试风暴",
                           f"{error_type.value}在{config['window_seconds']}秒内出现{count}次错误，触发风暴保护",
                           error_type,
                           {"error_count": count, "window_seconds": config["window_seconds"]})
        elif count < config["threshold"] and self._storm_active[error_type]:
            self._storm_active[error_type] = False
            self._add_alert(AlertLevel.INFO,
                           f"重试风暴已解除",
                           f"{error_type.value}错误频率已恢复正常",
                           error_type)

    def _check_retry_storm_protection(self, error_type: ErrorType,
                                       message: FailedMessage) -> Optional[ProcessSuggestion]:
        if self._storm_active[error_type]:
            config = self._storm_config
            window = self._storm_windows[error_type]
            count = len(window)
            delay = self._calculate_storm_delay(count)
            next_retry_at = datetime.now() + timedelta(seconds=delay)
            return ProcessSuggestion(
                message_id=message.message_id,
                strategy=RetryStrategy.CIRCUIT_BREAK,
                delay_seconds=delay,
                reason=f"检测到{error_type.value}重试风暴（{count}次/{config['window_seconds']}秒），已延长重试间隔",
                max_retries=0,
                circuit_break=True,
                alert_level=AlertLevel.WARNING,
                next_retry_at=next_retry_at
            )
        return None

    def _calculate_storm_delay(self, error_count: int) -> int:
        base_delay = 60
        factor = min(error_count / self._storm_config["threshold"], 3.0)
        return int(base_delay * factor)

    def _add_alert(self, level: AlertLevel, title: str, message: str,
                   error_type: Optional[ErrorType] = None, metrics: Optional[Dict] = None) -> None:
        alert = AlertEvent(
            timestamp=datetime.now(),
            level=level,
            title=title,
            message=message,
            error_type=error_type,
            metrics=metrics or {}
        )
        self._alerts.append(alert)

    def get_circuit_status(self, error_type: ErrorType) -> CircuitStatus:
        circuit = self._circuit_states[error_type]
        error_rate, total, failed = self._calculate_error_rate(error_type)
        return CircuitStatus(
            state=circuit["state"],
            error_rate=error_rate,
            total_requests=total,
            failed_requests=failed,
            opened_at=circuit["opened_at"],
            half_open_at=circuit["half_open_at"]
        )

    def get_all_circuit_statuses(self) -> Dict[ErrorType, CircuitStatus]:
        return {et: self.get_circuit_status(et) for et in ErrorType}

    def get_alerts(self, level: Optional[AlertLevel] = None) -> List[AlertEvent]:
        if level:
            return [a for a in self._alerts if a.level == level]
        return self._alerts.copy()

    def clear_alerts(self) -> None:
        self._alerts.clear()

    def reset_circuit(self, error_type: ErrorType) -> None:
        circuit = self._circuit_states[error_type]
        circuit["state"] = CircuitState.CLOSED
        circuit["opened_at"] = None
        circuit["half_open_at"] = None
        circuit["half_open_count"] = 0
        self._error_windows[error_type].clear()
        self._storm_windows[error_type].clear()
        self._storm_active[error_type] = False

    def reset_all_circuits(self) -> None:
        for error_type in ErrorType:
            self.reset_circuit(error_type)

    def _send_to_manual_queue(self, message: FailedMessage, max_retries: int) -> ProcessSuggestion:
        if message not in self._manual_queue:
            self._manual_queue.append(message)
        return ProcessSuggestion(
            message_id=message.message_id,
            strategy=RetryStrategy.MANUAL,
            delay_seconds=None,
            reason=f"已达到最大重试次数({max_retries}次)，已转入人工处理队列",
            max_retries=max_retries,
            manual_queue=True
        )

    def get_manual_queue(self) -> List[FailedMessage]:
        return self._manual_queue.copy()

    def clear_manual_queue(self) -> None:
        self._manual_queue.clear()

    def remove_from_manual_queue(self, message_id: str) -> bool:
        for i, msg in enumerate(self._manual_queue):
            if msg.message_id == message_id:
                del self._manual_queue[i]
                return True
        return False

    def analyze_batch(self, messages: List[FailedMessage]) -> List[ProcessSuggestion]:
        return [self.analyze(msg) for msg in messages]

    def print_suggestion(self, suggestion: ProcessSuggestion) -> None:
        print("=" * 60)
        print(f"消息ID: {suggestion.message_id}")
        print(f"处理策略: {suggestion.strategy.value}")
        if suggestion.delay_seconds:
            delay_str = str(timedelta(seconds=suggestion.delay_seconds))
            print(f"延迟时间: {delay_str} ({suggestion.delay_seconds}秒)")
        if suggestion.next_retry_at:
            print(f"下次重试时间: {suggestion.next_retry_at.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"最大重试次数: {suggestion.max_retries}")
        if suggestion.manual_queue:
            print("状态: 已转入人工处理队列")
        if suggestion.circuit_break:
            print("状态: 熔断器保护中")
        if suggestion.alert_level:
            level_icon = {"INFO": "ℹ", "WARNING": "⚠", "CRITICAL": "🔴"}
            print(f"告警级别: {level_icon.get(suggestion.alert_level.name, '')} {suggestion.alert_level.value}")
        print(f"建议原因: {suggestion.reason}")
        print("=" * 60)

    def print_summary(self, suggestions: List[ProcessSuggestion]) -> None:
        print("\n" + "=" * 60)
        print("处理建议汇总")
        print("=" * 60)

        summary: Dict[RetryStrategy, int] = {
            RetryStrategy.IMMEDIATE: 0,
            RetryStrategy.DELAYED: 0,
            RetryStrategy.SKIP: 0,
            RetryStrategy.MANUAL: 0,
            RetryStrategy.CIRCUIT_BREAK: 0
        }

        for s in suggestions:
            summary[s.strategy] += 1

        print(f"立即重试: {summary[RetryStrategy.IMMEDIATE]} 条")
        print(f"延迟重试: {summary[RetryStrategy.DELAYED]} 条")
        print(f"熔断暂停: {summary[RetryStrategy.CIRCUIT_BREAK]} 条")
        print(f"跳过: {summary[RetryStrategy.SKIP]} 条")
        print(f"人工处理: {summary[RetryStrategy.MANUAL]} 条")
        print(f"总计: {len(suggestions)} 条")
        print("=" * 60)

    def print_circuit_statuses(self) -> None:
        print("\n" + "=" * 60)
        print("熔断器状态")
        print("=" * 60)

        statuses = self.get_all_circuit_statuses()
        for error_type, status in statuses.items():
            state_icon = {
                CircuitState.CLOSED: "✅",
                CircuitState.HALF_OPEN: "🟡",
                CircuitState.OPEN: "🔴"
            }
            print(f"\n{error_type.value}:")
            print(f"  状态: {state_icon[status.state]} {status.state.value}")
            print(f"  错误率: {status.error_rate:.1%} ({status.failed_requests}/{status.total_requests})")
            if status.opened_at:
                print(f"  熔断时间: {status.opened_at.strftime('%Y-%m-%d %H:%M:%S')}")
            if status.half_open_at:
                print(f"  半开时间: {status.half_open_at.strftime('%Y-%m-%d %H:%M:%S')}")

        config = self._circuit_config
        print(f"\n配置: 错误率阈值={config['error_threshold']:.0%}, "
              f"窗口={config['window_seconds']}秒, "
              f"最小请求={config['min_requests']}")
        print("=" * 60)

    def print_alerts(self, level: Optional[AlertLevel] = None) -> None:
        alerts = self.get_alerts(level)
        if not alerts:
            print("\n暂无告警信息")
            return

        print("\n" + "=" * 60)
        title = "告警信息"
        if level:
            title += f" ({level.value})"
        print(f"{title} ({len(alerts)} 条)")
        print("=" * 60)

        level_icon = {
            AlertLevel.INFO: "ℹ",
            AlertLevel.WARNING: "⚠",
            AlertLevel.CRITICAL: "🔴"
        }

        for alert in alerts:
            icon = level_icon.get(alert.level, "•")
            print(f"\n{icon} [{alert.timestamp.strftime('%H:%M:%S')}] {alert.title}")
            print(f"   {alert.message}")
            if alert.error_type:
                print(f"   错误类型: {alert.error_type.value}")
            if alert.metrics:
                metrics_str = ", ".join(f"{k}={v}" for k, v in alert.metrics.items())
                print(f"   指标: {metrics_str}")

        print("=" * 60)

    def print_storm_status(self) -> None:
        print("\n" + "=" * 60)
        print("重试风暴状态")
        print("=" * 60)

        config = self._storm_config
        for error_type in ErrorType:
            window = self._storm_windows[error_type]
            count = len(window)
            active = self._storm_active[error_type]
            status_icon = "⚠️ 活跃" if active else "✅ 正常"
            print(f"\n{error_type.value}:")
            print(f"  状态: {status_icon}")
            print(f"  错误计数: {count}/{config['threshold']}")
            print(f"  时间窗口: {config['window_seconds']}秒")

        print("=" * 60)

    def print_manual_queue(self) -> None:
        queue = self.get_manual_queue()
        if not queue:
            print("\n人工处理队列为空")
            return

        print("\n" + "=" * 60)
        print(f"人工处理队列 ({len(queue)} 条)")
        print("=" * 60)
        for msg in queue:
            print(f"  消息ID: {msg.message_id}")
            print(f"  错误类型: {msg.error_type.value}")
            print(f"  失败次数: {msg.failed_count}")
            print(f"  错误详情: {msg.error_detail}")
            print(f"  首次失败: {msg.first_failed_at.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"  消息内容: {msg.content}")
            print("-" * 40)
        print("=" * 60)


def simulate_retry_process(analyzer: DeadLetterAnalyzer, message: FailedMessage, max_attempts: int = 8) -> None:
    print(f"\n{'=' * 60}")
    print(f"模拟消息 {message.message_id} 的重试过程")
    print(f"{'=' * 60}")

    current_msg = message
    for attempt in range(max_attempts):
        suggestion = analyzer.analyze(current_msg)
        print(f"\n第 {attempt + 1} 次分析 (失败次数: {current_msg.failed_count}):")
        print(f"  策略: {suggestion.strategy.value}")
        if suggestion.delay_seconds:
            delay_str = str(timedelta(seconds=suggestion.delay_seconds))
            print(f"  延迟: {delay_str} ({suggestion.delay_seconds}秒)")
        if suggestion.manual_queue:
            print(f"  已转入人工处理队列，停止重试")
            break
        print(f"  原因: {suggestion.reason}")

        current_msg = FailedMessage(
            message_id=current_msg.message_id,
            content=current_msg.content,
            error_type=current_msg.error_type,
            error_detail=current_msg.error_detail,
            failed_count=current_msg.failed_count + 1,
            first_failed_at=current_msg.first_failed_at
        )


def simulate_circuit_breaker(analyzer: DeadLetterAnalyzer) -> None:
    print(f"\n{'=' * 60}")
    print("熔断器熔断演示")
    print(f"{'=' * 60}")

    analyzer.reset_all_circuits()
    analyzer.clear_alerts()

    error_type = ErrorType.NETWORK_TIMEOUT
    config = analyzer._circuit_config

    print(f"\n模拟{error_type.value}高错误率场景...")
    print(f"配置: 错误率阈值={config['error_threshold']:.0%}, "
          f"最小请求数={config['min_requests']}, "
          f"窗口={config['window_seconds']}秒")

    for i in range(15):
        msg = FailedMessage(
            message_id=f"CIRCUIT-{i+1:03d}",
            content='{"test": true}',
            error_type=error_type,
            error_detail="模拟网络超时",
            failed_count=0
        )
        suggestion = analyzer.analyze(msg)
        status = analyzer.get_circuit_status(error_type)
        circuit = analyzer._circuit_states[error_type]
        half_open_count = circuit.get("half_open_count", 0)
        print(f"  请求 {i+1:2d}: {suggestion.strategy.value:<8} | "
              f"状态={status.state.value:<4} | "
              f"错误率={status.error_rate:.0%} ({status.failed_requests}/{status.total_requests}) | "
              f"半开计数={half_open_count}")
        if status.state == CircuitState.OPEN:
            print(f"  → 熔断器已打开，后续请求将被熔断")
            break

    analyzer.print_circuit_statuses()
    analyzer.print_alerts(AlertLevel.CRITICAL)

    print(f"\n{'=' * 60}")
    print("熔断器半开探测演示")
    print(f"{'=' * 60}")

    circuit = analyzer._circuit_states[error_type]
    circuit["state"] = CircuitState.HALF_OPEN
    circuit["opened_at"] = None
    circuit["half_open_at"] = datetime.now()
    circuit["half_open_count"] = 0
    analyzer._error_windows[error_type].clear()

    print(f"\n手动将熔断器切换到半开状态，允许{config['half_open_limit']}个探测请求...")
    print(f"前2次探测模拟成功，第3次模拟失败...")

    for i in range(5):
        msg = FailedMessage(
            message_id=f"PROBE-{i+1:03d}",
            content='{"probe": true}',
            error_type=error_type,
            error_detail="半开探测",
            failed_count=0
        )
        suggestion = analyzer.analyze(msg)

        if i < 2:
            analyzer.record_retry_success(error_type)

        status = analyzer.get_circuit_status(error_type)
        half_open_count = circuit.get("half_open_count", 0)
        print(f"  探测 {i+1:2d}: {suggestion.strategy.value:<8} | "
              f"状态={status.state.value:<4} | "
              f"错误率={status.error_rate:.0%} ({status.failed_requests}/{status.total_requests}) | "
              f"半开计数={half_open_count}")
        if status.state == CircuitState.OPEN:
            print(f"  → 半开探测失败，熔断器重新打开")
            break
        if status.state == CircuitState.CLOSED and i >= 2:
            print(f"  → 半开探测成功，熔断器恢复正常")
            break

    print(f"\n{'=' * 60}")
    print("熔断器恢复演示（半开探测全部成功）")
    print(f"{'=' * 60}")

    analyzer.reset_circuit(error_type)
    circuit["state"] = CircuitState.HALF_OPEN
    circuit["half_open_at"] = datetime.now()
    circuit["half_open_count"] = 0

    print(f"\n半开状态下连续{config['half_open_limit']}次探测成功...")

    for i in range(5):
        msg = FailedMessage(
            message_id=f"PROBE-SUCCESS-{i+1:03d}",
            content='{"probe": true}',
            error_type=error_type,
            error_detail="半开探测",
            failed_count=0
        )
        suggestion = analyzer.analyze(msg)
        analyzer.record_retry_success(error_type)

        status = analyzer.get_circuit_status(error_type)
        half_open_count = circuit.get("half_open_count", 0)
        print(f"  探测 {i+1:2d}: {suggestion.strategy.value:<8} | "
              f"状态={status.state.value:<4} | "
              f"错误率={status.error_rate:.0%} | "
              f"半开计数={half_open_count}")
        if status.state == CircuitState.CLOSED:
            print(f"  → 所有探测成功，熔断器恢复正常（关闭）")
            break

    analyzer.print_alerts()


def simulate_retry_storm(analyzer: DeadLetterAnalyzer) -> None:
    print(f"\n{'=' * 60}")
    print("重试风暴保护演示")
    print(f"{'=' * 60}")

    analyzer.reset_all_circuits()
    analyzer.clear_alerts()

    error_type = ErrorType.BUSINESS_EXCEPTION
    config = analyzer._storm_config

    print(f"\n模拟{error_type.value}重试风暴...")
    print(f"配置: 阈值={config['threshold']}次/{config['window_seconds']}秒")

    storm_count = config["threshold"] + 10

    for i in range(storm_count):
        msg = FailedMessage(
            message_id=f"STORM-{i+1:03d}",
            content='{"storm": true}',
            error_type=error_type,
            error_detail="模拟业务异常风暴",
            failed_count=i % 3
        )
        suggestion = analyzer.analyze(msg)
        active = analyzer._storm_active[error_type]
        window_count = len(analyzer._storm_windows[error_type])

        if (i + 1) % 10 == 0 or active:
            print(f"  请求 {i+1:3d}: {suggestion.strategy.value:<8} | "
                  f"风暴={active} | "
                  f"计数={window_count}/{config['threshold']}")
            if active and suggestion.strategy == RetryStrategy.CIRCUIT_BREAK:
                delay = suggestion.delay_seconds or 0
                print(f"    → 风暴保护已触发，延长重试间隔至 {delay} 秒")

    analyzer.print_storm_status()
    analyzer.print_alerts(AlertLevel.WARNING)


def main():
    random.seed(42)
    analyzer = DeadLetterAnalyzer(
        enable_jitter=True,
        jitter_range=0.3,
        circuit_error_threshold=0.5,
        circuit_window_seconds=60,
        circuit_min_requests=10,
        circuit_open_duration=60,
        circuit_half_open_limit=3,
        storm_window_seconds=30,
        storm_threshold=50
    )

    print("=" * 60)
    print("死信队列分析器 - 熔断与风暴保护版")
    print("=" * 60)
    print(f"全局最大重试次数: {DeadLetterAnalyzer.GLOBAL_MAX_RETRIES}")
    print(f"全局最大延迟时间: {timedelta(seconds=DeadLetterAnalyzer.GLOBAL_MAX_DELAY_SECONDS)}")
    print(f"随机抖动范围: ±{analyzer._jitter_range * 100:.0f}%")
    print(f"熔断错误率阈值: {analyzer._circuit_config['error_threshold']:.0%}")
    print(f"熔断窗口: {analyzer._circuit_config['window_seconds']}秒")
    print(f"风暴阈值: {analyzer._storm_config['threshold']}次/{analyzer._storm_config['window_seconds']}秒")
    print("=" * 60)

    messages = [
        FailedMessage(
            message_id="MSG-001",
            content='{"user_id": 1001, "action": "login"}',
            error_type=ErrorType.NETWORK_TIMEOUT,
            error_detail="Connection timeout after 30s",
            failed_count=1
        ),
        FailedMessage(
            message_id="MSG-002",
            content='{"order_id": 2001, "amount": 99.99}',
            error_type=ErrorType.BUSINESS_EXCEPTION,
            error_detail="库存不足，请稍后再试",
            failed_count=0
        ),
        FailedMessage(
            message_id="MSG-003",
            content='{"user_id": "abc", "age": 25}',
            error_type=ErrorType.DATA_FORMAT_ERROR,
            error_detail="JSON解析错误：预期数字类型",
            failed_count=2
        ),
        FailedMessage(
            message_id="MSG-004",
            content='{"event": "payment_success"}',
            error_type=ErrorType.NETWORK_TIMEOUT,
            error_detail="Read timed out",
            failed_count=3
        ),
        FailedMessage(
            message_id="MSG-005",
            content='{"order_id": 2002, "status": "paid"}',
            error_type=ErrorType.BUSINESS_EXCEPTION,
            error_detail="订单状态冲突",
            failed_count=3
        )
    ]

    print("\n" + "=" * 60)
    print("一、基础功能演示 - 批量分析结果")
    print("=" * 60)

    suggestions = analyzer.analyze_batch(messages)

    for msg, suggestion in zip(messages, suggestions):
        print(f"\n原始消息信息:")
        print(f"  错误类型: {msg.error_type.value}")
        print(f"  失败次数: {msg.failed_count}")
        print(f"  错误详情: {msg.error_detail}")
        analyzer.print_suggestion(suggestion)

    analyzer.print_summary(suggestions)
    analyzer.print_circuit_statuses()

    print("\n" + "=" * 60)
    print("二、指数退避+随机抖动 演示")
    print("=" * 60)

    base_delay = 30
    backoff_factor = 2
    print(f"\n基础延迟: {base_delay}秒, 退避因子: {backoff_factor}")
    print(f"{'失败次数':<10}{'理论延迟':<15}{'实际延迟(带抖动)':<20}{'抖动范围'}")
    print("-" * 60)
    for i in range(6):
        theoretical = base_delay * (backoff_factor ** i)
        capped = min(theoretical, DeadLetterAnalyzer.GLOBAL_MAX_DELAY_SECONDS)
        actual = analyzer._calculate_backoff_delay(i, base_delay, backoff_factor, True)
        jitter_min = max(0, int(capped * (1 - analyzer._jitter_range)))
        jitter_max = int(capped * (1 + analyzer._jitter_range))
        print(f"{i:<10}{theoretical:<15}{actual:<20}[{jitter_min}, {jitter_max}]")

    print("\n" + "=" * 60)
    print("三、重试流程模拟")
    print("=" * 60)

    simulate_retry_process(analyzer, FailedMessage(
        message_id="RETRY-001",
        content='{"order_id": 9999}',
        error_type=ErrorType.NETWORK_TIMEOUT,
        error_detail="持续网络超时",
        failed_count=0
    ))

    simulate_retry_process(analyzer, FailedMessage(
        message_id="RETRY-002",
        content='{"order_id": 8888}',
        error_type=ErrorType.BUSINESS_EXCEPTION,
        error_detail="持续业务异常",
        failed_count=0
    ))

    simulate_circuit_breaker(analyzer)

    simulate_retry_storm(analyzer)

    print("\n" + "=" * 60)
    print("四、人工处理队列")
    print("=" * 60)

    analyzer.print_manual_queue()

    print(f"\n当前人工队列大小: {len(analyzer.get_manual_queue())}")
    removed = analyzer.remove_from_manual_queue("MSG-003")
    print(f"移除 MSG-003: {'成功' if removed else '失败'}")
    print(f"当前人工队列大小: {len(analyzer.get_manual_queue())}")

    print("\n" + "=" * 60)
    print("五、告警汇总")
    print("=" * 60)
    analyzer.print_alerts()

    analyzer.clear_manual_queue()
    analyzer.clear_alerts()
    analyzer.reset_all_circuits()
    print(f"\n✅ 演示完成，所有状态已重置")


if __name__ == "__main__":
    main()
