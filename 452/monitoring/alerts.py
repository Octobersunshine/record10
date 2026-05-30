import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable, Any
from enum import Enum


class AlertSeverity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class AlertType(str, Enum):
    SLOW_REQUEST = "slow_request"
    ERROR_RATE = "error_rate"
    THROUGHPUT = "throughput"
    P99_EXCEEDED = "p99_exceeded"


@dataclass
class AlertRule:
    name: str
    alert_type: AlertType
    severity: AlertSeverity
    enabled: bool = True
    threshold: float = 0.0
    window_seconds: int = 60
    operator: str = "gt"
    path_pattern: Optional[str] = None
    cooldown_seconds: int = 60
    last_triggered: float = 0.0
    description: str = ""
    labels: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "alert_type": self.alert_type.value,
            "severity": self.severity.value,
            "enabled": self.enabled,
            "threshold": self.threshold,
            "window_seconds": self.window_seconds,
            "operator": self.operator,
            "path_pattern": self.path_pattern,
            "cooldown_seconds": self.cooldown_seconds,
            "description": self.description,
            "labels": self.labels
        }


@dataclass
class AlertEvent:
    rule_name: str
    alert_type: AlertType
    severity: AlertSeverity
    message: str
    value: float
    threshold: float
    timestamp: float
    labels: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rule_name": self.rule_name,
            "alert_type": self.alert_type.value,
            "severity": self.severity.value,
            "message": self.message,
            "value": round(self.value, 4),
            "threshold": self.threshold,
            "timestamp": self.timestamp,
            "time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(self.timestamp)),
            "labels": self.labels
        }


class AlertManager:
    def __init__(self, max_history: int = 1000):
        self._lock = threading.RLock()
        self._rules: Dict[str, AlertRule] = {}
        self._history: List[AlertEvent] = []
        self._max_history = max_history
        self._handlers: List[Callable[[AlertEvent], None]] = []

    def add_rule(self, rule: AlertRule) -> None:
        with self._lock:
            self._rules[rule.name] = rule

    def remove_rule(self, rule_name: str) -> bool:
        with self._lock:
            return self._rules.pop(rule_name, None) is not None

    def get_rule(self, rule_name: str) -> Optional[AlertRule]:
        with self._lock:
            return self._rules.get(rule_name)

    def list_rules(self) -> List[AlertRule]:
        with self._lock:
            return list(self._rules.values())

    def update_rule(self, rule_name: str, **kwargs) -> Optional[AlertRule]:
        with self._lock:
            rule = self._rules.get(rule_name)
            if rule is None:
                return None
            for key, value in kwargs.items():
                if hasattr(rule, key):
                    setattr(rule, key, value)
            return rule

    def add_handler(self, handler: Callable[[AlertEvent], None]) -> None:
        with self._lock:
            self._handlers.append(handler)

    def _trigger_alert(self, rule: AlertRule, value: float, message: str, labels: Optional[Dict[str, str]] = None) -> None:
        now = time.time()
        if now - rule.last_triggered < rule.cooldown_seconds:
            return

        rule.last_triggered = now
        event = AlertEvent(
            rule_name=rule.name,
            alert_type=rule.alert_type,
            severity=rule.severity,
            message=message,
            value=value,
            threshold=rule.threshold,
            timestamp=now,
            labels={**rule.labels, **(labels or {})}
        )

        with self._lock:
            self._history.append(event)
            while len(self._history) > self._max_history:
                self._history.pop(0)
            handlers = list(self._handlers)

        for handler in handlers:
            try:
                handler(event)
            except Exception as e:
                print(f"Alert handler error: {e}")

    def _evaluate_condition(self, rule: AlertRule, value: float) -> bool:
        if rule.operator == "gt":
            return value > rule.threshold
        elif rule.operator == "gte":
            return value >= rule.threshold
        elif rule.operator == "lt":
            return value < rule.threshold
        elif rule.operator == "lte":
            return value <= rule.threshold
        elif rule.operator == "eq":
            return value == rule.threshold
        return False

    def evaluate(
        self,
        avg_response_time: float = 0.0,
        p99_response_time: float = 0.0,
        error_rate: float = 0.0,
        throughput: float = 0.0,
        path_stats: Optional[Dict[str, Dict]] = None
    ) -> List[AlertEvent]:
        triggered = []
        path_stats = path_stats or {}

        with self._lock:
            rules = list(self._rules.values())

        for rule in rules:
            if not rule.enabled:
                continue

            value = 0.0
            should_check = True
            message = ""

            if rule.alert_type == AlertType.SLOW_REQUEST:
                value = avg_response_time
                message = f"Average response time {value:.2f}ms exceeds threshold {rule.threshold}ms"

            elif rule.alert_type == AlertType.P99_EXCEEDED:
                value = p99_response_time
                message = f"P99 response time {value:.2f}ms exceeds threshold {rule.threshold}ms"

            elif rule.alert_type == AlertType.ERROR_RATE:
                value = error_rate
                message = f"Error rate {value:.2f}% exceeds threshold {rule.threshold}%"

            elif rule.alert_type == AlertType.THROUGHPUT:
                value = throughput
                message = f"Throughput {value:.2f} req/s exceeds threshold {rule.threshold} req/s"

            else:
                should_check = False

            if rule.path_pattern and path_stats:
                path_value = path_stats.get(rule.path_pattern, {}).get("avg_response_time", 0)
                if path_value > 0:
                    value = path_value
                    message = f"Path {rule.path_pattern}: {message}"
                else:
                    should_check = False

            if should_check and self._evaluate_condition(rule, value):
                labels = {"path": rule.path_pattern} if rule.path_pattern else {}
                self._trigger_alert(rule, value, message, labels)
                triggered.append(AlertEvent(
                    rule_name=rule.name,
                    alert_type=rule.alert_type,
                    severity=rule.severity,
                    message=message,
                    value=value,
                    threshold=rule.threshold,
                    timestamp=time.time(),
                    labels=labels
                ))

        return triggered

    def get_history(self, limit: int = 100) -> List[AlertEvent]:
        with self._lock:
            return list(reversed(self._history[-limit:]))

    def clear_history(self) -> None:
        with self._lock:
            self._history.clear()
