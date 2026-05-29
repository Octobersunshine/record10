import json
import logging
import threading
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Callable
from pathlib import Path


logger = logging.getLogger("api_alert_manager")


class AlertRule:
    def __init__(
        self,
        name: str,
        status_code: Optional[int] = None,
        status_code_min: Optional[int] = None,
        status_code_max: Optional[int] = None,
        path_pattern: Optional[str] = None,
        method: Optional[str] = None,
        window_seconds: int = 300,
        threshold: int = 10,
        cooldown_seconds: int = 300,
    ):
        self.name = name
        self.status_code = status_code
        self.status_code_min = status_code_min
        self.status_code_max = status_code_max
        self.path_pattern = path_pattern
        self.method = method.upper() if method else None
        self.window_seconds = window_seconds
        self.threshold = threshold
        self.cooldown_seconds = cooldown_seconds
        self.last_triggered: Optional[datetime] = None

    def matches(self, log_entry: Dict[str, Any]) -> bool:
        response_status = log_entry.get("response_status")
        if self.status_code is not None and response_status != self.status_code:
            return False
        if self.status_code_min is not None and (response_status is None or response_status < self.status_code_min):
            return False
        if self.status_code_max is not None and (response_status is None or response_status > self.status_code_max):
            return False
        if self.path_pattern:
            request_path = log_entry.get("request_path") or ""
            if self.path_pattern not in request_path:
                return False
        if self.method:
            request_method = (log_entry.get("request_method") or "").upper()
            if request_method != self.method:
                return False
        return True

    def is_in_cooldown(self) -> bool:
        if self.last_triggered is None:
            return False
        elapsed = (datetime.now() - self.last_triggered).total_seconds()
        return elapsed < self.cooldown_seconds

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "status_code": self.status_code,
            "status_code_min": self.status_code_min,
            "status_code_max": self.status_code_max,
            "path_pattern": self.path_pattern,
            "method": self.method,
            "window_seconds": self.window_seconds,
            "threshold": self.threshold,
            "cooldown_seconds": self.cooldown_seconds,
            "last_triggered": self.last_triggered.isoformat() if self.last_triggered else None,
        }


class AlertRecord:
    def __init__(
        self,
        rule_name: str,
        message: str,
        count: int,
        window_seconds: int,
        triggered_at: Optional[datetime] = None,
    ):
        self.rule_name = rule_name
        self.message = message
        self.count = count
        self.window_seconds = window_seconds
        self.triggered_at = triggered_at or datetime.now()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rule_name": self.rule_name,
            "message": self.message,
            "count": self.count,
            "window_seconds": self.window_seconds,
            "triggered_at": self.triggered_at.isoformat(),
        }


class SlidingWindowCounter:
    def __init__(self, window_seconds: int):
        self.window_seconds = window_seconds
        self._events: List[datetime] = []
        self._lock = threading.Lock()

    def add(self, timestamp: Optional[datetime] = None) -> None:
        event_time = timestamp or datetime.now()
        with self._lock:
            self._events.append(event_time)
            self._cleanup(event_time)

    def count(self) -> int:
        now = datetime.now()
        with self._lock:
            self._cleanup(now)
            return len(self._events)

    def _cleanup(self, now: datetime) -> None:
        cutoff = now - timedelta(seconds=self.window_seconds)
        self._events = [e for e in self._events if e > cutoff]


class AlertManager:
    def __init__(
        self,
        rules: Optional[List[AlertRule]] = None,
        notifiers: Optional[List[Callable[[AlertRecord], None]]] = None,
        max_history: int = 1000,
    ):
        self.rules: List[AlertRule] = rules or []
        self.notifiers: List[Callable[[AlertRecord], None]] = notifiers or []
        self.max_history = max_history
        self._history: List[AlertRecord] = []
        self._counters: Dict[str, SlidingWindowCounter] = {}
        self._lock = threading.Lock()

        for rule in self.rules:
            self._counters[rule.name] = SlidingWindowCounter(rule.window_seconds)

    def add_rule(self, rule: AlertRule) -> None:
        with self._lock:
            self.rules.append(rule)
            self._counters[rule.name] = SlidingWindowCounter(rule.window_seconds)

    def remove_rule(self, rule_name: str) -> bool:
        with self._lock:
            for i, rule in enumerate(self.rules):
                if rule.name == rule_name:
                    self.rules.pop(i)
                    self._counters.pop(rule_name, None)
                    return True
        return False

    def add_notifier(self, notifier: Callable[[AlertRecord], None]) -> None:
        self.notifiers.append(notifier)

    def check(self, log_entry: Dict[str, Any]) -> Optional[AlertRecord]:
        matched_alert = None

        with self._lock:
            for rule in self.rules:
                if rule.matches(log_entry):
                    counter = self._counters.get(rule.name)
                    if counter:
                        counter.add()
                        current_count = counter.count()

                        if current_count >= rule.threshold and not rule.is_in_cooldown():
                            alert = AlertRecord(
                                rule_name=rule.name,
                                message=(
                                    f"Alert '{rule.name}': {current_count} matching requests "
                                    f"in the last {rule.window_seconds}s "
                                    f"(threshold: {rule.threshold})"
                                ),
                                count=current_count,
                                window_seconds=rule.window_seconds,
                            )
                            rule.last_triggered = datetime.now()
                            self._history.append(alert)

                            if len(self._history) > self.max_history:
                                self._history = self._history[-self.max_history:]

                            matched_alert = alert

        if matched_alert:
            self._notify(matched_alert)

        return matched_alert

    def _notify(self, alert: AlertRecord) -> None:
        for notifier in self.notifiers:
            try:
                notifier(alert)
            except Exception as e:
                logger.error(f"Notifier {notifier.__name__} failed: {e}")

    def get_alert_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        with self._lock:
            recent = self._history[-limit:]
            return [alert.to_dict() for alert in recent]

    def get_rule_status(self) -> List[Dict[str, Any]]:
        with self._lock:
            statuses = []
            for rule in self.rules:
                counter = self._counters.get(rule.name)
                current_count = counter.count() if counter else 0
                statuses.append({
                    **rule.to_dict(),
                    "current_count": current_count,
                    "is_in_cooldown": rule.is_in_cooldown(),
                })
            return statuses


class LoggingNotifier:
    _instance_counter = 0

    def __init__(self, log_file: Optional[str] = None, log_level: int = logging.WARNING):
        LoggingNotifier._instance_counter += 1
        self.logger = logging.getLogger(f"alert_notifier_{LoggingNotifier._instance_counter}")
        self.logger.setLevel(log_level)
        self.logger.propagate = False
        self.handler = None
        if log_file:
            self.handler = logging.FileHandler(log_file, encoding="utf-8")
            formatter = logging.Formatter(
                "%(asctime)s - %(levelname)s - %(message)s"
            )
            self.handler.setFormatter(formatter)
            self.logger.addHandler(self.handler)

    def __call__(self, alert: AlertRecord) -> None:
        self.logger.warning(alert.message)

    def close(self) -> None:
        if self.handler:
            self.handler.close()
            self.logger.removeHandler(self.handler)
            self.handler = None


class WebhookNotifier:
    def __init__(self, url: str, timeout: int = 5):
        self.url = url
        self.timeout = timeout

    def __call__(self, alert: AlertRecord) -> None:
        try:
            import urllib.request
            data = json.dumps(alert.to_dict()).encode("utf-8")
            req = urllib.request.Request(
                self.url,
                data=data,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            urllib.request.urlopen(req, timeout=self.timeout)
        except Exception as e:
            logger.error(f"Webhook notification failed: {e}")


class ConsoleNotifier:
    def __call__(self, alert: AlertRecord) -> None:
        print(f"\033[91m[ALERT] {alert.message}\033[0m")


def create_default_alert_manager(
    error_threshold: int = 10,
    window_seconds: int = 300,
    cooldown_seconds: int = 300,
    alert_log_file: Optional[str] = None,
) -> AlertManager:
    rules = [
        AlertRule(
            name="high_5xx_errors",
            status_code_min=500,
            status_code_max=599,
            window_seconds=window_seconds,
            threshold=error_threshold,
            cooldown_seconds=cooldown_seconds,
        ),
        AlertRule(
            name="high_4xx_errors",
            status_code_min=400,
            status_code_max=499,
            window_seconds=window_seconds,
            threshold=error_threshold * 2,
            cooldown_seconds=cooldown_seconds,
        ),
    ]

    notifiers = [ConsoleNotifier()]

    if alert_log_file:
        notifiers.append(LoggingNotifier(log_file=alert_log_file))

    return AlertManager(rules=rules, notifiers=notifiers)
