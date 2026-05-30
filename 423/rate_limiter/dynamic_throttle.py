import time
import os
import threading
from typing import Optional, Callable


class SystemLoadMonitor:
    def __init__(self, check_interval: float = 5.0, history_size: int = 12):
        self.check_interval = check_interval
        self.history_size = history_size
        self._cpu_history: list[float] = []
        self._request_times: list[float] = []
        self._request_count = 0
        self._error_count = 0
        self._last_check = time.time()
        self._current_qps = 0.0
        self._current_cpu = 0.0
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._monitor_thread: Optional[threading.Thread] = None

    def start(self) -> None:
        if self._monitor_thread is None or not self._monitor_thread.is_alive():
            self._stop_event.clear()
            self._monitor_thread = threading.Thread(
                target=self._monitor_loop, daemon=True
            )
            self._monitor_thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._monitor_thread:
            self._monitor_thread.join(timeout=2.0)

    def _monitor_loop(self) -> None:
        while not self._stop_event.is_set():
            self._collect_metrics()
            self._stop_event.wait(self.check_interval)

    def _collect_metrics(self) -> None:
        now = time.time()

        with self._lock:
            elapsed = now - self._last_check
            if elapsed > 0:
                self._current_qps = self._request_count / elapsed
            self._request_count = 0
            self._error_count = 0
            self._last_check = now

        try:
            cpu = self._get_cpu_usage()
            self._current_cpu = cpu
            self._cpu_history.append(cpu)
            if len(self._cpu_history) > self.history_size:
                self._cpu_history.pop(0)
        except Exception:
            self._current_cpu = 0.0

    def _get_cpu_usage(self) -> float:
        try:
            cpu_percent = os.cpu_count() or 1
            load_avg = os.getloadavg()[0] if hasattr(os, 'getloadavg') else 0
            return min(100.0, (load_avg / cpu_percent) * 100)
        except Exception:
            try:
                import psutil
                return psutil.cpu_percent(interval=0.1)
            except ImportError:
                return 0.0

    def record_request(self, is_error: bool = False) -> None:
        with self._lock:
            self._request_count += 1
            if is_error:
                self._error_count += 1

    def get_load(self) -> dict:
        with self._lock:
            avg_cpu = (
                sum(self._cpu_history) / len(self._cpu_history)
                if self._cpu_history else 0.0
            )
        return {
            'current_cpu': round(self._current_cpu, 1),
            'avg_cpu': round(avg_cpu, 1),
            'current_qps': round(self._current_qps, 1),
            'cpu_history_size': len(self._cpu_history),
            'timestamp': time.time()
        }

    def get_load_level(self) -> str:
        avg_cpu = (
            sum(self._cpu_history) / len(self._cpu_history)
            if self._cpu_history else 0.0
        )
        if avg_cpu < 50:
            return 'low'
        elif avg_cpu < 75:
            return 'medium'
        elif avg_cpu < 90:
            return 'high'
        else:
            return 'critical'


class DynamicThrottleManager:
    LEVEL_LOW = 'low'
    LEVEL_MEDIUM = 'medium'
    LEVEL_HIGH = 'high'
    LEVEL_CRITICAL = 'critical'

    DEFAULT_MULTIPLIERS = {
        'low': 1.0,
        'medium': 0.7,
        'high': 0.4,
        'critical': 0.2
    }

    def __init__(self, base_qps: int = 10,
                 multipliers: Optional[dict[str, float]] = None,
                 check_interval: float = 5.0,
                 min_qps: int = 1,
                 max_qps: Optional[int] = None,
                 cooldown_seconds: float = 30.0,
                 custom_evaluator: Optional[Callable[[dict], float]] = None):
        self.base_qps = base_qps
        self.multipliers = multipliers or dict(self.DEFAULT_MULTIPLIERS)
        self.min_qps = max(1, min_qps)
        self.max_qps = max_qps
        self.cooldown_seconds = cooldown_seconds
        self.custom_evaluator = custom_evaluator

        self._monitor = SystemLoadMonitor(check_interval=check_interval)
        self._current_qps = base_qps
        self._current_multiplier = 1.0
        self._current_level = self.LEVEL_LOW
        self._last_adjustment = 0.0
        self._adjustment_history: list[dict] = []
        self._lock = threading.Lock()
        self._enabled = True

    def start(self) -> None:
        self._monitor.start()

    def stop(self) -> None:
        self._monitor.stop()

    @property
    def enabled(self) -> bool:
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool) -> None:
        self._enabled = value
        if not value:
            with self._lock:
                self._current_qps = self.base_qps
                self._current_multiplier = 1.0
                self._current_level = self.LEVEL_LOW

    def record_request(self, is_error: bool = False) -> None:
        self._monitor.record_request(is_error)

    def get_effective_qps(self, base_qps: Optional[int] = None) -> int:
        if not self._enabled:
            return base_qps or self.base_qps

        effective_base = base_qps or self.base_qps
        self._adjust_if_needed()

        with self._lock:
            qps = int(effective_base * self._current_multiplier)
            qps = max(self.min_qps, qps)
            if self.max_qps is not None:
                qps = min(self.max_qps, qps)
            return qps

    def _adjust_if_needed(self) -> None:
        now = time.time()
        if now - self._last_adjustment < self.cooldown_seconds:
            return

        load = self._monitor.get_load()
        level = self._monitor.get_load_level()

        if self.custom_evaluator:
            multiplier = self.custom_evaluator(load)
            multiplier = max(0.1, min(2.0, multiplier))
        else:
            multiplier = self.multipliers.get(level, 1.0)

        with self._lock:
            old_qps = self._current_qps
            old_level = self._current_level
            old_mult = self._current_multiplier

            self._current_multiplier = multiplier
            self._current_level = level
            self._current_qps = max(
                self.min_qps,
                int(self.base_qps * multiplier)
            )
            if self.max_qps is not None:
                self._current_qps = min(self.max_qps, self._current_qps)
            self._last_adjustment = now

            if old_level != level or abs(old_mult - multiplier) > 0.01:
                self._adjustment_history.append({
                    'timestamp': now,
                    'old_level': old_level,
                    'new_level': level,
                    'old_multiplier': round(old_mult, 3),
                    'new_multiplier': round(multiplier, 3),
                    'old_qps': old_qps,
                    'new_qps': self._current_qps,
                    'cpu': load.get('current_cpu', 0),
                    'qps': load.get('current_qps', 0)
                })
                if len(self._adjustment_history) > 100:
                    self._adjustment_history.pop(0)

    def get_status(self) -> dict:
        load = self._monitor.get_load()
        with self._lock:
            return {
                'enabled': self._enabled,
                'base_qps': self.base_qps,
                'current_qps': self._current_qps,
                'current_multiplier': round(self._current_multiplier, 3),
                'current_level': self._current_level,
                'min_qps': self.min_qps,
                'max_qps': self.max_qps,
                'cooldown_seconds': self.cooldown_seconds,
                'multipliers': self.multipliers,
                'system_load': load,
                'last_adjustment': self._last_adjustment,
                'recent_adjustments': self._adjustment_history[-5:]
            }

    def force_adjust(self, multiplier: float, level: str = None) -> dict:
        now = time.time()
        with self._lock:
            old_qps = self._current_qps
            old_mult = self._current_multiplier
            old_level = self._current_level

            self._current_multiplier = max(0.1, min(2.0, multiplier))
            self._current_level = level or self._current_level
            self._current_qps = max(
                self.min_qps,
                int(self.base_qps * self._current_multiplier)
            )
            if self.max_qps is not None:
                self._current_qps = min(self.max_qps, self._current_qps)
            self._last_adjustment = now

            self._adjustment_history.append({
                'timestamp': now,
                'old_level': old_level,
                'new_level': self._current_level,
                'old_multiplier': round(old_mult, 3),
                'new_multiplier': round(self._current_multiplier, 3),
                'old_qps': old_qps,
                'new_qps': self._current_qps,
                'forced': True
            })

        return self.get_status()
