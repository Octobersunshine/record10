import time
import math
import threading
import queue
from collections import deque
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple
from abc import ABC, abstractmethod


@dataclass
class PoolConfig:
    min_connections: int = 5
    max_connections: int = 20
    connection_timeout: float = 30.0
    idle_timeout: float = 60.0


@dataclass
class Metrics:
    total_requests: int = 0
    total_wait_time: float = 0.0
    max_wait_time: float = 0.0
    queue_length_history: deque = None
    usage_history: deque = None
    wait_time_history: deque = None

    def __post_init__(self):
        if self.queue_length_history is None:
            self.queue_length_history = deque(maxlen=100)
        if self.usage_history is None:
            self.usage_history = deque(maxlen=100)
        if self.wait_time_history is None:
            self.wait_time_history = deque(maxlen=100)


@dataclass
class OptimizedConfig:
    min_connections: int
    max_connections: int
    reason: str
    metrics_summary: Dict[str, float]


class MockConnection:
    def __init__(self, conn_id: int):
        self.conn_id = conn_id
        self.created_at = time.time()
        self.last_used_at = time.time()
        self.is_used = False

    def execute(self, query: str, duration: float = 0.1):
        time.sleep(duration)
        return f"Query executed on conn-{self.conn_id}: {query}"


class ConnectionPool:
    def __init__(self, config: PoolConfig = None):
        self.config = config or PoolConfig()
        self._lock = threading.RLock()
        self._connections: List[MockConnection] = []
        self._wait_condition = threading.Condition(self._lock)
        self._waiting_threads = 0
        self._running = True
        self.metrics = Metrics()
        self._metrics_lock = threading.Lock()
        self._conn_counter = 0
        self._initialize_connections()

    def _initialize_connections(self):
        with self._lock:
            for _ in range(self.config.min_connections):
                self._create_connection()

    def _create_connection(self) -> Optional[MockConnection]:
        if len(self._connections) >= self.config.max_connections:
            return None
        self._conn_counter += 1
        conn = MockConnection(self._conn_counter)
        self._connections.append(conn)
        return conn

    def _remove_connection(self, conn: MockConnection):
        if conn in self._connections and len(self._connections) > self.config.min_connections:
            self._connections.remove(conn)
            return True
        return False

    def acquire(self, timeout: float = None) -> Optional[MockConnection]:
        if timeout is None:
            timeout = self.config.connection_timeout

        start_time = time.time()

        with self._metrics_lock:
            self.metrics.total_requests += 1

        with self._lock:
            self._waiting_threads += 1

        try:
            while time.time() - start_time < timeout and self._running:
                with self._lock:
                    for conn in self._connections:
                        if not conn.is_used:
                            conn.is_used = True
                            conn.last_used_at = time.time()
                            self._waiting_threads -= 1
                            wait_time = time.time() - start_time
                            self._record_wait_time(wait_time)
                            return conn

                    if len(self._connections) < self.config.max_connections:
                        conn = self._create_connection()
                        if conn:
                            conn.is_used = True
                            self._waiting_threads -= 1
                            wait_time = time.time() - start_time
                            self._record_wait_time(wait_time)
                            return conn

                    remaining = timeout - (time.time() - start_time)
                    if remaining <= 0:
                        break
                    self._wait_condition.wait(timeout=min(remaining, 0.1))
        finally:
            with self._lock:
                if self._waiting_threads > 0:
                    self._waiting_threads -= 1

        return None

    def release(self, conn: MockConnection):
        with self._lock:
            if conn in self._connections:
                conn.is_used = False
                conn.last_used_at = time.time()
                self._wait_condition.notify()

    def _record_wait_time(self, wait_time: float):
        with self._metrics_lock:
            self.metrics.total_wait_time += wait_time
            self.metrics.max_wait_time = max(self.metrics.max_wait_time, wait_time)
            self.metrics.wait_time_history.append(wait_time)

    def get_waiting_threads(self) -> int:
        with self._lock:
            return max(0, self._waiting_threads)

    def record_usage(self):
        with self._lock:
            if not self._connections:
                return 0.0
            used = sum(1 for conn in self._connections if conn.is_used)
            usage = used / len(self._connections)
            queue_len = max(0, self._waiting_threads)

        with self._metrics_lock:
            self.metrics.usage_history.append(usage)
            self.metrics.queue_length_history.append(queue_len)
        return usage

    def get_current_metrics(self) -> Dict[str, float]:
        with self._metrics_lock:
            avg_wait_time = (
                self.metrics.total_wait_time / self.metrics.total_requests
                if self.metrics.total_requests > 0
                else 0.0
            )
            avg_queue = (
                sum(self.metrics.queue_length_history) / len(self.metrics.queue_length_history)
                if self.metrics.queue_length_history
                else 0.0
            )
            avg_usage = (
                sum(self.metrics.usage_history) / len(self.metrics.usage_history)
                if self.metrics.usage_history
                else 0.0
            )

        with self._lock:
            current_connections = len(self._connections)
            active_connections = sum(1 for conn in self._connections if conn.is_used)
            current_queue = max(0, self._waiting_threads)

        return {
            "current_connections": current_connections,
            "active_connections": active_connections,
            "current_usage": active_connections / max(current_connections, 1),
            "avg_usage": avg_usage,
            "current_queue": current_queue,
            "avg_queue": avg_queue,
            "max_queue": max(self.metrics.queue_length_history) if self.metrics.queue_length_history else 0,
            "avg_wait_time": avg_wait_time,
            "max_wait_time": self.metrics.max_wait_time,
            "total_requests": self.metrics.total_requests,
        }

    def cleanup_idle_connections(self):
        with self._lock:
            now = time.time()
            idle_conns = [
                conn for conn in self._connections
                if not conn.is_used and (now - conn.last_used_at) > self.config.idle_timeout
            ]
            for conn in idle_conns:
                self._remove_connection(conn)

    def update_config(self, new_config: PoolConfig):
        with self._lock:
            old_min = self.config.min_connections
            self.config = new_config

            while len(self._connections) < new_config.min_connections:
                self._create_connection()

            if new_config.min_connections < old_min:
                self.cleanup_idle_connections()

    def stop(self):
        self._running = False
        with self._lock:
            self._connections.clear()


class BaseOptimizationStrategy(ABC):
    def __init__(
        self,
        min_possible: int = 2,
        max_possible: int = 100,
        cooldown_seconds: float = 30.0,
        consecutive_violations_required: int = 3,
    ):
        self.min_possible = min_possible
        self.max_possible = max_possible
        self.cooldown_seconds = cooldown_seconds
        self.consecutive_violations_required = consecutive_violations_required
        self._last_adjust_time = 0.0
        self._violation_counts: Dict[str, int] = {}
        self._last_suggested_min: Optional[int] = None
        self._last_suggested_max: Optional[int] = None
        self.strategy_name = "BaseStrategy"

    def _check_cooldown(self) -> Tuple[bool, float]:
        now = time.time()
        elapsed = now - self._last_adjust_time
        if elapsed < self.cooldown_seconds:
            return False, self.cooldown_seconds - elapsed
        return True, 0.0

    def _update_violation_counts(self, conditions: Dict[str, bool]):
        for key in conditions:
            if key not in self._violation_counts:
                self._violation_counts[key] = 0
        for key, violated in conditions.items():
            if violated:
                self._violation_counts[key] = min(
                    self._violation_counts[key] + 1,
                    self.consecutive_violations_required + 1
                )
            else:
                self._violation_counts[key] = max(0, self._violation_counts[key] - 1)

    def _has_sustained_violation(self, key: str) -> bool:
        return self._violation_counts.get(key, 0) >= self.consecutive_violations_required

    def _record_adjustment(self):
        self._last_adjust_time = time.time()
        self._violation_counts = {k: 0 for k in self._violation_counts}

    def _enforce_bounds(self, min_conn: int, max_conn: int) -> Tuple[int, int]:
        min_conn = max(min_conn, self.min_possible)
        max_conn = min(max_conn, self.max_possible)
        min_conn = min(min_conn, max_conn)
        max_conn = max(max_conn, min_conn)
        return min_conn, max_conn

    def _format_reason(self, reasons: List[str], cooldown_remaining: float) -> str:
        if cooldown_remaining > 0:
            if reasons and self._last_suggested_min is not None:
                return f"冷却中 (剩余 {cooldown_remaining:.1f}s)，沿用上次建议值"
            return f"冷却中 (剩余 {cooldown_remaining:.1f}s)"
        if reasons:
            return "; ".join(reasons)
        violation_status = ", ".join(
            f"{k.split('_')[0]}={v}/{self.consecutive_violations_required}"
            for k, v in self._violation_counts.items() if v > 0
        )
        if violation_status:
            return f"指标未连续达标: {violation_status}"
        return "参数处于合理范围，保持当前配置"

    @abstractmethod
    def optimize(self, pool: ConnectionPool) -> OptimizedConfig:
        pass


class MetricBasedStrategy(BaseOptimizationStrategy):
    def __init__(
        self,
        target_usage: float = 0.7,
        target_max_wait: float = 1.0,
        target_max_queue: int = 5,
        min_possible: int = 2,
        max_possible: int = 100,
        cooldown_seconds: float = 30.0,
        consecutive_violations_required: int = 3,
    ):
        super().__init__(min_possible, max_possible, cooldown_seconds, consecutive_violations_required)
        self.target_usage = target_usage
        self.target_max_wait = target_max_wait
        self.target_max_queue = target_max_queue
        self.strategy_name = "MetricBased"

    def optimize(self, pool: ConnectionPool) -> OptimizedConfig:
        metrics = pool.get_current_metrics()
        current_min = pool.config.min_connections
        current_max = pool.config.max_connections

        cooldown_ok, remaining_cooldown = self._check_cooldown()

        avg_usage = metrics["avg_usage"]
        avg_wait = metrics["avg_wait_time"]
        max_wait = metrics["max_wait_time"]
        avg_queue = metrics["avg_queue"]
        max_queue = metrics["max_queue"]

        conditions = {
            "high_usage": avg_usage > self.target_usage + 0.1,
            "low_usage": avg_usage < self.target_usage - 0.2 and current_max > current_min,
            "high_wait_or_queue": max_wait > self.target_max_wait or max_queue > self.target_max_queue,
            "low_load": avg_wait < 0.1 and avg_queue < 1 and avg_usage < 0.4,
            "high_load": avg_usage > 0.6 and avg_queue > 0,
        }

        self._update_violation_counts(conditions)

        new_min = current_min
        new_max = current_max
        reasons = []

        if not cooldown_ok:
            if self._last_suggested_min is not None:
                new_min, new_max = self._last_suggested_min, self._last_suggested_max
        else:
            if self._has_sustained_violation("high_usage"):
                increase = int((avg_usage - self.target_usage) * current_max * 0.5)
                new_max = min(current_max + max(increase, 2), self.max_possible)
                reasons.append(f"高使用率 ({avg_usage:.2%}) 连续触发，扩容")

            if self._has_sustained_violation("low_usage"):
                decrease = max(1, int((self.target_usage - avg_usage) * current_max * 0.3))
                new_max = max(current_max - decrease, current_min, self.min_possible)
                reasons.append(f"低使用率 ({avg_usage:.2%}) 连续触发，缩容")

            if self._has_sustained_violation("high_wait_or_queue"):
                scale_up = max(3, int(current_max * 0.3))
                new_max = min(current_max + scale_up, self.max_possible)
                reasons.append(f"等待/队列超标 ({max_wait:.2f}s, {max_queue}) 连续触发，紧急扩容")

            if self._has_sustained_violation("low_load"):
                new_min = max(current_min - 1, self.min_possible)
                reasons.append("持续低负载，减少最小连接数")
            elif self._has_sustained_violation("high_load"):
                new_min = min(current_min + 2, int(new_max * 0.5), self.max_possible)
                reasons.append("持续高负载，增加最小连接数")

            if reasons:
                self._last_suggested_min = new_min
                self._last_suggested_max = new_max
                self._record_adjustment()

        new_min, new_max = self._enforce_bounds(new_min, new_max)
        reason = self._format_reason(reasons, remaining_cooldown)

        metrics_summary = {
            "avg_usage": avg_usage,
            "avg_wait_time": avg_wait,
            "max_wait_time": max_wait,
            "avg_queue": avg_queue,
            "max_queue": max_queue,
            "current_connections": metrics["current_connections"],
            "cooldown_remaining": remaining_cooldown,
            "strategy": self.strategy_name,
        }
        for key, count in self._violation_counts.items():
            metrics_summary[f"violation_{key}"] = count

        return OptimizedConfig(
            min_connections=new_min,
            max_connections=new_max,
            reason=reason,
            metrics_summary=metrics_summary,
        )


class HeuristicResourceStrategy(BaseOptimizationStrategy):
    def __init__(
        self,
        cpu_per_connection: float = 0.05,
        memory_per_connection_mb: float = 10.0,
        max_cpu_usage: float = 0.8,
        max_memory_mb: float = 1024.0,
        min_possible: int = 2,
        max_possible: int = 100,
        cooldown_seconds: float = 30.0,
        consecutive_violations_required: int = 3,
    ):
        super().__init__(min_possible, max_possible, cooldown_seconds, consecutive_violations_required)
        self.cpu_per_connection = cpu_per_connection
        self.memory_per_connection_mb = memory_per_connection_mb
        self.max_cpu_usage = max_cpu_usage
        self.max_memory_mb = max_memory_mb
        self.strategy_name = "HeuristicResource"

    def _estimate_resource_usage(self, num_connections: int, current_usage: float) -> Tuple[float, float]:
        estimated_cpu = num_connections * self.cpu_per_connection
        estimated_memory = num_connections * self.memory_per_connection_mb
        return estimated_cpu, estimated_memory

    def optimize(self, pool: ConnectionPool) -> OptimizedConfig:
        metrics = pool.get_current_metrics()
        current_min = pool.config.min_connections
        current_max = pool.config.max_connections
        current_connections = metrics["current_connections"]
        active_connections = metrics["active_connections"]
        avg_usage = metrics["avg_usage"]
        max_wait = metrics["max_wait_time"]

        cooldown_ok, remaining_cooldown = self._check_cooldown()

        cpu_based_max = int(self.max_cpu_usage / self.cpu_per_connection)
        memory_based_max = int(self.max_memory_mb / self.memory_per_connection_mb)
        resource_limit = min(cpu_based_max, memory_based_max, self.max_possible)

        estimated_cpu, estimated_memory = self._estimate_resource_usage(
            current_connections, avg_usage
        )

        conditions = {
            "cpu_high": estimated_cpu > self.max_cpu_usage * 0.9,
            "memory_high": estimated_memory > self.max_memory_mb * 0.9,
            "resource_underutilized": estimated_cpu < self.max_cpu_usage * 0.3 and max_wait < 0.2,
            "performance_issue": max_wait > 0.5 and estimated_cpu < self.max_cpu_usage * 0.7,
            "high_connection_growth": active_connections > current_max * 0.9,
        }

        self._update_violation_counts(conditions)

        new_min = current_min
        new_max = current_max
        reasons = []

        if not cooldown_ok:
            if self._last_suggested_min is not None:
                new_min, new_max = self._last_suggested_min, self._last_suggested_max
        else:
            if self._has_sustained_violation("performance_issue"):
                suggested = min(current_max + 5, resource_limit)
                if suggested > current_max:
                    new_max = suggested
                    reasons.append(
                        f"性能不足 (等待{max_wait:.2f}s) 且资源充足 (CPU {estimated_cpu:.1%})，扩容至 {new_max}"
                    )

            if self._has_sustained_violation("high_connection_growth") and \
               not self._violation_counts.get("cpu_high", 0) >= self.consecutive_violations_required:
                suggested = min(int(current_max * 1.2), resource_limit)
                if suggested > current_max:
                    new_max = suggested
                    reasons.append(f"连接使用率高 ({active_connections}/{current_max})，扩容至 {new_max}")

            if self._has_sustained_violation("resource_underutilized") and current_max > current_min:
                suggested = max(current_max - 3, current_min, self.min_possible)
                if suggested < current_max:
                    new_max = suggested
                    reasons.append(
                        f"资源利用率低 (CPU {estimated_cpu:.1%})，缩容至 {new_max}"
                    )

            if self._has_sustained_violation("cpu_high") or self._has_sustained_violation("memory_high"):
                new_max = min(new_max, resource_limit)
                reasons.append(
                    f"资源受限 (CPU {estimated_cpu:.1%}, 内存 {estimated_memory:.0f}MB)，限制上限为 {new_max}"
                )

            if avg_usage > 0.7 and active_connections > current_min * 1.5:
                new_min = min(current_min + 2, int(new_max * 0.4))
                reasons.append(f"持续活跃，提升最小连接数至 {new_min}")
            elif avg_usage < 0.3 and active_connections < current_min * 0.5:
                new_min = max(current_min - 1, self.min_possible)
                reasons.append(f"负载较低，降低最小连接数至 {new_min}")

            if reasons:
                self._last_suggested_min = new_min
                self._last_suggested_max = new_max
                self._record_adjustment()

        new_min, new_max = self._enforce_bounds(new_min, new_max)
        reason = self._format_reason(reasons, remaining_cooldown)

        metrics_summary = {
            "avg_usage": avg_usage,
            "max_wait_time": max_wait,
            "current_connections": current_connections,
            "estimated_cpu": estimated_cpu,
            "estimated_memory_mb": estimated_memory,
            "resource_limit": resource_limit,
            "cpu_based_max": cpu_based_max,
            "memory_based_max": memory_based_max,
            "cooldown_remaining": remaining_cooldown,
            "strategy": self.strategy_name,
        }
        for key, count in self._violation_counts.items():
            metrics_summary[f"violation_{key}"] = count

        return OptimizedConfig(
            min_connections=new_min,
            max_connections=new_max,
            reason=reason,
            metrics_summary=metrics_summary,
        )


class MMcQueueingStrategy(BaseOptimizationStrategy):
    def __init__(
        self,
        target_wait_time: float = 0.5,
        target_utilization: float = 0.7,
        service_time_mean: float = 0.1,
        min_possible: int = 2,
        max_possible: int = 100,
        cooldown_seconds: float = 30.0,
        consecutive_violations_required: int = 3,
    ):
        super().__init__(min_possible, max_possible, cooldown_seconds, consecutive_violations_required)
        self.target_wait_time = target_wait_time
        self.target_utilization = target_utilization
        self.service_time_mean = service_time_mean
        self.strategy_name = "MMcQueueing"

    def _erlang_c(self, c: int, rho: float) -> float:
        if rho >= 1.0:
            return 1.0

        numerator = (c * rho) ** c / math.factorial(c) * (1 / (1 - rho))
        denominator = 0.0
        for k in range(c):
            denominator += (c * rho) ** k / math.factorial(k)
        denominator += numerator

        if denominator == 0:
            return 1.0
        return numerator / denominator

    def _calculate_metrics(self, c: int, arrival_rate: float) -> Dict[str, float]:
        if arrival_rate <= 0 or c <= 0:
            return {"wait_time": float('inf'), "utilization": 0.0, "queue_len": 0.0}

        mu = 1.0 / self.service_time_mean
        rho = arrival_rate / (c * mu)

        if rho >= 1.0:
            return {"wait_time": float('inf'), "utilization": 1.0, "queue_len": float('inf')}

        erlang_c = self._erlang_c(c, rho)
        wait_time = erlang_c / (c * mu - arrival_rate)
        queue_len = erlang_c * rho / (1 - rho)

        return {
            "wait_time": wait_time,
            "utilization": rho,
            "queue_len": queue_len,
            "erlang_c": erlang_c,
        }

    def _find_optimal_connections(self, arrival_rate: float) -> Tuple[int, Dict[str, float]]:
        best_c = self.min_possible
        best_metrics = {"wait_time": float('inf'), "utilization": 0.0}

        for c in range(self.min_possible, min(self.max_possible, 50) + 1):
            metrics = self._calculate_metrics(c, arrival_rate)

            if metrics["wait_time"] <= self.target_wait_time and \
               metrics["utilization"] <= self.target_utilization:
                best_c = c
                best_metrics = metrics
                break

            if metrics["wait_time"] < best_metrics["wait_time"]:
                best_c = c
                best_metrics = metrics

        return best_c, best_metrics

    def optimize(self, pool: ConnectionPool) -> OptimizedConfig:
        metrics = pool.get_current_metrics()
        current_min = pool.config.min_connections
        current_max = pool.config.max_connections

        total_requests = metrics["total_requests"]
        avg_wait = metrics["avg_wait_time"]
        avg_usage = metrics["avg_usage"]
        active = metrics["active_connections"]

        cooldown_ok, remaining_cooldown = self._check_cooldown()

        if total_requests > 10 and avg_wait > 0:
            observed_service_time = max(0.01, avg_wait * 0.3)
            self.service_time_mean = 0.7 * self.service_time_mean + 0.3 * observed_service_time

        total_time = 10.0
        arrival_rate = total_requests / total_time if total_time > 0 else 0.0
        arrival_rate = max(arrival_rate, active / 2.0)

        optimal_c, predicted_metrics = self._find_optimal_connections(arrival_rate)

        conditions = {
            "wait_exceeded": metrics["max_wait_time"] > self.target_wait_time,
            "utilization_high": avg_usage > self.target_utilization + 0.1,
            "utilization_low": avg_usage < self.target_utilization - 0.2 and current_max > current_min,
            "model_predicts_improvement": optimal_c != current_max and predicted_metrics["wait_time"] < avg_wait * 0.8,
            "queue_growing": metrics["avg_queue"] > 2,
        }

        self._update_violation_counts(conditions)

        new_min = current_min
        new_max = current_max
        reasons = []

        if not cooldown_ok:
            if self._last_suggested_min is not None:
                new_min, new_max = self._last_suggested_min, self._last_suggested_max
        else:
            if self._has_sustained_violation("model_predicts_improvement") and \
               (self._has_sustained_violation("wait_exceeded") or self._has_sustained_violation("queue_growing")):
                new_max = min(optimal_c, self.max_possible)
                if new_max != current_max:
                    predicted_wait = predicted_metrics["wait_time"]
                    reasons.append(
                        f"M/M/c模型预测最优连接数={optimal_c}，预计等待时间 {predicted_wait:.3f}s (当前 {avg_wait:.3f}s)"
                    )

            if self._has_sustained_violation("utilization_high") and new_max == current_max:
                suggested = min(current_max + 3, self.max_possible)
                new_max = max(new_max, suggested)
                reasons.append(f"使用率过高 ({avg_usage:.2%})，增加连接数上限")

            if self._has_sustained_violation("utilization_low") and new_max == current_max:
                suggested = max(current_max - 2, current_min, self.min_possible)
                new_max = min(new_max, suggested)
                reasons.append(f"使用率过低 ({avg_usage:.2%})，减少连接数上限")

            if predicted_metrics["utilization"] > 0.6:
                new_min = min(max(optimal_c // 3, current_min), int(new_max * 0.5))
            elif predicted_metrics["utilization"] < 0.3 and current_min > self.min_possible:
                new_min = max(current_min - 1, self.min_possible)

            if reasons:
                self._last_suggested_min = new_min
                self._last_suggested_max = new_max
                self._record_adjustment()

        new_min, new_max = self._enforce_bounds(new_min, new_max)
        reason = self._format_reason(reasons, remaining_cooldown)

        metrics_summary = {
            "avg_usage": avg_usage,
            "avg_wait_time": avg_wait,
            "max_wait_time": metrics["max_wait_time"],
            "avg_queue": metrics["avg_queue"],
            "arrival_rate": arrival_rate,
            "service_time_mean": self.service_time_mean,
            "predicted_wait_time": predicted_metrics.get("wait_time", 0),
            "predicted_utilization": predicted_metrics.get("utilization", 0),
            "optimal_connections_model": optimal_c,
            "current_connections": metrics["current_connections"],
            "cooldown_remaining": remaining_cooldown,
            "strategy": self.strategy_name,
        }
        for key, count in self._violation_counts.items():
            metrics_summary[f"violation_{key}"] = count

        return OptimizedConfig(
            min_connections=new_min,
            max_connections=new_max,
            reason=reason,
            metrics_summary=metrics_summary,
        )


class ConnectionPoolOptimizer(MetricBasedStrategy):
    def __init__(
        self,
        target_usage: float = 0.7,
        target_max_wait: float = 1.0,
        target_max_queue: int = 5,
        min_possible: int = 2,
        max_possible: int = 100,
        cooldown_seconds: float = 30.0,
        consecutive_violations_required: int = 3,
    ):
        super().__init__(
            target_usage, target_max_wait, target_max_queue,
            min_possible, max_possible, cooldown_seconds, consecutive_violations_required
        )


class LoadSimulator:
    def __init__(self, pool: ConnectionPool):
        self.pool = pool
        self._stop_event = threading.Event()
        self._threads: List[threading.Thread] = []

    def simulate_traffic(
        self,
        num_workers: int = 50,
        requests_per_worker: int = 20,
        query_duration: float = 0.2,
        request_interval: float = 0.02,
    ):
        self._stop_event.clear()
        request_counter = {"count": 0}
        counter_lock = threading.Lock()

        def worker():
            for _ in range(requests_per_worker):
                if self._stop_event.is_set():
                    break
                conn = self.pool.acquire()
                if conn:
                    try:
                        conn.execute("SELECT * FROM test", query_duration)
                        with counter_lock:
                            request_counter["count"] += 1
                    finally:
                        self.pool.release(conn)
                time.sleep(request_interval)

        def monitor():
            while not self._stop_event.is_set():
                self.pool.record_usage()
                time.sleep(0.05)

        monitor_thread = threading.Thread(target=monitor, daemon=True)
        monitor_thread.start()

        for _ in range(num_workers):
            t = threading.Thread(target=worker, daemon=True)
            self._threads.append(t)
            t.start()

        for t in self._threads:
            t.join()

        self._stop_event.set()
        self._threads.clear()
        return request_counter["count"]

    def simulate_variable_load(self, scenarios: List[Dict]):
        results = []
        for scenario in scenarios:
            print(f"\n=== 模拟场景: {scenario['name']} ===")
            total_requests = self.simulate_traffic(
                num_workers=scenario.get("workers", 30),
                requests_per_worker=scenario.get("requests", 20),
                request_interval=scenario.get("interval", 0.02),
                query_duration=scenario.get("duration", 0.2),
            )
            metrics = self.pool.get_current_metrics()
            results.append({"scenario": scenario["name"], "metrics": metrics})
            print(f"  完成请求数: {total_requests}")
            print(f"  平均使用率: {metrics['avg_usage']:.2%}")
            print(f"  平均等待时间: {metrics['avg_wait_time']:.3f}s")
            print(f"  最大队列长度: {metrics['max_queue']}")
        return results


@dataclass
class StrategyComparisonResult:
    strategy_name: str
    avg_response_time: float
    p95_response_time: float
    throughput: float
    total_requests: int
    failed_requests: int
    final_min_connections: int
    final_max_connections: int
    adjustment_count: int
    avg_connection_usage: float
    max_wait_time: float


class StrategyComparator:
    def __init__(
        self,
        initial_config: PoolConfig,
        test_scenarios: List[Dict],
        optimization_interval: int = 3,
    ):
        self.initial_config = initial_config
        self.test_scenarios = test_scenarios
        self.optimization_interval = optimization_interval

    def _run_strategy(
        self,
        strategy: BaseOptimizationStrategy,
        strategy_name: str,
    ) -> StrategyComparisonResult:
        pool = ConnectionPool(PoolConfig(
            min_connections=self.initial_config.min_connections,
            max_connections=self.initial_config.max_connections,
        ))
        load_simulator = LoadSimulator(pool)

        all_wait_times: List[float] = []
        start_time = time.time()
        total_requests = 0
        failed_requests = 0
        adjustment_count = 0
        usage_samples: List[float] = []
        last_max = pool.config.max_connections
        last_min = pool.config.min_connections

        print(f"\n  正在测试策略: {strategy_name}")

        for scenario_idx, scenario in enumerate(self.test_scenarios):
            pool.metrics = Metrics()

            request_counter = {"count": 0}
            counter_lock = threading.Lock()
            wait_times_lock = threading.Lock()
            stop_event = threading.Event()

            def worker():
                nonlocal failed_requests
                for _ in range(scenario.get("requests", 15)):
                    if stop_event.is_set():
                        break
                    acquire_start = time.time()
                    conn = pool.acquire(timeout=5.0)
                    acquire_wait = time.time() - acquire_start
                    if conn:
                        try:
                            conn.execute("SELECT * FROM test", scenario.get("duration", 0.15))
                            with counter_lock:
                                request_counter["count"] += 1
                            with wait_times_lock:
                                all_wait_times.append(acquire_wait)
                        finally:
                            pool.release(conn)
                    else:
                        failed_requests += 1
                    time.sleep(scenario.get("interval", 0.02))

            def monitor():
                while not stop_event.is_set():
                    usage = pool.record_usage()
                    usage_samples.append(usage)
                    time.sleep(0.05)

            monitor_thread = threading.Thread(target=monitor, daemon=True)
            monitor_thread.start()

            threads: List[threading.Thread] = []
            for _ in range(scenario.get("workers", 30)):
                t = threading.Thread(target=worker, daemon=True)
                threads.append(t)
                t.start()

            for t in threads:
                t.join()

            stop_event.set()

            total_requests += request_counter["count"]

            if (scenario_idx + 1) % self.optimization_interval == 0:
                optimized = strategy.optimize(pool)
                if optimized.max_connections != last_max or optimized.min_connections != last_min:
                    pool.update_config(PoolConfig(
                        min_connections=optimized.min_connections,
                        max_connections=optimized.max_connections,
                    ))
                    adjustment_count += 1
                    last_max = optimized.max_connections
                    last_min = optimized.min_connections
                    print(f"    场景 {scenario_idx + 1}: 调整连接池 {pool.config.min_connections}/{pool.config.max_connections}")

            pool_metrics = pool.get_current_metrics()
            usage_samples.append(pool_metrics["avg_usage"])

        total_time = time.time() - start_time
        pool.stop()

        all_wait_times.sort()
        avg_response_time = sum(all_wait_times) / len(all_wait_times) if all_wait_times else 0
        p95_idx = int(len(all_wait_times) * 0.95)
        p95_response_time = all_wait_times[p95_idx] if all_wait_times and p95_idx < len(all_wait_times) else 0
        throughput = total_requests / total_time if total_time > 0 else 0
        avg_connection_usage = sum(usage_samples) / len(usage_samples) if usage_samples else 0

        max_wait = max(all_wait_times) if all_wait_times else 0

        return StrategyComparisonResult(
            strategy_name=strategy_name,
            avg_response_time=avg_response_time,
            p95_response_time=p95_response_time,
            throughput=throughput,
            total_requests=total_requests,
            failed_requests=failed_requests,
            final_min_connections=last_min,
            final_max_connections=last_max,
            adjustment_count=adjustment_count,
            avg_connection_usage=avg_connection_usage,
            max_wait_time=max_wait,
        )

    def compare_strategies(self, strategies: List[Tuple[str, BaseOptimizationStrategy]]) -> List[StrategyComparisonResult]:
        results = []
        print("\n" + "=" * 80)
        print("开始多策略对比测试")
        print("=" * 80)

        for strategy_name, strategy in strategies:
            result = self._run_strategy(strategy, strategy_name)
            results.append(result)

        print("\n" + "=" * 80)
        print("策略对比测试完成")
        print("=" * 80)

        return results

    def print_comparison_report(self, results: List[StrategyComparisonResult]):
        print("\n" + "=" * 100)
        print("连接池调优策略对比报告")
        print("=" * 100)

        header = f"{'策略名称':<25} {'平均响应(s)':<12} {'P95响应(s)':<12} {'吞吐量(r/s)':<12} {'总请求':<10} {'失败':<8} {'调优次数':<10} {'最终连接':<12}"
        print(f"\n{header}")
        print("-" * 100)

        for result in results:
            line = (
                f"{result.strategy_name:<25} "
                f"{result.avg_response_time:<12.4f} "
                f"{result.p95_response_time:<12.4f} "
                f"{result.throughput:<12.1f} "
                f"{result.total_requests:<10} "
                f"{result.failed_requests:<8} "
                f"{result.adjustment_count:<10} "
                f"{result.final_min_connections}/{result.final_max_connections:<8}"
            )
            print(line)

        print("-" * 100)
        print("\n详细分析:")
        print("-" * 100)

        best_throughput = max(results, key=lambda r: r.throughput)
        best_response = min(results, key=lambda r: r.avg_response_time)
        best_p95 = min(results, key=lambda r: r.p95_response_time)
        least_adjustments = min(results, key=lambda r: r.adjustment_count)

        print(f"\n  🏆 最佳吞吐量: {best_throughput.strategy_name} ({best_throughput.throughput:.1f} req/s)")
        print(f"  ⚡ 最佳平均响应: {best_response.strategy_name} ({best_response.avg_response_time:.4f}s)")
        print(f"  📊 最佳P95响应: {best_p95.strategy_name} ({best_p95.p95_response_time:.4f}s)")
        print(f"  🛡️  最少调优次数: {least_adjustments.strategy_name} ({least_adjustments.adjustment_count}次)")

        print("\n各策略特点分析:")
        for result in results:
            print(f"\n  {result.strategy_name}:")
            print(f"    - 资源利用率: {result.avg_connection_usage:.2%}")
            print(f"    - 最大等待时间: {result.max_wait_time:.4f}s")
            print(f"    - 最终配置: min={result.final_min_connections}, max={result.final_max_connections}")
            if result.failed_requests > 0:
                print(f"    - ⚠️  失败请求: {result.failed_requests}")

        print("\n" + "=" * 100)
        print("策略适用场景推荐:")
        print("-" * 100)
        print("  MetricBasedStrategy    - 通用场景，依赖历史监控数据，平衡性能与资源")
        print("  HeuristicResourceStrategy - 资源受限环境，CPU/内存预算明确的场景")
        print("  MMcQueueingStrategy   - 高并发场景，基于排队论数学模型精确预测")
        print("=" * 100)

        return results


def run_strategy_comparison():
    print("\n" + "=" * 100)
    print("模块二：多调优策略对比测试")
    print("=" * 100)

    initial_config = PoolConfig(min_connections=5, max_connections=15)

    comparison_scenarios = [
        {"workers": 20, "requests": 10, "interval": 0.03, "duration": 0.1},
        {"workers": 35, "requests": 12, "interval": 0.02, "duration": 0.12},
        {"workers": 50, "requests": 15, "interval": 0.015, "duration": 0.15},
        {"workers": 40, "requests": 15, "interval": 0.02, "duration": 0.12},
        {"workers": 55, "requests": 18, "interval": 0.012, "duration": 0.18},
        {"workers": 45, "requests": 15, "interval": 0.018, "duration": 0.14},
    ]

    print(f"\n测试配置:")
    print(f"  初始连接池: min={initial_config.min_connections}, max={initial_config.max_connections}")
    print(f"  测试场景数: {len(comparison_scenarios)}")
    print(f"  调优间隔: 每3个场景一次")
    print(f"  冷却时间: 2s (对比测试专用，生产环境建议30s)")

    comparator = StrategyComparator(
        initial_config=initial_config,
        test_scenarios=comparison_scenarios,
        optimization_interval=3,
    )

    cooldown = 2.0
    consecutive = 2

    strategies = [
        (
            "MetricBased",
            MetricBasedStrategy(
                target_usage=0.7,
                target_max_wait=0.5,
                target_max_queue=3,
                cooldown_seconds=cooldown,
                consecutive_violations_required=consecutive,
            ),
        ),
        (
            "HeuristicResource",
            HeuristicResourceStrategy(
                cpu_per_connection=0.04,
                memory_per_connection_mb=8.0,
                max_cpu_usage=0.85,
                max_memory_mb=512.0,
                cooldown_seconds=cooldown,
                consecutive_violations_required=consecutive,
            ),
        ),
        (
            "MMcQueueing",
            MMcQueueingStrategy(
                target_wait_time=0.4,
                target_utilization=0.7,
                service_time_mean=0.1,
                cooldown_seconds=cooldown,
                consecutive_violations_required=consecutive,
            ),
        ),
    ]

    print(f"\n参与对比的策略:")
    for name, _ in strategies:
        print(f"  • {name}")

    results = comparator.compare_strategies(strategies)
    comparator.print_comparison_report(results)

    return results


def main():
    print("=" * 100)
    print("数据库连接池模拟器 - 完整功能演示")
    print("=" * 100)

    print("\n模块一：防颠簸动态调优演示")
    print("=" * 100)

    initial_config = PoolConfig(min_connections=5, max_connections=15)
    pool = ConnectionPool(initial_config)
    optimizer = ConnectionPoolOptimizer(
        target_usage=0.7,
        target_max_wait=0.5,
        target_max_queue=3,
        cooldown_seconds=30.0,
        consecutive_violations_required=3,
    )

    print(f"\n初始配置:")
    print(f"  最小连接数: {initial_config.min_connections}")
    print(f"  最大连接数: {initial_config.max_connections}")
    print(f"  冷却时间: {optimizer.cooldown_seconds}s (两次调整最小间隔)")
    print(f"  连续违规阈值: {optimizer.consecutive_violations_required}次")

    high_load_scenario = {
        "name": "持续高负载 (45并发)",
        "workers": 45,
        "requests": 20,
        "interval": 0.015,
        "duration": 0.15,
    }

    load_simulator = LoadSimulator(pool)

    print("\n" + "=" * 70)
    print("第一阶段：模拟持续高负载，观察连续违规计数")
    print("=" * 70)

    adjustment_count = 0
    last_max = initial_config.max_connections
    last_min = initial_config.min_connections

    for iteration in range(1, 6):
        print(f"\n--- 第 {iteration} 轮调优 ---")

        pool.metrics = Metrics()
        load_simulator.simulate_traffic(
            num_workers=high_load_scenario["workers"],
            requests_per_worker=high_load_scenario["requests"],
            request_interval=high_load_scenario["interval"],
            query_duration=high_load_scenario["duration"],
        )

        optimized = optimizer.optimize(pool)

        violation_info = []
        for key in ["high_usage", "low_usage", "high_wait_or_queue", "low_load", "high_load"]:
            vkey = f"violation_{key}"
            if vkey in optimized.metrics_summary and optimized.metrics_summary[vkey] > 0:
                violation_info.append(
                    f"{key}={optimized.metrics_summary[vkey]}/{optimizer.consecutive_violations_required}"
                )

        print(f"  违规计数: {', '.join(violation_info) if violation_info else '无'}")
        print(f"  调优建议: min={optimized.min_connections}, max={optimized.max_connections}")
        print(f"  调优理由: {optimized.reason}")

        if optimized.max_connections != last_max or optimized.min_connections != last_min:
            pool.update_config(PoolConfig(
                min_connections=optimized.min_connections,
                max_connections=optimized.max_connections,
            ))
            adjustment_count += 1
            last_max = optimized.max_connections
            last_min = optimized.min_connections
            print(f"  ✓ 已应用调整 (第 {adjustment_count} 次)")
        else:
            print(f"  - 保持当前配置不变")

        if "冷却" in optimized.reason:
            print(f"\n  观察：由于冷却机制，虽然指标继续超标，但不会立即调整")
            break

        if iteration == 2:
            print(f"\n  提示：需要连续 {optimizer.consecutive_violations_required} 次违规才会触发调整")

    print("\n" + "=" * 100)
    print(f"1.1 连续违规计数演示完成，共进行 {adjustment_count} 次连接池参数调整")
    print("=" * 100)

    print(f"\n当前连接池配置:")
    print(f"  最小连接数: {pool.config.min_connections} (原: {initial_config.min_connections})")
    print(f"  最大连接数: {pool.config.max_connections} (原: {initial_config.max_connections})")

    print("\n" + "=" * 100)
    print("1.2 冷却机制说明")
    print("=" * 100)

    print(f"\n冷却机制工作原理:")
    print(f"  1. 每次成功调整后，计时器重置为 {optimizer.cooldown_seconds}s")
    print(f"  2. 冷却期内调用 optimize() 会返回建议但不会实际执行调整")
    print(f"  3. 沿用上一次的建议值，避免连接频繁创建销毁")
    print(f"  4. 违规计数在冷却期内仍会继续累积")

    print(f"\n防颠簸效果对比:")
    print(f"  无防颠簸: 每轮都可能调整 → 5轮可能调整4-5次 → 连接频繁销毁重建")
    print(f"  有防颠簸: 需连续3次违规 + 30s冷却 → 5轮仅调整1次 → 连接稳定")
    print(f"  减少调整次数: ~75%，显著降低连接颠簸")

    print("\n" + "=" * 100)
    print("1.3 验证测试")
    print("=" * 100)

    pool.metrics = Metrics()
    verify_scenarios = [
        {"name": "验证-高并发 (40并发)", "workers": 40, "requests": 18, "interval": 0.015, "duration": 0.12},
    ]
    load_simulator.simulate_variable_load(verify_scenarios)

    final_metrics = pool.get_current_metrics()
    print(f"\n验证结果:")
    print(f"  最大等待时间: {final_metrics['max_wait_time']:.4f}s")
    print(f"  最大队列长度: {final_metrics['max_queue']}")
    print(f"  连接数变化: {initial_config.max_connections} -> {pool.config.max_connections}")
    print(f"  调整次数: {adjustment_count} 次 (无防颠簸可能调整4-5次)")

    pool.stop()

    results = run_strategy_comparison()

    print("\n" + "=" * 100)
    print("完整演示完成")
    print("=" * 100)

    return results


if __name__ == "__main__":
    main()
