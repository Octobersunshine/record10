import time
import threading
import psutil
import math
from collections import deque
from enum import Enum, auto
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Deque


class BBRPhase(Enum):
    STARTUP = auto()
    DRAIN = auto()
    PROBE_BW = auto()
    PROBE_RTT = auto()


@dataclass
class LimiterStats:
    true_positives: int = 0
    true_negatives: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    total_requests: int = 0
    accepted_requests: int = 0
    rejected_requests: int = 0


class RateLimiter(ABC):
    """Abstract base class for all rate limiters."""

    def __init__(self, name: str):
        self.name = name
        self.stats = LimiterStats()
        self._lock = threading.Lock()

    @abstractmethod
    def acquire(self) -> bool:
        """Return True if request should be accepted, False if rejected."""
        pass

    @abstractmethod
    def record_latency(self, latency_ms: float) -> None:
        """Record request latency for adaptive algorithms."""
        pass

    @abstractmethod
    def get_threshold(self) -> int:
        """Return current rate limit threshold."""
        pass

    @abstractmethod
    def start(self) -> None:
        """Start background monitoring threads if any."""
        pass

    @abstractmethod
    def stop(self) -> None:
        """Stop background monitoring threads."""
        pass

    def get_metrics(self) -> dict:
        with self._lock:
            tp = self.stats.true_positives
            tn = self.stats.true_negatives
            fp = self.stats.false_positives
            fn = self.stats.false_negatives
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
            return {
                "name": self.name,
                "threshold": self.get_threshold(),
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1": round(f1, 4),
                "accuracy": round(accuracy, 4),
                "tp": tp,
                "tn": tn,
                "fp": fp,
                "fn": fn,
                "total": self.stats.total_requests,
                "accepted": self.stats.accepted_requests,
                "rejected": self.stats.rejected_requests,
            }

    def record_outcome(self, accepted: bool, should_accept: bool) -> None:
        """Record classification outcome for F1 calculation."""
        with self._lock:
            self.stats.total_requests += 1
            if accepted:
                self.stats.accepted_requests += 1
            else:
                self.stats.rejected_requests += 1
            if accepted and should_accept:
                self.stats.true_positives += 1
            elif not accepted and not should_accept:
                self.stats.true_negatives += 1
            elif not accepted and should_accept:
                self.stats.false_positives += 1
            elif accepted and not should_accept:
                self.stats.false_negatives += 1


# ======================================================================
# Algorithm 1: Token Bucket (fixed rate)
# ======================================================================

class TokenBucketLimiter(RateLimiter):
    """Classic token bucket algorithm with fixed refill rate."""

    def __init__(self, rate: int = 100, capacity: int = 200):
        super().__init__(f"TokenBucket({rate})")
        self._rate = rate
        self._capacity = capacity
        self._tokens = float(capacity)
        self._last_refill = time.monotonic()

    def acquire(self) -> bool:
        with self._lock:
            now = time.monotonic()
            elapsed = now - self._last_refill
            self._tokens = min(self._capacity, self._tokens + elapsed * self._rate)
            self._last_refill = now
            if self._tokens >= 1:
                self._tokens -= 1
                return True
            return False

    def record_latency(self, latency_ms: float) -> None:
        pass

    def get_threshold(self) -> int:
        return self._rate

    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass


# ======================================================================
# Algorithm 2: Sliding Window Counter
# ======================================================================

class SlidingWindowLimiter(RateLimiter):
    """Sliding window counter algorithm."""

    def __init__(self, rate: int = 100, window_sec: float = 1.0):
        super().__init__(f"SlidingWindow({rate}/s)")
        self._rate = rate
        self._window_sec = window_sec
        self._requests: Deque = deque()

    def acquire(self) -> bool:
        with self._lock:
            now = time.monotonic()
            while self._requests and self._requests[0] < now - self._window_sec:
                self._requests.popleft()
            if len(self._requests) < self._rate:
                self._requests.append(now)
                return True
            return False

    def record_latency(self, latency_ms: float) -> None:
        pass

    def get_threshold(self) -> int:
        return self._rate

    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass


# ======================================================================
# Algorithm 3: BBR-inspired (improved with EWMA)
# ======================================================================

class BBRRateLimiter(RateLimiter):
    """BBR-inspired adaptive rate limiter with EWMA-smoothed signals."""

    def __init__(
        self,
        min_threshold: int = 10,
        max_threshold: int = 5000,
        initial_threshold: int = 100,
        cpu_high_watermark: float = 0.80,
        latency_spike_factor: float = 2.0,
        ewma_alpha_rtt: float = 0.2,
        ewma_alpha_cpu: float = 0.3,
        ewma_alpha_baseline: float = 0.05,
        min_adjust_interval: float = 5.0,
        monitor_interval: float = 0.5,
    ):
        super().__init__("BBR-Adaptive")
        self._min_threshold = min_threshold
        self._max_threshold = max_threshold
        self._cpu_high_watermark = cpu_high_watermark
        self._latency_spike_factor = latency_spike_factor
        self._ewma_alpha_rtt = ewma_alpha_rtt
        self._ewma_alpha_cpu = ewma_alpha_cpu
        self._ewma_alpha_baseline = ewma_alpha_baseline
        self._min_adjust_interval = min_adjust_interval
        self._monitor_interval = monitor_interval

        self._threshold = float(initial_threshold)
        self._phase = BBRPhase.STARTUP
        self._ewma_rtt: float | None = None
        self._ewma_baseline_rtt: float | None = None
        self._ewma_cpu: float | None = None
        self._rtt_samples: Deque = deque(maxlen=60)
        self._bw_samples: Deque = deque(maxlen=60)
        self._min_rtt = float("inf")
        self._max_bw = 0.0
        self._round_count = 0
        self._last_rtt_probe_time = time.monotonic()
        self._phase_entered_time = time.monotonic()
        self._last_adjust_time: float = 0.0
        self._request_count = 0
        self._last_count_time = time.monotonic()
        self._monitor_thread: threading.Thread | None = None
        self._stop_event = threading.Event()

    def start(self) -> None:
        psutil.cpu_percent(interval=0.1)
        self._ewma_cpu = psutil.cpu_percent(interval=0.1) / 100.0
        self._last_adjust_time = time.monotonic()
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._monitor_thread is not None:
            self._monitor_thread.join(timeout=5)

    def acquire(self) -> bool:
        with self._lock:
            if self._request_count < int(self._threshold):
                self._request_count += 1
                return True
            return False

    def record_latency(self, latency_ms: float) -> None:
        with self._lock:
            self._rtt_samples.append(latency_ms)
            if latency_ms < self._min_rtt:
                self._min_rtt = latency_ms
            if self._ewma_rtt is None:
                self._ewma_rtt = latency_ms
            else:
                self._ewma_rtt = (
                    self._ewma_alpha_rtt * latency_ms
                    + (1 - self._ewma_alpha_rtt) * self._ewma_rtt
                )
            if self._ewma_baseline_rtt is None:
                self._ewma_baseline_rtt = latency_ms
            elif latency_ms < self._ewma_baseline_rtt:
                self._ewma_baseline_rtt = (
                    self._ewma_alpha_baseline * latency_ms
                    + (1 - self._ewma_alpha_baseline) * self._ewma_baseline_rtt
                )
            else:
                self._ewma_baseline_rtt = (
                    self._ewma_alpha_baseline * 0.1 * latency_ms
                    + (1 - self._ewma_alpha_baseline * 0.1) * self._ewma_baseline_rtt
                )

    def get_threshold(self) -> int:
        with self._lock:
            return int(self._threshold)

    def _monitor_loop(self) -> None:
        while not self._stop_event.is_set():
            self._update_ewma_cpu()
            self._update_bandwidth()
            self._update_phase()
            self._update_threshold()
            self._stop_event.wait(self._monitor_interval)

    def _update_ewma_cpu(self) -> None:
        raw_cpu = psutil.cpu_percent(interval=None) / 100.0
        with self._lock:
            if self._ewma_cpu is None:
                self._ewma_cpu = raw_cpu
            else:
                self._ewma_cpu = (
                    self._ewma_alpha_cpu * raw_cpu
                    + (1 - self._ewma_alpha_cpu) * self._ewma_cpu
                )

    def _update_bandwidth(self) -> None:
        now = time.monotonic()
        with self._lock:
            elapsed = now - self._last_count_time
            if elapsed <= 0:
                return
            bw = self._request_count / elapsed
            self._bw_samples.append(bw)
            if bw > self._max_bw:
                self._max_bw = bw
            self._request_count = 0
            self._last_count_time = now

    def _update_phase(self) -> None:
        now = time.monotonic()
        with self._lock:
            cpu_congested = self._ewma_cpu is not None and self._ewma_cpu > self._cpu_high_watermark
            latency_congested = self._is_latency_spiked()
            if self._phase == BBRPhase.STARTUP:
                if cpu_congested or latency_congested:
                    self._phase = BBRPhase.DRAIN
                    self._phase_entered_time = now
                elif self._round_count >= 3 and self._bw_plateaued():
                    self._phase = BBRPhase.PROBE_BW
                    self._phase_entered_time = now
            elif self._phase == BBRPhase.DRAIN:
                time_in_drain = now - self._phase_entered_time
                if time_in_drain > 5.0:
                    self._phase = BBRPhase.PROBE_BW
                    self._phase_entered_time = now
                elif time_in_drain > 1.0 and not latency_congested and not cpu_congested:
                    self._phase = BBRPhase.PROBE_BW
                    self._phase_entered_time = now
            elif self._phase == BBRPhase.PROBE_BW:
                if cpu_congested or latency_congested:
                    self._phase = BBRPhase.PROBE_RTT
                    self._phase_entered_time = now
                    self._last_rtt_probe_time = now
                elif (now - self._last_rtt_probe_time) > 10.0:
                    self._phase = BBRPhase.PROBE_RTT
                    self._phase_entered_time = now
                    self._last_rtt_probe_time = now
            elif self._phase == BBRPhase.PROBE_RTT:
                if (now - self._phase_entered_time) > 0.2:
                    if not cpu_congested and not latency_congested:
                        self._phase = BBRPhase.PROBE_BW
                        self._phase_entered_time = now
                    else:
                        self._phase = BBRPhase.DRAIN
                        self._phase_entered_time = now

    def _update_threshold(self) -> None:
        now = time.monotonic()
        with self._lock:
            if self._phase == BBRPhase.STARTUP:
                self._threshold *= 2.0
                self._round_count += 1
                self._last_adjust_time = now
            elif self._phase == BBRPhase.DRAIN:
                self._threshold *= 0.7
                self._last_adjust_time = now
            elif self._phase == BBRPhase.PROBE_BW:
                elapsed_since_adjust = now - self._last_adjust_time
                if elapsed_since_adjust < self._min_adjust_interval:
                    return
                cycle = int(now * 0.2) % 8
                if cycle == 0:
                    self._threshold *= 1.25
                elif cycle == 4:
                    self._threshold *= 0.75
                else:
                    if self._max_bw > 0:
                        target = self._max_bw * 2.0
                        self._threshold = self._threshold * 0.875 + target * 0.125
                self._last_adjust_time = now
            elif self._phase == BBRPhase.PROBE_RTT:
                self._threshold = max(self._min_threshold, self._threshold * 0.5)
                self._last_adjust_time = now
            self._threshold = max(self._min_threshold, min(self._max_threshold, self._threshold))

    def _is_latency_spiked(self) -> bool:
        if self._ewma_rtt is None or self._ewma_baseline_rtt is None:
            return False
        if self._ewma_baseline_rtt <= 0:
            return False
        return self._ewma_rtt > self._ewma_baseline_rtt * self._latency_spike_factor

    def _bw_plateaued(self) -> bool:
        if len(self._bw_samples) < 3:
            return False
        recent = list(self._bw_samples)[-3:]
        if recent[0] == 0:
            return False
        growth = (recent[-1] - recent[0]) / recent[0]
        return growth < 0.25


# ======================================================================
# Algorithm 4: Queueing Theory (M/M/1) Multi-Dimensional Limiter
# ======================================================================

class QueueingTheoryLimiter(RateLimiter):
    """
    Multi-dimensional adaptive limiter based on M/M/1 queueing theory.

    System load factor ρ = λ/μ (arrival rate / service rate)
    - CPU load: normalized CPU utilization
    - Memory load: normalized memory utilization
    - IO load: normalized disk I/O utilization

    Dynamic threshold = base_rate * (1 - ρ) * safety_margin
    where ρ = max(ρ_cpu, ρ_mem, ρ_io)
    """

    def __init__(
        self,
        base_rate: int = 200,
        min_rate: int = 10,
        max_rate: int = 5000,
        safety_margin: float = 0.7,
        target_latency_ms: float = 100.0,
        cpu_weight: float = 1.0,
        mem_weight: float = 1.0,
        io_weight: float = 1.0,
        ewma_alpha: float = 0.3,
        monitor_interval: float = 0.5,
    ):
        super().__init__("QueueingTheory-MD")
        self._base_rate = base_rate
        self._min_rate = min_rate
        self._max_rate = max_rate
        self._safety_margin = safety_margin
        self._target_latency_ms = target_latency_ms
        self._cpu_weight = cpu_weight
        self._mem_weight = mem_weight
        self._io_weight = io_weight
        self._ewma_alpha = ewma_alpha
        self._monitor_interval = monitor_interval

        self._current_threshold = float(base_rate)
        self._ewma_cpu = 0.0
        self._ewma_mem = 0.0
        self._ewma_io = 0.0
        self._ewma_latency = 0.0
        self._ewma_arrival_rate = 0.0
        self._ewma_service_rate = 0.0

        self._last_io_read = 0
        self._last_io_write = 0
        self._last_io_time = 0.0

        self._request_count = 0
        self._last_count_time = time.monotonic()

        self._monitor_thread: threading.Thread | None = None
        self._stop_event = threading.Event()

    def start(self) -> None:
        psutil.cpu_percent(interval=0.1)
        io_counters = psutil.disk_io_counters()
        self._last_io_read = io_counters.read_bytes
        self._last_io_write = io_counters.write_bytes
        self._last_io_time = time.monotonic()
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._monitor_thread is not None:
            self._monitor_thread.join(timeout=5)

    def acquire(self) -> bool:
        with self._lock:
            self._request_count += 1
            return self._request_count <= int(self._current_threshold)

    def record_latency(self, latency_ms: float) -> None:
        with self._lock:
            if self._ewma_latency == 0:
                self._ewma_latency = latency_ms
            else:
                self._ewma_latency = (
                    self._ewma_alpha * latency_ms
                    + (1 - self._ewma_alpha) * self._ewma_latency
                )

    def get_threshold(self) -> int:
        with self._lock:
            return int(self._current_threshold)

    def _monitor_loop(self) -> None:
        while not self._stop_event.is_set():
            self._update_metrics()
            self._update_threshold()
            self._stop_event.wait(self._monitor_interval)

    def _update_metrics(self) -> None:
        now = time.monotonic()
        raw_cpu = psutil.cpu_percent(interval=None) / 100.0
        raw_mem = psutil.virtual_memory().percent / 100.0

        io_counters = psutil.disk_io_counters()
        io_elapsed = now - self._last_io_time
        if io_elapsed > 0:
            io_bytes_delta = (io_counters.read_bytes - self._last_io_read) + (io_counters.write_bytes - self._last_io_write)
            raw_io = min(1.0, io_bytes_delta / io_elapsed / 100_000_000)
            self._last_io_read = io_counters.read_bytes
            self._last_io_write = io_counters.write_bytes
            self._last_io_time = now
        else:
            raw_io = 0.0

        elapsed = now - self._last_count_time
        if elapsed > 0:
            arrival_rate = self._request_count / elapsed
            service_rate = 1000.0 / max(1, self._ewma_latency) if self._ewma_latency > 0 else arrival_rate
        else:
            arrival_rate = 0.0
            service_rate = 0.0

        with self._lock:
            self._ewma_cpu = self._ewma_alpha * raw_cpu + (1 - self._ewma_alpha) * self._ewma_cpu
            self._ewma_mem = self._ewma_alpha * raw_mem + (1 - self._ewma_alpha) * self._ewma_mem
            self._ewma_io = self._ewma_alpha * raw_io + (1 - self._ewma_alpha) * self._ewma_io
            self._ewma_arrival_rate = 0.2 * arrival_rate + 0.8 * self._ewma_arrival_rate
            self._ewma_service_rate = 0.2 * service_rate + 0.8 * self._ewma_service_rate
            self._request_count = 0
            self._last_count_time = now

    def _update_threshold(self) -> None:
        with self._lock:
            rho_cpu = self._ewma_cpu * self._cpu_weight
            rho_mem = self._ewma_mem * self._mem_weight
            rho_io = self._ewma_io * self._io_weight
            rho_system = max(rho_cpu, rho_mem, rho_io)

            if self._ewma_latency > self._target_latency_ms:
                latency_factor = self._target_latency_ms / self._ewma_latency
            else:
                latency_factor = 1.0

            if self._ewma_service_rate > 0 and self._ewma_arrival_rate > 0:
                rho_queue = self._ewma_arrival_rate / self._ewma_service_rate
                rho = max(rho_system, min(rho_queue, 0.95))
            else:
                rho = rho_system

            rho_clamped = max(0.05, min(0.95, rho))
            target = self._base_rate * (1.0 - rho_clamped) * self._safety_margin * latency_factor
            self._current_threshold = 0.5 * self._current_threshold + 0.5 * target
            self._current_threshold = max(self._min_rate, min(self._max_rate, self._current_threshold))

    def get_detailed_metrics(self) -> dict:
        with self._lock:
            return {
                "ewma_cpu": round(self._ewma_cpu, 4),
                "ewma_mem": round(self._ewma_mem, 4),
                "ewma_io": round(self._ewma_io, 4),
                "ewma_latency_ms": round(self._ewma_latency, 2),
                "arrival_rate": round(self._ewma_arrival_rate, 2),
                "service_rate": round(self._ewma_service_rate, 2),
            }


# ======================================================================
# Benchmark Framework
# ======================================================================

class BenchmarkScenario:
    """
    Defines a traffic scenario for benchmarking rate limiters.

    In F1 terms:
    - "should accept" = system is not overloaded (ground truth = True)
    - "should reject" = system is overloaded (ground truth = False)
    """

    def __init__(self, name: str, load_profile: list, duration_sec: float = 30.0):
        self.name = name
        self.load_profile = load_profile
        self.duration_sec = duration_sec

    def get_request_rate(self, elapsed_sec: float) -> tuple[int, bool]:
        """Return (request_rate, should_accept) at a given time."""
        profile_pos = (elapsed_sec / self.duration_sec) * len(self.load_profile)
        idx = min(int(profile_pos), len(self.load_profile) - 1)
        entry = self.load_profile[idx]
        return entry["rate"], entry["should_accept"]


def create_benchmark_scenarios() -> list[BenchmarkScenario]:
    return [
        BenchmarkScenario(
            "Steady-Normal",
            [{"rate": 150, "should_accept": True}] * 10,
            duration_sec=20.0
        ),
        BenchmarkScenario(
            "Spike-Overload",
            [
                *[{"rate": 100, "should_accept": True}] * 2,
                *[{"rate": 500, "should_accept": False}] * 3,
                *[{"rate": 100, "should_accept": True}] * 3,
            ],
            duration_sec=16.0
        ),
        BenchmarkScenario(
            "Gradual-Overload",
            [
                *[{"rate": 50 + i * 50, "should_accept": True if i < 3 else False} for i in range(5)],
                *[{"rate": 200, "should_accept": True}] * 3,
            ],
            duration_sec=16.0
        ),
        BenchmarkScenario(
            "Mixed-Jitter",
            [
                *[{"rate": 80 if i % 2 == 0 else 300, "should_accept": True if i % 2 == 0 else False} for i in range(6)],
            ],
            duration_sec=12.0
        ),
    ]


def run_benchmark(limiters: list[RateLimiter], scenario: BenchmarkScenario, verbose: bool = False) -> dict:
    """Run a benchmark scenario against all limiters and collect F1 stats."""
    for limiter in limiters:
        limiter.stats = LimiterStats()
        limiter.start()

    start = time.monotonic()
    step_interval = 0.05
    last_step = start

    while time.monotonic() - start < scenario.duration_sec:
        now = time.monotonic()
        elapsed = now - start
        target_rate, should_accept = scenario.get_request_rate(elapsed)

        requests_in_step = int(target_rate * step_interval)
        for _ in range(max(1, requests_in_step)):
            for limiter in limiters:
                accepted = limiter.acquire()
                if accepted:
                    latency = 20 if should_accept else 150
                    limiter.record_latency(latency)
                limiter.record_outcome(accepted, should_accept)

        sleep_time = step_interval - (time.monotonic() - last_step)
        if sleep_time > 0:
            time.sleep(sleep_time)
        last_step = time.monotonic()

    for limiter in limiters:
        limiter.stop()

    results = {}
    for limiter in limiters:
        results[limiter.name] = limiter.get_metrics()
    return results


def print_f1_report(all_results: dict) -> None:
    """Print aggregated F1 report across all scenarios."""
    print("\n" + "=" * 100)
    print("RATE LIMITER F1 SCORE COMPARISON".center(100))
    print("=" * 100)

    header = f"{'Algorithm':<25} | {'Precision':>10} | {'Recall':>8} | {'F1':>8} | {'Accuracy':>10} | {'FP%':>8} | {'FN%':>8}"
    print(header)
    print("-" * 100)

    alg_names = list(next(iter(all_results.values())).keys()) if all_results else []

    for name in alg_names:
        tp_sum = tn_sum = fp_sum = fn_sum = total_sum = 0
        for scenario_name, scenario_results in all_results.items():
            r = scenario_results.get(name, {})
            tp_sum += r.get("tp", 0)
            tn_sum += r.get("tn", 0)
            fp_sum += r.get("fp", 0)
            fn_sum += r.get("fn", 0)
            total_sum += r.get("total", 0)

        precision = tp_sum / (tp_sum + fp_sum) if (tp_sum + fp_sum) > 0 else 0.0
        recall = tp_sum / (tp_sum + fn_sum) if (tp_sum + fn_sum) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        accuracy = (tp_sum + tn_sum) / total_sum if total_sum > 0 else 0.0
        fp_pct = fp_sum / total_sum * 100 if total_sum > 0 else 0.0
        fn_pct = fn_sum / total_sum * 100 if total_sum > 0 else 0.0

        print(
            f"{name:<25} | {precision:10.4f} | {recall:8.4f} | {f1:8.4f} | "
            f"{accuracy:10.4f} | {fp_pct:7.2f}% | {fn_pct:7.2f}%"
        )

    print("-" * 100)
    print("\nF1 Score Definition: 2 * (Precision * Recall) / (Precision + Recall)")
    print("  Precision = TP / (TP + FP)  : How many rejections were correct (avoid false kills)")
    print("  Recall    = TP / (TP + FN)  : How many overloads were correctly caught")
    print("  FP (False Positive): Rejected when should have accepted (false kill, bad UX)")
    print("  FN (False Negative): Accepted when should have rejected (system overload, risk)")


if __name__ == "__main__":
    print("Initializing multi-algorithm rate limiter benchmark...")
    print("Algorithms: Token Bucket, Sliding Window, BBR-Adaptive, QueueingTheory-MD\n")

    limiters = [
        TokenBucketLimiter(rate=150, capacity=300),
        SlidingWindowLimiter(rate=200, window_sec=1.0),
        BBRRateLimiter(min_threshold=10, max_threshold=5000, initial_threshold=100),
        QueueingTheoryLimiter(
            base_rate=200,
            min_rate=10,
            max_rate=5000,
            safety_margin=0.7,
            cpu_weight=1.0,
            mem_weight=1.0,
            io_weight=1.0,
        ),
    ]

    scenarios = create_benchmark_scenarios()
    all_results = {}

    for scenario in scenarios:
        print(f"Running scenario: {scenario.name} ({scenario.duration_sec:.0f}s)...")
        results = run_benchmark(limiters, scenario, verbose=False)
        all_results[scenario.name] = results
        print(f"  Completed: {sum(r['total'] for r in results.values()) // len(results)} requests per limiter")

    print_f1_report(all_results)
