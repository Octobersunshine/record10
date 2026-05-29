import time
import threading
import psutil
from collections import deque
from enum import Enum, auto


class BBRPhase(Enum):
    STARTUP = auto()
    DRAIN = auto()
    PROBE_BW = auto()
    PROBE_RTT = auto()


class BBRRateLimiter:
    """
    BBR-inspired adaptive rate limiter with EWMA-smoothed signals.

    Mirrors TCP BBR's four phases:
      - STARTUP  : exponential increase to discover max capacity
      - DRAIN    : reduce rate to clear queued requests
      - PROBE_BW : steady-state, periodically gain/lose bandwidth
      - PROBE_RTT: probe minimum latency when congestion suspected

    Congestion signals (EWMA-smoothed to reject transient noise):
      - Smoothed CPU > cpu_high_watermark (default 80%)
      - Smoothed RTT > smoothed baseline * spike_factor

    Stability mechanisms:
      - EWMA on latency (alpha=0.2) and CPU (alpha=0.3) to filter jitter
      - Slow EWMA baseline RTT (alpha=0.05) for stable spike comparison
      - Minimum threshold adjustment interval (default 5s) in PROBE_BW
    """

    def __init__(
        self,
        min_threshold: int = 10,
        max_threshold: int = 10000,
        initial_threshold: int = 100,
        cpu_high_watermark: float = 0.80,
        latency_spike_factor: float = 2.0,
        startup_gain: float = 2.0,
        drain_gain: float = 0.7,
        probe_gain: float = 1.25,
        probe_cwnd_gain: float = 2.0,
        rtt_probe_interval: float = 10.0,
        rtt_probe_duration: float = 0.2,
        drain_timeout: float = 5.0,
        window_size: int = 60,
        monitor_interval: float = 1.0,
        ewma_alpha_rtt: float = 0.2,
        ewma_alpha_cpu: float = 0.3,
        ewma_alpha_baseline: float = 0.05,
        min_adjust_interval: float = 5.0,
    ):
        self._min_threshold = min_threshold
        self._max_threshold = max_threshold
        self._cpu_high_watermark = cpu_high_watermark
        self._latency_spike_factor = latency_spike_factor

        self._startup_gain = startup_gain
        self._drain_gain = drain_gain
        self._probe_gain = probe_gain
        self._probe_cwnd_gain = probe_cwnd_gain

        self._rtt_probe_interval = rtt_probe_interval
        self._rtt_probe_duration = rtt_probe_duration
        self._drain_timeout = drain_timeout

        self._window_size = window_size
        self._monitor_interval = monitor_interval

        self._ewma_alpha_rtt = ewma_alpha_rtt
        self._ewma_alpha_cpu = ewma_alpha_cpu
        self._ewma_alpha_baseline = ewma_alpha_baseline
        self._min_adjust_interval = min_adjust_interval

        self._threshold = float(initial_threshold)
        self._phase = BBRPhase.STARTUP

        self._rtt_samples: deque = deque(maxlen=window_size)
        self._bw_samples: deque = deque(maxlen=window_size)
        self._min_rtt = float("inf")
        self._max_bw = 0.0

        self._ewma_rtt: float | None = None
        self._ewma_baseline_rtt: float | None = None
        self._ewma_cpu: float | None = None

        self._round_count = 0
        self._last_rtt_probe_time = time.monotonic()
        self._phase_entered_time = time.monotonic()
        self._last_adjust_time: float = 0.0

        self._lock = threading.Lock()
        self._request_count = 0
        self._last_count_time = time.monotonic()

        self._monitor_thread: threading.Thread | None = None
        self._stop_event = threading.Event()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> None:
        psutil.cpu_percent(interval=0.1)
        self._ewma_cpu = psutil.cpu_percent(interval=0.1) / 100.0
        self._last_adjust_time = time.monotonic()
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop, daemon=True
        )
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

    def get_phase(self) -> BBRPhase:
        with self._lock:
            return self._phase

    def get_metrics(self) -> dict:
        with self._lock:
            return {
                "threshold": int(self._threshold),
                "phase": self._phase.name,
                "min_rtt_ms": round(self._min_rtt, 3) if self._min_rtt != float("inf") else None,
                "ewma_rtt_ms": round(self._ewma_rtt, 3) if self._ewma_rtt is not None else None,
                "ewma_baseline_ms": round(self._ewma_baseline_rtt, 3) if self._ewma_baseline_rtt is not None else None,
                "ewma_cpu": round(self._ewma_cpu, 3) if self._ewma_cpu is not None else None,
                "max_bw": round(self._max_bw, 3),
                "cpu_percent": psutil.cpu_percent(interval=None),
            }

    # ------------------------------------------------------------------
    # Core BBR logic
    # ------------------------------------------------------------------

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
            cpu_congested = (
                self._ewma_cpu is not None and self._ewma_cpu > self._cpu_high_watermark
            )
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
                if time_in_drain > self._drain_timeout:
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
                elif (now - self._last_rtt_probe_time) > self._rtt_probe_interval:
                    self._phase = BBRPhase.PROBE_RTT
                    self._phase_entered_time = now
                    self._last_rtt_probe_time = now

            elif self._phase == BBRPhase.PROBE_RTT:
                if (now - self._phase_entered_time) > self._rtt_probe_duration:
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
                self._threshold *= self._startup_gain
                self._round_count += 1
                self._last_adjust_time = now

            elif self._phase == BBRPhase.DRAIN:
                self._threshold *= self._drain_gain
                self._last_adjust_time = now

            elif self._phase == BBRPhase.PROBE_BW:
                elapsed_since_adjust = now - self._last_adjust_time
                if elapsed_since_adjust < self._min_adjust_interval:
                    return

                cycle = int(now * 0.2) % 8
                if cycle == 0:
                    self._threshold *= self._probe_gain
                elif cycle == 4:
                    self._threshold *= 0.75
                else:
                    if self._max_bw > 0:
                        target = self._max_bw * self._probe_cwnd_gain
                        self._threshold = self._threshold * 0.875 + target * 0.125

                self._last_adjust_time = now

            elif self._phase == BBRPhase.PROBE_RTT:
                self._threshold = max(self._min_threshold, self._threshold * 0.5)
                self._last_adjust_time = now

            self._threshold = max(self._min_threshold, min(self._max_threshold, self._threshold))

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

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
# Demo / integration example
# ======================================================================

if __name__ == "__main__":
    import random

    limiter = BBRRateLimiter(
        min_threshold=5,
        max_threshold=5000,
        initial_threshold=50,
        cpu_high_watermark=0.80,
        latency_spike_factor=2.0,
        drain_gain=0.7,
        monitor_interval=0.5,
        drain_timeout=5.0,
        ewma_alpha_rtt=0.2,
        ewma_alpha_cpu=0.3,
        ewma_alpha_baseline=0.05,
        min_adjust_interval=5.0,
    )
    limiter.start()

    header = (
        f"{'Time':>6}s | {'Threshold':>9} | {'Phase':>10} | "
        f"{'EwmaRTT':>8} | {'EwmaBase':>8} | {'EwmaCPU':>7} | {'Accepted':>8}"
    )
    print(header)
    print("-" * len(header))

    try:
        for step in range(60):
            time.sleep(0.5)

            accepted = 0
            for _ in range(200):
                if limiter.acquire():
                    accepted += 1
                    simulated_latency = random.uniform(10, 50)
                    cpu_now = psutil.cpu_percent(interval=None)
                    if cpu_now > 70:
                        simulated_latency += random.uniform(40, 120)
                    limiter.record_latency(simulated_latency)

            metrics = limiter.get_metrics()
            elapsed = (step + 1) * 0.5
            print(
                f"{elapsed:6.1f}s | {metrics['threshold']:9d} | {metrics['phase']:>10} | "
                f"{metrics['ewma_rtt_ms'] or 'N/A':>8} | {metrics['ewma_baseline_ms'] or 'N/A':>8} | "
                f"{metrics['ewma_cpu'] or 0:7.3f} | {accepted:8d}"
            )
    except KeyboardInterrupt:
        pass
    finally:
        limiter.stop()
        print("\nLimiter stopped.")
