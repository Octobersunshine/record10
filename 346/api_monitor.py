import time
import math
import inspect
import threading
import json
from collections import deque, defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from threading import Lock, Condition, RLock, Thread
from typing import Dict, List, Optional, Union, Callable


class AtomicCounter:
    def __init__(self, value: float = 0):
        self._value = value
        self._lock = Lock()

    def increment(self, n: float = 1) -> float:
        with self._lock:
            self._value += n
            return self._value

    def decrement(self, n: float = 1) -> float:
        with self._lock:
            self._value -= n
            return self._value

    def get(self) -> float:
        with self._lock:
            return self._value

    def get_and_reset(self) -> float:
        with self._lock:
            val = self._value
            self._value = 0
            return val


class ReadWriteLock:
    def __init__(self):
        self._lock = Lock()
        self._can_read = Condition(self._lock)
        self._can_write = Condition(self._lock)
        self._active_readers = 0
        self._active_writers = 0
        self._waiting_writers = 0

    @contextmanager
    def read_lock(self):
        with self._lock:
            while self._active_writers > 0 or self._waiting_writers > 0:
                self._can_read.wait()
            self._active_readers += 1
        try:
            yield
        finally:
            with self._lock:
                self._active_readers -= 1
                if self._active_readers == 0:
                    self._can_write.notify()

    @contextmanager
    def write_lock(self):
        with self._lock:
            self._waiting_writers += 1
            while self._active_readers > 0 or self._active_writers > 0:
                self._can_write.wait()
            self._waiting_writers -= 1
            self._active_writers += 1
        try:
            yield
        finally:
            with self._lock:
                self._active_writers -= 1
                if self._waiting_writers > 0:
                    self._can_write.notify()
                else:
                    self._can_read.notify_all()


@dataclass
class RequestRecord:
    timestamp: float
    latency: float


class SlidingWindow:
    def __init__(self, window_size: int = 60):
        self.window_size = window_size
        self._records: deque = deque()
        self._lock = Lock()
        self._count = AtomicCounter(0)
        self._total_latency = AtomicCounter(0.0)

    def add_record(self, latency: float, current_time: Optional[float] = None) -> None:
        current_time = current_time or time.time()
        with self._lock:
            self._records.append(RequestRecord(timestamp=current_time, latency=latency))
            self._count.increment()
            self._total_latency.increment(latency)

    def _cleanup(self, current_time: float) -> None:
        cutoff = current_time - self.window_size
        while self._records and self._records[0].timestamp < cutoff:
            record = self._records.popleft()
            self._count.decrement()
            self._total_latency.decrement(record.latency)

    def get_stats(self, current_time: Optional[float] = None) -> Dict:
        current_time = current_time or time.time()
        with self._lock:
            self._cleanup(current_time)
            latencies = [r.latency for r in self._records if r.timestamp < current_time]
            count = len(latencies)

        if count == 0:
            return {
                "count": 0,
                "qps": 0.0,
                "avg_latency": 0.0,
                "p99_latency": 0.0,
            }

        total_latency = sum(latencies)
        avg_latency = total_latency / count
        latencies.sort()
        p99_index = min(count - 1, max(0, int(count * 0.99) - 1))
        p99_latency = latencies[p99_index]
        qps = count / self.window_size

        return {
            "count": count,
            "qps": round(qps, 4),
            "avg_latency": round(avg_latency, 4),
            "p99_latency": round(p99_latency, 4),
        }

    def get_records_count(self, current_time: Optional[float] = None) -> int:
        current_time = current_time or time.time()
        with self._lock:
            self._cleanup(current_time)
            return sum(1 for r in self._records if r.timestamp < current_time)

    def get_latencies(self, current_time: Optional[float] = None) -> List[float]:
        current_time = current_time or time.time()
        with self._lock:
            self._cleanup(current_time)
            return [r.latency for r in self._records if r.timestamp < current_time]

    def get_time_series(self, bucket_size: int = 1, current_time: Optional[float] = None) -> Dict:
        current_time = current_time or time.time()
        cutoff = current_time - self.window_size
        buckets = defaultdict(lambda: {"count": 0, "total_latency": 0.0})

        with self._lock:
            self._cleanup(current_time)
            for r in self._records:
                if r.timestamp < current_time:
                    bucket_ts = int(r.timestamp / bucket_size) * bucket_size
                    buckets[bucket_ts]["count"] += 1
                    buckets[bucket_ts]["total_latency"] += r.latency

        sorted_ts = sorted(buckets.keys())
        timestamps = []
        qps_series = []
        latency_series = []
        count_series = []

        start_bucket = int(cutoff / bucket_size) * bucket_size
        end_bucket = int((current_time - 0.0001) / bucket_size) * bucket_size
        ts = start_bucket
        while ts <= end_bucket:
            timestamps.append(ts)
            data = buckets.get(ts, {"count": 0, "total_latency": 0.0})
            qps = data["count"] / bucket_size if bucket_size > 0 else 0
            avg_lat = data["total_latency"] / data["count"] if data["count"] > 0 else 0
            qps_series.append(round(qps, 4))
            latency_series.append(round(avg_lat, 4))
            count_series.append(data["count"])
            ts += bucket_size

        return {
            "timestamps": timestamps,
            "count": count_series,
            "qps": qps_series,
            "avg_latency": latency_series,
        }


WINDOW_PRESETS = {
    "1s": 1,
    "10s": 10,
    "30s": 30,
    "1m": 60,
    "5m": 300,
    "15m": 900,
    "30m": 1800,
    "1h": 3600,
    "2h": 7200,
    "6h": 21600,
    "12h": 43200,
    "24h": 86400,
}


class PathMetrics:
    def __init__(self, path: str, window_sizes: List[int]):
        self.path = path
        self._windows: Dict[int, SlidingWindow] = {
            size: SlidingWindow(window_size=size)
            for size in window_sizes
        }
        self._lock = Lock()

    def add_record(self, latency: float, current_time: Optional[float] = None) -> None:
        current_time = current_time or time.time()
        for window in self._windows.values():
            window.add_record(latency, current_time)

    def get_stats(self, window_size: Optional[int] = None, current_time: Optional[float] = None) -> Dict:
        current_time = current_time or time.time()
        if window_size is None:
            window_size = list(self._windows.keys())[0]

        if window_size not in self._windows:
            return {
                "path": self.path,
                "window_size": window_size,
                "count": 0,
                "qps": 0.0,
                "avg_latency": 0.0,
                "p99_latency": 0.0,
            }

        stats = self._windows[window_size].get_stats(current_time)
        return {
            "path": self.path,
            "window_size": window_size,
            **stats,
        }

    def get_all_window_stats(self, current_time: Optional[float] = None) -> Dict:
        return {
            str(size): self.get_stats(size, current_time)
            for size in self._windows
        }

    def get_time_series(self, window_size: int, bucket_size: int = 1, current_time: Optional[float] = None) -> Dict:
        if window_size in self._windows:
            return self._windows[window_size].get_time_series(bucket_size, current_time)
        return {"timestamps": [], "count": [], "qps": [], "avg_latency": []}


class ApiMonitor:
    def __init__(self, windows: Optional[List[Union[int, str]]] = None):
        if windows is None:
            windows = [1, 60, 3600]

        self.window_sizes = []
        for w in windows:
            if isinstance(w, str):
                if w in WINDOW_PRESETS:
                    self.window_sizes.append(WINDOW_PRESETS[w])
                else:
                    raise ValueError(f"Unknown window preset: {w}. Available: {list(WINDOW_PRESETS.keys())}")
            else:
                self.window_sizes.append(w)

        self.window_sizes.sort()
        self._path_metrics: Dict[str, PathMetrics] = {}
        self._global_metrics: Dict[int, SlidingWindow] = {
            size: SlidingWindow(window_size=size)
            for size in self.window_sizes
        }
        self._rwlock = ReadWriteLock()
        self._dict_lock = Lock()

        self._topn_cache: Dict[tuple, Dict] = {}
        self._topn_cache_lock = Lock()
        self._topn_update_interval = 1
        self._auto_refresh = False
        self._refresh_thread: Optional[Thread] = None
        self._stop_event = threading.Event()

    def _get_or_create_path(self, path: str) -> PathMetrics:
        with self._dict_lock:
            if path not in self._path_metrics:
                self._path_metrics[path] = PathMetrics(path, self.window_sizes)
            return self._path_metrics[path]

    def record(self, path: str, latency: float, current_time: Optional[float] = None) -> None:
        current_time = current_time or time.time()
        path_metrics = self._get_or_create_path(path)
        path_metrics.add_record(latency, current_time)
        for window in self._global_metrics.values():
            window.add_record(latency, current_time)

    def get_stats(self, path: Optional[str] = None, window: Optional[Union[int, str]] = None,
                  current_time: Optional[float] = None) -> Dict:
        current_time = current_time or time.time()
        window_size = self._resolve_window(window)

        if path is not None:
            with self._rwlock.read_lock():
                pm = self._path_metrics.get(path)
            if pm is None:
                return {
                    "path": path,
                    "window_size": window_size,
                    "count": 0,
                    "qps": 0.0,
                    "avg_latency": 0.0,
                    "p99_latency": 0.0,
                }
            return pm.get_stats(window_size, current_time)
        else:
            return self.get_all_stats(window, current_time)

    def _resolve_window(self, window: Optional[Union[int, str]]) -> int:
        if window is None:
            return self.window_sizes[0]
        if isinstance(window, str):
            if window in WINDOW_PRESETS:
                return WINDOW_PRESETS[window]
            raise ValueError(f"Unknown window preset: {window}")
        if isinstance(window, int):
            if window in self.window_sizes:
                return window
            raise ValueError(f"Window size {window} not in configured windows: {self.window_sizes}")
        raise ValueError(f"Invalid window: {window}")

    def get_all_stats(self, window: Optional[Union[int, str]] = None,
                      current_time: Optional[float] = None) -> Dict:
        current_time = current_time or time.time()
        window_size = self._resolve_window(window)

        with self._rwlock.read_lock():
            paths = list(self._path_metrics.keys())
            paths_snapshot = {p: self._path_metrics[p] for p in paths}

        path_stats = []
        for path in paths:
            stats = paths_snapshot[path].get_stats(window_size, current_time)
            path_stats.append(stats)

        path_stats.sort(key=lambda x: x["count"], reverse=True)

        global_stats = self._global_metrics[window_size].get_stats(current_time)

        return {
            "window_size": window_size,
            "total": {
                "path_count": len(paths),
                **global_stats,
            },
            "endpoints": path_stats,
        }

    def get_top_n(self, n: int = 10, by: str = "count",
                  window: Optional[Union[int, str]] = None,
                  current_time: Optional[float] = None,
                  use_cache: bool = True) -> List[Dict]:
        current_time = current_time or time.time()
        window_size = self._resolve_window(window)

        cache_key = (n, by, window_size)
        if use_cache:
            with self._topn_cache_lock:
                cached = self._topn_cache.get(cache_key)
                if cached and (time.time() - cached.get("timestamp", 0)) < self._topn_update_interval:
                    return cached["data"]

        stats = self.get_all_stats(window, current_time)
        endpoints = stats["endpoints"]
        if by not in ["count", "qps", "avg_latency", "p99_latency"]:
            by = "count"
        endpoints.sort(key=lambda x: x[by], reverse=True)
        result = endpoints[:n]

        if use_cache:
            with self._topn_cache_lock:
                self._topn_cache[cache_key] = {
                    "timestamp": time.time(),
                    "data": result,
                }

        return result

    def start_auto_refresh(self, interval: float = 1.0):
        if self._auto_refresh:
            return
        self._auto_refresh = True
        self._topn_update_interval = interval
        self._stop_event.clear()

        def refresh_loop():
            while not self._stop_event.is_set():
                try:
                    self._refresh_all_topn()
                except Exception:
                    pass
                self._stop_event.wait(interval)

        self._refresh_thread = Thread(target=refresh_loop, daemon=True)
        self._refresh_thread.start()

    def stop_auto_refresh(self):
        self._auto_refresh = False
        self._stop_event.set()
        if self._refresh_thread:
            self._refresh_thread.join(timeout=2)
            self._refresh_thread = None

    def _refresh_all_topn(self):
        current_time = time.time()
        for by in ["count", "qps", "avg_latency", "p99_latency"]:
            for window_size in self.window_sizes:
                for n in [3, 5, 10, 20]:
                    cache_key = (n, by, window_size)
                    stats = self.get_all_stats(window_size, current_time)
                    endpoints = stats["endpoints"]
                    endpoints.sort(key=lambda x: x[by], reverse=True)
                    result = endpoints[:n]
                    with self._topn_cache_lock:
                        self._topn_cache[cache_key] = {
                            "timestamp": time.time(),
                            "data": result,
                        }

    def get_time_series(self, path: Optional[str] = None,
                        window: Optional[Union[int, str]] = None,
                        bucket_size: int = 1,
                        current_time: Optional[float] = None) -> Dict:
        current_time = current_time or time.time()
        window_size = self._resolve_window(window)

        if path is not None:
            with self._rwlock.read_lock():
                pm = self._path_metrics.get(path)
            if pm is None:
                return {
                    "path": path,
                    "window_size": window_size,
                    "bucket_size": bucket_size,
                    "timestamps": [],
                    "count": [],
                    "qps": [],
                    "avg_latency": [],
                }
            ts = pm.get_time_series(window_size, bucket_size, current_time)
            return {
                "path": path,
                "window_size": window_size,
                "bucket_size": bucket_size,
                **ts,
            }
        else:
            if window_size in self._global_metrics:
                ts = self._global_metrics[window_size].get_time_series(bucket_size, current_time)
                return {
                    "path": "global",
                    "window_size": window_size,
                    "bucket_size": bucket_size,
                    **ts,
                }
            return {"timestamps": [], "count": [], "qps": [], "avg_latency": []}

    def get_echarts_pie(self, window: Optional[Union[int, str]] = None,
                        n: int = 10,
                        current_time: Optional[float] = None) -> Dict:
        current_time = current_time or time.time()
        stats = self.get_all_stats(window, current_time)
        endpoints = stats["endpoints"]

        endpoints.sort(key=lambda x: x["count"], reverse=True)
        top_endpoints = endpoints[:n]

        if len(endpoints) > n:
            others_count = sum(ep["count"] for ep in endpoints[n:])
            others_latency = sum(ep["count"] * ep["avg_latency"] for ep in endpoints[n:])
            others_avg_latency = others_latency / others_count if others_count > 0 else 0
            top_endpoints.append({
                "path": "others",
                "count": others_count,
                "qps": round(others_count / self._resolve_window(window), 4),
                "avg_latency": round(others_avg_latency, 4),
                "p99_latency": 0.0,
            })

        pie_data = []
        for ep in top_endpoints:
            pie_data.append({
                "name": ep["path"],
                "value": ep["count"],
                "itemStyle": {
                    "qps": ep["qps"],
                    "avg_latency": ep["avg_latency"],
                    "p99_latency": ep["p99_latency"],
                },
            })

        return {
            "title": {
                "text": f"API调用占比 (窗口:{stats['window_size']}s)",
                "left": "center",
            },
            "tooltip": {
                "trigger": "item",
                "formatter": "{b}: {c}次 ({d}%)<br/>QPS: {@[itemStyle.qps]}<br/>平均延迟: {@[itemStyle.avg_latency]}ms",
            },
            "legend": {
                "orient": "vertical",
                "left": "left",
            },
            "series": [{
                "name": "调用次数",
                "type": "pie",
                "radius": ["40%", "70%"],
                "avoidLabelOverlap": False,
                "itemStyle": {
                    "borderRadius": 10,
                    "borderColor": "#fff",
                    "borderWidth": 2,
                },
                "label": {
                    "show": True,
                    "formatter": "{b}\n{d}%",
                },
                "data": pie_data,
            }],
        }

    def get_echarts_line(self, path: Optional[str] = None,
                         window: Optional[Union[int, str]] = None,
                         bucket_size: int = 1,
                         metrics: List[str] = None,
                         current_time: Optional[float] = None) -> Dict:
        if metrics is None:
            metrics = ["qps", "avg_latency"]

        ts = self.get_time_series(path, window, bucket_size, current_time)
        window_size = self._resolve_window(window)

        timestamps = [time.strftime("%H:%M:%S", time.localtime(t)) for t in ts["timestamps"]]

        metric_configs = {
            "count": {"name": "调用次数", "type": "line", "yAxisIndex": 0, "color": "#5470c6"},
            "qps": {"name": "QPS", "type": "line", "yAxisIndex": 0, "color": "#91cc75"},
            "avg_latency": {"name": "平均延迟(ms)", "type": "line", "yAxisIndex": 1, "color": "#fac858"},
            "p99_latency": {"name": "P99延迟(ms)", "type": "line", "yAxisIndex": 1, "color": "#ee6666"},
        }

        series = []
        for m in metrics:
            if m in metric_configs and m in ts:
                config = metric_configs[m]
                series.append({
                    "name": config["name"],
                    "type": config["type"],
                    "yAxisIndex": config["yAxisIndex"],
                    "itemStyle": {"color": config["color"]},
                    "lineStyle": {"color": config["color"]},
                    "smooth": True,
                    "data": ts[m],
                })

        y_axis = [
            {"type": "value", "name": "QPS/次数", "position": "left"},
            {"type": "value", "name": "延迟(ms)", "position": "right"},
        ]

        path_name = path or "全局"
        return {
            "title": {
                "text": f"{path_name} - 性能趋势 (窗口:{window_size}s)",
                "left": "center",
            },
            "tooltip": {
                "trigger": "axis",
                "axisPointer": {"type": "cross"},
            },
            "legend": {
                "data": [metric_configs[m]["name"] for m in metrics if m in metric_configs],
                "bottom": 10,
            },
            "grid": {
                "left": "3%",
                "right": "4%",
                "bottom": "15%",
                "containLabel": True,
            },
            "xAxis": {
                "type": "category",
                "boundaryGap": False,
                "data": timestamps,
            },
            "yAxis": y_axis,
            "series": series,
        }

    def get_echarts_bar(self, window: Optional[Union[int, str]] = None,
                        n: int = 15,
                        by: str = "count",
                        current_time: Optional[float] = None) -> Dict:
        current_time = current_time or time.time()
        top_endpoints = self.get_top_n(n=n, by=by, window=window, current_time=current_time)
        window_size = self._resolve_window(window)

        by_labels = {
            "count": "调用次数",
            "qps": "QPS",
            "avg_latency": "平均延迟(ms)",
            "p99_latency": "P99延迟(ms)",
        }

        paths = [ep["path"] for ep in top_endpoints]
        values = [ep[by] for ep in top_endpoints]
        avg_latencies = [ep["avg_latency"] for ep in top_endpoints]
        p99_latencies = [ep["p99_latency"] for ep in top_endpoints]

        return {
            "title": {
                "text": f"Top {len(paths)} 接口 - {by_labels.get(by, by)} (窗口:{window_size}s)",
                "left": "center",
            },
            "tooltip": {
                "trigger": "axis",
                "axisPointer": {"type": "shadow"},
                "formatter": (
                    "{b}<br/>"
                    f"{by_labels.get(by, by)}: {{c0}}<br/>"
                    "平均延迟: {c1}ms<br/>"
                    "P99延迟: {c2}ms"
                ),
            },
            "grid": {
                "left": "3%",
                "right": "4%",
                "bottom": "15%",
                "containLabel": True,
            },
            "xAxis": {
                "type": "category",
                "data": paths,
                "axisLabel": {
                    "rotate": 45,
                    "interval": 0,
                    "fontSize": 10,
                },
            },
            "yAxis": [
                {"type": "value", "name": by_labels.get(by, by), "position": "left"},
                {"type": "value", "name": "延迟(ms)", "position": "right"},
            ],
            "series": [
                {
                    "name": by_labels.get(by, by),
                    "type": "bar",
                    "data": values,
                    "itemStyle": {"color": "#5470c6"},
                    "yAxisIndex": 0,
                },
                {
                    "name": "平均延迟",
                    "type": "line",
                    "data": avg_latencies,
                    "smooth": True,
                    "itemStyle": {"color": "#91cc75"},
                    "lineStyle": {"color": "#91cc75"},
                    "yAxisIndex": 1,
                },
                {
                    "name": "P99延迟",
                    "type": "line",
                    "data": p99_latencies,
                    "smooth": True,
                    "itemStyle": {"color": "#ee6666"},
                    "lineStyle": {"color": "#ee6666"},
                    "yAxisIndex": 1,
                },
            ],
        }

    def get_echarts_dashboard(self, window: Optional[Union[int, str]] = None,
                              n: int = 10,
                              current_time: Optional[float] = None) -> Dict:
        window_size = self._resolve_window(window)
        stats = self.get_all_stats(window, current_time)

        return {
            "timestamp": int(time.time() * 1000),
            "window_size": window_size,
            "summary": {
                "total_requests": stats["total"]["count"],
                "total_paths": stats["total"]["path_count"],
                "avg_qps": stats["total"]["qps"],
                "avg_latency": stats["total"]["avg_latency"],
                "p99_latency": stats["total"]["p99_latency"],
            },
            "charts": {
                "pie": self.get_echarts_pie(window, n, current_time),
                "bar_count": self.get_echarts_bar(window, n, "count", current_time),
                "bar_p99": self.get_echarts_bar(window, n, "p99_latency", current_time),
            },
            "top_endpoints": {
                "by_count": self.get_top_n(n, "count", window, current_time),
                "by_qps": self.get_top_n(n, "qps", window, current_time),
                "by_p99": self.get_top_n(n, "p99_latency", window, current_time),
            },
            "time_series_global": self.get_time_series(window=window, bucket_size=max(1, window_size // 60), current_time=current_time),
        }


def monitor_decorator(monitor: ApiMonitor):
    def decorator(func):
        sig = inspect.signature(func)
        default_path = None
        if "path" in sig.parameters:
            param = sig.parameters["path"]
            if param.default is not inspect.Parameter.empty:
                default_path = param.default

        def wrapper(*args, **kwargs):
            path = kwargs.get("path", default_path or func.__name__)
            start = time.time()
            try:
                result = func(*args, **kwargs)
                latency = (time.time() - start) * 1000
                monitor.record(path, latency)
                return result
            except Exception as e:
                latency = (time.time() - start) * 1000
                monitor.record(path, latency)
                raise e

        return wrapper

    return decorator
