import threading
import time
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from collections import defaultdict
import httpx
from .storage import MetricsStorage, InstanceMetrics

logger = logging.getLogger(__name__)


@dataclass
class AggregatedStats:
    timestamp: float
    window_start: str
    window_end: str
    total_requests: int = 0
    total_errors: int = 0
    avg_response_time: float = 0.0
    p99_response_time: float = 0.0
    error_rate: float = 0.0
    instance_count: int = 0
    instances: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    path_stats: List[Dict[str, Any]] = field(default_factory=list)
    slow_requests: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(self.timestamp)),
            "window_start": self.window_start,
            "window_end": self.window_end,
            "total_requests": self.total_requests,
            "total_errors": self.total_errors,
            "avg_response_time": round(self.avg_response_time, 4),
            "p99_response_time": round(self.p99_response_time, 4),
            "error_rate": round(self.error_rate, 2),
            "instance_count": self.instance_count,
            "instances": self.instances,
            "path_stats": self.path_stats,
            "slow_requests": self.slow_requests
        }


class MetricsAggregator:
    def __init__(self, is_central: bool = False, central_url: Optional[str] = None, report_interval: int = 10):
        self.is_central = is_central
        self.central_url = central_url
        self.report_interval = report_interval

        self._lock = threading.RLock()
        self._instance_metrics: Dict[str, InstanceMetrics] = {}
        self._report_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._handlers: List = []

    def add_handler(self, handler):
        with self._lock:
            self._handlers.append(handler)

    def report_metrics(self, metrics: InstanceMetrics) -> None:
        with self._lock:
            self._instance_metrics[metrics.instance_id] = metrics

        for handler in self._handlers:
            try:
                handler(metrics)
            except Exception as e:
                logger.error(f"Aggregator handler error: {e}")

    def get_aggregated_stats(self) -> Optional[AggregatedStats]:
        with self._lock:
            instances = list(self._instance_metrics.values())

        if not instances:
            return None

        latest = max(instances, key=lambda x: x.timestamp)

        total_requests = sum(m.request_count for m in instances)
        total_errors = sum(m.error_count for m in instances)

        all_response_times = []
        path_agg: Dict[str, Dict] = {}
        all_slow_requests: List[Dict] = []

        for m in instances:
            for path_stat in m.path_stats:
                key = f"{path_stat['method']}:{path_stat['path']}"
                if key not in path_agg:
                    path_agg[key] = {
                        "path": path_stat["path"],
                        "method": path_stat["method"],
                        "total_time": 0.0,
                        "count": 0,
                        "error_count": 0,
                        "sorted_times": []
                    }
                path_agg[key]["total_time"] += path_stat["avg_response_time"] * path_stat["request_count"]
                path_agg[key]["count"] += path_stat["request_count"]
                path_agg[key]["error_count"] += path_stat["error_count"]
                all_response_times.extend([path_stat["p99_response_time"]] * min(path_stat["request_count"], 10))

            all_slow_requests.extend(m.slow_requests)

        all_response_times.sort()
        if all_response_times:
            n = len(all_response_times)
            p99_index = int(n * 0.99)
            if p99_index >= n:
                p99_index = n - 1
            p99_response_time = all_response_times[p99_index]
        else:
            p99_response_time = 0.0

        avg_response_time = (
            sum(m.avg_response_time * m.request_count for m in instances) / total_requests
            if total_requests > 0 else 0.0
        )

        path_stats_list = []
        for key, agg in path_agg.items():
            path_stats_list.append({
                "path": agg["path"],
                "method": agg["method"],
                "request_count": agg["count"],
                "error_count": agg["error_count"],
                "error_rate": round((agg["error_count"] / agg["count"] * 100) if agg["count"] > 0 else 0.0, 2),
                "avg_response_time": round(agg["total_time"] / agg["count"] if agg["count"] > 0 else 0.0, 4),
            })
        path_stats_list.sort(key=lambda x: x["request_count"], reverse=True)

        instances_dict = {}
        for m in instances:
            instances_dict[m.instance_id] = {
                "hostname": m.hostname,
                "last_report": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(m.timestamp)),
                "request_count": m.request_count,
                "error_count": m.error_count,
                "error_rate": m.error_rate,
                "avg_response_time": m.avg_response_time,
                "p99_response_time": m.p99_response_time
            }

        all_slow_requests.sort(key=lambda x: x["timestamp"], reverse=True)

        return AggregatedStats(
            timestamp=time.time(),
            window_start=latest.window_start,
            window_end=latest.window_end,
            total_requests=total_requests,
            total_errors=total_errors,
            avg_response_time=avg_response_time,
            p99_response_time=p99_response_time,
            error_rate=(total_errors / total_requests * 100) if total_requests > 0 else 0.0,
            instance_count=len(instances),
            instances=instances_dict,
            path_stats=path_stats_list,
            slow_requests=all_slow_requests[:100]
        )

    def get_instance_ids(self) -> List[str]:
        with self._lock:
            return list(self._instance_metrics.keys())

    def get_instance_metrics(self, instance_id: str) -> Optional[InstanceMetrics]:
        with self._lock:
            return self._instance_metrics.get(instance_id)

    async def send_to_central(self, storage: MetricsStorage) -> None:
        if not self.central_url:
            return

        metrics = storage.get_instance_metrics()
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.post(
                    f"{self.central_url}/monitoring/central/report",
                    json={
                        "instance_id": metrics.instance_id,
                        "hostname": metrics.hostname,
                        "timestamp": metrics.timestamp,
                        "window_start": metrics.window_start,
                        "window_end": metrics.window_end,
                        "avg_response_time": metrics.avg_response_time,
                        "p99_response_time": metrics.p99_response_time,
                        "request_count": metrics.request_count,
                        "error_count": metrics.error_count,
                        "error_rate": metrics.error_rate,
                        "path_stats": metrics.path_stats,
                        "slow_requests": metrics.slow_requests
                    }
                )
                if response.status_code != 200:
                    logger.warning(f"Failed to report to central: {response.status_code}")
        except Exception as e:
            logger.warning(f"Report to central failed: {e}")

    def _report_loop(self, storage: MetricsStorage):
        import asyncio
        while not self._stop_event.is_set():
            try:
                asyncio.run(self.send_to_central(storage))
            except Exception as e:
                logger.error(f"Report loop error: {e}")
            self._stop_event.wait(self.report_interval)

    def start_reporting(self, storage: MetricsStorage) -> None:
        if self.is_central or not self.central_url:
            return
        if self._report_thread and self._report_thread.is_alive():
            return

        self._stop_event.clear()
        self._report_thread = threading.Thread(
            target=self._report_loop,
            args=(storage,),
            daemon=True
        )
        self._report_thread.start()
        logger.info(f"Started reporting to central at {self.central_url} every {self.report_interval}s")

    def stop_reporting(self) -> None:
        self._stop_event.set()
        if self._report_thread:
            self._report_thread.join(timeout=5.0)
            logger.info("Stopped reporting to central")

    def clear(self) -> None:
        with self._lock:
            self._instance_metrics.clear()
