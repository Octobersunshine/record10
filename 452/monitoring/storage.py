import time
import bisect
import threading
import uuid
import socket
from collections import deque, defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any


@dataclass
class RequestRecord:
    path: str
    method: str
    response_time: float
    status_code: int
    timestamp: float
    request_body: Optional[str] = None
    response_body: Optional[str] = None
    request_id: Optional[str] = None
    instance_id: Optional[str] = None


@dataclass
class SlowRequestRecord:
    path: str
    method: str
    response_time: float
    status_code: int
    timestamp: float
    threshold: float
    request_body: Optional[str] = None
    response_body: Optional[str] = None
    request_id: Optional[str] = None
    instance_id: Optional[str] = None
    headers: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "path": self.path,
            "method": self.method,
            "response_time": round(self.response_time, 4),
            "status_code": self.status_code,
            "timestamp": self.timestamp,
            "time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(self.timestamp)),
            "threshold": self.threshold,
            "request_body": self.request_body,
            "response_body": self.response_body,
            "request_id": self.request_id,
            "instance_id": self.instance_id,
            "headers": self.headers
        }


@dataclass
class PathStats:
    path: str
    method: str
    total_time: float = 0.0
    count: int = 0
    error_count: int = 0
    sorted_times: List[float] = field(default_factory=list)

    @property
    def avg_response_time(self) -> float:
        return self.total_time / self.count if self.count > 0 else 0.0

    @property
    def p99_response_time(self) -> float:
        if not self.sorted_times:
            return 0.0
        n = len(self.sorted_times)
        index = int(n * 0.99)
        if index >= n:
            index = n - 1
        return self.sorted_times[index]

    @property
    def error_rate(self) -> float:
        return (self.error_count / self.count * 100) if self.count > 0 else 0.0


@dataclass
class SecondBucket:
    timestamp: float
    total_time: float = 0.0
    count: int = 0
    error_count: int = 0
    sorted_times: List[float] = field(default_factory=list)
    requests: List[RequestRecord] = field(default_factory=list)
    path_stats: Dict[str, PathStats] = field(default_factory=lambda: defaultdict(lambda: PathStats(path="", method="")))


@dataclass
class WindowStats:
    window_start: str
    window_end: str
    avg_response_time: float
    p99_response_time: float
    request_count: int
    error_count: int
    error_rate: float
    requests: List[RequestRecord] = field(default_factory=list)
    path_stats: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class InstanceMetrics:
    instance_id: str
    hostname: str
    timestamp: float
    window_start: str
    window_end: str
    avg_response_time: float
    p99_response_time: float
    request_count: int
    error_count: int
    error_rate: float
    path_stats: List[Dict[str, Any]] = field(default_factory=list)
    slow_requests: List[Dict[str, Any]] = field(default_factory=list)


class MetricsStorage:
    def __init__(
        self,
        window_seconds: int = 60,
        bucket_seconds: int = 1,
        slow_request_threshold_ms: float = 1000.0,
        max_slow_requests: int = 1000,
        instance_id: Optional[str] = None
    ):
        self.window_seconds = window_seconds
        self.bucket_seconds = bucket_seconds
        self._num_buckets = window_seconds // bucket_seconds
        self.slow_request_threshold_ms = slow_request_threshold_ms

        self.instance_id = instance_id or str(uuid.uuid4())
        self.hostname = socket.gethostname()

        self._lock = threading.RLock()
        self._buckets: deque = deque(maxlen=self._num_buckets)
        self._bucket_index: Dict[int, SecondBucket] = {}

        self._slow_requests: deque = deque(maxlen=max_slow_requests)

    @staticmethod
    def _get_bucket_key(timestamp: float, bucket_seconds: int) -> int:
        return int(timestamp // bucket_seconds) * bucket_seconds

    def _cleanup_expired(self, current_time: float):
        cutoff = current_time - self.window_seconds
        while self._buckets and self._buckets[0].timestamp < cutoff:
            old_bucket = self._buckets.popleft()
            self._bucket_index.pop(int(old_bucket.timestamp), None)

    def _get_path_key(self, path: str, method: str) -> str:
        return f"{method}:{path}"

    def add_record(
        self,
        path: str,
        method: str,
        response_time: float,
        status_code: int,
        request_body: Optional[str] = None,
        response_body: Optional[str] = None,
        request_id: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None
    ):
        timestamp = time.time()
        bucket_key = self._get_bucket_key(timestamp, self.bucket_seconds)
        path_key = self._get_path_key(path, method)
        is_error = status_code >= 400

        record = RequestRecord(
            path=path,
            method=method,
            response_time=response_time,
            status_code=status_code,
            timestamp=timestamp,
            request_body=request_body,
            response_body=response_body,
            request_id=request_id,
            instance_id=self.instance_id
        )

        with self._lock:
            self._cleanup_expired(timestamp)

            bucket = self._bucket_index.get(bucket_key)
            if bucket is None:
                bucket = SecondBucket(timestamp=bucket_key)
                self._buckets.append(bucket)
                self._bucket_index[bucket_key] = bucket

            bucket.total_time += response_time
            bucket.count += 1
            if is_error:
                bucket.error_count += 1
            bisect.insort(bucket.sorted_times, response_time)
            bucket.requests.append(record)

            path_stat = bucket.path_stats[path_key]
            if path_stat.path == "":
                path_stat.path = path
                path_stat.method = method
            path_stat.total_time += response_time
            path_stat.count += 1
            if is_error:
                path_stat.error_count += 1
            bisect.insort(path_stat.sorted_times, response_time)

        if response_time >= self.slow_request_threshold_ms:
            slow_record = SlowRequestRecord(
                path=path,
                method=method,
                response_time=response_time,
                status_code=status_code,
                timestamp=timestamp,
                threshold=self.slow_request_threshold_ms,
                request_body=request_body,
                response_body=response_body,
                request_id=request_id,
                instance_id=self.instance_id,
                headers=headers or {}
            )
            with self._lock:
                self._slow_requests.append(slow_record)

    def _calculate_p99(self, sorted_times: List[float]) -> float:
        if not sorted_times:
            return 0.0
        n = len(sorted_times)
        index = int(n * 0.99)
        if index >= n:
            index = n - 1
        return sorted_times[index]

    def _get_current_window(self) -> Tuple[List[SecondBucket], float, float]:
        current_time = time.time()
        cutoff = current_time - self.window_seconds

        with self._lock:
            self._cleanup_expired(current_time)
            buckets = list(self._buckets)

        valid_buckets = [b for b in buckets if b.timestamp >= cutoff]
        return valid_buckets, cutoff, current_time

    def get_window_stats(self) -> Optional[WindowStats]:
        valid_buckets, window_start_ts, window_end_ts = self._get_current_window()

        if not valid_buckets:
            return None

        total_time = 0.0
        total_count = 0
        total_error_count = 0
        all_sorted_times: List[float] = []
        all_requests: List[RequestRecord] = []
        path_agg: Dict[str, Dict] = {}

        for bucket in valid_buckets:
            total_time += bucket.total_time
            total_count += bucket.count
            total_error_count += bucket.error_count
            all_sorted_times.extend(bucket.sorted_times)
            all_requests.extend(bucket.requests)

            for path_key, stat in bucket.path_stats.items():
                if stat.count == 0:
                    continue
                if path_key not in path_agg:
                    path_agg[path_key] = {
                        "path": stat.path,
                        "method": stat.method,
                        "total_time": 0.0,
                        "count": 0,
                        "error_count": 0,
                        "sorted_times": []
                    }
                path_agg[path_key]["total_time"] += stat.total_time
                path_agg[path_key]["count"] += stat.count
                path_agg[path_key]["error_count"] += stat.error_count
                path_agg[path_key]["sorted_times"].extend(stat.sorted_times)

        all_sorted_times.sort()
        avg = total_time / total_count if total_count > 0 else 0.0
        p99 = self._calculate_p99(all_sorted_times)
        error_rate = (total_error_count / total_count * 100) if total_count > 0 else 0.0

        path_stats_list = []
        for path_key, agg in path_agg.items():
            sorted_times = sorted(agg["sorted_times"])
            path_p99 = self._calculate_p99(sorted_times)
            path_stats_list.append({
                "path": agg["path"],
                "method": agg["method"],
                "request_count": agg["count"],
                "error_count": agg["error_count"],
                "error_rate": round((agg["error_count"] / agg["count"] * 100) if agg["count"] > 0 else 0.0, 2),
                "avg_response_time": round(agg["total_time"] / agg["count"] if agg["count"] > 0 else 0.0, 4),
                "p99_response_time": round(path_p99, 4)
            })

        path_stats_list.sort(key=lambda x: x["request_count"], reverse=True)

        return WindowStats(
            window_start=time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(window_start_ts)),
            window_end=time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(window_end_ts)),
            avg_response_time=round(avg, 4),
            p99_response_time=round(p99, 4),
            request_count=total_count,
            error_count=total_error_count,
            error_rate=round(error_rate, 2),
            requests=all_requests,
            path_stats=path_stats_list
        )

    def get_path_stats(self) -> List[Dict[str, Any]]:
        stats = self.get_window_stats()
        return stats.path_stats if stats else []

    def get_slow_requests(self, limit: int = 100, path: Optional[str] = None) -> List[SlowRequestRecord]:
        with self._lock:
            if path:
                filtered = [r for r in reversed(self._slow_requests) if r.path == path]
                return filtered[:limit]
            return list(reversed(list(self._slow_requests)))[:limit]

    def get_recent_requests(self, limit: int = 100) -> List[RequestRecord]:
        valid_buckets, _, _ = self._get_current_window()

        recent: List[RequestRecord] = []
        for bucket in reversed(valid_buckets):
            for record in reversed(bucket.requests):
                recent.append(record)
                if len(recent) >= limit:
                    return recent
        return recent

    def get_per_second_stats(self) -> List[Dict]:
        valid_buckets, _, _ = self._get_current_window()

        stats = []
        for bucket in valid_buckets:
            avg = bucket.total_time / bucket.count if bucket.count > 0 else 0.0
            p99 = self._calculate_p99(bucket.sorted_times)
            error_rate = (bucket.error_count / bucket.count * 100) if bucket.count > 0 else 0.0
            stats.append({
                "timestamp": bucket.timestamp,
                "time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(bucket.timestamp)),
                "avg_response_time": round(avg, 4),
                "p99_response_time": round(p99, 4),
                "request_count": bucket.count,
                "error_count": bucket.error_count,
                "error_rate": round(error_rate, 2)
            })
        return stats

    def get_instance_metrics(self) -> InstanceMetrics:
        stats = self.get_window_stats()
        slow_requests = self.get_slow_requests(limit=100)

        return InstanceMetrics(
            instance_id=self.instance_id,
            hostname=self.hostname,
            timestamp=time.time(),
            window_start=stats.window_start if stats else "",
            window_end=stats.window_end if stats else "",
            avg_response_time=stats.avg_response_time if stats else 0.0,
            p99_response_time=stats.p99_response_time if stats else 0.0,
            request_count=stats.request_count if stats else 0,
            error_count=stats.error_count if stats else 0,
            error_rate=stats.error_rate if stats else 0.0,
            path_stats=stats.path_stats if stats else [],
            slow_requests=[r.to_dict() for r in slow_requests]
        )

    def set_slow_request_threshold(self, threshold_ms: float) -> None:
        with self._lock:
            self.slow_request_threshold_ms = threshold_ms

    def clear(self):
        with self._lock:
            self._buckets.clear()
            self._bucket_index.clear()
            self._slow_requests.clear()

    def get_bucket_count(self) -> int:
        with self._lock:
            return len(self._buckets)

    def get_slow_request_count(self) -> int:
        with self._lock:
            return len(self._slow_requests)
