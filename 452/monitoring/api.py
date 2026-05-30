from fastapi import APIRouter, Query, HTTPException, Body
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from .storage import MetricsStorage, RequestRecord
from .alerts import AlertManager, AlertRule, AlertType, AlertSeverity
from .aggregator import MetricsAggregator


class RequestRecordResponse(BaseModel):
    path: str
    method: str
    response_time: float
    status_code: int
    timestamp: float
    request_id: Optional[str] = None
    instance_id: Optional[str] = None


class WindowStatsResponse(BaseModel):
    window_start: str
    window_end: str
    avg_response_time: float
    p99_response_time: float
    request_count: int
    error_count: int
    error_rate: float


class WindowStatsDetailResponse(WindowStatsResponse):
    requests: List[RequestRecordResponse]
    path_stats: List[Dict[str, Any]]


class SecondBucketStats(BaseModel):
    timestamp: float
    time: str
    avg_response_time: float
    p99_response_time: float
    request_count: int
    error_count: int
    error_rate: float


class PathStatsResponse(BaseModel):
    path: str
    method: str
    request_count: int
    error_count: int
    error_rate: float
    avg_response_time: float
    p99_response_time: float


class SlowRequestResponse(BaseModel):
    path: str
    method: str
    response_time: float
    status_code: int
    timestamp: float
    time: str
    threshold: float
    request_body: Optional[str] = None
    response_body: Optional[str] = None
    request_id: Optional[str] = None
    instance_id: Optional[str] = None
    headers: Dict[str, str]


class AlertRuleCreate(BaseModel):
    name: str
    alert_type: str
    severity: str
    threshold: float
    operator: str = "gt"
    window_seconds: int = 60
    path_pattern: Optional[str] = None
    cooldown_seconds: int = 60
    enabled: bool = True
    description: str = ""
    labels: Dict[str, str] = {}


class AlertRuleResponse(BaseModel):
    name: str
    alert_type: str
    severity: str
    threshold: float
    operator: str
    window_seconds: int
    path_pattern: Optional[str]
    cooldown_seconds: int
    enabled: bool
    description: str
    labels: Dict[str, str]


class AlertEventResponse(BaseModel):
    rule_name: str
    alert_type: str
    severity: str
    message: str
    value: float
    threshold: float
    timestamp: float
    time: str
    labels: Dict[str, str]


class InstanceMetricsReport(BaseModel):
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
    path_stats: List[Dict[str, Any]] = []
    slow_requests: List[Dict[str, Any]] = []


class AggregatedStatsResponse(BaseModel):
    timestamp: float
    time: str
    window_start: str
    window_end: str
    total_requests: int
    total_errors: int
    avg_response_time: float
    p99_response_time: float
    error_rate: float
    instance_count: int
    instances: Dict[str, Dict[str, Any]]
    path_stats: List[Dict[str, Any]]
    slow_requests: List[Dict[str, Any]]


class ThresholdUpdate(BaseModel):
    threshold_ms: float


def create_monitoring_router(
    storage: MetricsStorage,
    alert_manager: Optional[AlertManager] = None,
    aggregator: Optional[MetricsAggregator] = None
) -> APIRouter:
    router = APIRouter(prefix="/monitoring", tags=["monitoring"])

    @router.get("/stats/current", response_model=WindowStatsDetailResponse)
    async def get_current_window_stats():
        stats = storage.get_window_stats()
        if stats is None:
            raise HTTPException(status_code=404, detail="No statistics available")
        return WindowStatsDetailResponse(
            window_start=stats.window_start,
            window_end=stats.window_end,
            avg_response_time=stats.avg_response_time,
            p99_response_time=stats.p99_response_time,
            request_count=stats.request_count,
            error_count=stats.error_count,
            error_rate=stats.error_rate,
            requests=[
                RequestRecordResponse(
                    path=r.path,
                    method=r.method,
                    response_time=r.response_time,
                    status_code=r.status_code,
                    timestamp=r.timestamp,
                    request_id=r.request_id,
                    instance_id=r.instance_id
                )
                for r in stats.requests
            ],
            path_stats=stats.path_stats
        )

    @router.get("/stats/per-second", response_model=List[SecondBucketStats])
    async def get_per_second_stats():
        return storage.get_per_second_stats()

    @router.get("/stats/summary", response_model=WindowStatsResponse)
    async def get_window_summary():
        stats = storage.get_window_stats()
        if stats is None:
            raise HTTPException(status_code=404, detail="No statistics available")
        return WindowStatsResponse(
            window_start=stats.window_start,
            window_end=stats.window_end,
            avg_response_time=stats.avg_response_time,
            p99_response_time=stats.p99_response_time,
            request_count=stats.request_count,
            error_count=stats.error_count,
            error_rate=stats.error_rate
        )

    @router.get("/stats/paths", response_model=List[PathStatsResponse])
    async def get_path_statistics():
        path_stats = storage.get_path_stats()
        return [
            PathStatsResponse(**p)
            for p in path_stats
        ]

    @router.get("/requests/recent", response_model=List[RequestRecordResponse])
    async def get_recent_requests(limit: int = Query(default=100, ge=1, le=1000)):
        records = storage.get_recent_requests(limit=limit)
        return [
            RequestRecordResponse(
                path=r.path,
                method=r.method,
                response_time=r.response_time,
                status_code=r.status_code,
                timestamp=r.timestamp,
                request_id=r.request_id,
                instance_id=r.instance_id
            )
            for r in records
        ]

    @router.get("/slow-requests", response_model=List[SlowRequestResponse])
    async def get_slow_requests(
        limit: int = Query(default=100, ge=1, le=1000),
        path: Optional[str] = None
    ):
        records = storage.get_slow_requests(limit=limit, path=path)
        return [SlowRequestResponse(**r.to_dict()) for r in records]

    @router.get("/slow-requests/count")
    async def get_slow_request_count():
        return {
            "count": storage.get_slow_request_count(),
            "threshold_ms": storage.slow_request_threshold_ms
        }

    @router.put("/slow-requests/threshold")
    async def update_slow_request_threshold(body: ThresholdUpdate):
        if body.threshold_ms <= 0:
            raise HTTPException(status_code=400, detail="Threshold must be positive")
        storage.set_slow_request_threshold(body.threshold_ms)
        return {"status": "ok", "new_threshold_ms": body.threshold_ms}

    @router.get("/alerts/rules", response_model=List[AlertRuleResponse])
    async def list_alert_rules():
        if alert_manager is None:
            raise HTTPException(status_code=501, detail="Alert manager not configured")
        rules = alert_manager.list_rules()
        return [
            AlertRuleResponse(**rule.to_dict())
            for rule in rules
        ]

    @router.post("/alerts/rules", response_model=AlertRuleResponse)
    async def create_alert_rule(rule_data: AlertRuleCreate):
        if alert_manager is None:
            raise HTTPException(status_code=501, detail="Alert manager not configured")
        try:
            alert_type = AlertType(rule_data.alert_type)
            severity = AlertSeverity(rule_data.severity)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=f"Invalid alert type or severity: {e}")

        rule = AlertRule(
            name=rule_data.name,
            alert_type=alert_type,
            severity=severity,
            threshold=rule_data.threshold,
            operator=rule_data.operator,
            window_seconds=rule_data.window_seconds,
            path_pattern=rule_data.path_pattern,
            cooldown_seconds=rule_data.cooldown_seconds,
            enabled=rule_data.enabled,
            description=rule_data.description,
            labels=rule_data.labels
        )
        alert_manager.add_rule(rule)
        return AlertRuleResponse(**rule.to_dict())

    @router.get("/alerts/rules/{rule_name}", response_model=AlertRuleResponse)
    async def get_alert_rule(rule_name: str):
        if alert_manager is None:
            raise HTTPException(status_code=501, detail="Alert manager not configured")
        rule = alert_manager.get_rule(rule_name)
        if rule is None:
            raise HTTPException(status_code=404, detail="Rule not found")
        return AlertRuleResponse(**rule.to_dict())

    @router.put("/alerts/rules/{rule_name}")
    async def update_alert_rule(rule_name: str, updates: Dict[str, Any]):
        if alert_manager is None:
            raise HTTPException(status_code=501, detail="Alert manager not configured")
        rule = alert_manager.update_rule(rule_name, **updates)
        if rule is None:
            raise HTTPException(status_code=404, detail="Rule not found")
        return {"status": "ok", "rule": rule.to_dict()}

    @router.delete("/alerts/rules/{rule_name}")
    async def delete_alert_rule(rule_name: str):
        if alert_manager is None:
            raise HTTPException(status_code=501, detail="Alert manager not configured")
        if not alert_manager.remove_rule(rule_name):
            raise HTTPException(status_code=404, detail="Rule not found")
        return {"status": "ok"}

    @router.get("/alerts/history", response_model=List[AlertEventResponse])
    async def get_alert_history(limit: int = Query(default=100, ge=1, le=1000)):
        if alert_manager is None:
            raise HTTPException(status_code=501, detail="Alert manager not configured")
        events = alert_manager.get_history(limit=limit)
        return [AlertEventResponse(**e.to_dict()) for e in events]

    @router.post("/central/report")
    async def report_instance_metrics(report: InstanceMetricsReport):
        if aggregator is None or not aggregator.is_central:
            raise HTTPException(status_code=501, detail="Not a central aggregator node")
        from .storage import InstanceMetrics
        metrics = InstanceMetrics(
            instance_id=report.instance_id,
            hostname=report.hostname,
            timestamp=report.timestamp,
            window_start=report.window_start,
            window_end=report.window_end,
            avg_response_time=report.avg_response_time,
            p99_response_time=report.p99_response_time,
            request_count=report.request_count,
            error_count=report.error_count,
            error_rate=report.error_rate,
            path_stats=report.path_stats,
            slow_requests=report.slow_requests
        )
        aggregator.report_metrics(metrics)
        return {"status": "received"}

    @router.get("/central/aggregated", response_model=AggregatedStatsResponse)
    async def get_aggregated_stats():
        if aggregator is None or not aggregator.is_central:
            raise HTTPException(status_code=501, detail="Not a central aggregator node")
        stats = aggregator.get_aggregated_stats()
        if stats is None:
            raise HTTPException(status_code=404, detail="No aggregated data available")
        return AggregatedStatsResponse(**stats.to_dict())

    @router.get("/central/instances")
    async def list_instances():
        if aggregator is None or not aggregator.is_central:
            raise HTTPException(status_code=501, detail="Not a central aggregator node")
        instance_ids = aggregator.get_instance_ids()
        return {
            "instance_count": len(instance_ids),
            "instances": instance_ids
        }

    @router.get("/central/instances/{instance_id}")
    async def get_instance_detail(instance_id: str):
        if aggregator is None or not aggregator.is_central:
            raise HTTPException(status_code=501, detail="Not a central aggregator node")
        metrics = aggregator.get_instance_metrics(instance_id)
        if metrics is None:
            raise HTTPException(status_code=404, detail="Instance not found")
        return {
            "instance_id": metrics.instance_id,
            "hostname": metrics.hostname,
            "last_report": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(metrics.timestamp)),
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

    @router.get("/health")
    async def health_check():
        stats = storage.get_window_stats()
        result = {
            "status": "healthy",
            "instance_id": storage.instance_id,
            "hostname": storage.hostname,
            "window_seconds": storage.window_seconds,
            "bucket_seconds": storage.bucket_seconds,
            "active_buckets": storage.get_bucket_count(),
            "total_requests": stats.request_count if stats else 0,
            "slow_requests_count": storage.get_slow_request_count(),
            "slow_request_threshold_ms": storage.slow_request_threshold_ms
        }
        if aggregator:
            result["is_central"] = aggregator.is_central
            if aggregator.central_url:
                result["central_url"] = aggregator.central_url
        if alert_manager:
            result["alert_rules_count"] = len(alert_manager.list_rules())
        return result

    return router


import time
