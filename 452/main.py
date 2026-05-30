import logging
import random
import asyncio
import os
from fastapi import FastAPI
from monitoring import (
    MetricsStorage,
    ResponseTimeMiddleware,
    create_monitoring_router,
    AlertManager,
    AlertRule,
    AlertType,
    AlertSeverity,
    MetricsAggregator
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

app = FastAPI(title="API Response Time Monitoring", version="3.0.0")

IS_CENTRAL = os.environ.get("MONITORING_CENTRAL", "false").lower() == "true"
CENTRAL_URL = os.environ.get("MONITORING_CENTRAL_URL")
REPORT_INTERVAL = int(os.environ.get("MONITORING_REPORT_INTERVAL", "10"))

storage = MetricsStorage(
    window_seconds=60,
    bucket_seconds=1,
    slow_request_threshold_ms=500.0,
    max_slow_requests=1000
)

alert_manager = AlertManager(max_history=1000)

default_rules = [
    AlertRule(
        name="high_avg_response_time",
        alert_type=AlertType.SLOW_REQUEST,
        severity=AlertSeverity.WARNING,
        threshold=300.0,
        operator="gt",
        description="Average response time exceeds 300ms",
        labels={"category": "performance"}
    ),
    AlertRule(
        name="high_p99_response_time",
        alert_type=AlertType.P99_EXCEEDED,
        severity=AlertSeverity.CRITICAL,
        threshold=1000.0,
        operator="gt",
        description="P99 response time exceeds 1000ms",
        labels={"category": "performance"}
    ),
    AlertRule(
        name="high_error_rate",
        alert_type=AlertType.ERROR_RATE,
        severity=AlertSeverity.CRITICAL,
        threshold=5.0,
        operator="gt",
        description="Error rate exceeds 5%",
        labels={"category": "availability"}
    ),
    AlertRule(
        name="high_throughput",
        alert_type=AlertType.THROUGHPUT,
        severity=AlertSeverity.INFO,
        threshold=100.0,
        operator="gt",
        description="Throughput exceeds 100 req/s",
        labels={"category": "capacity"}
    ),
    AlertRule(
        name="slow_api_slow_endpoint",
        alert_type=AlertType.SLOW_REQUEST,
        severity=AlertSeverity.WARNING,
        threshold=800.0,
        operator="gt",
        path_pattern="GET:/api/slow",
        cooldown_seconds=120,
        description="Slow endpoint /api/slow exceeds 800ms",
        labels={"category": "performance", "path": "/api/slow"}
    )
]

for rule in default_rules:
    alert_manager.add_rule(rule)
    logger.info(f"Loaded alert rule: {rule.name}")


def alert_handler(event):
    logger.warning(
        f"[ALERT] [{event.severity.value.upper()}] {event.rule_name}: "
        f"{event.message} (value={event.value:.2f}, threshold={event.threshold})"
    )


alert_manager.add_handler(alert_handler)

aggregator = MetricsAggregator(
    is_central=IS_CENTRAL,
    central_url=CENTRAL_URL,
    report_interval=REPORT_INTERVAL
)

app.add_middleware(
    ResponseTimeMiddleware,
    storage=storage,
    alert_manager=alert_manager,
    exclude_paths=["/docs", "/openapi.json", "/favicon.ico"],
    log_requests=True,
    capture_body=True,
    capture_headers=True,
    max_body_size=4096
)

monitoring_router = create_monitoring_router(storage, alert_manager, aggregator)
app.include_router(monitoring_router)


@app.on_event("startup")
async def startup_event():
    if not IS_CENTRAL and CENTRAL_URL:
        aggregator.start_reporting(storage)
        logger.info(f"Monitoring started. Instance ID: {storage.instance_id}")
        logger.info(f"Reporting to central: {CENTRAL_URL} every {REPORT_INTERVAL}s")
    elif IS_CENTRAL:
        logger.info(f"Central monitoring node started. Instance ID: {storage.instance_id}")
    else:
        logger.info(f"Monitoring started. Instance ID: {storage.instance_id} (standalone mode)")


@app.on_event("shutdown")
async def shutdown_event():
    aggregator.stop_reporting()


@app.get("/")
async def root():
    return {
        "message": "API Response Time Monitoring Demo v3",
        "instance_id": storage.instance_id,
        "hostname": storage.hostname,
        "features": [
            "Response time monitoring with sliding window",
            "Thread-safe metrics storage",
            "Slow request tracking with request/response body capture",
            "Per-path statistics aggregation",
            "Alert rules engine with 4 default rules",
            "Multi-instance aggregation support",
            "Central node for multi-deployment monitoring"
        ],
        "is_central": IS_CENTRAL,
        "slow_request_threshold_ms": storage.slow_request_threshold_ms
    }


@app.get("/api/users")
async def get_users():
    await asyncio.sleep(random.uniform(0.01, 0.1))
    return {"users": [{"id": i, "name": f"User{i}"} for i in range(10)]}


@app.get("/api/users/{user_id}")
async def get_user(user_id: int):
    await asyncio.sleep(random.uniform(0.005, 0.05))
    return {"id": user_id, "name": f"User{user_id}", "email": f"user{user_id}@example.com"}


@app.post("/api/users")
async def create_user(user: dict):
    await asyncio.sleep(random.uniform(0.02, 0.15))
    return {"id": random.randint(1, 1000), **user}


@app.get("/api/products")
async def get_products():
    await asyncio.sleep(random.uniform(0.03, 0.2))
    return {"products": [{"id": i, "name": f"Product{i}", "price": i * 10.5} for i in range(20)]}


@app.get("/api/orders")
async def get_orders():
    await asyncio.sleep(random.uniform(0.05, 0.3))
    return {"orders": [{"id": i, "status": "pending", "total": i * 100} for i in range(5)]}


@app.get("/api/slow")
async def slow_endpoint():
    await asyncio.sleep(random.uniform(0.5, 2.0))
    return {"message": "This is a slow endpoint"}


@app.get("/api/random-status")
async def random_status():
    status_codes = [200, 200, 200, 201, 400, 401, 404, 500]
    code = random.choice(status_codes)
    await asyncio.sleep(random.uniform(0.01, 0.1))
    from fastapi import HTTPException
    if code >= 400:
        raise HTTPException(status_code=code, detail=f"Simulated error {code}")
    return {"status": "success", "code": code}


@app.get("/api/echo", tags=["test"])
async def echo(message: str = "hello"):
    return {"echo": message, "timestamp": asyncio.get_event_loop().time()}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
