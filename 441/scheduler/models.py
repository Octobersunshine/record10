from __future__ import annotations

import uuid
from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class TaskMode(str, Enum):
    CRON = "cron"
    INTERVAL = "interval"


class HttpMethod(str, Enum):
    GET = "GET"
    POST = "POST"
    PUT = "PUT"
    DELETE = "DELETE"


class HttpCallback(BaseModel):
    url: str
    method: HttpMethod = HttpMethod.POST
    headers: dict[str, str] = Field(default_factory=dict)
    body: Optional[str] = None
    timeout: int = 30


class TaskStatus(str, Enum):
    ACTIVE = "active"
    PAUSED = "paused"


class ExecutionStatus(str, Enum):
    SUCCESS = "success"
    FAILED = "failed"
    RETRY = "retry"


class Task(BaseModel):
    id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    name: str
    mode: TaskMode
    cron_expr: Optional[str] = None
    interval_seconds: Optional[int] = None
    callback: HttpCallback
    status: TaskStatus = TaskStatus.ACTIVE
    depends_on: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    last_run_at: Optional[datetime] = None
    next_run_at: Optional[datetime] = None
    run_count: int = 0
    success_count: int = 0
    failed_count: int = 0
    total_duration_ms: int = 0


class TaskCreateRequest(BaseModel):
    name: str
    mode: TaskMode
    cron_expr: Optional[str] = None
    interval_seconds: Optional[int] = None
    callback: HttpCallback
    status: TaskStatus = TaskStatus.ACTIVE
    depends_on: Optional[str] = None


class TaskUpdateRequest(BaseModel):
    name: Optional[str] = None
    mode: Optional[TaskMode] = None
    cron_expr: Optional[str] = None
    interval_seconds: Optional[int] = None
    callback: Optional[HttpCallback] = None
    status: Optional[TaskStatus] = None
    depends_on: Optional[str] = None


class ExecutionHistory(BaseModel):
    id: str
    task_id: str
    task_name: str
    status: ExecutionStatus
    retry_attempt: int
    duration_ms: int
    error_message: Optional[str] = None
    started_at: datetime
    ended_at: datetime


class TaskStats(BaseModel):
    task_id: str
    task_name: str
    total_runs: int
    success_count: int
    failed_count: int
    avg_duration_ms: float
    min_duration_ms: int
    max_duration_ms: int
    last_success_at: Optional[datetime] = None
    last_failed_at: Optional[datetime] = None
