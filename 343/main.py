import asyncio
import json
import logging
import random
import sqlite3
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Optional
from uuid import uuid4

import httpx
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.date import DateTrigger
from apscheduler.triggers.interval import IntervalTrigger
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DB_PATH = Path(__file__).parent / "tasks.db"

app = FastAPI(title="定时任务管理API", version="3.0.0")

scheduler = BackgroundScheduler(timezone="Asia/Shanghai")


class TaskType(str, Enum):
    delay = "delay"
    interval = "interval"
    cron = "cron"


class HttpMethod(str, Enum):
    GET = "GET"
    POST = "POST"
    PUT = "PUT"
    PATCH = "PATCH"
    DELETE = "DELETE"


class ExecutionStatus(str, Enum):
    success = "success"
    failed = "failed"
    retrying = "retrying"
    skipped = "skipped"


class CallbackConfig(BaseModel):
    url: str
    method: HttpMethod = HttpMethod.POST
    headers: Optional[dict[str, str]] = None
    body: Optional[Any] = None
    timeout: int = Field(default=30, ge=1, le=300)


class RetryConfig(BaseModel):
    max_attempts: int = Field(default=3, ge=1, le=20, description="最大重试次数（包含首次执行）")
    base_delay_seconds: int = Field(default=2, ge=1, le=3600, description="初始退避延迟秒数")
    multiplier: float = Field(default=2.0, ge=1.0, le=10.0, description="退避乘数")
    max_delay_seconds: int = Field(default=60, ge=1, le=3600, description="最大退避延迟秒数")
    jitter: bool = Field(default=True, description="是否添加随机抖动")


class TaskCreate(BaseModel):
    name: str
    type: TaskType
    delay_seconds: Optional[int] = Field(None, ge=1, description="延迟执行的秒数(type=delay时必填)")
    interval_seconds: Optional[int] = Field(None, ge=1, description="间隔执行的秒数(type=interval时必填)")
    cron_expression: Optional[str] = Field(None, description="Cron表达式(type=cron时必填, 格式: 秒 分 时 日 月 星期)")
    callback: CallbackConfig
    enabled: bool = True
    retry_config: Optional[RetryConfig] = None
    depends_on: Optional[list[str]] = Field(None, description="依赖的前置任务ID列表，所有前置任务成功后才执行")


class TaskUpdate(BaseModel):
    name: Optional[str] = None
    delay_seconds: Optional[int] = Field(None, ge=1)
    interval_seconds: Optional[int] = Field(None, ge=1)
    cron_expression: Optional[str] = None
    callback: Optional[CallbackConfig] = None
    enabled: Optional[bool] = None
    retry_config: Optional[RetryConfig] = None
    depends_on: Optional[list[str]] = None


class TaskResponse(BaseModel):
    id: str
    name: str
    type: TaskType
    delay_seconds: Optional[int] = None
    interval_seconds: Optional[int] = None
    cron_expression: Optional[str] = None
    callback: CallbackConfig
    enabled: bool
    next_run_time: Optional[str] = None
    run_date: Optional[str] = None
    retry_config: Optional[RetryConfig] = None
    depends_on: Optional[list[str]] = None
    created_at: str
    updated_at: str


class TaskListResponse(BaseModel):
    total: int
    tasks: list[TaskResponse]


class ExecutionHistoryRecord(BaseModel):
    id: int
    task_id: str
    task_name: str
    status: ExecutionStatus
    attempt: int
    max_attempts: int
    status_code: Optional[int] = None
    error_message: Optional[str] = None
    duration_ms: Optional[float] = None
    started_at: str
    finished_at: Optional[str] = None
    trigger_type: str
    parent_execution_id: Optional[int] = None


class ExecutionHistoryListResponse(BaseModel):
    total: int
    records: list[ExecutionHistoryRecord]


def _get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def _column_exists(conn, table_name, column_name) -> bool:
    rows = conn.execute(f"PRAGMA table_info({table_name})").fetchall()
    return any(r["name"] == column_name for r in rows)


def init_db():
    conn = _get_conn()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS tasks (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            type TEXT NOT NULL,
            delay_seconds INTEGER,
            interval_seconds INTEGER,
            cron_expression TEXT,
            callback TEXT NOT NULL,
            enabled INTEGER NOT NULL DEFAULT 1,
            run_date TEXT,
            retry_config TEXT,
            depends_on TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
    """)
    if not _column_exists(conn, "tasks", "retry_config"):
        conn.execute("ALTER TABLE tasks ADD COLUMN retry_config TEXT")
        logger.info("数据库迁移: 添加 retry_config 列")
    if not _column_exists(conn, "tasks", "depends_on"):
        conn.execute("ALTER TABLE tasks ADD COLUMN depends_on TEXT")
        logger.info("数据库迁移: 添加 depends_on 列")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS execution_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            task_id TEXT NOT NULL,
            task_name TEXT NOT NULL,
            status TEXT NOT NULL,
            attempt INTEGER NOT NULL DEFAULT 1,
            max_attempts INTEGER NOT NULL DEFAULT 1,
            status_code INTEGER,
            error_message TEXT,
            duration_ms REAL,
            started_at TEXT NOT NULL,
            finished_at TEXT,
            trigger_type TEXT NOT NULL,
            parent_execution_id INTEGER,
            FOREIGN KEY (task_id) REFERENCES tasks(id) ON DELETE CASCADE
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_history_task_id ON execution_history(task_id)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_history_status ON execution_history(status)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_history_started_at ON execution_history(started_at)")
    conn.commit()
    conn.close()
    logger.info("数据库初始化完成: %s", DB_PATH)


def _db_insert(task_data: dict):
    conn = _get_conn()
    conn.execute(
        "INSERT INTO tasks (id, name, type, delay_seconds, interval_seconds, cron_expression, callback, enabled, run_date, retry_config, depends_on, created_at, updated_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            task_data["id"],
            task_data["name"],
            task_data["type"],
            task_data.get("delay_seconds"),
            task_data.get("interval_seconds"),
            task_data.get("cron_expression"),
            json.dumps(task_data["callback"], ensure_ascii=False),
            1 if task_data["enabled"] else 0,
            task_data.get("run_date"),
            json.dumps(task_data.get("retry_config"), ensure_ascii=False) if task_data.get("retry_config") else None,
            json.dumps(task_data.get("depends_on"), ensure_ascii=False) if task_data.get("depends_on") else None,
            task_data["created_at"],
            task_data["updated_at"],
        ),
    )
    conn.commit()
    conn.close()


def _db_get(task_id: str) -> Optional[dict]:
    conn = _get_conn()
    row = conn.execute("SELECT * FROM tasks WHERE id = ?", (task_id,)).fetchone()
    conn.close()
    if not row:
        return None
    return _row_to_dict(row)


def _db_list(task_type: Optional[str] = None, enabled: Optional[bool] = None) -> list[dict]:
    conn = _get_conn()
    sql = "SELECT * FROM tasks WHERE 1=1"
    params: list[Any] = []
    if task_type is not None:
        sql += " AND type = ?"
        params.append(task_type)
    if enabled is not None:
        sql += " AND enabled = ?"
        params.append(1 if enabled else 0)
    rows = conn.execute(sql, params).fetchall()
    conn.close()
    return [_row_to_dict(r) for r in rows]


def _db_update(task_id: str, updates: dict):
    conn = _get_conn()
    sets = []
    vals: list[Any] = []
    for key, val in updates.items():
        if key == "callback":
            sets.append("callback = ?")
            vals.append(json.dumps(val, ensure_ascii=False))
        elif key == "enabled":
            sets.append("enabled = ?")
            vals.append(1 if val else 0)
        elif key == "retry_config":
            sets.append("retry_config = ?")
            vals.append(json.dumps(val, ensure_ascii=False) if val else None)
        elif key == "depends_on":
            sets.append("depends_on = ?")
            vals.append(json.dumps(val, ensure_ascii=False) if val else None)
        else:
            sets.append(f"{key} = ?")
            vals.append(val)
    vals.append(task_id)
    conn.execute(f"UPDATE tasks SET {', '.join(sets)} WHERE id = ?", vals)
    conn.commit()
    conn.close()


def _db_delete(task_id: str):
    conn = _get_conn()
    conn.execute("DELETE FROM tasks WHERE id = ?", (task_id,))
    conn.commit()
    conn.close()


def _row_to_dict(row: sqlite3.Row) -> dict:
    d = dict(row)
    d["callback"] = json.loads(d["callback"]) if d.get("callback") else None
    d["retry_config"] = json.loads(d["retry_config"]) if d.get("retry_config") else None
    d["depends_on"] = json.loads(d["depends_on"]) if d.get("depends_on") else None
    d["enabled"] = bool(d["enabled"])
    return d


def _history_insert(history_data: dict) -> int:
    conn = _get_conn()
    cur = conn.execute(
        "INSERT INTO execution_history (task_id, task_name, status, attempt, max_attempts, status_code, error_message, duration_ms, started_at, finished_at, trigger_type, parent_execution_id) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            history_data["task_id"],
            history_data["task_name"],
            history_data["status"],
            history_data["attempt"],
            history_data["max_attempts"],
            history_data.get("status_code"),
            history_data.get("error_message"),
            history_data.get("duration_ms"),
            history_data["started_at"],
            history_data.get("finished_at"),
            history_data["trigger_type"],
            history_data.get("parent_execution_id"),
        ),
    )
    conn.commit()
    last_id = cur.lastrowid
    conn.close()
    return last_id


def _history_update(history_id: int, updates: dict):
    conn = _get_conn()
    sets = []
    vals: list[Any] = []
    for key, val in updates.items():
        sets.append(f"{key} = ?")
        vals.append(val)
    vals.append(history_id)
    conn.execute(f"UPDATE execution_history SET {', '.join(sets)} WHERE id = ?", vals)
    conn.commit()
    conn.close()


def _history_list(task_id: Optional[str] = None, status: Optional[ExecutionStatus] = None,
                  limit: int = 100, offset: int = 0) -> tuple[int, list[dict]]:
    conn = _get_conn()
    sql = "FROM execution_history WHERE 1=1"
    count_sql = "SELECT COUNT(*) as cnt " + sql
    data_sql = "SELECT * " + sql
    params: list[Any] = []
    if task_id is not None:
        sql += " AND task_id = ?"
        count_sql += " AND task_id = ?"
        data_sql += " AND task_id = ?"
        params.append(task_id)
    if status is not None:
        sql += " AND status = ?"
        count_sql += " AND status = ?"
        data_sql += " AND status = ?"
        params.append(status.value)
    data_sql += " ORDER BY started_at DESC LIMIT ? OFFSET ?"
    data_params = params + [limit, offset]
    total_row = conn.execute(count_sql, params).fetchone()
    total = total_row["cnt"] if total_row else 0
    rows = conn.execute(data_sql, data_params).fetchall()
    conn.close()
    return total, [dict(r) for r in rows]


def _build_trigger(task_data: dict, for_recovery: bool = False):
    task_type = task_data["type"]
    if task_type == TaskType.delay:
        delay = task_data.get("delay_seconds")
        if not delay:
            raise ValueError("delay类型任务必须指定delay_seconds")
        if for_recovery and task_data.get("run_date"):
            run_date = datetime.fromisoformat(task_data["run_date"])
        else:
            run_date = datetime.now() + timedelta(seconds=delay)
        return DateTrigger(run_date=run_date)
    elif task_type == TaskType.interval:
        interval = task_data.get("interval_seconds")
        if not interval:
            raise ValueError("interval类型任务必须指定interval_seconds")
        return IntervalTrigger(seconds=interval)
    elif task_type == TaskType.cron:
        cron_expr = task_data.get("cron_expression")
        if not cron_expr:
            raise ValueError("cron类型任务必须指定cron_expression")
        parts = cron_expr.strip().split()
        if len(parts) != 6:
            raise ValueError("Cron表达式必须包含6个字段: 秒 分 时 日 月 星期")
        return CronTrigger(
            second=parts[0], minute=parts[1], hour=parts[2],
            day=parts[3], month=parts[4], day_of_week=parts[5],
        )
    else:
        raise ValueError(f"不支持的任务类型: {task_type}")


def _calculate_backoff(retry_cfg: dict, attempt: int) -> float:
    base = retry_cfg.get("base_delay_seconds", 2)
    multiplier = retry_cfg.get("multiplier", 2.0)
    max_delay = retry_cfg.get("max_delay_seconds", 60)
    delay = base * (multiplier ** (attempt - 1))
    delay = min(delay, max_delay)
    if retry_cfg.get("jitter", True):
        delay = delay * (0.5 + random.random() * 0.5)
    return delay


def _check_dependencies_met(task_data: dict) -> tuple[bool, list[str]]:
    depends_on = task_data.get("depends_on") or []
    if not depends_on:
        return True, []
    unmet = []
    for dep_id in depends_on:
        dep_task = _db_get(dep_id)
        if not dep_task:
            unmet.append(f"任务{dep_id}不存在")
            continue
        total, records = _history_list(task_id=dep_id, limit=1000)
        has_success = any(r["status"] == ExecutionStatus.success.value for r in records)
        if not has_success:
            unmet.append(f"依赖任务{dep_id}({dep_task['name']})尚未成功执行")
    return len(unmet) == 0, unmet


async def _execute_callback(task_id: str, trigger_type: str = "schedule",
                            parent_execution_id: Optional[int] = None, run_once: bool = False):
    task_data = _db_get(task_id)
    if not task_data:
        return
    if not task_data["enabled"]:
        logger.info("任务已禁用，跳过执行: task_id=%s", task_id)
        return

    deps_met, unmet = _check_dependencies_met(task_data)
    if not deps_met:
        logger.info("依赖未满足，跳过任务: task_id=%s, unmet=%s", task_id, unmet)
        hid = _history_insert({
            "task_id": task_id,
            "task_name": task_data["name"],
            "status": ExecutionStatus.skipped.value,
            "attempt": 1,
            "max_attempts": 1,
            "error_message": "; ".join(unmet),
            "started_at": datetime.now().isoformat(),
            "trigger_type": trigger_type,
            "parent_execution_id": parent_execution_id,
        })
        _history_update(hid, {"finished_at": datetime.now().isoformat(), "duration_ms": 0})
        return

    retry_cfg = task_data.get("retry_config") or {"max_attempts": 1}
    max_attempts = retry_cfg.get("max_attempts", 1)
    cb = task_data["callback"]
    success = False
    last_error = None
    last_status_code = None
    final_attempt = 1
    history_id = None

    for attempt in range(1, max_attempts + 1):
        final_attempt = attempt
        started_at = datetime.now()
        status = ExecutionStatus.retrying.value if attempt < max_attempts else ExecutionStatus.failed.value

        if history_id is None:
            history_id = _history_insert({
                "task_id": task_id,
                "task_name": task_data["name"],
                "status": status,
                "attempt": attempt,
                "max_attempts": max_attempts,
                "started_at": started_at.isoformat(),
                "trigger_type": trigger_type,
                "parent_execution_id": parent_execution_id,
            })
        else:
            _history_update(history_id, {
                "status": status,
                "attempt": attempt,
                "started_at": started_at.isoformat(),
                "finished_at": None,
                "error_message": None,
                "status_code": None,
                "duration_ms": None,
            })

        logger.info("执行任务回调: task_id=%s, name=%s, attempt=%d/%d, url=%s",
                    task_id, task_data["name"], attempt, max_attempts, cb["url"])
        try:
            async with httpx.AsyncClient(timeout=cb.get("timeout", 30)) as client:
                resp = await client.request(
                    method=cb["method"],
                    url=cb["url"],
                    headers=cb.get("headers"),
                    json=cb.get("body"),
                )
            last_status_code = resp.status_code
            duration_ms = (datetime.now() - started_at).total_seconds() * 1000
            if resp.status_code < 400:
                success = True
                _history_update(history_id, {
                    "status": ExecutionStatus.success.value,
                    "status_code": resp.status_code,
                    "duration_ms": duration_ms,
                    "finished_at": datetime.now().isoformat(),
                })
                logger.info("回调成功: task_id=%s, attempt=%d, status=%d", task_id, attempt, resp.status_code)
                break
            else:
                last_error = f"HTTP {resp.status_code}"
                _history_update(history_id, {
                    "status": status,
                    "status_code": resp.status_code,
                    "error_message": f"HTTP {resp.status_code}",
                    "duration_ms": duration_ms,
                    "finished_at": datetime.now().isoformat(),
                })
                logger.warning("回调返回错误状态: task_id=%s, attempt=%d, status=%d", task_id, attempt, resp.status_code)
        except Exception as e:
            last_error = str(e)
            duration_ms = (datetime.now() - started_at).total_seconds() * 1000
            _history_update(history_id, {
                "status": status,
                "error_message": str(e),
                "duration_ms": duration_ms,
                "finished_at": datetime.now().isoformat(),
            })
            logger.error("回调异常: task_id=%s, attempt=%d, error=%s", task_id, attempt, str(e))

        if attempt < max_attempts:
            backoff = _calculate_backoff(retry_cfg, attempt)
            logger.info("等待重试: task_id=%s, attempt=%d, backoff=%.2fs", task_id, attempt, backoff)
            await asyncio.sleep(backoff)

    if success:
        await _trigger_downstream_tasks(task_id, history_id)
    else:
        logger.error("任务最终失败: task_id=%s, attempt=%d, error=%s", task_id, final_attempt, last_error)

    if run_once or task_data["type"] == TaskType.delay:
        _db_update(task_id, {"enabled": False, "updated_at": datetime.now().isoformat()})
        job_id = f"task_{task_id}"
        if scheduler.get_job(job_id):
            scheduler.remove_job(job_id)


async def _trigger_downstream_tasks(completed_task_id: str, parent_execution_id: int):
    all_tasks = _db_list(enabled=True)
    triggered = []
    for task in all_tasks:
        deps = task.get("depends_on") or []
        if completed_task_id not in deps:
            continue
        deps_met, _ = _check_dependencies_met(task)
        if not deps_met:
            continue
        logger.info("触发下游任务: %s(%s) 因为依赖 %s 已完成", task["id"], task["name"], completed_task_id)
        asyncio.ensure_future(_execute_callback(
            task["id"],
            trigger_type="dependency",
            parent_execution_id=parent_execution_id,
            run_once=(task["type"] == TaskType.delay),
        ))
        triggered.append(task["id"])
    if triggered:
        logger.info("下游任务触发完成: completed_task_id=%s, triggered=%s", completed_task_id, triggered)


def _sync_callback_wrapper(task_id: str):
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            asyncio.ensure_future(_execute_callback(task_id))
        else:
            loop.run_until_complete(_execute_callback(task_id))
    except RuntimeError:
        loop = asyncio.new_event_loop()
        loop.run_until_complete(_execute_callback(task_id))


def _schedule_task(task_id: str, task_data: dict, for_recovery: bool = False):
    try:
        trigger = _build_trigger(task_data, for_recovery=for_recovery)
    except ValueError:
        return
    job_id = f"task_{task_id}"
    existing = scheduler.get_job(job_id)
    if existing:
        scheduler.remove_job(job_id)
    if not task_data["enabled"]:
        return
    scheduler.add_job(
        _sync_callback_wrapper,
        trigger=trigger,
        id=job_id,
        args=[task_id],
        replace_existing=True,
    )


def _task_to_response(task_data: dict) -> TaskResponse:
    job_id = f"task_{task_data['id']}"
    job = scheduler.get_job(job_id)
    next_run = str(job.next_run_time) if job and job.next_run_time else None
    return TaskResponse(
        id=task_data["id"],
        name=task_data["name"],
        type=task_data["type"],
        delay_seconds=task_data.get("delay_seconds"),
        interval_seconds=task_data.get("interval_seconds"),
        cron_expression=task_data.get("cron_expression"),
        callback=task_data["callback"],
        enabled=task_data["enabled"],
        next_run_time=next_run,
        run_date=task_data.get("run_date"),
        retry_config=RetryConfig(**task_data["retry_config"]) if task_data.get("retry_config") else None,
        depends_on=task_data.get("depends_on"),
        created_at=task_data["created_at"],
        updated_at=task_data["updated_at"],
    )


def _history_to_record(row: dict) -> ExecutionHistoryRecord:
    return ExecutionHistoryRecord(**row)


@app.on_event("startup")
def startup_event():
    init_db()
    scheduler.start()
    all_tasks = _db_list()
    now = datetime.now()
    recovered = 0
    expired = 0
    for task_data in all_tasks:
        if not task_data["enabled"]:
            continue
        if task_data["type"] == TaskType.delay:
            run_date_str = task_data.get("run_date")
            if not run_date_str:
                continue
            run_date = datetime.fromisoformat(run_date_str)
            if run_date <= now:
                _db_update(task_data["id"], {"enabled": False, "updated_at": now.isoformat()})
                expired += 1
                logger.info("跳过已过期的delay任务: id=%s, name=%s, run_date=%s", task_data["id"], task_data["name"], run_date_str)
                continue
        _schedule_task(task_data["id"], task_data, for_recovery=True)
        recovered += 1
    logger.info("启动恢复完成: 共%d个任务, 恢复调度%d个, 过期跳过%d个", len(all_tasks), recovered, expired)


@app.post("/tasks", response_model=TaskResponse, status_code=201)
def create_task(body: TaskCreate):
    task_id = uuid4().hex
    now = datetime.now()
    run_date = None
    if body.type == TaskType.delay:
        if not body.delay_seconds:
            raise HTTPException(status_code=400, detail="delay类型任务必须指定delay_seconds")
        run_date = (now + timedelta(seconds=body.delay_seconds)).isoformat()
    if body.depends_on:
        for dep_id in body.depends_on:
            if not _db_get(dep_id):
                raise HTTPException(status_code=400, detail=f"依赖任务不存在: {dep_id}")
    task_data = {
        "id": task_id,
        "name": body.name,
        "type": body.type,
        "delay_seconds": body.delay_seconds,
        "interval_seconds": body.interval_seconds,
        "cron_expression": body.cron_expression,
        "callback": body.callback.model_dump(),
        "enabled": body.enabled,
        "run_date": run_date,
        "retry_config": body.retry_config.model_dump() if body.retry_config else None,
        "depends_on": body.depends_on,
        "created_at": now.isoformat(),
        "updated_at": now.isoformat(),
    }
    try:
        _build_trigger(task_data)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    _db_insert(task_data)
    _schedule_task(task_id, task_data)
    return _task_to_response(task_data)


@app.get("/tasks", response_model=TaskListResponse)
def list_tasks(type: Optional[TaskType] = None, enabled: Optional[bool] = None):
    results = _db_list(task_type=type.value if type else None, enabled=enabled)
    return TaskListResponse(
        total=len(results),
        tasks=[_task_to_response(t) for t in results],
    )


@app.get("/tasks/{task_id}", response_model=TaskResponse)
def get_task(task_id: str):
    task_data = _db_get(task_id)
    if not task_data:
        raise HTTPException(status_code=404, detail="任务不存在")
    return _task_to_response(task_data)


@app.put("/tasks/{task_id}", response_model=TaskResponse)
def update_task(task_id: str, body: TaskUpdate):
    task_data = _db_get(task_id)
    if not task_data:
        raise HTTPException(status_code=404, detail="任务不存在")
    updates = {}
    if body.name is not None:
        updates["name"] = body.name
    if body.delay_seconds is not None:
        updates["delay_seconds"] = body.delay_seconds
        if task_data["type"] == TaskType.delay:
            updates["run_date"] = (datetime.now() + timedelta(seconds=body.delay_seconds)).isoformat()
    if body.interval_seconds is not None:
        updates["interval_seconds"] = body.interval_seconds
    if body.cron_expression is not None:
        updates["cron_expression"] = body.cron_expression
    if body.callback is not None:
        updates["callback"] = body.callback.model_dump()
    if body.enabled is not None:
        updates["enabled"] = body.enabled
    if body.retry_config is not None:
        updates["retry_config"] = body.retry_config.model_dump()
    if body.depends_on is not None:
        for dep_id in body.depends_on:
            if not _db_get(dep_id):
                raise HTTPException(status_code=400, detail=f"依赖任务不存在: {dep_id}")
        updates["depends_on"] = body.depends_on
    updates["updated_at"] = datetime.now().isoformat()
    _db_update(task_id, updates)
    updated_data = _db_get(task_id)
    try:
        _build_trigger(updated_data)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    _schedule_task(task_id, updated_data)
    return _task_to_response(updated_data)


@app.delete("/tasks/{task_id}")
def delete_task(task_id: str):
    task_data = _db_get(task_id)
    if not task_data:
        raise HTTPException(status_code=404, detail="任务不存在")
    job_id = f"task_{task_id}"
    job = scheduler.get_job(job_id)
    if job:
        scheduler.remove_job(job_id)
    _db_delete(task_id)
    return {"detail": "任务已删除"}


@app.post("/tasks/{task_id}/toggle", response_model=TaskResponse)
def toggle_task(task_id: str):
    task_data = _db_get(task_id)
    if not task_data:
        raise HTTPException(status_code=404, detail="任务不存在")
    new_enabled = not task_data["enabled"]
    _db_update(task_id, {"enabled": new_enabled, "updated_at": datetime.now().isoformat()})
    updated_data = _db_get(task_id)
    _schedule_task(task_id, updated_data)
    return _task_to_response(updated_data)


@app.post("/tasks/{task_id}/trigger", response_model=dict)
async def trigger_task_now(task_id: str):
    task_data = _db_get(task_id)
    if not task_data:
        raise HTTPException(status_code=404, detail="任务不存在")
    asyncio.ensure_future(_execute_callback(task_id, trigger_type="manual"))
    return {"detail": f"任务 {task_data['name']} 已手动触发"}


@app.get("/tasks/{task_id}/history", response_model=ExecutionHistoryListResponse)
def get_task_history(
    task_id: str,
    status: Optional[ExecutionStatus] = None,
    limit: int = Query(100, ge=1, le=500),
    offset: int = Query(0, ge=0),
):
    if not _db_get(task_id):
        raise HTTPException(status_code=404, detail="任务不存在")
    total, records = _history_list(task_id=task_id, status=status, limit=limit, offset=offset)
    return ExecutionHistoryListResponse(
        total=total,
        records=[_history_to_record(r) for r in records],
    )


@app.get("/execution-history", response_model=ExecutionHistoryListResponse)
def list_execution_history(
    task_id: Optional[str] = None,
    status: Optional[ExecutionStatus] = None,
    limit: int = Query(100, ge=1, le=500),
    offset: int = Query(0, ge=0),
):
    if task_id and not _db_get(task_id):
        raise HTTPException(status_code=404, detail="任务不存在")
    total, records = _history_list(task_id=task_id, status=status, limit=limit, offset=offset)
    return ExecutionHistoryListResponse(
        total=total,
        records=[_history_to_record(r) for r in records],
    )


@app.on_event("shutdown")
def shutdown_event():
    scheduler.shutdown(wait=False)
