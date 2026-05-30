from __future__ import annotations

import logging
from datetime import datetime
from typing import Optional

from croniter import croniter
from fastapi import APIRouter, HTTPException, Query

from scheduler.engine import SchedulerEngine
from scheduler.models import (
    ExecutionHistory,
    Task,
    TaskCreateRequest,
    TaskMode,
    TaskStatus,
    TaskStats,
    TaskUpdateRequest,
)
from scheduler.store import TaskStore

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/tasks", tags=["tasks"])

_store: TaskStore | None = None
_engine: SchedulerEngine | None = None


def init_router(store: TaskStore, engine: SchedulerEngine):
    global _store, _engine
    _store = store
    _engine = engine


def _validate_schedule(req: TaskCreateRequest | TaskUpdateRequest):
    if isinstance(req, TaskCreateRequest):
        if req.mode == TaskMode.CRON and not req.cron_expr:
            raise HTTPException(400, "cron_expr is required for cron mode")
        if req.mode == TaskMode.INTERVAL and (
            not req.interval_seconds or req.interval_seconds <= 0
        ):
            raise HTTPException(400, "interval_seconds must be > 0 for interval mode")
        if req.mode == TaskMode.CRON and req.cron_expr:
            try:
                croniter(req.cron_expr)
            except (ValueError, KeyError) as e:
                raise HTTPException(400, f"Invalid cron expression: {e}")
    if isinstance(req, TaskUpdateRequest):
        if req.cron_expr is not None:
            try:
                croniter(req.cron_expr)
            except (ValueError, KeyError) as e:
                    raise HTTPException(400, f"Invalid cron expression: {e}")
        if req.interval_seconds is not None and req.interval_seconds <= 0:
            raise HTTPException(400, "interval_seconds must be > 0")


def _validate_dependency(depends_on: Optional[str]):
    if depends_on and not _store.get(depends_on):
        raise HTTPException(400, f"Depends_on task '{depends_on}' not found")


@router.post("", response_model=Task, status_code=201)
async def create_task(req: TaskCreateRequest):
    _validate_schedule(req)
    _validate_dependency(req.depends_on)
    task = Task(
        name=req.name,
        mode=req.mode,
        cron_expr=req.cron_expr,
        interval_seconds=req.interval_seconds,
        callback=req.callback,
        status=req.status,
        depends_on=req.depends_on,
    )
    _store.add(task)
    if task.status == TaskStatus.ACTIVE and not task.depends_on:
        _engine.register_task(task)
    logger.info("Created task %s: %s", task.id, task.name)
    return _store.get(task.id)


@router.get("", response_model=list[Task])
async def list_tasks():
    return _store.list_all()


@router.get("/{task_id}", response_model=Task)
async def get_task(task_id: str):
    task = _store.get(task_id)
    if not task:
        raise HTTPException(404, "Task not found")
    return task


@router.put("/{task_id}", response_model=Task)
async def update_task(task_id: str, req: TaskUpdateRequest):
    _validate_schedule(req)
    existing = _store.get(task_id)
    if not existing:
        raise HTTPException(404, "Task not found")

    if req.depends_on is not None:
        _validate_dependency(req.depends_on)

    update_kwargs = {}
    if req.name is not None:
        update_kwargs["name"] = req.name
    if req.cron_expr is not None:
        update_kwargs["cron_expr"] = req.cron_expr
    if req.interval_seconds is not None:
        update_kwargs["interval_seconds"] = req.interval_seconds
    if req.callback is not None:
        update_kwargs["callback"] = req.callback
    if req.status is not None:
        update_kwargs["status"] = req.status
    if req.depends_on is not None:
        update_kwargs["depends_on"] = req.depends_on
    update_kwargs["updated_at"] = datetime.utcnow()

    if req.mode is not None:
        update_kwargs["mode"] = req.mode

    updated = _store.update(task_id, **update_kwargs)
    if updated:
        _engine.reschedule_task(updated)
    return _store.get(task_id)


@router.delete("/{task_id}", status_code=204)
async def delete_task(task_id: str):
    existing = _store.get(task_id)
    if not existing:
        raise HTTPException(404, "Task not found")
    _engine.unregister_task(task_id)
    _store.delete(task_id)
    logger.info("Deleted task %s", task_id)


@router.get("/{task_id}/history", response_model=list[ExecutionHistory])
async def get_task_history(task_id: str, limit: int = Query(100, ge=1, le=1000)):
    task = _store.get(task_id)
    if not task:
        raise HTTPException(404, "Task not found")
    return _store.get_execution_history(task_id=task_id, limit=limit)


@router.get("/{task_id}/stats", response_model=TaskStats)
async def get_task_stats(task_id: str):
    stats = _store.get_task_stats(task_id)
    if not stats:
        raise HTTPException(404, "Task not found")
    return stats


@router.get("/history/all", response_model=list[ExecutionHistory])
async def get_all_history(limit: int = Query(100, ge=1, le=1000)):
    return _store.get_execution_history(limit=limit)
