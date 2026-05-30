from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta

from croniter import croniter

from scheduler.models import ExecutionStatus, Task, TaskMode, TaskStatus
from scheduler.store import TaskStore

logger = logging.getLogger(__name__)

MAX_RETRY_ATTEMPTS = 3
RETRY_BASE_DELAY_SECONDS = 1


class SchedulerEngine:
    def __init__(self, store: TaskStore, callback_fn=None):
        self._store = store
        self._callback_fn = callback_fn
        self._running = False
        self._task_handles: dict[str, asyncio.Task] = {}
        self._loop: asyncio.AbstractEventLoop | None = None

    def set_callback(self, fn):
        self._callback_fn = fn

    async def start(self):
        self._running = True
        self._loop = asyncio.get_running_loop()
        logger.info("Scheduler engine started")
        for task in self._store.list_all():
            if task.status == TaskStatus.ACTIVE:
                self._schedule_task(task)

    async def stop(self):
        self._running = False
        for task_id, handle in list(self._task_handles.items()):
            handle.cancel()
        self._task_handles.clear()
        logger.info("Scheduler engine stopped")

    def register_task(self, task: Task):
        if task.id in self._task_handles:
            logger.info("Task %s already registered, skipping duplicate registration", task.id)
            return
        if task.status == TaskStatus.ACTIVE:
            self._schedule_task(task)

    def unregister_task(self, task_id: str):
        handle = self._task_handles.pop(task_id, None)
        if handle and not handle.done():
            handle.cancel()

    def reschedule_task(self, task: Task):
        self.unregister_task(task.id)
        if task.status == TaskStatus.ACTIVE:
            self._schedule_task(task)

    def _compute_next_run(self, task: Task) -> datetime | None:
        now = datetime.utcnow()
        if task.mode == TaskMode.CRON:
            if not task.cron_expr:
                return None
            cron = croniter(task.cron_expr, now)
            return cron.get_next(datetime)
        elif task.mode == TaskMode.INTERVAL:
            if not task.interval_seconds or task.interval_seconds <= 0:
                return None
            base = task.last_run_at or now
            next_run = base + timedelta(seconds=task.interval_seconds)
            if next_run < now:
                logger.info("Task %s missed scheduled run during downtime, rescheduling", task.id)
                next_run = now + timedelta(seconds=task.interval_seconds)
            return next_run
        return None

    def _schedule_task(self, task: Task):
        if task.id in self._task_handles:
            return
        next_run = self._compute_next_run(task)
        if next_run is None:
            logger.warning("Task %s has no valid schedule, skipping", task.id)
            return
        self._store.update_runtime(task.id, next_run_at=next_run)
        delay = max((next_run - datetime.utcnow()).total_seconds(), 0.1)
        handle = asyncio.ensure_future(self._run_after(task.id, delay))
        self._task_handles[task.id] = handle
        logger.info("Scheduled task %s (%s) to run at %s (in %.1fs)", task.id, task.name, next_run, delay)

    async def _run_after(self, task_id: str, delay: float):
        try:
            await asyncio.sleep(delay)
        except asyncio.CancelledError:
            return
        if not self._running:
            return
        task = self._store.get(task_id)
        if not task or task.status != TaskStatus.ACTIVE:
            return
        await self._execute_task(task)

    async def _execute_task(self, task: Task):
        logger.info("Executing task %s (%s)", task.id, task.name)
        started_at = datetime.utcnow()
        self._store.update_runtime(
            task.id,
            last_run_at=started_at,
            run_count=task.run_count + 1,
        )

        success = False
        error_msg = None
        retry_attempt = 0

        for attempt in range(MAX_RETRY_ATTEMPTS):
            retry_attempt = attempt
            try:
                if self._callback_fn:
                    await self._callback_fn(task)
                success = True
                break
            except Exception as e:
                error_msg = str(e)
                logger.warning("Task %s failed (attempt %d/%d): %s", task.id, attempt + 1, MAX_RETRY_ATTEMPTS, e)

                self._store.add_execution_history(
                    task_id=task.id,
                    task_name=task.name,
                    status=ExecutionStatus.RETRY,
                    retry_attempt=attempt,
                    duration_ms=0,
                    started_at=started_at,
                    ended_at=datetime.utcnow(),
                    error_message=error_msg,
                )

                if attempt < MAX_RETRY_ATTEMPTS - 1:
                    delay = RETRY_BASE_DELAY_SECONDS * (2 ** attempt)
                    logger.info("Task %s will retry in %d seconds", task.id, delay)
                    await asyncio.sleep(delay)
                else:
                    logger.error("Task %s failed after %d attempts", task.id, MAX_RETRY_ATTEMPTS)

        ended_at = datetime.utcnow()
        duration_ms = int((ended_at - started_at).total_seconds() * 1000)

        task = self._store.get(task.id)
        if not task:
            return

        if success:
            final_status = ExecutionStatus.SUCCESS
            new_success_count = task.success_count + 1
            new_failed_count = task.failed_count
            new_total_duration = task.total_duration_ms + duration_ms

            self._trigger_dependent_tasks(task.id)
        else:
            final_status = ExecutionStatus.FAILED
            new_success_count = task.success_count
            new_failed_count = task.failed_count + 1
            new_total_duration = task.total_duration_ms + duration_ms

        self._store.add_execution_history(
            task_id=task.id,
            task_name=task.name,
            status=final_status,
            retry_attempt=retry_attempt,
            duration_ms=duration_ms,
            started_at=started_at,
            ended_at=ended_at,
            error_message=error_msg,
        )

        self._store.update_runtime(
            task.id,
            success_count=new_success_count,
            failed_count=new_failed_count,
            total_duration_ms=new_total_duration,
        )

        refreshed = self._store.get(task.id)
        if refreshed and refreshed.status == TaskStatus.ACTIVE and refreshed.depends_on is None:
            self._schedule_task(refreshed)

    def _trigger_dependent_tasks(self, completed_task_id: str):
        dependents = self._store.get_dependent_tasks(completed_task_id)
        for dep_task in dependents:
            if dep_task.status == TaskStatus.ACTIVE:
                logger.info("Triggering dependent task %s after %s completed", dep_task.id, completed_task_id)
                asyncio.ensure_future(self._execute_task(dep_task))
