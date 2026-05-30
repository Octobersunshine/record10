from __future__ import annotations

import json
import threading
import uuid
from datetime import datetime
from typing import Optional

from sqlalchemy import Column, DateTime, Integer, String, Text, create_engine, desc
from sqlalchemy.orm import declarative_base, sessionmaker

from scheduler.models import ExecutionHistory, ExecutionStatus, HttpCallback, Task, TaskMode, TaskStatus, TaskStats

_STORE_LOCK = threading.Lock()
Base = declarative_base()


class TaskDAO(Base):
    __tablename__ = "tasks"

    id = Column(String(32), primary_key=True)
    name = Column(String(255), nullable=False)
    mode = Column(String(32), nullable=False)
    cron_expr = Column(String(64), nullable=True)
    interval_seconds = Column(Integer, nullable=True)
    callback_url = Column(String(1024), nullable=False)
    callback_method = Column(String(16), nullable=False)
    callback_headers = Column(Text, nullable=False)
    callback_body = Column(Text, nullable=True)
    callback_timeout = Column(Integer, nullable=False)
    status = Column(String(32), nullable=False)
    depends_on = Column(String(32), nullable=True)
    created_at = Column(DateTime, nullable=False)
    updated_at = Column(DateTime, nullable=False)
    last_run_at = Column(DateTime, nullable=True)
    next_run_at = Column(DateTime, nullable=True)
    run_count = Column(Integer, nullable=False, default=0)
    success_count = Column(Integer, nullable=False, default=0)
    failed_count = Column(Integer, nullable=False, default=0)
    total_duration_ms = Column(Integer, nullable=False, default=0)


class ExecutionHistoryDAO(Base):
    __tablename__ = "execution_history"

    id = Column(String(32), primary_key=True)
    task_id = Column(String(32), nullable=False, index=True)
    task_name = Column(String(255), nullable=False)
    status = Column(String(32), nullable=False)
    retry_attempt = Column(Integer, nullable=False, default=0)
    duration_ms = Column(Integer, nullable=False, default=0)
    error_message = Column(Text, nullable=True)
    started_at = Column(DateTime, nullable=False)
    ended_at = Column(DateTime, nullable=False)


def _dao_to_task(dao: TaskDAO) -> Task:
    return Task(
        id=dao.id,
        name=dao.name,
        mode=TaskMode(dao.mode),
        cron_expr=dao.cron_expr,
        interval_seconds=dao.interval_seconds,
        callback=HttpCallback(
            url=dao.callback_url,
            method=dao.callback_method,
            headers=json.loads(dao.callback_headers) if dao.callback_headers else {},
            body=dao.callback_body,
            timeout=dao.callback_timeout,
        ),
        status=TaskStatus(dao.status),
        depends_on=dao.depends_on,
        created_at=dao.created_at,
        updated_at=dao.updated_at,
        last_run_at=dao.last_run_at,
        next_run_at=dao.next_run_at,
        run_count=dao.run_count,
        success_count=dao.success_count,
        failed_count=dao.failed_count,
        total_duration_ms=dao.total_duration_ms,
    )


def _task_to_dao(task: Task) -> TaskDAO:
    return TaskDAO(
        id=task.id,
        name=task.name,
        mode=task.mode.value,
        cron_expr=task.cron_expr,
        interval_seconds=task.interval_seconds,
        callback_url=task.callback.url,
        callback_method=task.callback.method.value,
        callback_headers=json.dumps(task.callback.headers, ensure_ascii=False),
        callback_body=task.callback.body,
        callback_timeout=task.callback.timeout,
        status=task.status.value,
        depends_on=task.depends_on,
        created_at=task.created_at,
        updated_at=task.updated_at,
        last_run_at=task.last_run_at,
        next_run_at=task.next_run_at,
        run_count=task.run_count,
        success_count=task.success_count,
        failed_count=task.failed_count,
        total_duration_ms=task.total_duration_ms,
    )


def _dao_to_history(dao: ExecutionHistoryDAO) -> ExecutionHistory:
    return ExecutionHistory(
        id=dao.id,
        task_id=dao.task_id,
        task_name=dao.task_name,
        status=ExecutionStatus(dao.status),
        retry_attempt=dao.retry_attempt,
        duration_ms=dao.duration_ms,
        error_message=dao.error_message,
        started_at=dao.started_at,
        ended_at=dao.ended_at,
    )


class TaskStore:
    def __init__(self, db_url: str = "sqlite:///tasks.db"):
        self._tasks: dict[str, Task] = {}
        self._engine = create_engine(db_url, connect_args={"check_same_thread": False})
        Base.metadata.create_all(self._engine)
        self._SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self._engine)
        self._load()

    def _load(self):
        session = self._SessionLocal()
        try:
            daos = session.query(TaskDAO).all()
            for dao in daos:
                task = _dao_to_task(dao)
                self._tasks[task.id] = task
            logger = __import__("logging").getLogger(__name__)
            logger.info("Loaded %d tasks from SQLite", len(daos))
        finally:
            session.close()

    def add(self, task: Task) -> Task:
        with _STORE_LOCK:
            if task.id in self._tasks:
                logger = __import__("logging").getLogger(__name__)
                logger.warning("Task %s already exists, skipping duplicate add", task.id)
                return self._tasks[task.id]
            self._tasks[task.id] = task
            session = self._SessionLocal()
            try:
                dao = _task_to_dao(task)
                session.add(dao)
                session.commit()
            except Exception as e:
                session.rollback()
                del self._tasks[task.id]
                raise e
            finally:
                session.close()
        return task

    def get(self, task_id: str) -> Optional[Task]:
        return self._tasks.get(task_id)

    def list_all(self) -> list[Task]:
        return list(self._tasks.values())

    def get_dependent_tasks(self, task_id: str) -> list[Task]:
        return [t for t in self._tasks.values() if t.depends_on == task_id]

    def update(self, task_id: str, **kwargs) -> Optional[Task]:
        with _STORE_LOCK:
            task = self._tasks.get(task_id)
            if not task:
                return None
            update_data = {k: v for k, v in kwargs.items() if v is not None}
            updated = task.model_copy(update=update_data)
            self._tasks[task_id] = updated

            session = self._SessionLocal()
            try:
                dao = session.query(TaskDAO).filter(TaskDAO.id == task_id).first()
                if dao:
                    skip_keys = {"callback", "status", "mode", "depends_on"}
                    for key, value in update_data.items():
                        if key in skip_keys:
                            continue
                        if hasattr(dao, key):
                            setattr(dao, key, value)
                    if "callback" in update_data:
                        cb = update_data["callback"]
                        dao.callback_url = cb.url
                        dao.callback_method = cb.method.value
                        dao.callback_headers = json.dumps(cb.headers, ensure_ascii=False)
                        dao.callback_body = cb.body
                        dao.callback_timeout = cb.timeout
                    if "status" in update_data:
                        dao.status = update_data["status"].value
                    if "mode" in update_data:
                        dao.mode = update_data["mode"].value
                    if "depends_on" in update_data:
                        dao.depends_on = update_data["depends_on"]
                session.commit()
            finally:
                session.close()
        return updated

    def delete(self, task_id: str) -> bool:
        with _STORE_LOCK:
            if task_id in self._tasks:
                del self._tasks[task_id]
                session = self._SessionLocal()
                try:
                    dao = session.query(TaskDAO).filter(TaskDAO.id == task_id).first()
                    if dao:
                        session.delete(dao)
                    session.query(ExecutionHistoryDAO).filter(ExecutionHistoryDAO.task_id == task_id).delete()
                    session.commit()
                finally:
                    session.close()
                return True
        return False

    def update_runtime(self, task_id: str, last_run_at=None, next_run_at=None, run_count=None,
                       success_count=None, failed_count=None, total_duration_ms=None):
        with _STORE_LOCK:
            task = self._tasks.get(task_id)
            if not task:
                return
            updates = {}
            if last_run_at is not None:
                updates["last_run_at"] = last_run_at
            if next_run_at is not None:
                updates["next_run_at"] = next_run_at
            if run_count is not None:
                updates["run_count"] = run_count
            if success_count is not None:
                updates["success_count"] = success_count
            if failed_count is not None:
                updates["failed_count"] = failed_count
            if total_duration_ms is not None:
                updates["total_duration_ms"] = total_duration_ms
            updates["updated_at"] = datetime.utcnow()
            updated = task.model_copy(update=updates)
            self._tasks[task_id] = updated

            session = self._SessionLocal()
            try:
                dao = session.query(TaskDAO).filter(TaskDAO.id == task_id).first()
                if dao:
                    if last_run_at is not None:
                        dao.last_run_at = last_run_at
                    if next_run_at is not None:
                        dao.next_run_at = next_run_at
                    if run_count is not None:
                        dao.run_count = run_count
                    if success_count is not None:
                        dao.success_count = success_count
                    if failed_count is not None:
                        dao.failed_count = failed_count
                    if total_duration_ms is not None:
                        dao.total_duration_ms = total_duration_ms
                    dao.updated_at = updates["updated_at"]
                session.commit()
            finally:
                session.close()

    def add_execution_history(self, task_id: str, task_name: str, status: ExecutionStatus,
                              retry_attempt: int, duration_ms: int, started_at: datetime,
                              ended_at: datetime, error_message: Optional[str] = None):
        with _STORE_LOCK:
            session = self._SessionLocal()
            try:
                dao = ExecutionHistoryDAO(
                    id=uuid.uuid4().hex[:12],
                    task_id=task_id,
                    task_name=task_name,
                    status=status.value,
                    retry_attempt=retry_attempt,
                    duration_ms=duration_ms,
                    error_message=error_message,
                    started_at=started_at,
                    ended_at=ended_at,
                )
                session.add(dao)
                session.commit()
            finally:
                session.close()

    def get_execution_history(self, task_id: Optional[str] = None, limit: int = 100) -> list[ExecutionHistory]:
        session = self._SessionLocal()
        try:
            query = session.query(ExecutionHistoryDAO)
            if task_id:
                query = query.filter(ExecutionHistoryDAO.task_id == task_id)
            daos = query.order_by(desc(ExecutionHistoryDAO.started_at)).limit(limit).all()
            return [_dao_to_history(dao) for dao in daos]
        finally:
            session.close()

    def get_task_stats(self, task_id: str) -> Optional[TaskStats]:
        task = self.get(task_id)
        if not task:
            return None

        session = self._SessionLocal()
        try:
            daos = session.query(ExecutionHistoryDAO).filter(
                ExecutionHistoryDAO.task_id == task_id,
                ExecutionHistoryDAO.status != ExecutionStatus.RETRY.value
            ).all()

            durations = [d.duration_ms for d in daos if d.duration_ms > 0]
            avg_duration = sum(durations) / len(durations) if durations else 0.0
            min_duration = min(durations) if durations else 0
            max_duration = max(durations) if durations else 0

            success_daos = [d for d in daos if d.status == ExecutionStatus.SUCCESS.value]
            failed_daos = [d for d in daos if d.status == ExecutionStatus.FAILED.value]

            last_success = max([d.ended_at for d in success_daos], default=None) if success_daos else None
            last_failed = max([d.ended_at for d in failed_daos], default=None) if failed_daos else None

            return TaskStats(
                task_id=task_id,
                task_name=task.name,
                total_runs=task.run_count,
                success_count=task.success_count,
                failed_count=task.failed_count,
                avg_duration_ms=round(avg_duration, 2),
                min_duration_ms=min_duration,
                max_duration_ms=max_duration,
                last_success_at=last_success,
                last_failed_at=last_failed,
            )
        finally:
            session.close()
