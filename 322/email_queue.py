import uuid
import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Union, Any
from queue import Queue

from email_sender import EmailSender, SendResult
from config import EmailConfig

logger = logging.getLogger(__name__)


class TaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class EmailTask:
    task_id: str
    status: TaskStatus = TaskStatus.PENDING
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    result: Optional[SendResult] = None
    error: Optional[str] = None

    @property
    def duration(self) -> Optional[float]:
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        return None


@dataclass
class QueueStats:
    total_tasks: int = 0
    pending: int = 0
    running: int = 0
    completed: int = 0
    failed: int = 0


class EmailQueue:
    def __init__(
        self,
        config: EmailConfig,
        batch_size: int = 20,
        batch_delay: float = 1.0,
        max_workers: int = 2,
    ):
        self._sender = EmailSender(config, batch_size=batch_size, batch_delay=batch_delay)
        self._max_workers = max_workers
        self._task_queue: Queue = Queue()
        self._tasks: Dict[str, EmailTask] = {}
        self._lock = threading.Lock()
        self._workers: List[threading.Thread] = []
        self._running = False

    def start(self) -> None:
        if self._running:
            logger.warning("Email queue is already running")
            return
        self._running = True
        for i in range(self._max_workers):
            worker = threading.Thread(
                target=self._worker_loop,
                name=f"email-worker-{i}",
                daemon=True,
            )
            worker.start()
            self._workers.append(worker)
        logger.info(f"Email queue started with {self._max_workers} workers")

    def stop(self) -> None:
        self._running = False
        for _ in self._workers:
            self._task_queue.put(None)
        for worker in self._workers:
            worker.join(timeout=10)
        self._workers.clear()
        logger.info("Email queue stopped")

    def submit(
        self,
        to: Union[str, List[str]],
        subject: str,
        body: str,
        is_html: bool = False,
        cc: Optional[Union[str, List[str]]] = None,
        bcc: Optional[Union[str, List[str]]] = None,
        reply_to: Optional[str] = None,
        attachments: Optional[List[str]] = None,
        inline_attachments: Optional[Dict[str, bytes]] = None,
    ) -> str:
        task_id = str(uuid.uuid4())

        task = EmailTask(
            task_id=task_id,
            status=TaskStatus.PENDING,
            created_at=datetime.now(),
        )

        with self._lock:
            self._tasks[task_id] = task

        self._task_queue.put({
            "task_id": task_id,
            "to": to,
            "subject": subject,
            "body": body,
            "is_html": is_html,
            "cc": cc,
            "bcc": bcc,
            "reply_to": reply_to,
            "attachments": attachments,
            "inline_attachments": inline_attachments,
        })

        logger.info(f"Task {task_id} submitted to queue")
        return task_id

    def get_status(self, task_id: str) -> Optional[EmailTask]:
        with self._lock:
            return self._tasks.get(task_id)

    def get_result(self, task_id: str) -> Optional[SendResult]:
        with self._lock:
            task = self._tasks.get(task_id)
            if task:
                return task.result
        return None

    def get_stats(self) -> QueueStats:
        with self._lock:
            stats = QueueStats(total_tasks=len(self._tasks))
            for task in self._tasks.values():
                if task.status == TaskStatus.PENDING:
                    stats.pending += 1
                elif task.status == TaskStatus.RUNNING:
                    stats.running += 1
                elif task.status == TaskStatus.COMPLETED:
                    stats.completed += 1
                elif task.status == TaskStatus.FAILED:
                    stats.failed += 1
        return stats

    def list_tasks(self, status: Optional[TaskStatus] = None) -> List[EmailTask]:
        with self._lock:
            tasks = list(self._tasks.values())
        if status:
            tasks = [t for t in tasks if t.status == status]
        return sorted(tasks, key=lambda t: t.created_at)

    def clear_completed(self) -> int:
        with self._lock:
            to_remove = [
                tid for tid, task in self._tasks.items()
                if task.status in (TaskStatus.COMPLETED, TaskStatus.FAILED)
            ]
            for tid in to_remove:
                del self._tasks[tid]
        logger.info(f"Cleared {len(to_remove)} completed/failed tasks")
        return len(to_remove)

    def _worker_loop(self) -> None:
        thread_name = threading.current_thread().name
        logger.debug(f"Worker {thread_name} started")

        while self._running:
            try:
                item = self._task_queue.get(timeout=1.0)
            except Exception:
                continue

            if item is None:
                break

            task_id = item["task_id"]
            self._execute_task(task_id, item)

        logger.debug(f"Worker {thread_name} stopped")

    def _execute_task(self, task_id: str, payload: dict) -> None:
        with self._lock:
            task = self._tasks.get(task_id)
            if not task:
                logger.error(f"Task {task_id} not found")
                return
            task.status = TaskStatus.RUNNING
            task.started_at = datetime.now()

        logger.info(f"Task {task_id}: Starting execution")

        try:
            result = self._sender.send(
                to=payload["to"],
                subject=payload["subject"],
                body=payload["body"],
                is_html=payload.get("is_html", False),
                cc=payload.get("cc"),
                bcc=payload.get("bcc"),
                reply_to=payload.get("reply_to"),
                attachments=payload.get("attachments"),
                inline_attachments=payload.get("inline_attachments"),
            )

            with self._lock:
                task.result = result
                task.completed_at = datetime.now()
                if result.success:
                    task.status = TaskStatus.COMPLETED
                    logger.info(f"Task {task_id}: Completed successfully ({task.duration:.2f}s)")
                else:
                    task.status = TaskStatus.FAILED
                    task.error = result.message
                    logger.warning(f"Task {task_id}: Failed - {result.message}")

        except Exception as e:
            with self._lock:
                task.status = TaskStatus.FAILED
                task.error = str(e)
                task.completed_at = datetime.now()
            logger.exception(f"Task {task_id}: Unexpected error")

    def __enter__(self) -> "EmailQueue":
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.stop()
