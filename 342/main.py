import asyncio
import json
import logging
import uuid
from enum import Enum
from typing import Dict, Optional, Set
from datetime import datetime, timedelta

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel

logger = logging.getLogger("task_progress")
logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)

app = FastAPI(title="Long Task Progress API")


class TaskStatus(str, Enum):
    PENDING = "等待中"
    RUNNING = "执行中"
    COMPLETED = "已完成"
    FAILED = "失败"


class Task(BaseModel):
    task_id: str
    status: TaskStatus
    progress: int
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None

    def calculate_eta(self) -> Optional[float]:
        if self.status != TaskStatus.RUNNING or not self.started_at or self.progress <= 0:
            return None
        elapsed = (datetime.now() - self.started_at).total_seconds()
        estimated_total = elapsed * 100 / self.progress
        remaining = estimated_total - elapsed
        return max(0, remaining)


class TaskResponse(BaseModel):
    task_id: str


class TaskProgressResponse(BaseModel):
    task_id: str
    status: TaskStatus
    progress: int
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    eta_seconds: Optional[float] = None


class ConnectionManager:
    def __init__(self):
        self.active_connections: Set[WebSocket] = set()

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.add(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.discard(websocket)

    async def broadcast(self, message: dict):
        for connection in list(self.active_connections):
            try:
                await connection.send_json(message)
            except Exception:
                self.disconnect(connection)


manager = ConnectionManager()
tasks: Dict[str, Task] = {}
task_locks: Dict[str, asyncio.Lock] = {}


def clamp_progress(task_id: str, value: int) -> int:
    if value < 0:
        logger.warning("任务 %s 进度值 %d 小于0，强制截断为0", task_id, value)
        return 0
    if value > 100:
        logger.warning("任务 %s 进度值 %d 超过100，强制截断为100", task_id, value)
        return 100
    return value


async def notify_progress(task: Task):
    message = {
        "type": "task_update",
        "task_id": task.task_id,
        "status": task.status,
        "progress": task.progress,
        "eta_seconds": task.calculate_eta()
    }
    await manager.broadcast(message)


async def simulate_task(task_id: str, duration_seconds: int = 10):
    await asyncio.sleep(1)

    async with task_locks[task_id]:
        if task_id not in tasks:
            return
        task = tasks[task_id]
        task.status = TaskStatus.RUNNING
        task.started_at = datetime.now()
        await notify_progress(task)

    total_steps = 100
    delay_per_step = duration_seconds / total_steps

    for step in range(1, total_steps + 1):
        await asyncio.sleep(delay_per_step)

        async with task_locks[task_id]:
            if task_id not in tasks:
                return
            task = tasks[task_id]
            if task.status == TaskStatus.FAILED:
                return
            task.progress = clamp_progress(task_id, step)
            await notify_progress(task)

    await asyncio.sleep(0.5)

    async with task_locks[task_id]:
        if task_id not in tasks:
            return
        task = tasks[task_id]
        task.status = TaskStatus.COMPLETED
        task.completed_at = datetime.now()
        await notify_progress(task)


@app.post("/tasks", response_model=TaskResponse, summary="创建新任务")
async def create_task():
    task_id = str(uuid.uuid4())

    task = Task(
        task_id=task_id,
        status=TaskStatus.PENDING,
        progress=0,
        created_at=datetime.now()
    )

    tasks[task_id] = task
    task_locks[task_id] = asyncio.Lock()

    asyncio.create_task(notify_progress(task))
    asyncio.create_task(simulate_task(task_id))

    return TaskResponse(task_id=task_id)


@app.get("/tasks/{task_id}", response_model=TaskProgressResponse, summary="查询任务进度")
async def get_task_progress(task_id: str):
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="任务不存在")

    task = tasks[task_id]
    return TaskProgressResponse(
        task_id=task.task_id,
        status=task.status,
        progress=task.progress,
        created_at=task.created_at,
        started_at=task.started_at,
        completed_at=task.completed_at,
        error_message=task.error_message,
        eta_seconds=task.calculate_eta()
    )


@app.get("/tasks", summary="获取所有任务列表")
async def list_tasks():
    return [
        TaskProgressResponse(
            task_id=task.task_id,
            status=task.status,
            progress=task.progress,
            created_at=task.created_at,
            started_at=task.started_at,
            completed_at=task.completed_at,
            error_message=task.error_message,
            eta_seconds=task.calculate_eta()
        )
        for task in tasks.values()
    ]


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        for task in tasks.values():
            await websocket.send_json({
                "type": "task_update",
                "task_id": task.task_id,
                "status": task.status,
                "progress": task.progress,
                "eta_seconds": task.calculate_eta()
            })

        while True:
            data = await websocket.receive_text()
            try:
                msg = json.loads(data)
                if msg.get("action") == "create_task":
                    task_id = str(uuid.uuid4())
                    task = Task(
                        task_id=task_id,
                        status=TaskStatus.PENDING,
                        progress=0,
                        created_at=datetime.now()
                    )
                    tasks[task_id] = task
                    task_locks[task_id] = asyncio.Lock()
                    await notify_progress(task)
                    asyncio.create_task(simulate_task(task_id))
                    await websocket.send_json({
                        "type": "task_created",
                        "task_id": task_id
                    })
            except json.JSONDecodeError:
                pass
    except WebSocketDisconnect:
        manager.disconnect(websocket)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
