import time
import random
import uuid
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Dict, Any
from enum import Enum


class TaskStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class Task:
    name: str
    func: Callable
    args: tuple = ()
    kwargs: Dict[str, Any] = field(default_factory=dict)
    dependencies: List["Task"] = field(default_factory=list)
    max_retries: int = 3
    retry_backoff: float = 1.0
    task_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    status: TaskStatus = TaskStatus.PENDING
    result: Any = None
    error: Optional[Exception] = None
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    retry_count: int = 0

    def depends_on(self, task: "Task") -> None:
        if task not in self.dependencies:
            self.dependencies.append(task)

    @property
    def duration(self) -> Optional[float]:
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return None


@dataclass
class ExecutionRecord:
    task_id: str
    task_name: str
    status: TaskStatus
    start_time: float
    end_time: float
    duration: float
    retry_count: int
    error: Optional[str] = None
    result: Optional[Any] = None


class TaskScheduler:
    def __init__(self, max_history: int = 100):
        self.tasks: List[Task] = []
        self.history: deque[ExecutionRecord] = deque(maxlen=max_history)
        self.max_history = max_history

    def add_task(self, task: Task) -> None:
        self.tasks.append(task)

    def _check_dependencies(self, task: Task) -> bool:
        for dep in task.dependencies:
            if dep.status != TaskStatus.SUCCESS:
                return False
        return True

    def _execute_with_retry(self, task: Task) -> bool:
        task.start_time = time.time()
        task.status = TaskStatus.RUNNING
        task.retry_count = 0

        while task.retry_count <= task.max_retries:
            try:
                task.result = task.func(*task.args, **task.kwargs)
                task.status = TaskStatus.SUCCESS
                task.end_time = time.time()
                self._add_to_history(task)
                return True
            except Exception as e:
                task.retry_count += 1
                task.error = e

                if task.retry_count > task.max_retries:
                    task.status = TaskStatus.FAILED
                    task.end_time = time.time()
                    self._add_to_history(task)
                    return False

                wait_time = task.retry_backoff * (2 ** (task.retry_count - 1))
                print(f"  任务 [{task.name}] 失败，第 {task.retry_count}/{task.max_retries} 次重试，"
                      f"等待 {wait_time:.2f} 秒...")
                time.sleep(wait_time)

        return False

    def _add_to_history(self, task: Task) -> None:
        record = ExecutionRecord(
            task_id=task.task_id,
            task_name=task.name,
            status=task.status,
            start_time=task.start_time,
            end_time=task.end_time,
            duration=task.duration,
            retry_count=task.retry_count,
            error=str(task.error) if task.error else None,
            result=task.result
        )
        self.history.append(record)

    def _get_executable_tasks(self) -> List[Task]:
        executable = []
        for task in self.tasks:
            if task.status == TaskStatus.PENDING and self._check_dependencies(task):
                executable.append(task)
        return executable

    def run(self) -> None:
        print("=" * 70)
        print("开始任务调度...")
        print("=" * 70)

        while True:
            executable = self._get_executable_tasks()

            if not executable:
                pending = [t for t in self.tasks if t.status == TaskStatus.PENDING]
                if pending:
                    print(f"\n警告: 有 {len(pending)} 个任务无法执行（依赖失败或循环依赖）")
                    for task in pending:
                        task.status = TaskStatus.SKIPPED
                        task.start_time = task.end_time = time.time()
                        self._add_to_history(task)
                break

            for task in executable:
                print(f"\n执行任务: {task.name} (ID: {task.task_id})")
                print(f"  依赖: {[d.name for d in task.dependencies] if task.dependencies else '无'}")

                success = self._execute_with_retry(task)

                if success:
                    print(f"  ✓ 任务 [{task.name}] 成功完成，"
                          f"耗时: {task.duration:.4f} 秒，"
                          f"结果: {task.result}")
                else:
                    print(f"  ✗ 任务 [{task.name}] 失败，"
                          f"耗时: {task.duration:.4f} 秒，"
                          f"错误: {task.error}")

        print("\n" + "=" * 70)
        print("任务调度完成")
        print("=" * 70)

    def get_history(self, limit: Optional[int] = None) -> List[ExecutionRecord]:
        records = list(self.history)
        if limit:
            return records[-limit:]
        return records

    def get_statistics(self) -> Dict[str, Any]:
        if not self.history:
            return {"message": "暂无执行记录"}

        total_tasks = len(self.history)
        success_tasks = sum(1 for r in self.history if r.status == TaskStatus.SUCCESS)
        failed_tasks = sum(1 for r in self.history if r.status == TaskStatus.FAILED)
        skipped_tasks = sum(1 for r in self.history if r.status == TaskStatus.SKIPPED)

        durations = [r.duration for r in self.history if r.duration]
        total_duration = sum(durations)
        avg_duration = total_duration / len(durations) if durations else 0
        max_duration = max(durations) if durations else 0
        min_duration = min(durations) if durations else 0

        total_retries = sum(r.retry_count for r in self.history)

        return {
            "total_tasks": total_tasks,
            "success_count": success_tasks,
            "failed_count": failed_tasks,
            "skipped_count": skipped_tasks,
            "success_rate": (success_tasks / total_tasks * 100) if total_tasks else 0,
            "total_duration": total_duration,
            "avg_duration": avg_duration,
            "max_duration": max_duration,
            "min_duration": min_duration,
            "total_retries": total_retries,
        }

    def print_history(self, limit: Optional[int] = None) -> None:
        records = self.get_history(limit)
        print("\n" + "=" * 70)
        print(f"执行历史记录 (最近 {len(records)} 条)")
        print("=" * 70)
        print(f"{'ID':<10} {'任务名':<15} {'状态':<10} {'耗时(s)':<10} {'重试':<6} {'结果/错误'}")
        print("-" * 70)

        for record in records:
            status_display = {
                TaskStatus.SUCCESS: "成功",
                TaskStatus.FAILED: "失败",
                TaskStatus.SKIPPED: "跳过",
                TaskStatus.RUNNING: "运行中",
                TaskStatus.PENDING: "等待",
            }.get(record.status, str(record.status))

            info = record.result if record.status == TaskStatus.SUCCESS else record.error
            info_str = str(info)[:30] if info else "-"

            print(f"{record.task_id:<10} {record.task_name:<15} {status_display:<10} "
                  f"{record.duration:<10.4f} {record.retry_count:<6} {info_str}")

    def print_statistics(self) -> None:
        stats = self.get_statistics()
        print("\n" + "=" * 70)
        print("执行统计")
        print("=" * 70)

        if "message" in stats:
            print(stats["message"])
            return

        print(f"总任务数:      {stats['total_tasks']}")
        print(f"成功:          {stats['success_count']} ({stats['success_rate']:.1f}%)")
        print(f"失败:          {stats['failed_count']}")
        print(f"跳过:          {stats['skipped_count']}")
        print(f"总重试次数:    {stats['total_retries']}")
        print(f"总耗时:        {stats['total_duration']:.4f} 秒")
        print(f"平均耗时:      {stats['avg_duration']:.4f} 秒")
        print(f"最短耗时:      {stats['min_duration']:.4f} 秒")
        print(f"最长耗时:      {stats['max_duration']:.4f} 秒")


def flaky_task(task_name: str, failure_prob: float = 0.0) -> str:
    time.sleep(random.uniform(0.1, 0.5))
    if random.random() < failure_prob:
        raise RuntimeError(f"任务 {task_name} 随机失败")
    return f"{task_name}_result"


if __name__ == "__main__":
    random.seed(42)
    scheduler = TaskScheduler(max_history=100)

    task_a = Task(
        name="Task_A",
        func=flaky_task,
        args=("Task_A", 0.0),
        max_retries=3,
        retry_backoff=0.5
    )

    task_b = Task(
        name="Task_B",
        func=flaky_task,
        args=("Task_B", 0.5),
        max_retries=3,
        retry_backoff=0.5
    )
    task_b.depends_on(task_a)

    task_c = Task(
        name="Task_C",
        func=flaky_task,
        args=("Task_C", 0.0),
        max_retries=3,
        retry_backoff=0.5
    )
    task_c.depends_on(task_a)

    task_d = Task(
        name="Task_D",
        func=flaky_task,
        args=("Task_D", 0.0),
        max_retries=3,
        retry_backoff=0.5
    )
    task_d.depends_on(task_b)
    task_d.depends_on(task_c)

    task_e = Task(
        name="Task_E",
        func=flaky_task,
        args=("Task_E", 0.9),
        max_retries=2,
        retry_backoff=0.3
    )
    task_e.depends_on(task_d)

    scheduler.add_task(task_a)
    scheduler.add_task(task_b)
    scheduler.add_task(task_c)
    scheduler.add_task(task_d)
    scheduler.add_task(task_e)

    print("任务依赖关系:")
    print("  A → B → D → E")
    print("    ↘ C ↗")
    print()

    scheduler.run()

    scheduler.print_history()
    scheduler.print_statistics()

    print("\n" + "=" * 70)
    print("查询最近 3 条历史记录:")
    print("=" * 70)
    recent = scheduler.get_history(3)
    for r in recent:
        print(f"  [{r.task_id}] {r.task_name}: {r.status.value} (耗时: {r.duration:.4f}s)")
