import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class MockConnection:
    conn_id: int
    created_at: float = field(default_factory=time.time)
    is_alive: bool = True

    def execute(self, query: str) -> str:
        if not self.is_alive:
            raise RuntimeError(f"Connection {self.conn_id} is dead")
        return f"Connection {self.conn_id} executed: {query}"

    def validate(self) -> bool:
        return self.is_alive

    def close(self) -> None:
        self.is_alive = False

    def __repr__(self) -> str:
        state = "alive" if self.is_alive else "dead"
        return f"MockConnection(id={self.conn_id}, {state})"


class PooledConnection:
    def __init__(self, conn: MockConnection, pool: "DatabaseConnectionPool"):
        self._conn = conn
        self._pool = pool
        self._released = False

    @property
    def conn_id(self) -> int:
        return self._conn.conn_id

    def execute(self, query: str) -> str:
        return self._conn.execute(query)

    def release(self) -> None:
        if not self._released:
            self._pool.release(self._conn)
            self._released = True

    def __enter__(self) -> "PooledConnection":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.release()

    def __repr__(self) -> str:
        return f"PooledConnection({self._conn})"


@dataclass
class PoolStats:
    created_at: float
    total_acquires: int = 0
    total_releases: int = 0
    total_wait_time: float = 0.0
    max_wait_time: float = 0.0
    wait_times: deque = field(default_factory=lambda: deque(maxlen=1000))
    connection_busy_time: dict[int, float] = field(default_factory=dict)
    scale_up_events: int = 0
    scale_down_events: int = 0

    def reset(self) -> None:
        self.total_acquires = 0
        self.total_releases = 0
        self.total_wait_time = 0.0
        self.max_wait_time = 0.0
        self.wait_times.clear()
        self.connection_busy_time.clear()
        self.scale_up_events = 0
        self.scale_down_events = 0


class DatabaseConnectionPool:
    def __init__(
        self,
        max_connections: int,
        min_connections: int = 0,
        max_usage_time: Optional[float] = None,
        health_check: bool = True,
        recycle_check_interval: float = 1.0,
        lazy_init: bool = False,
        scale_up_threshold: int = 2,
        scale_down_threshold: int = 2,
        scale_down_idle_time: float = 30.0,
        scale_interval: float = 5.0,
        auto_scale: bool = False,
    ):
        if max_connections <= 0:
            raise ValueError("max_connections must be greater than 0")
        if min_connections < 0 or min_connections > max_connections:
            raise ValueError(
                "min_connections must be between 0 and max_connections"
            )

        self.max_connections = max_connections
        self.min_connections = min_connections
        self.max_usage_time = max_usage_time
        self.health_check = health_check
        self.recycle_check_interval = recycle_check_interval
        self.lazy_init = lazy_init
        self.scale_up_threshold = scale_up_threshold
        self.scale_down_threshold = scale_down_threshold
        self.scale_down_idle_time = scale_down_idle_time
        self.scale_interval = scale_interval
        self.auto_scale = auto_scale

        self._lock = threading.Lock()
        self._condition = threading.Condition(self._lock)
        self._idle_connections: list[MockConnection] = []
        self._active_connections: list[MockConnection] = []
        self._acquire_times: dict[int, float] = {}
        self._idle_since: dict[int, float] = {}
        self._next_conn_id = 1
        self._shutdown = False
        self._waiting_count = 0
        self._stats = PoolStats(created_at=time.time())

        if not self.lazy_init:
            for _ in range(min_connections):
                self._idle_connections.append(self._create_connection())
                self._idle_since[self._idle_connections[-1].conn_id] = time.time()

        if self.max_usage_time is not None:
            self._recycler_thread = threading.Thread(
                target=self._recycle_loop, daemon=True
            )
            self._recycler_thread.start()

        if self.auto_scale:
            self._scaler_thread = threading.Thread(
                target=self._auto_scale_loop, daemon=True
            )
            self._scaler_thread.start()

    def _create_connection(self) -> MockConnection:
        conn = MockConnection(conn_id=self._next_conn_id)
        self._next_conn_id += 1
        return conn

    def _validate_and_get(self, conn: MockConnection) -> MockConnection:
        if not self.health_check or conn.validate():
            return conn
        new_conn = self._create_connection()
        return new_conn

    def acquire(self, timeout: Optional[float] = None) -> PooledConnection:
        with self._condition:
            if self._shutdown:
                raise RuntimeError("Connection pool is shut down")

            self._waiting_count += 1
            wait_start = time.time()

            try:
                start_time = time.time()
                while not self._idle_connections:
                    total_current = (
                        len(self._active_connections) + len(self._idle_connections)
                    )
                    if not self.auto_scale and total_current < self.max_connections:
                        conn = self._create_connection()
                        self._active_connections.append(conn)
                        self._acquire_times[conn.conn_id] = time.time()
                        wait_time = time.time() - wait_start
                        self._record_wait_time(wait_time)
                        self._stats.total_acquires += 1
                        return PooledConnection(conn, self)

                    remaining = None
                    if timeout is not None:
                        elapsed = time.time() - start_time
                        remaining = timeout - elapsed
                        if remaining <= 0:
                            raise TimeoutError("Timeout waiting for connection")

                    self._condition.wait(timeout=remaining)

                wait_time = time.time() - wait_start
                self._record_wait_time(wait_time)

                conn = self._idle_connections.pop()
                self._idle_since.pop(conn.conn_id, None)
                conn = self._validate_and_get(conn)
                self._active_connections.append(conn)
                self._acquire_times[conn.conn_id] = time.time()
                self._stats.total_acquires += 1
                return PooledConnection(conn, self)
            finally:
                self._waiting_count -= 1

    def release(self, conn: MockConnection) -> None:
        with self._condition:
            if conn not in self._active_connections:
                raise ValueError(f"Connection {conn.conn_id} not in active pool")

            acquire_time = self._acquire_times.pop(conn.conn_id, None)
            if acquire_time is not None:
                busy_duration = time.time() - acquire_time
                self._stats.connection_busy_time[conn.conn_id] = (
                    self._stats.connection_busy_time.get(conn.conn_id, 0.0)
                    + busy_duration
                )

            self._active_connections.remove(conn)
            self._stats.total_releases += 1

            if self.health_check and not conn.validate():
                new_conn = self._create_connection()
                self._idle_connections.append(new_conn)
                self._idle_since[new_conn.conn_id] = time.time()
            else:
                self._idle_connections.append(conn)
                self._idle_since[conn.conn_id] = time.time()

            self._condition.notify()

    def connection(self, timeout: Optional[float] = None) -> PooledConnection:
        return self.acquire(timeout=timeout)

    def warm_up(self, count: int) -> int:
        if count <= 0:
            return 0
        with self._condition:
            total_current = (
                len(self._active_connections) + len(self._idle_connections)
            )
            can_create = min(count, self.max_connections - total_current)
            for _ in range(can_create):
                conn = self._create_connection()
                self._idle_connections.append(conn)
                self._idle_since[conn.conn_id] = time.time()
            self._condition.notify(can_create)
            return can_create

    def _record_wait_time(self, wait_time: float) -> None:
        self._stats.total_wait_time += wait_time
        self._stats.wait_times.append(wait_time)
        if wait_time > self._stats.max_wait_time:
            self._stats.max_wait_time = wait_time

    def _recycle_loop(self) -> None:
        while not self._shutdown:
            time.sleep(self.recycle_check_interval)
            self._recycle_expired_connections()

    def _recycle_expired_connections(self) -> None:
        with self._condition:
            now = time.time()
            expired = [
                conn
                for conn in self._active_connections
                if (now - self._acquire_times.get(conn.conn_id, now))
                >= self.max_usage_time
            ]
            for conn in expired:
                self._active_connections.remove(conn)
                self._acquire_times.pop(conn.conn_id, None)
                if self.health_check and not conn.validate():
                    new_conn = self._create_connection()
                    self._idle_connections.append(new_conn)
                    self._idle_since[new_conn.conn_id] = now
                else:
                    self._idle_connections.append(conn)
                    self._idle_since[conn.conn_id] = now

            if expired:
                self._condition.notify(len(expired))

    def _auto_scale_loop(self) -> None:
        while not self._shutdown:
            time.sleep(self.scale_interval)
            self._auto_scale()

    def _auto_scale(self) -> None:
        with self._condition:
            total_current = (
                len(self._active_connections) + len(self._idle_connections)
            )
            now = time.time()

            if self._waiting_count >= self.scale_up_threshold:
                if total_current < self.max_connections:
                    new_conn = self._create_connection()
                    self._idle_connections.append(new_conn)
                    self._idle_since[new_conn.conn_id] = now
                    self._stats.scale_up_events += 1
                    self._condition.notify()
                    return

            if (
                len(self._idle_connections) > self.scale_down_threshold
                and total_current > self.min_connections
            ):
                idle_to_remove = [
                    conn
                    for conn in self._idle_connections
                    if (now - self._idle_since.get(conn.conn_id, now))
                    >= self.scale_down_idle_time
                ]
                if idle_to_remove:
                    conn = idle_to_remove[0]
                    self._idle_connections.remove(conn)
                    self._idle_since.pop(conn.conn_id, None)
                    self._stats.scale_down_events += 1
                    return

    def scale_up(self, count: int = 1) -> int:
        if count <= 0:
            return 0
        with self._condition:
            total_current = (
                len(self._active_connections) + len(self._idle_connections)
            )
            can_create = min(count, self.max_connections - total_current)
            now = time.time()
            for _ in range(can_create):
                conn = self._create_connection()
                self._idle_connections.append(conn)
                self._idle_since[conn.conn_id] = now
            self._condition.notify(can_create)
            return can_create

    def scale_down(self, count: int = 1) -> int:
        if count <= 0:
            return 0
        with self._condition:
            total_current = (
                len(self._active_connections) + len(self._idle_connections)
            )
            can_remove = min(
                count,
                len(self._idle_connections),
                total_current - self.min_connections,
            )
            for _ in range(can_remove):
                conn = self._idle_connections.pop()
                self._idle_since.pop(conn.conn_id, None)
            return can_remove

    def shutdown(self) -> None:
        with self._condition:
            self._shutdown = True
            self._condition.notify_all()

    def status(self) -> dict:
        with self._lock:
            now = time.time()
            active_durations = {
                conn.conn_id: now - self._acquire_times.get(conn.conn_id, now)
                for conn in self._active_connections
            }
            idle_durations = {
                conn.conn_id: now - self._idle_since.get(conn.conn_id, now)
                for conn in self._idle_connections
            }
            total = len(self._active_connections) + len(self._idle_connections)
            return {
                "max_connections": self.max_connections,
                "min_connections": self.min_connections,
                "active_connections": len(self._active_connections),
                "idle_connections": len(self._idle_connections),
                "total_connections": total,
                "waiting_count": self._waiting_count,
                "active_durations": active_durations,
                "idle_durations": idle_durations,
            }

    def stats(self) -> dict:
        with self._lock:
            now = time.time()
            uptime = now - self._stats.created_at
            total = len(self._active_connections) + len(self._idle_connections)
            total_busy = sum(self._stats.connection_busy_time.values(), 0.0)
            total_capacity = total * uptime if total > 0 else 1.0
            utilization = (total_busy / total_capacity) * 100 if total_capacity > 0 else 0.0

            avg_wait = (
                self._stats.total_wait_time / self._stats.total_acquires
                if self._stats.total_acquires > 0
                else 0.0
            )
            recent_wait_times = list(self._stats.wait_times)
            recent_avg = (
                sum(recent_wait_times) / len(recent_wait_times)
                if recent_wait_times
                else 0.0
            )

            return {
                "uptime_seconds": uptime,
                "total_acquires": self._stats.total_acquires,
                "total_releases": self._stats.total_releases,
                "utilization_percent": round(utilization, 2),
                "avg_wait_time_seconds": round(avg_wait, 6),
                "max_wait_time_seconds": round(self._stats.max_wait_time, 6),
                "recent_avg_wait_seconds": round(recent_avg, 6),
                "recent_wait_times": recent_wait_times,
                "scale_up_events": self._stats.scale_up_events,
                "scale_down_events": self._stats.scale_down_events,
            }

    def reset_stats(self) -> None:
        with self._lock:
            self._stats.reset()

    def __repr__(self) -> str:
        s = self.status()
        return (
            f"DatabaseConnectionPool(max={s['max_connections']}, "
            f"active={s['active_connections']}, idle={s['idle_connections']}, "
            f"waiting={s['waiting_count']})"
        )


if __name__ == "__main__":
    print("=" * 70)
    print("Test 1: Context manager prevents connection leak")
    print("=" * 70)
    pool = DatabaseConnectionPool(max_connections=3, min_connections=1)
    with pool.connection() as conn:
        print(f"  Acquired {conn}")
        print(f"  Status: {pool.status()}")
        result = conn.execute("SELECT 1")
        print(f"  {result}")
    print(f"  After with-block: {pool.status()}")
    assert pool.status()["active_connections"] == 0
    assert pool.status()["idle_connections"] == 1
    print("  PASSED\n")

    print("=" * 70)
    print("Test 2: Lazy init - no connections created until first acquire")
    print("=" * 70)
    pool2 = DatabaseConnectionPool(max_connections=5, min_connections=0, lazy_init=True)
    s = pool2.status()
    print(f"  Initial status (lazy): {s}")
    assert s["total_connections"] == 0, f"Expected 0, got {s['total_connections']}"
    with pool2.connection() as conn:
        print(f"  After first acquire: {conn}")
        s2 = pool2.status()
        assert s2["total_connections"] == 1
    print(f"  Status after release: {pool2.status()}")
    print("  PASSED\n")

    print("=" * 70)
    print("Test 3: Warm up - pre-create connections")
    print("=" * 70)
    pool3 = DatabaseConnectionPool(max_connections=5, min_connections=0, lazy_init=True)
    s_before = pool3.status()
    print(f"  Before warm_up: total={s_before['total_connections']}")
    created = pool3.warm_up(3)
    print(f"  warm_up(3) returned: {created}")
    s_after = pool3.status()
    print(f"  After warm_up: total={s_after['total_connections']}, idle={s_after['idle_connections']}")
    assert s_after["total_connections"] == 3
    assert s_after["idle_connections"] == 3
    print("  PASSED\n")

    print("=" * 70)
    print("Test 4: Manual scale up / scale down")
    print("=" * 70)
    pool4 = DatabaseConnectionPool(max_connections=5, min_connections=1, lazy_init=True)
    pool4.warm_up(2)
    print(f"  Initial: {pool4.status()}")
    scaled = pool4.scale_up(2)
    print(f"  scale_up(2) created: {scaled}")
    s_up = pool4.status()
    print(f"  After scale_up: {s_up}")
    assert s_up["total_connections"] == 4
    scaled_down = pool4.scale_down(2)
    print(f"  scale_down(2) removed: {scaled_down}")
    s_down = pool4.status()
    print(f"  After scale_down: {s_down}")
    assert s_down["total_connections"] == 2
    print("  PASSED\n")

    print("=" * 70)
    print("Test 5: Monitoring stats - wait times and utilization")
    print("=" * 70)
    pool5 = DatabaseConnectionPool(max_connections=2)
    pool5.warm_up(2)
    print(f"  Initial stats: {pool5.stats()}")

    def quick_worker(p, wid):
        with p.connection(timeout=3) as conn:
            time.sleep(0.1)
            conn.execute(f"SELECT {wid}")

    threads = []
    for i in range(6):
        t = threading.Thread(target=quick_worker, args=(pool5, i))
        threads.append(t)
        t.start()
    for t in threads:
        t.join()

    stats = pool5.stats()
    print(f"  After 6 acquires (2 concurrent slots):")
    print(f"    total_acquires: {stats['total_acquires']}")
    print(f"    utilization: {stats['utilization_percent']}%")
    print(f"    avg_wait_time: {stats['avg_wait_time_seconds']}s")
    print(f"    max_wait_time: {stats['max_wait_time_seconds']}s")
    assert stats["total_acquires"] == 6
    assert stats["total_releases"] == 6
    assert stats["utilization_percent"] > 0
    print("  PASSED\n")

    print("=" * 70)
    print("Test 6: Auto-scaling based on waiting queue")
    print("=" * 70)
    pool6 = DatabaseConnectionPool(
        max_connections=8,
        min_connections=2,
        lazy_init=True,
        auto_scale=True,
        scale_up_threshold=2,
        scale_down_threshold=1,
        scale_down_idle_time=0.5,
        scale_interval=0.2,
    )
    pool6.warm_up(3)
    print(f"  Starting (3 pre-created, max=8): {pool6.status()}")

    held_conns = []
    for _ in range(3):
        held_conns.append(pool6.acquire())
    print(f"  Acquired all 3, pool exhausted: {pool6.status()}")

    waiter_connections = []

    def waiter_func():
        conn = pool6.acquire(timeout=10)
        waiter_connections.append(conn)
        return conn

    waiters = []
    for _ in range(3):
        wt = threading.Thread(target=waiter_func)
        waiters.append(wt)
        wt.start()

    time.sleep(0.2)
    s_waiting = pool6.status()
    print(f"  With 3 waiters: waiting={s_waiting['waiting_count']}, total={s_waiting['total_connections']}")

    time.sleep(1.5)
    s_scaled = pool6.status()
    st_scaled = pool6.stats()
    print(f"  After auto-scale: total={s_scaled['total_connections']}, scale_ups={st_scaled['scale_up_events']}")
    assert st_scaled["scale_up_events"] > 0, "Expected scale up events"

    for pc in held_conns:
        pc.release()
    for wt in waiters:
        wt.join()
    for wc in waiter_connections:
        wc.release()

    print(f"  After releasing: {pool6.status()}")
    time.sleep(2.0)
    s_final = pool6.status()
    st_final = pool6.stats()
    print(f"  After auto-scale down: total={s_final['total_connections']}, scale_downs={st_final['scale_down_events']}")
    assert st_final["scale_down_events"] > 0, f"Expected scale down events, got {st_final['scale_down_events']}. idle={s_final['idle_connections']}"
    print("  PASSED\n")

    print("=" * 70)
    print("Test 7: Timeout auto-recycling (max_usage_time=1.5s)")
    print("=" * 70)
    pool7 = DatabaseConnectionPool(
        max_connections=2, max_usage_time=1.5, recycle_check_interval=0.5
    )
    pool7.warm_up(2)
    leaked_conn = pool7.acquire()
    print(f"  Acquired {leaked_conn} (intentionally NOT released)")
    print(f"  Status immediately: {pool7.status()}")
    print("  Waiting 2.5s for auto-recycle...")
    time.sleep(2.5)
    s = pool7.status()
    print(f"  Status after 2.5s: {s}")
    assert s["active_connections"] == 0
    print("  PASSED\n")

    print("=" * 70)
    print("Test 8: Health check - dead connection rebuilt")
    print("=" * 70)
    pool8 = DatabaseConnectionPool(max_connections=3, health_check=True)
    pool8.warm_up(3)
    raw_conn = pool8._idle_connections[0]
    print(f"  Killing {raw_conn}...")
    raw_conn.close()
    with pool8.connection() as conn:
        print(f"  Acquired {conn} (should be a new healthy connection)")
        result = conn.execute("SELECT healthy")
        print(f"  {result}")
    print(f"  Status: {pool8.status()}")
    print("  PASSED\n")

    print("=" * 70)
    print("Test 9: reset_stats clears counters")
    print("=" * 70)
    pool9 = DatabaseConnectionPool(max_connections=2)
    pool9.warm_up(2)
    with pool9.connection() as conn:
        conn.execute("SELECT 1")
    print(f"  Before reset: total_acquires={pool9.stats()['total_acquires']}")
    pool9.reset_stats()
    print(f"  After reset: total_acquires={pool9.stats()['total_acquires']}")
    assert pool9.stats()["total_acquires"] == 0
    print("  PASSED\n")

    print("All tests passed!")
