import threading
import time
import uuid
from typing import Optional, Callable, List, Tuple, Dict, Any

import redis


class LockLostError(Exception):
    pass


class RedisDistributedLock:
    def __init__(
        self,
        redis_client: redis.Redis,
        lock_key: str,
        timeout: int,
        auto_renew: bool = True,
        on_lock_lost: Optional[Callable[[], None]] = None,
        max_renew_failures: int = 3,
        reentrant: bool = True
    ):
        self.redis = redis_client
        self.lock_key = lock_key
        self.timeout = timeout
        self.auto_renew = auto_renew
        self.on_lock_lost = on_lock_lost
        self.max_renew_failures = max_renew_failures
        self.reentrant = reentrant
        self.lock_value: Optional[str] = None
        self._renew_thread: Optional[threading.Thread] = None
        self._stop_renew_event = threading.Event()
        self._locked = False
        self._lock_lost = False
        self._renew_failures = 0
        self._state_lock = threading.Lock()
        self._lock_owner_thread: Optional[int] = None
        self._reentrant_count = 0

    def _lua_unlock_script(self) -> str:
        return """
        if redis.call('GET', KEYS[1]) == ARGV[1] then
            return redis.call('DEL', KEYS[1])
        else
            return 0
        end
        """

    def _lua_renew_script(self) -> str:
        return """
        if redis.call('GET', KEYS[1]) == ARGV[1] then
            return redis.call('EXPIRE', KEYS[1], ARGV[2])
        else
            return 0
        end
        """

    def _get_current_thread_id(self) -> int:
        return threading.get_ident()

    def is_owner(self) -> bool:
        if not self._locked or self.lock_value is None:
            return False
        try:
            current_value = self.redis.get(self.lock_key)
            return current_value == self.lock_value
        except Exception:
            return False

    def _is_same_thread(self) -> bool:
        return self._lock_owner_thread == self._get_current_thread_id()

    def acquire(self, blocking: bool = True, retry_interval: float = 0.1) -> bool:
        current_thread = self._get_current_thread_id()
        
        with self._state_lock:
            if self.reentrant and self._locked and self._is_same_thread():
                self._reentrant_count += 1
                return True
        
        lock_value = str(uuid.uuid4())
        
        while True:
            result = self.redis.set(
                self.lock_key,
                lock_value,
                nx=True,
                ex=self.timeout
            )
            
            if result:
                with self._state_lock:
                    self.lock_value = lock_value
                    self._locked = True
                    self._lock_lost = False
                    self._renew_failures = 0
                    self._lock_owner_thread = current_thread
                    self._reentrant_count = 1
                if self.auto_renew:
                    self._start_auto_renew()
                return True
            
            if not blocking:
                return False
            
            time.sleep(retry_interval)

    def release(self, force: bool = False) -> bool:
        current_thread = self._get_current_thread_id()
        
        with self._state_lock:
            if not self._locked:
                return False
            
            if self.reentrant and self._is_same_thread() and self._reentrant_count > 1:
                self._reentrant_count -= 1
                return True
            
            lock_value = self.lock_value
            self._stop_auto_renew()
            
            if self._lock_lost and not force:
                self._locked = False
                self.lock_value = None
                self._lock_lost = False
                self._lock_owner_thread = None
                self._reentrant_count = 0
                return False
        
        if lock_value is None:
            return False
            
        script = self._lua_unlock_script()
        result = self.redis.eval(script, 1, self.lock_key, lock_value)
        
        if result == 1 or force:
            with self._state_lock:
                self._locked = False
                self.lock_value = None
                self._lock_lost = False
                self._lock_owner_thread = None
                self._reentrant_count = 0
            return True
        
        return False

    def _handle_lock_lost(self) -> None:
        with self._state_lock:
            if self._lock_lost:
                return
            self._lock_lost = True
            self._renew_failures = 0
        
        if self.on_lock_lost is not None:
            try:
                self.on_lock_lost()
            except Exception:
                pass

    def _start_auto_renew(self) -> None:
        self._stop_renew_event.clear()
        
        def renew_worker():
            renew_interval = self.timeout / 3
            script = self._lua_renew_script()
            
            while not self._stop_renew_event.is_set():
                time.sleep(renew_interval)
                
                if self._stop_renew_event.is_set():
                    break
                
                with self._state_lock:
                    if self._lock_lost or self.lock_value is None:
                        break
                    lock_value = self.lock_value
                
                try:
                    result = self.redis.eval(
                        script, 1, self.lock_key, lock_value, str(self.timeout)
                    )
                    
                    if result == 1:
                        with self._state_lock:
                            self._renew_failures = 0
                    else:
                        with self._state_lock:
                            self._renew_failures += 1
                            if self._renew_failures >= self.max_renew_failures:
                                self._handle_lock_lost()
                                break
                except Exception:
                    with self._state_lock:
                        self._renew_failures += 1
                        if self._renew_failures >= self.max_renew_failures:
                            self._handle_lock_lost()
                            break

        self._renew_thread = threading.Thread(target=renew_worker, daemon=True)
        self._renew_thread.start()

    def _stop_auto_renew(self) -> None:
        self._stop_renew_event.set()
        if self._renew_thread and self._renew_thread.is_alive():
            self._renew_thread.join(timeout=1)
        self._renew_thread = None

    def __enter__(self) -> bool:
        return self.acquire()

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.release()

    @property
    def is_locked(self) -> bool:
        with self._state_lock:
            return self._locked and not self._lock_lost

    @property
    def is_lock_lost(self) -> bool:
        with self._state_lock:
            return self._lock_lost

    @property
    def reentrant_count(self) -> int:
        with self._state_lock:
            return self._reentrant_count

    def check_and_raise_if_lost(self) -> None:
        if self.is_lock_lost:
            raise LockLostError(f"Lock '{self.lock_key}' has been lost")

    @staticmethod
    def with_lock(
        redis_client: redis.Redis,
        lock_key: str,
        timeout: int,
        auto_renew: bool = True,
        on_lock_lost: Optional[Callable[[], None]] = None,
        max_renew_failures: int = 3,
        reentrant: bool = True
    ) -> Callable:
        def decorator(func: Callable) -> Callable:
            def wrapper(*args, **kwargs):
                lock = RedisDistributedLock(
                    redis_client, lock_key, timeout, auto_renew,
                    on_lock_lost, max_renew_failures, reentrant
                )
                with lock:
                    return func(*args, **kwargs)
            return wrapper
        return decorator


class RedLock:
    def __init__(
        self,
        redis_clients: List[redis.Redis],
        lock_key: str,
        timeout: int,
        auto_renew: bool = True,
        on_lock_lost: Optional[Callable[[], None]] = None,
        max_renew_failures: int = 3,
        reentrant: bool = True,
        drift_factor: float = 0.01
    ):
        if len(redis_clients) < 3:
            raise ValueError("RedLock requires at least 3 Redis nodes")
        
        self.redis_clients = redis_clients
        self.lock_key = lock_key
        self.timeout = timeout
        self.auto_renew = auto_renew
        self.on_lock_lost = on_lock_lost
        self.max_renew_failures = max_renew_failures
        self.reentrant = reentrant
        self.drift_factor = drift_factor
        
        self.quorum = len(redis_clients) // 2 + 1
        self.lock_value: Optional[str] = None
        self._renew_thread: Optional[threading.Thread] = None
        self._stop_renew_event = threading.Event()
        self._locked = False
        self._lock_lost = False
        self._renew_failures = 0
        self._state_lock = threading.Lock()
        self._lock_owner_thread: Optional[int] = None
        self._reentrant_count = 0
        
        self._lua_unlock = """
        if redis.call('GET', KEYS[1]) == ARGV[1] then
            return redis.call('DEL', KEYS[1])
        else
            return 0
        end
        """
        
        self._lua_renew = """
        if redis.call('GET', KEYS[1]) == ARGV[1] then
            return redis.call('EXPIRE', KEYS[1], ARGV[2])
        else
            return 0
        end
        """

    def _get_current_thread_id(self) -> int:
        return threading.get_ident()

    def _acquire_single_instance(self, client: redis.Redis, lock_value: str) -> bool:
        try:
            result = client.set(
                self.lock_key,
                lock_value,
                nx=True,
                ex=self.timeout
            )
            return result is True
        except Exception:
            return False

    def _release_single_instance(self, client: redis.Redis, lock_value: str) -> bool:
        try:
            result = client.eval(self._lua_unlock, 1, self.lock_key, lock_value)
            return result == 1
        except Exception:
            return False

    def _renew_single_instance(self, client: redis.Redis, lock_value: str) -> bool:
        try:
            result = client.eval(self._lua_renew, 1, self.lock_key, lock_value, str(self.timeout))
            return result == 1
        except Exception:
            return False

    def is_owner(self) -> bool:
        if not self._locked or self.lock_value is None:
            return False
        
        success_count = 0
        for client in self.redis_clients:
            try:
                current_value = client.get(self.lock_key)
                if current_value == self.lock_value:
                    success_count += 1
            except Exception:
                pass
        
        return success_count >= self.quorum

    def acquire(self, blocking: bool = True, retry_interval: float = 0.2, retry_count: int = 3) -> bool:
        current_thread = self._get_current_thread_id()
        
        with self._state_lock:
            if self.reentrant and self._locked and self._lock_owner_thread == current_thread:
                self._reentrant_count += 1
                return True
        
        lock_value = str(uuid.uuid4())
        drift = int(self.timeout * self.drift_factor) + 2
        
        for attempt in range(retry_count if not blocking else 1000):
            start_time = int(time.time() * 1000)
            success_count = 0
            
            for client in self.redis_clients:
                if self._acquire_single_instance(client, lock_value):
                    success_count += 1
            
            elapsed_time = int(time.time() * 1000) - start_time
            validity_time = self.timeout * 1000 - elapsed_time - drift
            
            if success_count >= self.quorum and validity_time > 0:
                with self._state_lock:
                    self.lock_value = lock_value
                    self._locked = True
                    self._lock_lost = False
                    self._renew_failures = 0
                    self._lock_owner_thread = current_thread
                    self._reentrant_count = 1
                if self.auto_renew:
                    self._start_auto_renew()
                return True
            
            else:
                for client in self.redis_clients:
                    self._release_single_instance(client, lock_value)
            
            if not blocking:
                return False
            
            time.sleep(retry_interval)
        
        return False

    def release(self, force: bool = False) -> bool:
        current_thread = self._get_current_thread_id()
        
        with self._state_lock:
            if not self._locked:
                return False
            
            if self.reentrant and self._lock_owner_thread == current_thread and self._reentrant_count > 1:
                self._reentrant_count -= 1
                return True
            
            lock_value = self.lock_value
            self._stop_auto_renew()
            
            if self._lock_lost and not force:
                self._locked = False
                self.lock_value = None
                self._lock_lost = False
                self._lock_owner_thread = None
                self._reentrant_count = 0
                return False
        
        if lock_value is None:
            return False
        
        success_count = 0
        for client in self.redis_clients:
            if self._release_single_instance(client, lock_value):
                success_count += 1
        
        if success_count >= self.quorum or force:
            with self._state_lock:
                self._locked = False
                self.lock_value = None
                self._lock_lost = False
                self._lock_owner_thread = None
                self._reentrant_count = 0
            return True
        
        return False

    def _handle_lock_lost(self) -> None:
        with self._state_lock:
            if self._lock_lost:
                return
            self._lock_lost = True
            self._renew_failures = 0
        
        if self.on_lock_lost is not None:
            try:
                self.on_lock_lost()
            except Exception:
                pass

    def _start_auto_renew(self) -> None:
        self._stop_renew_event.clear()
        
        def renew_worker():
            renew_interval = self.timeout / 3
            
            while not self._stop_renew_event.is_set():
                time.sleep(renew_interval)
                
                if self._stop_renew_event.is_set():
                    break
                
                with self._state_lock:
                    if self._lock_lost or self.lock_value is None:
                        break
                    lock_value = self.lock_value
                
                success_count = 0
                for client in self.redis_clients:
                    if self._renew_single_instance(client, lock_value):
                        success_count += 1
                
                if success_count >= self.quorum:
                    with self._state_lock:
                        self._renew_failures = 0
                else:
                    with self._state_lock:
                        self._renew_failures += 1
                        if self._renew_failures >= self.max_renew_failures:
                            self._handle_lock_lost()
                            break

        self._renew_thread = threading.Thread(target=renew_worker, daemon=True)
        self._renew_thread.start()

    def _stop_auto_renew(self) -> None:
        self._stop_renew_event.set()
        if self._renew_thread and self._renew_thread.is_alive():
            self._renew_thread.join(timeout=1)
        self._renew_thread = None

    def __enter__(self) -> bool:
        return self.acquire()

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.release()

    @property
    def is_locked(self) -> bool:
        with self._state_lock:
            return self._locked and not self._lock_lost

    @property
    def is_lock_lost(self) -> bool:
        with self._state_lock:
            return self._lock_lost

    @property
    def reentrant_count(self) -> int:
        with self._state_lock:
            return self._reentrant_count

    @property
    def node_count(self) -> int:
        return len(self.redis_clients)

    @property
    def quorum_size(self) -> int:
        return self.quorum

    def check_and_raise_if_lost(self) -> None:
        if self.is_lock_lost:
            raise LockLostError(f"RedLock '{self.lock_key}' has been lost")


if __name__ == "__main__":
    redis_client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)
    
    print("=== Test 1: Basic lock acquire and release ===")
    lock = RedisDistributedLock(redis_client, "test:lock", timeout=10)
    print(f"Before acquire: is_locked={lock.is_locked}")
    lock.acquire()
    print(f"After acquire: is_locked={lock.is_locked}")
    print(f"Lock value in Redis: {redis_client.get('test:lock')}")
    lock.release()
    print(f"After release: is_locked={lock.is_locked}")
    print(f"Lock value in Redis: {redis_client.get('test:lock')}")
    
    print("\n=== Test 2: Context manager ===")
    lock2 = RedisDistributedLock(redis_client, "test:lock:cm", timeout=10)
    with lock2:
        print(f"Inside context: is_locked={lock2.is_locked}")
        print(f"Lock value in Redis: {redis_client.get('test:lock:cm')}")
    print(f"Outside context: is_locked={lock2.is_locked}")
    
    print("\n=== Test 3: Non-blocking acquire ===")
    lock3 = RedisDistributedLock(redis_client, "test:lock:nb", timeout=10)
    lock4 = RedisDistributedLock(redis_client, "test:lock:nb", timeout=10)
    print(f"lock3 acquire: {lock3.acquire()}")
    print(f"lock4 acquire (non-blocking): {lock4.acquire(blocking=False)}")
    lock3.release()
    
    print("\n=== Test 4: Auto renew (watchdog) ===")
    lock5 = RedisDistributedLock(redis_client, "test:lock:renew", timeout=3, auto_renew=True)
    lock5.acquire()
    print(f"Acquired lock, TTL: {redis_client.ttl('test:lock:renew')}")
    print("Waiting 2 seconds for renew...")
    time.sleep(2)
    print(f"After 2s, TTL: {redis_client.ttl('test:lock:renew')}")
    print("Waiting 2 more seconds...")
    time.sleep(2)
    print(f"After 4s, TTL: {redis_client.ttl('test:lock:renew')}")
    lock5.release()
    print("Released lock")
    
    print("\n=== Test 5: Decorator ===")
    @RedisDistributedLock.with_lock(redis_client, "test:lock:decorator", timeout=10)
    def critical_section():
        print("Inside critical section")
        print(f"Lock exists: {redis_client.exists('test:lock:decorator')}")
        return "success"
    
    result = critical_section()
    print(f"Decorator result: {result}")
    print(f"Lock released: {not redis_client.exists('test:lock:decorator')}")
    
    print("\n=== Test 6: Lock safety - cannot release others' lock ===")
    lock6 = RedisDistributedLock(redis_client, "test:lock:safety", timeout=10)
    lock7 = RedisDistributedLock(redis_client, "test:lock:safety", timeout=10)
    lock6.acquire()
    print(f"lock6 lock_value: {lock6.lock_value}")
    print(f"lock7 trying to release lock6's lock: {lock7.release()}")
    print(f"Lock still exists: {redis_client.exists('test:lock:safety')}")
    lock6.release()
    print(f"lock6 released, lock exists: {redis_client.exists('test:lock:safety')}")
    
    print("\n=== Test 7: Blocking acquire ===")
    lock8 = RedisDistributedLock(redis_client, "test:lock:blocking", timeout=2)
    lock9 = RedisDistributedLock(redis_client, "test:lock:blocking", timeout=10)
    lock8.acquire()
    print("lock8 acquired lock")
    
    start_time = time.time()
    
    def release_later():
        time.sleep(1.5)
        lock8.release()
        print("lock8 released lock after 1.5s")
    
    threading.Thread(target=release_later, daemon=True).start()
    
    print("lock9 waiting to acquire lock...")
    lock9.acquire(blocking=True)
    wait_time = time.time() - start_time
    print(f"lock9 acquired lock after waiting {wait_time:.2f}s")
    lock9.release()
    
    print("\n=== Test 8: is_owner() verification ===")
    lock10 = RedisDistributedLock(redis_client, "test:lock:owner", timeout=10)
    lock11 = RedisDistributedLock(redis_client, "test:lock:owner", timeout=10)
    print(f"lock10 is_owner before acquire: {lock10.is_owner()}")
    lock10.acquire()
    print(f"lock10 is_owner after acquire: {lock10.is_owner()}")
    print(f"lock11 is_owner (not acquired): {lock11.is_owner()}")
    lock10.release()
    print(f"lock10 is_owner after release: {lock10.is_owner()}")
    
    print("\n=== Test 9: Lock timeout - cannot delete others' lock ===")
    lock12 = RedisDistributedLock(redis_client, "test:lock:timeout", timeout=2, auto_renew=False)
    lock12.acquire()
    lock12_value = lock12.lock_value
    print(f"lock12 acquired with value: {lock12_value}")
    print(f"lock12 is_owner: {lock12.is_owner()}")
    print("Waiting 3 seconds for lock to expire...")
    time.sleep(3)
    print(f"Lock exists in Redis: {redis_client.exists('test:lock:timeout')}")
    print(f"lock12 is_owner after timeout: {lock12.is_owner()}")
    
    lock13 = RedisDistributedLock(redis_client, "test:lock:timeout", timeout=10, auto_renew=False)
    lock13.acquire()
    lock13_value = lock13.lock_value
    print(f"lock13 acquired with value: {lock13_value}")
    print(f"Current lock value in Redis: {redis_client.get('test:lock:timeout')}")
    
    print(f"lock12 trying to release (should fail, not owner): {lock12.release()}")
    print(f"Lock still exists: {redis_client.exists('test:lock:timeout')}")
    print(f"Current value in Redis: {redis_client.get('test:lock:timeout')}")
    
    print(f"lock13 is_owner: {lock13.is_owner()}")
    print(f"lock13 release: {lock13.release()}")
    print(f"Lock exists after lock13 release: {redis_client.exists('test:lock:timeout')}")
    
    print("\n=== Test 10: Reentrant lock - same thread multiple acquire ===")
    lock14 = RedisDistributedLock(redis_client, "test:lock:reentrant", timeout=10, reentrant=True)
    print(f"First acquire: {lock14.acquire()}, reentrant_count={lock14.reentrant_count}")
    print(f"Second acquire: {lock14.acquire()}, reentrant_count={lock14.reentrant_count}")
    print(f"Third acquire: {lock14.acquire()}, reentrant_count={lock14.reentrant_count}")
    print(f"Lock value in Redis: {redis_client.get('test:lock:reentrant')}")
    print(f"First release: {lock14.release()}, reentrant_count={lock14.reentrant_count}")
    print(f"Second release: {lock14.release()}, reentrant_count={lock14.reentrant_count}")
    print(f"Lock still exists: {redis_client.exists('test:lock:reentrant')}")
    print(f"Third release: {lock14.release()}, reentrant_count={lock14.reentrant_count}")
    print(f"Lock exists after final release: {redis_client.exists('test:lock:reentrant')}")
    
    print("\n=== Test 11: Non-reentrant lock - cannot reacquire ===")
    lock15 = RedisDistributedLock(redis_client, "test:lock:nonreentrant", timeout=10, reentrant=False)
    print(f"First acquire: {lock15.acquire()}")
    print("Second acquire (non-blocking, should block/fail)...")
    result = lock15.acquire(blocking=False)
    print(f"Second acquire result: {result} (False is correct for non-reentrant)")
    lock15.release()
    
    print("\n=== Test 12: Reentrant with context manager ===")
    lock16 = RedisDistributedLock(redis_client, "test:lock:reentrant:cm", timeout=10)
    print(f"Initial count: {lock16.reentrant_count}")
    with lock16:
        print(f"After first with: count={lock16.reentrant_count}")
        with lock16:
            print(f"After nested with: count={lock16.reentrant_count}")
            print(f"Lock exists: {redis_client.exists('test:lock:reentrant:cm')}")
        print(f"After inner exit: count={lock16.reentrant_count}")
        print(f"Lock still exists: {redis_client.exists('test:lock:reentrant:cm')}")
    print(f"After outer exit: count={lock16.reentrant_count}")
    print(f"Lock released: {not redis_client.exists('test:lock:reentrant:cm')}")
    
    print("\n=== Test 13: RedLock initialization ===")
    try:
        bad_redlock = RedLock([redis_client], "test:redlock:bad", timeout=10)
    except ValueError as e:
        print(f"Correctly raised ValueError for <3 nodes: {e}")
    
    redis_nodes = [
        redis.Redis(host='localhost', port=6379, db=1, decode_responses=True),
        redis.Redis(host='localhost', port=6379, db=2, decode_responses=True),
        redis.Redis(host='localhost', port=6379, db=3, decode_responses=True),
        redis.Redis(host='localhost', port=6379, db=4, decode_responses=True),
        redis.Redis(host='localhost', port=6379, db=5, decode_responses=True),
    ]
    
    redlock = RedLock(redis_nodes, "test:redlock", timeout=10)
    print(f"RedLock created with {redlock.node_count} nodes")
    print(f"Quorum size: {redlock.quorum_size}")
    
    print("\n=== Test 14: RedLock acquire and release ===")
    print(f"Before acquire: is_locked={redlock.is_locked}")
    success = redlock.acquire()
    print(f"Acquire result: {success}")
    if success:
        print(f"Lock values across nodes:")
        for i, node in enumerate(redis_nodes):
            print(f"  Node {i+1}: {node.get('test:redlock')}")
    redlock.release()
    print(f"After release: is_locked={redlock.is_locked}")
    for i, node in enumerate(redis_nodes):
        node.delete("test:redlock")
    
    print("\n=== Test 15: RedLock with quorum failure simulation ===")
    redlock2 = RedLock(redis_nodes[:3], "test:redlock:quorum", timeout=10, auto_renew=False)
    print(f"RedLock2 with {redlock2.node_count} nodes, quorum={redlock2.quorum_size}")
    redlock2.acquire()
    print("Acquired lock on all 3 nodes")
    
    print("Manually deleting lock on 2 nodes (simulating node failure)...")
    redis_nodes[0].delete("test:redlock:quorum")
    redis_nodes[1].delete("test:redlock:quorum")
    
    print(f"is_owner after losing majority: {redlock2.is_owner()}")
    print(f"Lock on node 0: {redis_nodes[0].get('test:redlock:quorum')}")
    print(f"Lock on node 1: {redis_nodes[1].get('test:redlock:quorum')}")
    print(f"Lock on node 2: {redis_nodes[2].get('test:redlock:quorum')}")
    
    redlock2.release()
    for i, node in enumerate(redis_nodes):
        node.delete("test:redlock:quorum")
    
    print("\n=== Test 16: RedLock reentrant ===")
    redlock3 = RedLock(redis_nodes[:3], "test:redlock:reentrant", timeout=10)
    print(f"First acquire: {redlock3.acquire()}, count={redlock3.reentrant_count}")
    print(f"Second acquire: {redlock3.acquire()}, count={redlock3.reentrant_count}")
    print(f"First release: {redlock3.release()}, count={redlock3.reentrant_count}")
    print(f"Second release: {redlock3.release()}, count={redlock3.reentrant_count}")
    for i, node in enumerate(redis_nodes):
        node.delete("test:redlock:reentrant")
    
    print("\n=== Test 17: Force release ===")
    lock17 = RedisDistributedLock(redis_client, "test:lock:force", timeout=10, auto_renew=False)
    lock17.acquire()
    lock17_value = lock17.lock_value
    print(f"lock17 acquired with value: {lock17_value}")
    
    print("Manually changing lock value in Redis...")
    redis_client.set("test:lock:force", "other-client-value", ex=10)
    
    print(f"lock17 release (normal): {lock17.release()}")
    print(f"Lock still exists: {redis_client.exists('test:lock:force')}")
    print(f"lock17._locked: {lock17._locked}")
    print(f"lock17 release (force=True): {lock17.release(force=True)}")
    print(f"lock17._locked: {lock17._locked}")
    redis_client.delete("test:lock:force")
    
    print("\nAll tests completed!")
