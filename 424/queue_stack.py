import logging
import threading
import time

logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class Queue:
    def __init__(self):
        self.items = []
        self._lock = threading.Lock()

    def push(self, item):
        with self._lock:
            self.items.append(item)
        return f"Pushed {item} to queue. Queue: {self.items}"

    def pop(self):
        with self._lock:
            if self.is_empty():
                logger.warning("Attempted to pop from an empty queue")
                return None
            item = self.items.pop(0)
        return f"Popped {item} from queue. Queue: {self.items}"

    def peek(self):
        with self._lock:
            if self.is_empty():
                logger.warning("Attempted to peek an empty queue")
                return None
        return f"Front of queue: {self.items[0]}"

    def is_empty(self):
        return len(self.items) == 0

    def size(self):
        return f"Queue size: {len(self.items)}"

    def get_state(self):
        return f"Current queue: {self.items}"


class Stack:
    def __init__(self):
        self.items = []
        self._lock = threading.Lock()

    def push(self, item):
        with self._lock:
            self.items.append(item)
        return f"Pushed {item} to stack. Stack: {self.items}"

    def pop(self):
        with self._lock:
            if self.is_empty():
                logger.warning("Attempted to pop from an empty stack")
                return None
            item = self.items.pop()
        return f"Popped {item} from stack. Stack: {self.items}"

    def peek(self):
        with self._lock:
            if self.is_empty():
                logger.warning("Attempted to peek an empty stack")
                return None
        return f"Top of stack: {self.items[-1]}"

    def is_empty(self):
        return len(self.items) == 0

    def size(self):
        return f"Stack size: {len(self.items)}"

    def get_state(self):
        return f"Current stack: {self.items}"


class TwoStacksQueue:
    def __init__(self):
        self.stack_in = []
        self.stack_out = []
        self._lock = threading.Lock()

    def _transfer(self):
        while self.stack_in:
            self.stack_out.append(self.stack_in.pop())

    def push(self, item):
        with self._lock:
            self.stack_in.append(item)
        return f"Pushed {item} to queue. in: {self.stack_in}, out: {self.stack_out}"

    def pop(self):
        with self._lock:
            if self.is_empty():
                logger.warning("Attempted to pop from an empty queue")
                return None
            if not self.stack_out:
                self._transfer()
            item = self.stack_out.pop()
        return f"Popped {item} from queue. in: {self.stack_in}, out: {self.stack_out}"

    def peek(self):
        with self._lock:
            if self.is_empty():
                logger.warning("Attempted to peek an empty queue")
                return None
            if not self.stack_out:
                self._transfer()
        return f"Front of queue: {self.stack_out[-1]}"

    def is_empty(self):
        return len(self.stack_in) == 0 and len(self.stack_out) == 0

    def size(self):
        return f"Queue size: {len(self.stack_in) + len(self.stack_out)}"

    def get_state(self):
        return f"Current queue - stack_in: {self.stack_in}, stack_out: {self.stack_out}"


class TwoQueuesStack:
    def __init__(self):
        self.queue1 = []
        self.queue2 = []
        self._lock = threading.Lock()

    def push(self, item):
        with self._lock:
            self.queue1.append(item)
        return f"Pushed {item} to stack. q1: {self.queue1}, q2: {self.queue2}"

    def pop(self):
        with self._lock:
            if self.is_empty():
                logger.warning("Attempted to pop from an empty stack")
                return None
            while len(self.queue1) > 1:
                self.queue2.append(self.queue1.pop(0))
            item = self.queue1.pop(0)
            self.queue1, self.queue2 = self.queue2, self.queue1
        return f"Popped {item} from stack. q1: {self.queue1}, q2: {self.queue2}"

    def peek(self):
        with self._lock:
            if self.is_empty():
                logger.warning("Attempted to peek an empty stack")
                return None
            while len(self.queue1) > 1:
                self.queue2.append(self.queue1.pop(0))
            item = self.queue1[0]
            self.queue2.append(self.queue1.pop(0))
            self.queue1, self.queue2 = self.queue2, self.queue1
        return f"Top of stack: {item}"

    def is_empty(self):
        return len(self.queue1) == 0

    def size(self):
        return f"Stack size: {len(self.queue1)}"

    def get_state(self):
        return f"Current stack - q1: {self.queue1}, q2: {self.queue2}"


def benchmark_queue(n=10000):
    print(f"\nQueue Performance Benchmark (n={n})")
    print("-" * 60)

    q1 = Queue()
    start = time.time()
    for i in range(n):
        q1.push(i)
    for _ in range(n):
        q1.pop()
    t1 = time.time() - start
    print(f"Single-list Queue: {t1:.6f}s")

    q2 = TwoStacksQueue()
    start = time.time()
    for i in range(n):
        q2.push(i)
    for _ in range(n):
        q2.pop()
    t2 = time.time() - start
    print(f"Two-stacks Queue:  {t2:.6f}s")
    print(f"Speedup: {t1/t2:.2f}x" if t2 < t1 else f"Slowdown: {t2/t1:.2f}x")


def benchmark_stack(n=10000):
    print(f"\nStack Performance Benchmark (n={n})")
    print("-" * 60)

    s1 = Stack()
    start = time.time()
    for i in range(n):
        s1.push(i)
    for _ in range(n):
        s1.pop()
    t1 = time.time() - start
    print(f"Single-list Stack: {t1:.6f}s")

    s2 = TwoQueuesStack()
    start = time.time()
    for i in range(n):
        s2.push(i)
    for _ in range(n):
        s2.pop()
    t2 = time.time() - start
    print(f"Two-queues Stack:  {t2:.6f}s")
    print(f"Slowdown: {t2/t1:.2f}x")


if __name__ == "__main__":
    print("=== Queue Demo (FIFO) ===")
    q = Queue()
    print(q.push(1))
    print(q.push(2))
    print(q.push(3))
    print(q.peek())
    print(q.size())
    print(q.pop())
    print(q.pop())
    print(q.is_empty())
    print(q.get_state())
    print(q.pop())
    print(q.pop())

    print("\n=== Stack Demo (LIFO) ===")
    s = Stack()
    print(s.push('a'))
    print(s.push('b'))
    print(s.push('c'))
    print(s.peek())
    print(s.size())
    print(s.pop())
    print(s.pop())
    print(s.is_empty())
    print(s.get_state())
    print(s.pop())
    print(s.pop())

    print("\n=== Two Stacks Queue Demo ===")
    tq = TwoStacksQueue()
    print(tq.push(1))
    print(tq.push(2))
    print(tq.push(3))
    print(tq.peek())
    print(tq.pop())
    print(tq.push(4))
    print(tq.push(5))
    print(tq.pop())
    print(tq.pop())
    print(tq.pop())
    print(tq.pop())
    print(tq.pop())

    print("\n=== Two Queues Stack Demo ===")
    ts = TwoQueuesStack()
    print(ts.push('a'))
    print(ts.push('b'))
    print(ts.push('c'))
    print(ts.peek())
    print(ts.pop())
    print(ts.push('d'))
    print(ts.push('e'))
    print(ts.pop())
    print(ts.pop())
    print(ts.pop())
    print(ts.pop())
    print(ts.pop())

    print("\n=== Thread Safety Demo ===")
    q = TwoStacksQueue()
    s = TwoQueuesStack()
    results = []

    def queue_worker(data_list):
        for val in data_list:
            q.push(val)
        for _ in data_list:
            r = q.pop()
            if r is not None:
                results.append(r)

    def stack_worker(data_list):
        for val in data_list:
            s.push(val)
        for _ in data_list:
            r = s.pop()
            if r is not None:
                results.append(r)

    t1 = threading.Thread(target=queue_worker, args=([1, 2, 3],))
    t2 = threading.Thread(target=queue_worker, args=([4, 5, 6],))
    t3 = threading.Thread(target=stack_worker, args=(['a', 'b'],))
    t4 = threading.Thread(target=stack_worker, args=(['c', 'd'],))

    for t in [t1, t2, t3, t4]:
        t.start()
    for t in [t1, t2, t3, t4]:
        t.join()

    print(f"Thread results count: {len(results)}")
    print(f"Queue empty after threads: {q.is_empty()}")
    print(f"Stack empty after threads: {s.is_empty()}")

    benchmark_queue(10000)
    benchmark_stack(2000)
