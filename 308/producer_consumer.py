import threading
import queue
import time
import random
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from collections import deque

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


@dataclass
class Message:
    id: int
    content: str
    created_at: float
    processed_at: float = None
    processing_latency: float = None


@dataclass
class Metrics:
    backpressure_triggered: int = 0
    total_messages_produced: int = 0
    total_messages_consumed: int = 0
    throughput_history: List[Dict] = field(default_factory=list)
    queue_length_history: List[Dict] = field(default_factory=list)
    latency_history: List[float] = field(default_factory=list)
    rejected_messages: int = 0
    backpressure_active: bool = False
    backpressure_start_time: float = None
    total_backpressure_duration: float = 0
    producer_rate_history: List[Dict] = field(default_factory=list)
    pid_output_history: List[Dict] = field(default_factory=list)


class PIDController:
    def __init__(
        self,
        kp: float = 0.06,
        ki: float = 0.008,
        kd: float = 0.03,
        setpoint: float = 25.0,
        output_min: float = 0.0,
        output_max: float = 1.0,
        integral_limit: float = 3.0,
        deadband: float = 2.0,
        derivative_filter: float = 0.5,
    ):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.setpoint = setpoint
        self.output_min = output_min
        self.output_max = output_max
        self.integral_limit = integral_limit
        self.deadband = deadband
        self.derivative_filter = derivative_filter

        self._integral = 0.0
        self._prev_error = 0.0
        self._prev_time: Optional[float] = None
        self._prev_derivative = 0.0

    def compute(self, process_variable: float, current_time: float) -> Dict:
        error = process_variable - self.setpoint

        if self._prev_time is None:
            self._prev_time = current_time
            self._prev_error = error
            return self._build_result(0.0, error, 0.0, 0.0, 0.0, current_time)

        dt = current_time - self._prev_time
        if dt <= 0:
            dt = 0.001

        if error > 0:
            self._integral += error * dt
        else:
            self._integral += error * dt * 0.3
        self._integral = max(0.0, min(self.integral_limit, self._integral))
        i_term = self.ki * self._integral

        p_term = self.kp * error

        raw_derivative = (error - self._prev_error) / dt
        derivative = self.derivative_filter * self._prev_derivative + (1 - self.derivative_filter) * raw_derivative
        d_term = self.kd * derivative
        self._prev_derivative = derivative

        output = p_term + i_term + d_term
        output = max(self.output_min, min(self.output_max, output))

        if abs(error) <= self.deadband and abs(raw_derivative) < 1.0:
            output = max(0.0, min(0.05, output))

        self._prev_error = error
        self._prev_time = current_time

        return self._build_result(output, error, p_term, i_term, d_term, current_time)

    def _build_result(self, output, error, p_term, i_term, d_term, t) -> Dict:
        return {
            "time": t,
            "output": output,
            "error": error,
            "p_term": p_term,
            "i_term": i_term,
            "d_term": d_term,
            "setpoint": self.setpoint,
        }

    def reset(self):
        self._integral = 0.0
        self._prev_error = 0.0
        self._prev_time = None
        self._prev_derivative = 0.0


class Producer(threading.Thread):
    def __init__(
        self,
        q: queue.Queue,
        metrics: Metrics,
        pid_controller: PIDController,
        produce_rate: float = 10,
        max_queue_size: int = 100,
        run_duration: float = 30,
        min_rate_ratio: float = 0.15,
        rate_adjust_interval: float = 0.5,
        name: str = "Producer",
    ):
        super().__init__(name=name)
        self.q = q
        self.metrics = metrics
        self.pid = pid_controller
        self.base_produce_rate = produce_rate
        self.current_produce_rate = produce_rate
        self.max_queue_size = max_queue_size
        self.run_duration = run_duration
        self.min_produce_rate = produce_rate * min_rate_ratio
        self.rate_adjust_interval = rate_adjust_interval
        self.stop_event = threading.Event()
        self.message_id = 0
        self.last_rate_adjust_time = 0

    def run(self):
        start_time = time.time()
        while not self.stop_event.is_set() and (time.time() - start_time) < self.run_duration:
            self._pid_control()

            if self.q.qsize() >= self.max_queue_size:
                self.metrics.rejected_messages += 1
                time.sleep(0.1)
                continue

            interval = 1.0 / self.current_produce_rate
            time.sleep(interval)

            message = Message(
                id=self.message_id,
                content=f"message-{self.message_id}",
                created_at=time.time()
            )
            self.q.put(message)
            self.message_id += 1
            self.metrics.total_messages_produced += 1

    def _pid_control(self):
        current_queue_size = self.q.qsize()
        current_time = time.time()

        if current_time - self.last_rate_adjust_time < self.rate_adjust_interval:
            return

        self.last_rate_adjust_time = current_time

        pid_result = self.pid.compute(current_queue_size, current_time)

        throttling_ratio = pid_result["output"]

        was_backpressure = self.metrics.backpressure_active
        self.metrics.backpressure_active = throttling_ratio > 0.01

        if self.metrics.backpressure_active and not was_backpressure:
            self.metrics.backpressure_start_time = current_time
            self.metrics.backpressure_triggered += 1
        elif not self.metrics.backpressure_active and was_backpressure:
            if self.metrics.backpressure_start_time:
                self.metrics.total_backpressure_duration += current_time - self.metrics.backpressure_start_time
                self.metrics.backpressure_start_time = None

        effective_rate = self.base_produce_rate * (1.0 - throttling_ratio)
        effective_rate = max(effective_rate, self.min_produce_rate)
        effective_rate = min(effective_rate, self.base_produce_rate)
        self.current_produce_rate = effective_rate

        self.metrics.producer_rate_history.append({
            "time": current_time,
            "rate": self.current_produce_rate,
            "queue_size": current_queue_size,
            "backpressure_active": self.metrics.backpressure_active,
            "throttling_ratio": throttling_ratio,
        })

        self.metrics.pid_output_history.append(pid_result)

    def stop(self):
        self.stop_event.set()


class Consumer(threading.Thread):
    def __init__(
        self,
        q: queue.Queue,
        metrics: Metrics,
        process_rate: float = 8,
        producer_stopped: threading.Event = None,
        name: str = "Consumer",
    ):
        super().__init__(name=name)
        self.q = q
        self.metrics = metrics
        self.base_process_rate = process_rate
        self.current_process_rate = process_rate
        self.stop_event = threading.Event()
        self.producer_stopped = producer_stopped
        self.processing_times = deque(maxlen=100)

    def run(self):
        while not self.stop_event.is_set():
            try:
                message = self.q.get(timeout=0.2)
                self._process_message(message)
                self.q.task_done()
                self.metrics.total_messages_consumed += 1
            except queue.Empty:
                if self.producer_stopped and self.producer_stopped.is_set() and self.q.empty():
                    break
                continue

    def _process_message(self, message: Message):
        process_time = 1.0 / self.current_process_rate
        time.sleep(process_time)

        message.processed_at = time.time()
        latency = message.processed_at - message.created_at
        message.processing_latency = latency

        self.processing_times.append(latency)
        self.metrics.latency_history.append(latency)

    def stop(self):
        self.stop_event.set()


class Monitor(threading.Thread):
    def __init__(
        self,
        q: queue.Queue,
        metrics: Metrics,
        consumer: Consumer,
        sample_interval: float = 1.0,
    ):
        super().__init__(name="Monitor")
        self.q = q
        self.metrics = metrics
        self.consumer = consumer
        self.sample_interval = sample_interval
        self.stop_event = threading.Event()
        self.last_consumed_count = 0
        self.last_produced_count = 0

    def run(self):
        while not self.stop_event.is_set():
            current_time = time.time()
            queue_size = self.q.qsize()

            current_consumed = self.metrics.total_messages_consumed
            consumer_throughput = (current_consumed - self.last_consumed_count) / self.sample_interval
            self.last_consumed_count = current_consumed

            current_produced = self.metrics.total_messages_produced
            producer_throughput = (current_produced - self.last_produced_count) / self.sample_interval
            self.last_produced_count = current_produced

            self.metrics.queue_length_history.append({
                "time": current_time,
                "length": queue_size
            })

            self.metrics.throughput_history.append({
                "time": current_time,
                "consumer_throughput": consumer_throughput,
                "producer_throughput": producer_throughput,
                "backpressure_active": self.metrics.backpressure_active
            })

            time.sleep(self.sample_interval)

    def stop(self):
        self.stop_event.set()


def run_simulation(
    produce_rate: float = 10,
    consume_rate: float = 8,
    pid_kp: float = 0.06,
    pid_ki: float = 0.008,
    pid_kd: float = 0.03,
    pid_setpoint: float = 25.0,
    max_queue_size: int = 100,
    min_rate_ratio: float = 0.15,
    rate_adjust_interval: float = 0.5,
    run_duration: float = 30,
) -> Metrics:
    q = queue.Queue(maxsize=max_queue_size)
    metrics = Metrics()
    producer_stopped = threading.Event()

    pid = PIDController(
        kp=pid_kp,
        ki=pid_ki,
        kd=pid_kd,
        setpoint=pid_setpoint,
        output_min=0.0,
        output_max=1.0,
        integral_limit=2.0,
        deadband=2.0,
    )

    producer = Producer(
        q=q,
        metrics=metrics,
        pid_controller=pid,
        produce_rate=produce_rate,
        max_queue_size=max_queue_size,
        run_duration=run_duration,
        min_rate_ratio=min_rate_ratio,
        rate_adjust_interval=rate_adjust_interval,
    )

    consumer = Consumer(
        q=q,
        metrics=metrics,
        process_rate=consume_rate,
        producer_stopped=producer_stopped,
    )

    monitor = Monitor(
        q=q,
        metrics=metrics,
        consumer=consumer,
        sample_interval=1.0,
    )

    monitor.start()
    consumer.start()
    producer.start()

    producer.join()
    producer_stopped.set()

    consumer.join(timeout=20)
    if consumer.is_alive():
        consumer.stop()
        consumer.join(timeout=5)

    monitor.stop()
    monitor.join(timeout=5)

    if metrics.backpressure_start_time:
        metrics.total_backpressure_duration += time.time() - metrics.backpressure_start_time

    return metrics


def print_results(metrics: Metrics):
    print("=" * 60)
    print("生产者-消费者模拟结果 (PID自适应背压)")
    print("=" * 60)
    print(f"总生产消息数: {metrics.total_messages_produced}")
    print(f"总消费消息数: {metrics.total_messages_consumed}")
    print(f"拒绝消息数: {metrics.rejected_messages}")
    print(f"背压触发次数: {metrics.backpressure_triggered}")
    print(f"背压总持续时间: {metrics.total_backpressure_duration:.2f} 秒")
    print("-" * 60)

    if metrics.latency_history:
        avg_latency = sum(metrics.latency_history) / len(metrics.latency_history)
        max_latency = max(metrics.latency_history)
        min_latency = min(metrics.latency_history)
        print(f"平均处理延迟: {avg_latency:.4f} 秒")
        print(f"最大处理延迟: {max_latency:.4f} 秒")
        print(f"最小处理延迟: {min_latency:.4f} 秒")
    print("-" * 60)

    normal_producer_tp = [t["producer_throughput"] for t in metrics.throughput_history if not t["backpressure_active"]]
    backpressure_producer_tp = [t["producer_throughput"] for t in metrics.throughput_history if t["backpressure_active"]]
    normal_consumer_tp = [t["consumer_throughput"] for t in metrics.throughput_history if not t["backpressure_active"]]
    backpressure_consumer_tp = [t["consumer_throughput"] for t in metrics.throughput_history if t["backpressure_active"]]

    print("生产者吞吐量变化:")
    if normal_producer_tp:
        avg_normal = sum(normal_producer_tp) / len(normal_producer_tp)
        print(f"  正常状态平均: {avg_normal:.2f} msg/s")
    if backpressure_producer_tp:
        avg_backpressure = sum(backpressure_producer_tp) / len(backpressure_producer_tp)
        print(f"  背压状态平均: {avg_backpressure:.2f} msg/s")
    if normal_producer_tp and backpressure_producer_tp:
        avg_normal = sum(normal_producer_tp) / len(normal_producer_tp)
        avg_backpressure = sum(backpressure_producer_tp) / len(backpressure_producer_tp)
        if avg_normal > 0:
            drop_percent = ((avg_normal - avg_backpressure) / avg_normal) * 100
            print(f"  吞吐量下降率: {drop_percent:.2f}%")

    print()
    print("消费者吞吐量变化:")
    if normal_consumer_tp:
        avg_normal = sum(normal_consumer_tp) / len(normal_consumer_tp)
        print(f"  正常状态平均: {avg_normal:.2f} msg/s")
    if backpressure_consumer_tp:
        avg_backpressure = sum(backpressure_consumer_tp) / len(backpressure_consumer_tp)
        print(f"  背压状态平均: {avg_backpressure:.2f} msg/s")

    print("-" * 60)
    print("生产者速率变化 (PID控制):")
    for i, entry in enumerate(metrics.producer_rate_history[:15]):
        bp_marker = " [背压]" if entry["backpressure_active"] else ""
        print(f"  时间 {entry['time']:.1f}s: 速率 = {entry['rate']:.2f} msg/s, "
              f"队列 = {entry['queue_size']}, 限流比 = {entry['throttling_ratio']:.3f}{bp_marker}")
    if len(metrics.producer_rate_history) > 15:
        print(f"  ... (共 {len(metrics.producer_rate_history)} 个速率调整点)")

    print("-" * 60)
    print("PID控制器输出 (前10个采样点):")
    for i, entry in enumerate(metrics.pid_output_history[:10]):
        print(f"  误差 = {entry['error']:.2f}, P = {entry['p_term']:.3f}, "
              f"I = {entry['i_term']:.3f}, D = {entry['d_term']:.3f}, 输出 = {entry['output']:.3f}")
    if len(metrics.pid_output_history) > 10:
        print(f"  ... (共 {len(metrics.pid_output_history)} 个PID采样点)")


def plot_control_curves(metrics: Metrics, output_path: str = "pid_control_curves.png"):
    fig, axes = plt.subplots(4, 1, figsize=(12, 14), sharex=True)
    fig.suptitle("PID Adaptive Backpressure Control Curves", fontsize=16, fontweight="bold")

    if not metrics.producer_rate_history:
        print("无数据可绘制")
        return

    base_time = metrics.producer_rate_history[0]["time"]
    rate_times = [e["time"] - base_time for e in metrics.producer_rate_history]
    rates = [e["rate"] for e in metrics.producer_rate_history]
    queue_sizes = [e["queue_size"] for e in metrics.producer_rate_history]
    throttling_ratios = [e["throttling_ratio"] for e in metrics.producer_rate_history]

    pid_times = [e["time"] - base_time for e in metrics.pid_output_history]
    errors = [e["error"] for e in metrics.pid_output_history]
    p_terms = [e["p_term"] for e in metrics.pid_output_history]
    i_terms = [e["i_term"] for e in metrics.pid_output_history]
    d_terms = [e["d_term"] for e in metrics.pid_output_history]
    pid_outputs = [e["output"] for e in metrics.pid_output_history]
    setpoint = metrics.pid_output_history[0]["setpoint"] if metrics.pid_output_history else 0

    ax1 = axes[0]
    ax1.plot(rate_times, queue_sizes, "b-", linewidth=1.5, label="Queue Length")
    ax1.axhline(y=setpoint, color="r", linestyle="--", linewidth=1.2, label=f"Setpoint ({setpoint:.0f})")
    ax1.fill_between(rate_times, setpoint, queue_sizes,
                     where=[q > setpoint for q in queue_sizes],
                     alpha=0.2, color="red", label="Over Setpoint")
    ax1.set_ylabel("Queue Length", fontsize=12)
    ax1.legend(loc="upper right", fontsize=9)
    ax1.set_title("Queue Length vs Setpoint", fontsize=13)
    ax1.grid(True, alpha=0.3)

    ax2 = axes[1]
    ax2.plot(pid_times, errors, "r-", linewidth=1.2, label="Error (e)")
    ax2.axhline(y=0, color="gray", linestyle="-", linewidth=0.8)
    ax2.set_ylabel("Error", fontsize=12)
    ax2.legend(loc="upper right", fontsize=9)
    ax2.set_title("PID Error Signal (Queue - Setpoint)", fontsize=13)
    ax2.grid(True, alpha=0.3)

    ax3 = axes[2]
    ax3.plot(pid_times, p_terms, "r-", linewidth=1.2, label="P term")
    ax3.plot(pid_times, i_terms, "g-", linewidth=1.2, label="I term")
    ax3.plot(pid_times, d_terms, "b-", linewidth=1.2, label="D term")
    ax3.plot(pid_times, pid_outputs, "k-", linewidth=2.0, label="PID Output")
    ax3.axhline(y=0, color="gray", linestyle="-", linewidth=0.8)
    ax3.set_ylabel("PID Terms / Output", fontsize=12)
    ax3.legend(loc="upper right", fontsize=9)
    ax3.set_title("PID Controller Output Decomposition", fontsize=13)
    ax3.grid(True, alpha=0.3)

    ax4 = axes[3]
    ax4_rate = ax4
    ax4_throttle = ax4.twinx()

    ln1 = ax4_rate.plot(rate_times, rates, "g-", linewidth=1.8, label="Produce Rate")
    ln2 = ax4_throttle.plot(rate_times, [r * 100 for r in throttling_ratios], "m-", linewidth=1.2, alpha=0.7, label="Throttling %")

    ax4_rate.set_xlabel("Time (s)", fontsize=12)
    ax4_rate.set_ylabel("Rate (msg/s)", fontsize=12, color="green")
    ax4_throttle.set_ylabel("Throttling (%)", fontsize=12, color="purple")
    ax4_rate.tick_params(axis="y", labelcolor="green")
    ax4_throttle.tick_params(axis="y", labelcolor="purple")

    lns = ln1 + ln2
    labs = [l.get_label() for l in lns]
    ax4_rate.legend(lns, labs, loc="upper right", fontsize=9)
    ax4_rate.set_title("Producer Rate & Throttling Ratio", fontsize=13)
    ax4_rate.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\n控制曲线图已保存: {output_path}")
    plt.close()


if __name__ == "__main__":
    random.seed(42)

    print("开始模拟: 生产速率 15 msg/s, 消费速率 8 msg/s")
    print("PID参数: Kp=0.06, Ki=0.008, Kd=0.03, Setpoint=25")
    print("最大队列: 60, 最小速率比: 15%, 调整间隔: 0.5s")
    print("模拟时长: 15 秒")
    print()

    metrics = run_simulation(
        produce_rate=15,
        consume_rate=8,
        pid_kp=0.06,
        pid_ki=0.008,
        pid_kd=0.03,
        pid_setpoint=25.0,
        max_queue_size=60,
        min_rate_ratio=0.15,
        rate_adjust_interval=0.5,
        run_duration=15,
    )

    print_results(metrics)
    plot_control_curves(metrics, output_path="pid_control_curves.png")
