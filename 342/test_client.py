import time
import threading
import requests
import json

BASE_URL = "http://localhost:8000"
WS_URL = "ws://localhost:8000/ws"


def format_eta(seconds):
    if seconds is None:
        return "--"
    seconds = int(seconds)
    if seconds < 60:
        return f"{seconds}s"
    minutes = seconds // 60
    seconds = seconds % 60
    return f"{minutes}m {seconds}s"


def create_task():
    response = requests.post(f"{BASE_URL}/tasks")
    response.raise_for_status()
    data = response.json()
    return data["task_id"]


def get_task_progress(task_id):
    response = requests.get(f"{BASE_URL}/tasks/{task_id}")
    response.raise_for_status()
    return response.json()


def poll_task_progress(task_id, interval=0.5):
    print(f"\n开始轮询任务: {task_id}")
    print("-" * 80)

    while True:
        task = get_task_progress(task_id)
        status = task["status"]
        progress = task["progress"]
        eta = task.get("eta_seconds")

        bar_length = 40
        filled = int(bar_length * progress // 100)
        bar = "█" * filled + "░" * (bar_length - filled)

        print(f"\r进度: [{bar}] {progress:3d}% | 状态: {status:4s} | 剩余: {format_eta(eta):6s}", end="", flush=True)

        if status in ["已完成", "失败"]:
            print("\n" + "-" * 80)
            if status == "已完成":
                print("✓ 任务完成!")
            else:
                print(f"✗ 任务失败: {task.get('error_message', '未知错误')}")
            break

        time.sleep(interval)


def list_all_tasks():
    response = requests.get(f"{BASE_URL}/tasks")
    response.raise_for_status()
    return response.json()


class WebSocketTaskMonitor:
    def __init__(self):
        import websocket
        self.websocket = websocket
        self.tasks = {}
        self.ws = None
        self.running = False

    def on_message(self, ws, message):
        data = json.loads(message)
        if data.get("type") == "task_update":
            task_id = data["task_id"]
            self.tasks[task_id] = {
                "status": data["status"],
                "progress": data["progress"],
                "eta": data.get("eta_seconds")
            }
            self.display_tasks()

    def on_error(self, ws, error):
        print(f"WebSocket 错误: {error}")

    def on_open(self, ws):
        print("WebSocket 连接成功!\n")

    def display_tasks(self):
        print("\033c", end="")
        print("=== WebSocket 实时任务监控 ===")
        print("-" * 80)
        print(f"{'任务ID':<12} {'状态':<8} {'进度':<6} {'剩余时间':<10}")
        print("-" * 80)

        for task_id, info in sorted(self.tasks.items()):
            bar_length = 20
            filled = int(bar_length * info["progress"] // 100)
            bar = "█" * filled + "░" * (bar_length - filled)
            progress_str = f"[{bar}] {info['progress']:3d}%"
            print(f"{task_id[:10]}...  {info['status']:<6} {progress_str:<26} {format_eta(info['eta']):<6}")

        print("-" * 80)
        print(f"当前监控任务数: {len(self.tasks)}")

    def run(self):
        self.running = True
        self.ws = self.websocket.WebSocketApp(
            WS_URL,
            on_open=self.on_open,
            on_message=self.on_message,
            on_error=self.on_error
        )
        self.ws.run_forever()

    def stop(self):
        self.running = False
        if self.ws:
            self.ws.close()


def run_websocket_monitor():
    monitor = WebSocketTaskMonitor()

    def input_thread():
        input("\n按回车键创建新任务...\n")
        while monitor.running:
            task_id = create_task()
            print(f"已创建任务: {task_id}")
            input("\n按回车键创建更多任务，或按 Ctrl+C 退出...\n")

    try:
        t = threading.Thread(target=input_thread, daemon=True)
        t.start()
        monitor.run()
    except KeyboardInterrupt:
        print("\n停止监控...")
        monitor.stop()


if __name__ == "__main__":
    import sys

    print("=== 长时间任务进度API测试 ===")
    print("1. 轮询模式测试")
    print("2. WebSocket实时监控模式")

    choice = input("\n请选择模式 (1/2, 默认1): ").strip() or "1"

    if choice == "2":
        print("\n启动WebSocket实时监控...")
        print("提示: 运行期间按回车键可创建新任务")
        try:
            import websocket
        except ImportError:
            print("安装 websocket-client...")
            import subprocess
            subprocess.check_call([sys.executable, "-m", "pip", "install", "websocket-client"])
        run_websocket_monitor()
    else:
        try:
            print("\n1. 创建新任务...")
            task_id = create_task()
            print(f"✓ 任务已创建, ID: {task_id}")

            print("\n2. 创建第二个任务...")
            task_id2 = create_task()
            print(f"✓ 任务已创建, ID: {task_id2}")

            print("\n3. 查看所有任务列表...")
            tasks = list_all_tasks()
            print(f"当前共有 {len(tasks)} 个任务")
            for t in tasks:
                eta = format_eta(t.get("eta_seconds"))
                print(f"  - {t['task_id'][:8]}... | {t['status']} | {t['progress']}% | 剩余: {eta}")

            poll_task_progress(task_id)

            print("\n4. 再次查看所有任务...")
            tasks = list_all_tasks()
            for t in tasks:
                eta = format_eta(t.get("eta_seconds"))
                print(f"  - {t['task_id'][:8]}... | {t['status']} | {t['progress']}% | 剩余: {eta}")

        except requests.exceptions.ConnectionError:
            print("错误: 无法连接到服务器, 请先运行 `python main.py` 启动服务")
        except Exception as e:
            print(f"错误: {e}")
