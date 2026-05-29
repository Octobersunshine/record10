import time
import random
import json
from api_monitor import ApiMonitor, monitor_decorator


def multi_window_example():
    print("=== 多时间窗口示例 ===\n")

    monitor = ApiMonitor(windows=["1s", "1m", "1h"])
    print(f"配置的窗口大小: {monitor.window_sizes} 秒")
    print()

    base_time = time.time()
    for i in range(100):
        path = random.choice(["/api/users", "/api/orders", "/api/products"])
        latency = random.uniform(10, 200)
        monitor.record(path, latency, current_time=base_time + i * 0.1)

    for window in ["1s", "1m", "1h"]:
        stats = monitor.get_all_stats(window=window)
        print(f"窗口 {window} ({stats['window_size']}秒):")
        print(f"  总调用: {stats['total']['count']}, QPS: {stats['total']['qps']:.4f}")
        print(f"  平均延迟: {stats['total']['avg_latency']:.2f}ms, P99: {stats['total']['p99_latency']:.2f}ms")
        print()


def realtime_topn_example():
    print("=== 实时Top N热点接口监控 ===\n")

    monitor = ApiMonitor(windows=["1m"])

    paths = ["/api/users", "/api/orders", "/api/products",
             "/api/cart", "/api/payment", "/api/search"]

    base_time = time.time()
    for i in range(500):
        path = random.choices(paths, weights=[30, 25, 20, 10, 10, 5])[0]
        latency = random.uniform(5, 150)
        monitor.record(path, latency, current_time=base_time + i * 0.01)

    print("启动自动刷新（每秒更新Top N缓存）...")
    monitor.start_auto_refresh(interval=1.0)
    time.sleep(0.5)

    print("\n按调用次数 Top 5:")
    top_count = monitor.get_top_n(5, by="count", window="1m", use_cache=True)
    for i, ep in enumerate(top_count, 1):
        print(f"  {i}. {ep['path']}: {ep['count']}次, QPS: {ep['qps']:.2f}")

    print("\n按P99延迟 Top 5:")
    top_p99 = monitor.get_top_n(5, by="p99_latency", window="1m", use_cache=True)
    for i, ep in enumerate(top_p99, 1):
        print(f"  {i}. {ep['path']}: P99 {ep['p99_latency']:.2f}ms")

    monitor.stop_auto_refresh()
    print("\n已停止自动刷新")
    print()


def visualization_example():
    print("=== ECharts可视化JSON示例 ===\n")

    monitor = ApiMonitor(windows=["1m"])

    for i in range(200):
        path = random.choice(["/api/users", "/api/orders", "/api/products",
                              "/api/cart", "/api/payment"])
        latency = random.uniform(10, 100)
        monitor.record(path, latency)
        time.sleep(0.01)

    print("1. 饼图数据 (API调用占比):")
    pie = monitor.get_echarts_pie(window="1m", n=5)
    print(f"   包含 {len(pie['series'][0]['data'])} 个数据项")
    print(f"   示例项: {json.dumps(pie['series'][0]['data'][0], ensure_ascii=False, indent=6)}")

    print("\n2. 折线图数据 (性能趋势):")
    line = monitor.get_echarts_line(window="1m", bucket_size=2, metrics=["qps", "avg_latency"])
    print(f"   时间序列长度: {len(line['xAxis']['data'])}")
    print(f"   包含指标: {[s['name'] for s in line['series']]}")

    print("\n3. 柱状图数据 (Top接口对比):")
    bar = monitor.get_echarts_bar(window="1m", n=5, by="count")
    print(f"   对比接口数: {len(bar['xAxis']['data'])}")
    print(f"   包含系列: {[s['name'] for s in bar['series']]}")

    print("\n4. 完整Dashboard数据:")
    dashboard = monitor.get_echarts_dashboard(window="1m", n=5)
    print(f"   时间戳: {dashboard['timestamp']}")
    print(f"   汇总: 总请求 {dashboard['summary']['total_requests']}, "
          f"平均QPS {dashboard['summary']['avg_qps']:.2f}")
    print(f"   包含图表: {list(dashboard['charts'].keys())}")
    print(f"   包含Top排名: {list(dashboard['top_endpoints'].keys())}")

    print("\n所有JSON可直接传入ECharts.setOption()使用")
    print()


def time_series_example():
    print("=== 时间序列数据示例 ===\n")

    monitor = ApiMonitor(windows=["1m"])
    base_time = time.time()

    for i in range(60):
        monitor.record("/api/users", random.uniform(10, 50),
                       current_time=base_time + i)
        if i % 2 == 0:
            monitor.record("/api/orders", random.uniform(20, 80),
                           current_time=base_time + i)

    ts = monitor.get_time_series("/api/users", window="1m", bucket_size=10)
    print(f"接口: {ts['path']}, 窗口: {ts['window_size']}s, 分桶: {ts['bucket_size']}s")
    print(f"时间点: {[time.strftime('%H:%M:%S', time.localtime(t)) for t in ts['timestamps']]}")
    print(f"QPS序列: {ts['qps']}")
    print(f"平均延迟: {ts['avg_latency']}")
    print()

    global_ts = monitor.get_time_series(window="1m", bucket_size=10)
    print(f"全局时间序列 - 总请求: {sum(global_ts['count'])}")
    print()


if __name__ == "__main__":
    multi_window_example()
    time.sleep(0.5)
    realtime_topn_example()
    time.sleep(0.5)
    visualization_example()
    time.sleep(0.5)
    time_series_example()
    print("=== 示例运行完成 ===")
