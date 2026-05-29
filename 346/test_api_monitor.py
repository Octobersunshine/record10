import unittest
import time
import threading
import json
from api_monitor import ApiMonitor, SlidingWindow, AtomicCounter, ReadWriteLock, PathMetrics, monitor_decorator, WINDOW_PRESETS


class TestAtomicCounter(unittest.TestCase):
    def test_increment(self):
        counter = AtomicCounter(0)
        counter.increment(1)
        self.assertEqual(counter.get(), 1)

    def test_decrement(self):
        counter = AtomicCounter(10)
        counter.decrement(3)
        self.assertEqual(counter.get(), 7)

    def test_get_and_reset(self):
        counter = AtomicCounter(42)
        val = counter.get_and_reset()
        self.assertEqual(val, 42)
        self.assertEqual(counter.get(), 0)

    def test_concurrent_increment(self):
        counter = AtomicCounter(0)
        num_threads = 20
        increments_per_thread = 10000

        def worker():
            for _ in range(increments_per_thread):
                counter.increment(1)

        threads = []
        for _ in range(num_threads):
            t = threading.Thread(target=worker)
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        self.assertEqual(counter.get(), num_threads * increments_per_thread)

    def test_concurrent_mixed_ops(self):
        counter = AtomicCounter(0)
        num_threads = 10
        ops_per_thread = 5000
        errors = []

        def increment_worker():
            try:
                for _ in range(ops_per_thread):
                    counter.increment(1)
            except Exception as e:
                errors.append(e)

        def decrement_worker():
            try:
                for _ in range(ops_per_thread):
                    counter.decrement(1)
            except Exception as e:
                errors.append(e)

        threads = []
        for _ in range(num_threads):
            threads.append(threading.Thread(target=increment_worker))
            threads.append(threading.Thread(target=decrement_worker))

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(len(errors), 0)
        self.assertEqual(counter.get(), 0)

    def test_float_precision(self):
        counter = AtomicCounter(0.0)
        for _ in range(1000):
            counter.increment(0.1)
        val = counter.get()
        self.assertAlmostEqual(val, 100.0, places=2)


class TestReadWriteLock(unittest.TestCase):
    def test_concurrent_reads(self):
        rwlock = ReadWriteLock()
        max_concurrent = AtomicCounter(0)
        active_readers = AtomicCounter(0)
        iterations = 100

        def reader():
            for _ in range(iterations):
                with rwlock.read_lock():
                    active_readers.increment(1)
                    current = active_readers.get()
                    while True:
                        old = max_concurrent.get()
                        if current <= old:
                            break
                        if max_concurrent.increment(current - old) >= current:
                            break
                    time.sleep(0.0001)
                    active_readers.decrement(1)

        threads = [threading.Thread(target=reader) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertGreater(max_concurrent.get(), 1)

    def test_write_exclusivity(self):
        rwlock = ReadWriteLock()
        active_writers = AtomicCounter(0)
        errors = []

        def writer():
            for _ in range(50):
                with rwlock.write_lock():
                    active_writers.increment(1)
                    if active_writers.get() > 1:
                        errors.append("Multiple writers active!")
                    time.sleep(0.0001)
                    active_writers.decrement(1)

        threads = [threading.Thread(target=writer) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(len(errors), 0)


class TestSlidingWindow(unittest.TestCase):
    def test_add_record(self):
        window = SlidingWindow(window_size=60)
        window.add_record(100.0)
        self.assertEqual(window.get_records_count(), 1)

    def test_cleanup_expired_records(self):
        window = SlidingWindow(window_size=5)
        base_time = 1000.0

        for i in range(10):
            window.add_record(50.0, current_time=base_time + i)

        self.assertEqual(window.get_records_count(current_time=base_time + 10), 5)
        self.assertEqual(window.get_records_count(current_time=base_time + 20), 0)

    def test_get_stats_empty(self):
        window = SlidingWindow(window_size=60)
        stats = window.get_stats()
        self.assertEqual(stats["count"], 0)
        self.assertEqual(stats["qps"], 0.0)
        self.assertEqual(stats["avg_latency"], 0.0)
        self.assertEqual(stats["p99_latency"], 0.0)

    def test_get_stats_with_data(self):
        window = SlidingWindow(window_size=60)
        base_time = 1000.0

        latencies = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        for i, lat in enumerate(latencies):
            window.add_record(lat, current_time=base_time + i)

        stats = window.get_stats(current_time=base_time + 60)
        self.assertEqual(stats["count"], 10)
        self.assertAlmostEqual(stats["avg_latency"], 55.0, places=2)
        self.assertEqual(stats["p99_latency"], 90)

        expected_qps = 10 / 60
        self.assertAlmostEqual(stats["qps"], expected_qps, places=4)

    def test_p99_calculation(self):
        window = SlidingWindow(window_size=60)
        base_time = 1000.0

        for i in range(100):
            window.add_record(i + 1, current_time=base_time + i * 0.1)

        stats = window.get_stats(current_time=base_time + 60)
        self.assertEqual(stats["p99_latency"], 99)

    def test_get_time_series(self):
        window = SlidingWindow(window_size=10)
        base_time = 1000.0

        for i in range(10):
            window.add_record(10.0 * (i + 1), current_time=base_time + i)

        ts = window.get_time_series(bucket_size=2, current_time=base_time + 10)
        self.assertIn("timestamps", ts)
        self.assertIn("count", ts)
        self.assertIn("qps", ts)
        self.assertIn("avg_latency", ts)
        self.assertEqual(len(ts["timestamps"]), 5)
        self.assertEqual(len(ts["count"]), 5)
        self.assertEqual(sum(ts["count"]), 10)

    def test_atomic_counters_consistent_after_cleanup(self):
        window = SlidingWindow(window_size=5)
        base_time = 1000.0

        for i in range(10):
            window.add_record(10.0 * (i + 1), current_time=base_time + i)

        stats = window.get_stats(current_time=base_time + 10)
        self.assertEqual(stats["count"], 5)
        expected_total = sum(10.0 * (i + 1) for i in range(5, 10))
        self.assertAlmostEqual(stats["avg_latency"], expected_total / 5, places=2)

    def test_concurrent_add_and_cleanup(self):
        window = SlidingWindow(window_size=10)
        base_time = 1000.0
        num_threads = 10
        records_per_thread = 1000
        errors = []

        def worker(thread_id):
            try:
                for i in range(records_per_thread):
                    latency = (thread_id * 100 + i) % 200 + 1
                    window.add_record(latency, current_time=base_time + 5)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(tid,)) for tid in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(len(errors), 0)
        stats = window.get_stats(current_time=base_time + 10)
        expected_count = num_threads * records_per_thread
        self.assertEqual(stats["count"], expected_count)


class TestWindowPresets(unittest.TestCase):
    def test_presets_values(self):
        self.assertEqual(WINDOW_PRESETS["1s"], 1)
        self.assertEqual(WINDOW_PRESETS["1m"], 60)
        self.assertEqual(WINDOW_PRESETS["1h"], 3600)
        self.assertEqual(WINDOW_PRESETS["24h"], 86400)

    def test_all_presets_positive(self):
        for name, value in WINDOW_PRESETS.items():
            self.assertGreater(value, 0)


class TestPathMetrics(unittest.TestCase):
    def test_multi_window_support(self):
        pm = PathMetrics("/api/test", [10, 60, 3600])
        base_time = 1000.0

        for i in range(100):
            pm.add_record(50.0, current_time=base_time + i * 0.5)

        stats_10s = pm.get_stats(10, current_time=base_time + 50)
        stats_60s = pm.get_stats(60, current_time=base_time + 50)
        stats_1h = pm.get_stats(3600, current_time=base_time + 50)

        self.assertEqual(stats_10s["window_size"], 10)
        self.assertEqual(stats_60s["window_size"], 60)
        self.assertEqual(stats_1h["window_size"], 3600)

        self.assertGreater(stats_60s["count"], stats_10s["count"])
        self.assertEqual(stats_1h["count"], stats_60s["count"])

    def test_unknown_window_size(self):
        pm = PathMetrics("/api/test", [60])
        stats = pm.get_stats(120)
        self.assertEqual(stats["count"], 0)
        self.assertEqual(stats["window_size"], 120)


class TestApiMonitorMultiWindow(unittest.TestCase):
    def test_init_with_presets(self):
        monitor = ApiMonitor(windows=["1s", "1m", "1h"])
        self.assertIn(1, monitor.window_sizes)
        self.assertIn(60, monitor.window_sizes)
        self.assertIn(3600, monitor.window_sizes)

    def test_init_with_integers(self):
        monitor = ApiMonitor(windows=[10, 120, 7200])
        self.assertIn(10, monitor.window_sizes)
        self.assertIn(120, monitor.window_sizes)
        self.assertIn(7200, monitor.window_sizes)

    def test_init_with_mixed(self):
        monitor = ApiMonitor(windows=[30, "5m"])
        self.assertIn(30, monitor.window_sizes)
        self.assertIn(300, monitor.window_sizes)

    def test_invalid_preset(self):
        with self.assertRaises(ValueError):
            ApiMonitor(windows=["invalid"])

    def test_window_not_configured(self):
        monitor = ApiMonitor(windows=[60])
        with self.assertRaises(ValueError):
            monitor.get_stats(window=120)

    def test_record_updates_all_windows(self):
        monitor = ApiMonitor(windows=[5, 10, 20])
        base_time = 1000.0

        for i in range(15):
            monitor.record("/api/test", 10.0, current_time=base_time + i)

        stats_5s = monitor.get_stats("/api/test", window=5, current_time=base_time + 15)
        stats_10s = monitor.get_stats("/api/test", window=10, current_time=base_time + 15)
        stats_20s = monitor.get_stats("/api/test", window=20, current_time=base_time + 15)

        self.assertEqual(stats_5s["count"], 5)
        self.assertEqual(stats_10s["count"], 10)
        self.assertEqual(stats_20s["count"], 15)

    def test_get_all_stats_with_window(self):
        monitor = ApiMonitor(windows=[5, 10])
        base_time = 1000.0

        for i in range(10):
            monitor.record("/api/a", 10.0, current_time=base_time + i)
            monitor.record("/api/b", 20.0, current_time=base_time + i)

        stats_5s = monitor.get_all_stats(window=5, current_time=base_time + 10)
        stats_10s = monitor.get_all_stats(window=10, current_time=base_time + 10)

        self.assertEqual(stats_5s["window_size"], 5)
        self.assertEqual(stats_10s["window_size"], 10)
        self.assertEqual(stats_5s["total"]["count"], 10)
        self.assertEqual(stats_10s["total"]["count"], 20)


class TestApiMonitorTopN(unittest.TestCase):
    def test_get_top_n_by_count(self):
        monitor = ApiMonitor(windows=[60])
        base_time = 1000.0

        for i in range(10):
            monitor.record("/api/a", 10.0, current_time=base_time + i)
        for i in range(5):
            monitor.record("/api/b", 20.0, current_time=base_time + i)
        for i in range(15):
            monitor.record("/api/c", 30.0, current_time=base_time + i)

        top2 = monitor.get_top_n(2, by="count", current_time=base_time + 60, use_cache=False)
        self.assertEqual(len(top2), 2)
        self.assertEqual(top2[0]["path"], "/api/c")
        self.assertEqual(top2[0]["count"], 15)
        self.assertEqual(top2[1]["path"], "/api/a")
        self.assertEqual(top2[1]["count"], 10)

    def test_get_top_n_by_qps(self):
        monitor = ApiMonitor(windows=[10])
        base_time = 1000.0

        for i in range(5):
            monitor.record("/api/high", 10.0, current_time=base_time + i)
        for i in range(2):
            monitor.record("/api/low", 10.0, current_time=base_time + i)

        top = monitor.get_top_n(5, by="qps", current_time=base_time + 10, use_cache=False)
        self.assertGreater(top[0]["qps"], top[1]["qps"])

    def test_topn_cache(self):
        monitor = ApiMonitor(windows=[60])
        base_time = 1000.0

        monitor.record("/api/a", 10.0, current_time=base_time)
        top1 = monitor.get_top_n(1, by="count", current_time=base_time + 60, use_cache=True)
        self.assertEqual(top1[0]["count"], 1)

        monitor.record("/api/b", 10.0, current_time=base_time + 1)
        top2 = monitor.get_top_n(1, by="count", current_time=base_time + 60, use_cache=True)
        self.assertEqual(top2[0]["count"], 1)

        top3 = monitor.get_top_n(1, by="count", current_time=base_time + 60, use_cache=False)
        self.assertEqual(top3[0]["count"], 1)

    def test_auto_refresh(self):
        monitor = ApiMonitor(windows=[60])
        monitor.start_auto_refresh(interval=0.1)
        time.sleep(0.3)

        self.assertTrue(monitor._auto_refresh)
        self.assertIsNotNone(monitor._refresh_thread)

        monitor.stop_auto_refresh()
        self.assertFalse(monitor._auto_refresh)
        self.assertIsNone(monitor._refresh_thread)

    def test_topn_with_different_windows(self):
        monitor = ApiMonitor(windows=[5, 10])
        base_time = 1000.0

        for i in range(8):
            monitor.record("/api/a", 10.0, current_time=base_time + i)
        for i in range(12):
            monitor.record("/api/b", 10.0, current_time=base_time + i)

        top_5s = monitor.get_top_n(5, window=5, current_time=base_time + 9, use_cache=False)
        top_10s = monitor.get_top_n(5, window=10, current_time=base_time + 9, use_cache=False)

        a_5s = next(ep for ep in top_5s if ep["path"] == "/api/a")
        b_5s = next(ep for ep in top_5s if ep["path"] == "/api/b")
        self.assertEqual(a_5s["count"], 4)
        self.assertEqual(b_5s["count"], 5)

        a_10s = next(ep for ep in top_10s if ep["path"] == "/api/a")
        b_10s = next(ep for ep in top_10s if ep["path"] == "/api/b")
        self.assertEqual(a_10s["count"], 8)
        self.assertEqual(b_10s["count"], 9)


class TestApiMonitorTimeSeries(unittest.TestCase):
    def test_global_time_series(self):
        monitor = ApiMonitor(windows=[10])
        base_time = 1000.0

        for i in range(10):
            monitor.record("/api/test", 10.0, current_time=base_time + i)

        ts = monitor.get_time_series(window=10, bucket_size=2, current_time=base_time + 10)
        self.assertEqual(ts["path"], "global")
        self.assertEqual(ts["window_size"], 10)
        self.assertEqual(ts["bucket_size"], 2)
        self.assertEqual(len(ts["timestamps"]), 5)
        self.assertEqual(sum(ts["count"]), 10)

    def test_path_time_series(self):
        monitor = ApiMonitor(windows=[10])
        base_time = 1000.0

        for i in range(10):
            monitor.record("/api/a", 10.0, current_time=base_time + i)
            monitor.record("/api/b", 20.0, current_time=base_time + i)

        ts_a = monitor.get_time_series("/api/a", window=10, bucket_size=2, current_time=base_time + 10)
        ts_b = monitor.get_time_series("/api/b", window=10, bucket_size=2, current_time=base_time + 10)

        self.assertEqual(ts_a["path"], "/api/a")
        self.assertEqual(ts_b["path"], "/api/b")
        self.assertEqual(sum(ts_a["count"]), 10)
        self.assertEqual(sum(ts_b["count"]), 10)

    def test_nonexistent_path_time_series(self):
        monitor = ApiMonitor(windows=[60])
        ts = monitor.get_time_series("/api/nonexistent")
        self.assertEqual(ts["path"], "/api/nonexistent")
        self.assertEqual(len(ts["timestamps"]), 0)


class TestEChartsVisualization(unittest.TestCase):
    def setUp(self):
        self.monitor = ApiMonitor(windows=[60])
        self.base_time = 1000.0

        for i in range(50):
            self.monitor.record("/api/users", 20.0, current_time=self.base_time + i * 0.5)
        for i in range(30):
            self.monitor.record("/api/orders", 50.0, current_time=self.base_time + i * 0.5)
        for i in range(20):
            self.monitor.record("/api/products", 30.0, current_time=self.base_time + i * 0.5)

    def test_pie_chart_structure(self):
        pie = self.monitor.get_echarts_pie(window=60, n=3, current_time=self.base_time + 60)

        self.assertIn("title", pie)
        self.assertIn("tooltip", pie)
        self.assertIn("legend", pie)
        self.assertIn("series", pie)
        self.assertGreater(len(pie["series"]), 0)
        self.assertEqual(pie["series"][0]["type"], "pie")
        self.assertIn("data", pie["series"][0])
        self.assertEqual(len(pie["series"][0]["data"]), 3)

    def test_pie_chart_data(self):
        pie = self.monitor.get_echarts_pie(window=60, n=3, current_time=self.base_time + 60)
        data = pie["series"][0]["data"]

        paths = [d["name"] for d in data]
        self.assertIn("/api/users", paths)
        self.assertIn("/api/orders", paths)
        self.assertIn("/api/products", paths)

        users_item = next(d for d in data if d["name"] == "/api/users")
        self.assertEqual(users_item["value"], 50)

    def test_pie_chart_with_others(self):
        for i in range(5):
            self.monitor.record(f"/api/other_{i}", 10.0, current_time=self.base_time)

        pie = self.monitor.get_echarts_pie(window=60, n=3, current_time=self.base_time + 60)
        data = pie["series"][0]["data"]
        names = [d["name"] for d in data]
        self.assertIn("others", names)

    def test_line_chart_structure(self):
        line = self.monitor.get_echarts_line(
            path="/api/users",
            window=60,
            bucket_size=10,
            metrics=["qps", "avg_latency"],
            current_time=self.base_time + 60
        )

        self.assertIn("title", line)
        self.assertIn("xAxis", line)
        self.assertIn("yAxis", line)
        self.assertIn("series", line)
        self.assertIn("legend", line)
        self.assertEqual(len(line["series"]), 2)
        self.assertEqual(len(line["yAxis"]), 2)

    def test_line_chart_global(self):
        line = self.monitor.get_echarts_line(
            window=60,
            bucket_size=10,
            metrics=["count", "qps"],
            current_time=self.base_time + 60
        )

        self.assertIn("title", line)
        self.assertEqual(len(line["series"]), 2)
        series_names = [s["name"] for s in line["series"]]
        self.assertIn("调用次数", series_names)
        self.assertIn("QPS", series_names)

    def test_bar_chart_structure(self):
        bar = self.monitor.get_echarts_bar(
            window=60,
            n=5,
            by="count",
            current_time=self.base_time + 60
        )

        self.assertIn("title", bar)
        self.assertIn("xAxis", bar)
        self.assertIn("yAxis", bar)
        self.assertIn("series", bar)
        self.assertEqual(len(bar["series"]), 3)

        series_types = [s["type"] for s in bar["series"]]
        self.assertIn("bar", series_types)
        self.assertIn("line", series_types)

    def test_bar_chart_sorting(self):
        bar = self.monitor.get_echarts_bar(
            window=60,
            n=5,
            by="count",
            current_time=self.base_time + 60
        )

        x_data = bar["xAxis"]["data"]
        bar_series = next(s for s in bar["series"] if s["type"] == "bar")
        bar_data = bar_series["data"]

        self.assertEqual(x_data[0], "/api/users")
        self.assertEqual(bar_data[0], 50)

    def test_bar_chart_by_p99(self):
        bar = self.monitor.get_echarts_bar(
            window=60,
            n=5,
            by="p99_latency",
            current_time=self.base_time + 60
        )

        x_data = bar["xAxis"]["data"]
        bar_series = next(s for s in bar["series"] if s["type"] == "bar")

        self.assertIn("/api/orders", x_data)

    def test_dashboard_structure(self):
        dashboard = self.monitor.get_echarts_dashboard(
            window=60,
            n=10,
            current_time=self.base_time + 60
        )

        self.assertIn("timestamp", dashboard)
        self.assertIn("window_size", dashboard)
        self.assertIn("summary", dashboard)
        self.assertIn("charts", dashboard)
        self.assertIn("top_endpoints", dashboard)
        self.assertIn("time_series_global", dashboard)

        self.assertIn("total_requests", dashboard["summary"])
        self.assertIn("avg_qps", dashboard["summary"])
        self.assertIn("p99_latency", dashboard["summary"])

        self.assertIn("pie", dashboard["charts"])
        self.assertIn("bar_count", dashboard["charts"])
        self.assertIn("bar_p99", dashboard["charts"])

        self.assertIn("by_count", dashboard["top_endpoints"])
        self.assertIn("by_qps", dashboard["top_endpoints"])
        self.assertIn("by_p99", dashboard["top_endpoints"])

    def test_dashboard_values(self):
        dashboard = self.monitor.get_echarts_dashboard(
            window=60,
            n=10,
            current_time=self.base_time + 60
        )

        self.assertEqual(dashboard["summary"]["total_requests"], 100)
        self.assertEqual(len(dashboard["top_endpoints"]["by_count"]), 3)
        self.assertEqual(dashboard["top_endpoints"]["by_count"][0]["path"], "/api/users")

    def test_chart_json_serializable(self):
        pie = self.monitor.get_echarts_pie(window=60, current_time=self.base_time + 60)
        line = self.monitor.get_echarts_line(window=60, current_time=self.base_time + 60)
        bar = self.monitor.get_echarts_bar(window=60, current_time=self.base_time + 60)
        dashboard = self.monitor.get_echarts_dashboard(window=60, current_time=self.base_time + 60)

        for chart in [pie, line, bar, dashboard]:
            json_str = json.dumps(chart)
            parsed = json.loads(json_str)
            self.assertEqual(parsed, chart)


class TestApiMonitor(unittest.TestCase):
    def test_record_single_path(self):
        monitor = ApiMonitor(windows=[60])
        monitor.record("/api/test", 50.0)

        stats = monitor.get_stats("/api/test")
        self.assertEqual(stats["path"], "/api/test")
        self.assertEqual(stats["count"], 1)
        self.assertEqual(stats["avg_latency"], 50.0)

    def test_record_multiple_paths(self):
        monitor = ApiMonitor(windows=[60])
        base_time = 1000.0

        for i in range(10):
            monitor.record("/api/a", 10.0, current_time=base_time + i)
            monitor.record("/api/b", 20.0, current_time=base_time + i)

        all_stats = monitor.get_all_stats(current_time=base_time + 60)
        self.assertEqual(all_stats["total"]["path_count"], 2)
        self.assertEqual(all_stats["total"]["count"], 20)

        path_a = next(ep for ep in all_stats["endpoints"] if ep["path"] == "/api/a")
        path_b = next(ep for ep in all_stats["endpoints"] if ep["path"] == "/api/b")

        self.assertEqual(path_a["count"], 10)
        self.assertEqual(path_a["avg_latency"], 10.0)
        self.assertEqual(path_b["count"], 10)
        self.assertEqual(path_b["avg_latency"], 20.0)

    def test_get_stats_nonexistent_path(self):
        monitor = ApiMonitor(windows=[60])
        stats = monitor.get_stats("/api/nonexistent")
        self.assertEqual(stats["count"], 0)
        self.assertEqual(stats["qps"], 0.0)

    def test_get_top_n(self):
        monitor = ApiMonitor(windows=[60])
        base_time = 1000.0

        paths = ["/api/1", "/api/2", "/api/3", "/api/4", "/api/5"]
        for i, path in enumerate(paths):
            for j in range(i + 1):
                monitor.record(path, 10.0, current_time=base_time + j)

        top3 = monitor.get_top_n(3, by="count", current_time=base_time + 60, use_cache=False)
        self.assertEqual(len(top3), 3)
        self.assertEqual(top3[0]["path"], "/api/5")
        self.assertEqual(top3[0]["count"], 5)
        self.assertEqual(top3[1]["path"], "/api/4")
        self.assertEqual(top3[1]["count"], 4)
        self.assertEqual(top3[2]["path"], "/api/3")
        self.assertEqual(top3[2]["count"], 3)

    def test_sliding_window_expiration(self):
        monitor = ApiMonitor(windows=[5])
        base_time = 1000.0

        for i in range(10):
            monitor.record("/api/test", 10.0, current_time=base_time + i)

        stats = monitor.get_stats("/api/test", window=5, current_time=base_time + 7)
        self.assertEqual(stats["count"], 5)

        stats = monitor.get_stats("/api/test", window=5, current_time=base_time + 15)
        self.assertEqual(stats["count"], 0)


class TestMonitorDecorator(unittest.TestCase):
    def test_decorator_records_success(self):
        monitor = ApiMonitor(windows=[60])

        @monitor_decorator(monitor)
        def test_func(path="/api/test"):
            return "success"

        result = test_func()
        self.assertEqual(result, "success")

        stats = monitor.get_stats("/api/test")
        self.assertEqual(stats["count"], 1)
        self.assertGreater(stats["avg_latency"], 0)

    def test_decorator_records_exception(self):
        monitor = ApiMonitor(windows=[60])

        @monitor_decorator(monitor)
        def error_func(path="/api/error"):
            raise ValueError("test error")

        with self.assertRaises(ValueError):
            error_func()

        stats = monitor.get_stats("/api/error")
        self.assertEqual(stats["count"], 1)
        self.assertGreater(stats["avg_latency"], 0)


class TestThreadSafety(unittest.TestCase):
    def test_concurrent_records(self):
        monitor = ApiMonitor(windows=[60])
        base_time = 1000.0
        num_threads = 10
        records_per_thread = 1000

        def record_worker(thread_id):
            for i in range(records_per_thread):
                path = f"/api/thread_{thread_id % 3}"
                monitor.record(path, 10.0, current_time=base_time + i * 0.001)

        threads = []
        for i in range(num_threads):
            t = threading.Thread(target=record_worker, args=(i,))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        stats = monitor.get_all_stats(current_time=base_time + 60)
        expected = num_threads * records_per_thread
        self.assertEqual(stats["total"]["count"], expected)

    def test_concurrent_read_write(self):
        monitor = ApiMonitor(windows=[60])
        base_time = 1000.0
        num_writers = 5
        num_readers = 5
        records_per_writer = 500
        errors = []

        def writer(thread_id):
            try:
                for i in range(records_per_writer):
                    path = f"/api/path_{thread_id}"
                    monitor.record(path, 10.0 + i, current_time=base_time + 5)
            except Exception as e:
                errors.append(("writer", e))

        def reader(thread_id):
            try:
                for _ in range(records_per_writer):
                    stats = monitor.get_all_stats(current_time=base_time + 60)
                    self.assertIn("total", stats)
                    self.assertIn("endpoints", stats)
                    self.assertGreaterEqual(stats["total"]["path_count"], 0)
            except Exception as e:
                errors.append(("reader", e))

        threads = []
        for i in range(num_writers):
            threads.append(threading.Thread(target=writer, args=(i,)))
        for i in range(num_readers):
            threads.append(threading.Thread(target=reader, args=(i,)))

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(len(errors), 0, f"Errors during concurrent read/write: {errors}")

    def test_high_concurrency_stress(self):
        monitor = ApiMonitor(windows=[60])
        base_time = 1000.0
        num_threads = 50
        records_per_thread = 200

        def worker(thread_id):
            for i in range(records_per_thread):
                path = f"/api/ep_{thread_id % 5}"
                latency = (thread_id + i) % 500 + 1
                monitor.record(path, latency, current_time=base_time + 5)

        threads = [threading.Thread(target=worker, args=(tid,)) for tid in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        stats = monitor.get_all_stats(current_time=base_time + 60)
        expected = num_threads * records_per_thread
        self.assertEqual(stats["total"]["count"], expected)

        per_path_total = sum(ep["count"] for ep in stats["endpoints"])
        self.assertEqual(per_path_total, expected)

    def test_no_count_loss_under_contention(self):
        monitor = ApiMonitor(windows=[60])
        base_time = 1000.0
        num_threads = 20
        total_records = 10000
        records_per_thread = total_records // num_threads

        barrier = threading.Barrier(num_threads)

        def worker():
            barrier.wait()
            for i in range(records_per_thread):
                monitor.record("/api/stress", 1.0, current_time=base_time + 5)

        threads = [threading.Thread(target=worker) for _ in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        stats = monitor.get_stats("/api/stress", current_time=base_time + 60)
        self.assertEqual(stats["count"], total_records)
        self.assertEqual(stats["avg_latency"], 1.0)


if __name__ == "__main__":
    unittest.main()
