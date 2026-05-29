import unittest
import os
import json
import tempfile
from datetime import datetime, timedelta
from unittest.mock import Mock

from alerting import (
    AlertRule,
    AlertRecord,
    SlidingWindowCounter,
    AlertManager,
    LoggingNotifier,
    ConsoleNotifier,
    create_default_alert_manager,
)
from storage import DatabaseLogStorage


class TestAlertRule(unittest.TestCase):
    def test_matches_status_code(self):
        rule = AlertRule(name="test_500", status_code=500)
        self.assertTrue(rule.matches({"response_status": 500}))
        self.assertFalse(rule.matches({"response_status": 200}))
        self.assertFalse(rule.matches({"response_status": 404}))

    def test_matches_status_code_range(self):
        rule = AlertRule(name="test_5xx", status_code_min=500, status_code_max=599)
        self.assertTrue(rule.matches({"response_status": 500}))
        self.assertTrue(rule.matches({"response_status": 503}))
        self.assertFalse(rule.matches({"response_status": 404}))
        self.assertFalse(rule.matches({"response_status": 200}))

    def test_matches_path_pattern(self):
        rule = AlertRule(name="test_path", path_pattern="/api/login")
        self.assertTrue(rule.matches({"response_status": 200, "request_path": "/api/login"}))
        self.assertTrue(rule.matches({"response_status": 200, "request_path": "/api/login/callback"}))
        self.assertFalse(rule.matches({"response_status": 200, "request_path": "/api/users"}))

    def test_matches_method(self):
        rule = AlertRule(name="test_post", method="POST", status_code=500)
        self.assertTrue(rule.matches({"response_status": 500, "request_method": "POST"}))
        self.assertFalse(rule.matches({"response_status": 500, "request_method": "GET"}))

    def test_matches_combined(self):
        rule = AlertRule(
            name="test_combined",
            status_code_min=500,
            status_code_max=599,
            path_pattern="/api",
            method="POST",
        )
        self.assertTrue(rule.matches({
            "response_status": 500,
            "request_path": "/api/upload",
            "request_method": "POST",
        }))
        self.assertFalse(rule.matches({
            "response_status": 500,
            "request_path": "/api/upload",
            "request_method": "GET",
        }))
        self.assertFalse(rule.matches({
            "response_status": 200,
            "request_path": "/api/upload",
            "request_method": "POST",
        }))

    def test_matches_none_status(self):
        rule = AlertRule(name="test", status_code=500)
        self.assertFalse(rule.matches({"response_status": None}))

    def test_cooldown(self):
        rule = AlertRule(name="test", status_code=500, cooldown_seconds=60)
        self.assertFalse(rule.is_in_cooldown())
        rule.last_triggered = datetime.now()
        self.assertTrue(rule.is_in_cooldown())

    def test_cooldown_expired(self):
        rule = AlertRule(name="test", status_code=500, cooldown_seconds=1)
        rule.last_triggered = datetime.now() - timedelta(seconds=2)
        self.assertFalse(rule.is_in_cooldown())

    def test_to_dict(self):
        rule = AlertRule(name="test_500", status_code=500, window_seconds=300, threshold=10)
        d = rule.to_dict()
        self.assertEqual(d["name"], "test_500")
        self.assertEqual(d["status_code"], 500)
        self.assertEqual(d["window_seconds"], 300)
        self.assertEqual(d["threshold"], 10)


class TestAlertRecord(unittest.TestCase):
    def test_to_dict(self):
        record = AlertRecord(
            rule_name="test_alert",
            message="Test alert message",
            count=15,
            window_seconds=300,
        )
        d = record.to_dict()
        self.assertEqual(d["rule_name"], "test_alert")
        self.assertEqual(d["message"], "Test alert message")
        self.assertEqual(d["count"], 15)
        self.assertEqual(d["window_seconds"], 300)
        self.assertIn("triggered_at", d)


class TestSlidingWindowCounter(unittest.TestCase):
    def test_count_within_window(self):
        counter = SlidingWindowCounter(window_seconds=60)
        counter.add()
        counter.add()
        counter.add()
        self.assertEqual(counter.count(), 3)

    def test_expired_events(self):
        counter = SlidingWindowCounter(window_seconds=1)
        counter.add(datetime.now() - timedelta(seconds=5))
        self.assertEqual(counter.count(), 0)

    def test_mixed_events(self):
        counter = SlidingWindowCounter(window_seconds=60)
        counter.add(datetime.now() - timedelta(seconds=120))
        counter.add()
        counter.add()
        self.assertEqual(counter.count(), 2)


class TestAlertManager(unittest.TestCase):
    def _create_log_entry(self, status_code=200, path="/api/test", method="GET"):
        return {
            "timestamp": datetime.now(),
            "ip_address": "127.0.0.1",
            "request_method": method,
            "request_path": path,
            "response_status": status_code,
            "duration_ms": 100.0,
        }

    def test_no_alert_below_threshold(self):
        rule = AlertRule(name="test_5xx", status_code_min=500, status_code_max=599, threshold=5)
        manager = AlertManager(rules=[rule])
        for _ in range(4):
            result = manager.check(self._create_log_entry(status_code=500))
            self.assertIsNone(result)

    def test_alert_at_threshold(self):
        rule = AlertRule(
            name="test_5xx",
            status_code_min=500,
            status_code_max=599,
            threshold=3,
            cooldown_seconds=0,
        )
        manager = AlertManager(rules=[rule])
        for _ in range(2):
            manager.check(self._create_log_entry(status_code=500))
        result = manager.check(self._create_log_entry(status_code=500))
        self.assertIsNotNone(result)
        self.assertEqual(result.rule_name, "test_5xx")
        self.assertGreaterEqual(result.count, 3)

    def test_cooldown_prevents_duplicate_alerts(self):
        rule = AlertRule(
            name="test_5xx",
            status_code_min=500,
            status_code_max=599,
            threshold=2,
            cooldown_seconds=60,
        )
        manager = AlertManager(rules=[rule])
        manager.check(self._create_log_entry(status_code=500))
        alert1 = manager.check(self._create_log_entry(status_code=500))
        self.assertIsNotNone(alert1)
        alert2 = manager.check(self._create_log_entry(status_code=500))
        self.assertIsNone(alert2)

    def test_no_alert_for_non_matching(self):
        rule = AlertRule(name="test_5xx", status_code_min=500, status_code_max=599, threshold=3)
        manager = AlertManager(rules=[rule])
        for _ in range(10):
            result = manager.check(self._create_log_entry(status_code=200))
            self.assertIsNone(result)

    def test_multiple_rules(self):
        rule_5xx = AlertRule(name="5xx", status_code_min=500, threshold=2, cooldown_seconds=0)
        rule_4xx = AlertRule(name="4xx", status_code_min=400, status_code_max=499, threshold=3, cooldown_seconds=0)
        manager = AlertManager(rules=[rule_5xx, rule_4xx])

        manager.check(self._create_log_entry(status_code=500))
        alert = manager.check(self._create_log_entry(status_code=500))
        self.assertIsNotNone(alert)
        self.assertEqual(alert.rule_name, "5xx")

    def test_alert_history(self):
        rule = AlertRule(
            name="test_5xx",
            status_code_min=500,
            threshold=1,
            cooldown_seconds=0,
        )
        manager = AlertManager(rules=[rule])
        manager.check(self._create_log_entry(status_code=500))
        history = manager.get_alert_history()
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0]["rule_name"], "test_5xx")

    def test_add_rule(self):
        manager = AlertManager()
        rule = AlertRule(name="new_rule", status_code=500, threshold=1)
        manager.add_rule(rule)
        self.assertEqual(len(manager.rules), 1)
        alert = manager.check(self._create_log_entry(status_code=500))
        self.assertIsNotNone(alert)

    def test_remove_rule(self):
        rule = AlertRule(name="removable", status_code=500, threshold=1)
        manager = AlertManager(rules=[rule])
        self.assertTrue(manager.remove_rule("removable"))
        self.assertEqual(len(manager.rules), 0)

    def test_remove_nonexistent_rule(self):
        manager = AlertManager()
        self.assertFalse(manager.remove_rule("nonexistent"))

    def test_notifier_called(self):
        notifier_mock = Mock()
        rule = AlertRule(name="test", status_code=500, threshold=1, cooldown_seconds=0)
        manager = AlertManager(rules=[rule], notifiers=[notifier_mock])
        manager.check(self._create_log_entry(status_code=500))
        self.assertEqual(notifier_mock.call_count, 1)

    def test_notifier_failure_handled(self):
        def bad_notifier(alert):
            raise RuntimeError("Notifier failed")

        rule = AlertRule(name="test", status_code=500, threshold=1, cooldown_seconds=0)
        manager = AlertManager(rules=[rule], notifiers=[bad_notifier])
        result = manager.check(self._create_log_entry(status_code=500))
        self.assertIsNotNone(result)

    def test_get_rule_status(self):
        rule = AlertRule(name="test", status_code=500, threshold=5)
        manager = AlertManager(rules=[rule])
        statuses = manager.get_rule_status()
        self.assertEqual(len(statuses), 1)
        self.assertEqual(statuses[0]["name"], "test")
        self.assertEqual(statuses[0]["current_count"], 0)
        self.assertFalse(statuses[0]["is_in_cooldown"])


class TestCreateDefaultAlertManager(unittest.TestCase):
    def _cleanup_notifiers(self, manager):
        for notifier in manager.notifiers:
            if hasattr(notifier, 'close'):
                notifier.close()

    def test_default_creation(self):
        manager = create_default_alert_manager()
        try:
            self.assertEqual(len(manager.rules), 2)
            self.assertEqual(manager.rules[0].name, "high_5xx_errors")
            self.assertEqual(manager.rules[1].name, "high_4xx_errors")
            self.assertEqual(manager.rules[0].threshold, 10)
            self.assertEqual(manager.rules[1].threshold, 20)
        finally:
            self._cleanup_notifiers(manager)

    def test_custom_threshold(self):
        manager = create_default_alert_manager(error_threshold=5)
        try:
            self.assertEqual(manager.rules[0].threshold, 5)
            self.assertEqual(manager.rules[1].threshold, 10)
        finally:
            self._cleanup_notifiers(manager)

    def test_with_alert_log_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = os.path.join(tmpdir, "alerts.log")
            manager = create_default_alert_manager(alert_log_file=log_file)
            try:
                self.assertTrue(len(manager.notifiers) >= 2)
            finally:
                self._cleanup_notifiers(manager)


class TestDatabaseQueryAndExport(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.tmpdir, "test.db")
        self.storage = DatabaseLogStorage(db_path=self.db_path)

    def tearDown(self):
        self.storage.close()
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _insert_log_entries(self, count=5, status_code=200, ip="127.0.0.1", path="/api/test"):
        for i in range(count):
            log_entry = {
                "timestamp": datetime.now() - timedelta(seconds=count - i),
                "ip_address": ip,
                "request_method": "GET",
                "request_path": path,
                "response_status": status_code,
                "duration_ms": 100.0 + i,
            }
            self.storage.save(log_entry)

    def test_query_by_status_code(self):
        self._insert_log_entries(3, status_code=200)
        self._insert_log_entries(2, status_code=500)

        results = self.storage.query(status_code=500)
        self.assertEqual(len(results), 2)
        for r in results:
            self.assertEqual(r["response_status"], 500)

    def test_query_by_status_range(self):
        self._insert_log_entries(2, status_code=200)
        self._insert_log_entries(3, status_code=404)
        self._insert_log_entries(1, status_code=500)

        results = self.storage.query(status_code_min=400, status_code_max=499)
        self.assertEqual(len(results), 3)

    def test_query_by_ip(self):
        self._insert_log_entries(3, ip="192.168.1.1")
        self._insert_log_entries(2, ip="10.0.0.1")

        results = self.storage.query(ip="192.168.1.1")
        self.assertEqual(len(results), 3)

    def test_query_by_time_range(self):
        now = datetime.now()
        old_entry = {
            "timestamp": now - timedelta(hours=2),
            "ip_address": "127.0.0.1",
            "request_method": "GET",
            "request_path": "/api/old",
            "response_status": 200,
            "duration_ms": 100.0,
        }
        self.storage.save(old_entry)
        self._insert_log_entries(3)

        results = self.storage.query(
            start_time=now - timedelta(minutes=5),
            end_time=now + timedelta(minutes=1),
        )
        self.assertEqual(len(results), 3)

    def test_query_by_method(self):
        self._insert_log_entries(3)
        post_entry = {
            "timestamp": datetime.now(),
            "ip_address": "127.0.0.1",
            "request_method": "POST",
            "request_path": "/api/users",
            "response_status": 201,
            "duration_ms": 150.0,
        }
        self.storage.save(post_entry)

        results = self.storage.query(method="POST")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["request_method"], "POST")

    def test_count_with_filters(self):
        self._insert_log_entries(5, status_code=200)
        self._insert_log_entries(3, status_code=500)

        total = self.storage.count()
        self.assertEqual(total, 8)

        error_count = self.storage.count(status_code_min=500)
        self.assertEqual(error_count, 3)

    def test_status_distribution(self):
        self._insert_log_entries(5, status_code=200)
        self._insert_log_entries(3, status_code=404)
        self._insert_log_entries(2, status_code=500)

        distribution = self.storage.get_status_distribution()
        dist_dict = {d["response_status"]: d["count"] for d in distribution}
        self.assertEqual(dist_dict[200], 5)
        self.assertEqual(dist_dict[404], 3)
        self.assertEqual(dist_dict[500], 2)

    def test_export_csv(self):
        self._insert_log_entries(3, status_code=200)
        self._insert_log_entries(2, status_code=500)

        csv_path = os.path.join(self.tmpdir, "export.csv")
        result_path = self.storage.export_csv(output_path=csv_path)

        self.assertTrue(os.path.exists(result_path))

        with open(result_path, "r", encoding="utf-8-sig") as f:
            lines = f.readlines()
            self.assertEqual(len(lines), 6)  # 1 header + 5 data
            header = lines[0].strip()
            self.assertIn("timestamp", header)
            self.assertIn("ip_address", header)
            self.assertIn("response_status", header)

    def test_export_csv_with_filters(self):
        self._insert_log_entries(3, status_code=200)
        self._insert_log_entries(2, status_code=500)

        csv_path = os.path.join(self.tmpdir, "errors.csv")
        self.storage.export_csv(output_path=csv_path, status_code_min=500)

        with open(csv_path, "r", encoding="utf-8-sig") as f:
            lines = f.readlines()
            self.assertEqual(len(lines), 3)  # 1 header + 2 data

    def test_query_with_offset(self):
        self._insert_log_entries(5, status_code=200)

        page1 = self.storage.query(limit=2, offset=0)
        page2 = self.storage.query(limit=2, offset=2)

        self.assertEqual(len(page1), 2)
        self.assertEqual(len(page2), 2)
        self.assertNotEqual(page1[0]["id"], page2[0]["id"])

    def test_query_combined_filters(self):
        self._insert_log_entries(3, status_code=500, ip="192.168.1.1", path="/api/error")
        self._insert_log_entries(2, status_code=500, ip="10.0.0.1", path="/api/other")
        self._insert_log_entries(2, status_code=200, ip="192.168.1.1", path="/api/ok")

        results = self.storage.query(
            status_code=500,
            ip="192.168.1.1",
        )
        self.assertEqual(len(results), 3)

        results = self.storage.query(
            status_code_min=400,
            path="/api/other",
        )
        self.assertEqual(len(results), 2)


class TestLoggingNotifier(unittest.TestCase):
    def test_logging_notifier(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = os.path.join(tmpdir, "alerts.log")
            notifier = LoggingNotifier(log_file=log_file)
            try:
                alert = AlertRecord(
                    rule_name="test",
                    message="Test alert",
                    count=5,
                    window_seconds=300,
                )
                notifier(alert)

                self.assertTrue(os.path.exists(log_file))
                with open(log_file, "r", encoding="utf-8") as f:
                    content = f.read()
                    self.assertIn("Test alert", content)
            finally:
                notifier.close()


class TestConsoleNotifier(unittest.TestCase):
    def test_console_notifier_no_exception(self):
        notifier = ConsoleNotifier()
        alert = AlertRecord(
            rule_name="test",
            message="Test alert",
            count=5,
            window_seconds=300,
        )
        notifier(alert)


if __name__ == "__main__":
    unittest.main(verbosity=2)
