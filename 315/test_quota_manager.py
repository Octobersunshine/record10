import unittest
from datetime import datetime, timedelta, timezone
from quota_manager import (
    HierarchicalQuotaManager, QuotaPeriod, ResetStrategy, QuotaLevel,
    QuotaResult, TrendPrediction, BorrowRecord, QuotaConfig, QuotaUsage,
    _ensure_timezone, _to_timezone, ZONEINFO_AVAILABLE,
)


class TestQuotaLevel(unittest.TestCase):
    def test_quota_level_values(self):
        self.assertEqual(QuotaLevel.GLOBAL.value, "global")
        self.assertEqual(QuotaLevel.TENANT.value, "tenant")
        self.assertEqual(QuotaLevel.USER.value, "user")


class TestQuotaResultExtended(unittest.TestCase):
    def test_usage_rate(self):
        result = QuotaResult(
            allowed=True, remaining=5, limit=10,
            reset_time=datetime(2024, 1, 16, 0, 0, tzinfo=timezone.utc),
            used=5
        )
        self.assertEqual(result.usage_rate, 0.5)

    def test_usage_rate_zero_limit(self):
        result = QuotaResult(
            allowed=True, remaining=0, limit=0,
            reset_time=datetime(2024, 1, 16, 0, 0, tzinfo=timezone.utc),
            used=0
        )
        self.assertEqual(result.usage_rate, 0.0)

    def test_effective_remaining(self):
        result = QuotaResult(
            allowed=True, remaining=3, limit=10,
            reset_time=datetime(2024, 1, 16, 0, 0, tzinfo=timezone.utc),
            used=7, borrowed=2
        )
        self.assertEqual(result.effective_remaining, 1)

    def test_to_dict_with_borrow(self):
        result = QuotaResult(
            allowed=True, remaining=3, limit=10,
            reset_time=datetime(2024, 1, 16, 0, 0, tzinfo=timezone.utc),
            used=7, borrowed=2, max_borrow=5
        )
        d = result.to_dict()
        self.assertEqual(d["borrowed"], 2)
        self.assertEqual(d["max_borrow"], 5)
        self.assertEqual(d["usage_rate"], 0.7)
        self.assertEqual(d["effective_remaining"], 1)

    def test_level_results_in_dict(self):
        inner = QuotaResult(
            allowed=True, remaining=80, limit=100,
            reset_time=datetime(2024, 1, 16, 0, 0, tzinfo=timezone.utc),
            used=20
        )
        result = QuotaResult(
            allowed=True, remaining=5, limit=10,
            reset_time=datetime(2024, 1, 16, 0, 0, tzinfo=timezone.utc),
            used=5, level_results={"global": inner}
        )
        d = result.to_dict()
        self.assertIn("level_results", d)
        self.assertIn("global", d["level_results"])


class TestHierarchicalQuota(unittest.TestCase):
    def setUp(self):
        self.manager = HierarchicalQuotaManager()

    def test_create_hierarchical_quotas(self):
        self.manager.create_quota(
            "global_api", 1000, QuotaPeriod.DAILY,
            ResetStrategy.FIXED_TIME,
            level=QuotaLevel.GLOBAL
        )
        self.manager.create_quota(
            "tenant_a", 500, QuotaPeriod.DAILY,
            ResetStrategy.FIXED_TIME,
            level=QuotaLevel.TENANT,
            parent_id="global_api"
        )
        self.manager.create_quota(
            "user_1", 100, QuotaPeriod.DAILY,
            ResetStrategy.FIXED_TIME,
            level=QuotaLevel.USER,
            parent_id="tenant_a"
        )
        children = self.manager.get_children("global_api")
        self.assertIn("tenant_a", children)

    def test_consume_propagates_to_parent(self):
        self.manager.create_quota(
            "global_api", 1000, QuotaPeriod.DAILY,
            ResetStrategy.FIXED_TIME,
            level=QuotaLevel.GLOBAL
        )
        self.manager.create_quota(
            "tenant_a", 500, QuotaPeriod.DAILY,
            ResetStrategy.FIXED_TIME,
            level=QuotaLevel.TENANT,
            parent_id="global_api"
        )
        self.manager.create_quota(
            "user_1", 100, QuotaPeriod.DAILY,
            ResetStrategy.FIXED_TIME,
            level=QuotaLevel.USER,
            parent_id="tenant_a"
        )
        result = self.manager.consume("user_1")
        self.assertTrue(result.allowed)
        self.assertEqual(result.used, 1)

        global_result = result.level_results.get("global_api")
        self.assertIsNotNone(global_result)
        self.assertEqual(global_result.used, 1)

        tenant_result = result.level_results.get("tenant_a")
        self.assertIsNotNone(tenant_result)
        self.assertEqual(tenant_result.used, 1)

    def test_parent_limit_blocks_child(self):
        self.manager.create_quota(
            "global_api", 5, QuotaPeriod.DAILY,
            ResetStrategy.FIXED_TIME,
            level=QuotaLevel.GLOBAL
        )
        self.manager.create_quota(
            "user_1", 100, QuotaPeriod.DAILY,
            ResetStrategy.FIXED_TIME,
            level=QuotaLevel.USER,
            parent_id="global_api"
        )
        for _ in range(5):
            result = self.manager.consume("user_1")
            self.assertTrue(result.allowed)
        result = self.manager.consume("user_1")
        self.assertFalse(result.allowed)

    def test_ancestor_chain(self):
        self.manager.create_quota(
            "global", 1000, QuotaPeriod.DAILY,
            ResetStrategy.FIXED_TIME, level=QuotaLevel.GLOBAL
        )
        self.manager.create_quota(
            "tenant", 500, QuotaPeriod.DAILY,
            ResetStrategy.FIXED_TIME,
            level=QuotaLevel.TENANT, parent_id="global"
        )
        self.manager.create_quota(
            "user", 100, QuotaPeriod.DAILY,
            ResetStrategy.FIXED_TIME,
            level=QuotaLevel.USER, parent_id="tenant"
        )
        chain = self.manager._get_ancestor_chain("user")
        self.assertEqual(chain, ["user", "tenant", "global"])

    def test_delete_removes_from_parent_children(self):
        self.manager.create_quota(
            "global", 1000, QuotaPeriod.DAILY,
            ResetStrategy.FIXED_TIME, level=QuotaLevel.GLOBAL
        )
        self.manager.create_quota(
            "tenant", 500, QuotaPeriod.DAILY,
            ResetStrategy.FIXED_TIME,
            level=QuotaLevel.TENANT, parent_id="global"
        )
        self.assertIn("tenant", self.manager.get_children("global"))
        self.manager.delete_quota("tenant")
        self.assertNotIn("tenant", self.manager.get_children("global"))


class TestBorrowRepay(unittest.TestCase):
    def setUp(self):
        self.manager = HierarchicalQuotaManager()

    def test_borrow_within_limit(self):
        self.manager.create_quota(
            "api", 10, QuotaPeriod.DAILY,
            ResetStrategy.FIXED_TIME,
            max_borrow=5
        )
        for _ in range(10):
            self.manager.consume("api")
        result = self.manager.borrow("api", 3, reason="burst")
        self.assertTrue(result.allowed)
        self.assertEqual(result.borrowed, 3)
        self.assertEqual(result.used, 13)

    def test_borrow_exceeds_max(self):
        self.manager.create_quota(
            "api", 10, QuotaPeriod.DAILY,
            ResetStrategy.FIXED_TIME,
            max_borrow=2
        )
        for _ in range(10):
            self.manager.consume("api")
        result = self.manager.borrow("api", 3, reason="burst")
        self.assertFalse(result.allowed)

    def test_repay_reduces_borrowed(self):
        self.manager.create_quota(
            "api", 10, QuotaPeriod.DAILY,
            ResetStrategy.FIXED_TIME,
            max_borrow=5
        )
        for _ in range(10):
            self.manager.consume("api")
        self.manager.borrow("api", 3, reason="burst")
        result = self.manager.repay("api", 2)
        self.assertEqual(result.borrowed, 1)
        self.assertEqual(result.used, 11)

    def test_repay_more_than_borrowed(self):
        self.manager.create_quota(
            "api", 10, QuotaPeriod.DAILY,
            ResetStrategy.FIXED_TIME,
            max_borrow=5
        )
        for _ in range(10):
            self.manager.consume("api")
        self.manager.borrow("api", 3)
        result = self.manager.repay("api", 10)
        self.assertEqual(result.borrowed, 0)
        self.assertEqual(result.used, 10)

    def test_auto_borrow_on_consume(self):
        self.manager.create_quota(
            "api", 10, QuotaPeriod.DAILY,
            ResetStrategy.FIXED_TIME,
            max_borrow=5
        )
        for _ in range(10):
            self.manager.consume("api")
        result = self.manager.consume("api", amount=3)
        self.assertTrue(result.allowed)
        self.assertEqual(result.borrowed, 3)

    def test_auto_borrow_exceeds_max(self):
        self.manager.create_quota(
            "api", 10, QuotaPeriod.DAILY,
            ResetStrategy.FIXED_TIME,
            max_borrow=2
        )
        for _ in range(10):
            self.manager.consume("api")
        result = self.manager.consume("api", amount=3)
        self.assertFalse(result.allowed)

    def test_no_borrow_allowed(self):
        self.manager.create_quota(
            "api", 10, QuotaPeriod.DAILY,
            ResetStrategy.FIXED_TIME
        )
        for _ in range(10):
            self.manager.consume("api")
        result = self.manager.consume("api")
        self.assertFalse(result.allowed)
        with self.assertRaises(ValueError):
            self.manager.borrow("api", 1)

    def test_borrow_records(self):
        self.manager.create_quota(
            "api", 10, QuotaPeriod.DAILY,
            ResetStrategy.FIXED_TIME,
            max_borrow=5
        )
        for _ in range(10):
            self.manager.consume("api")
        self.manager.borrow("api", 3, reason="peak traffic")
        records = self.manager.get_borrow_records("api")
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["amount"], 3)
        self.assertEqual(records[0]["reason"], "peak traffic")
        self.assertEqual(records[0]["outstanding"], 3)

    def test_repay_updates_records(self):
        self.manager.create_quota(
            "api", 10, QuotaPeriod.DAILY,
            ResetStrategy.FIXED_TIME,
            max_borrow=5
        )
        for _ in range(10):
            self.manager.consume("api")
        self.manager.borrow("api", 3)
        self.manager.repay("api", 1)
        records = self.manager.get_borrow_records("api")
        self.assertEqual(records[0]["repaid"], 1)
        self.assertEqual(records[0]["outstanding"], 2)

    def test_borrow_resets_with_quota(self):
        self.manager.create_quota(
            "api", 5, QuotaPeriod.DAILY,
            ResetStrategy.FIXED_TIME,
            max_borrow=3
        )
        now = datetime(2024, 1, 15, 12, 0, tzinfo=timezone.utc)
        for _ in range(5):
            self.manager.consume("api", current_time=now)
        self.manager.borrow("api", 2, current_time=now)
        next_day = datetime(2024, 1, 16, 12, 0, tzinfo=timezone.utc)
        result = self.manager.consume("api", current_time=next_day)
        self.assertEqual(result.borrowed, 0)
        self.assertEqual(result.used, 1)


class TestTrendPrediction(unittest.TestCase):
    def setUp(self):
        self.manager = HierarchicalQuotaManager()

    def test_predict_with_no_history(self):
        self.manager.create_quota(
            "api", 1000, QuotaPeriod.DAILY,
            ResetStrategy.FIXED_TIME
        )
        for _ in range(100):
            self.manager.consume("api")
        prediction = self.manager.predict_trend("api")
        self.assertGreater(prediction.current_rate, 0)
        self.assertGreater(prediction.avg_consumption_per_hour, 0)

    def test_predict_with_history(self):
        self.manager.create_quota(
            "api", 1000, QuotaPeriod.DAILY,
            ResetStrategy.FIXED_TIME
        )
        now = datetime(2024, 1, 15, 12, 0, tzinfo=timezone.utc)
        for i in range(10):
            t = now + timedelta(hours=i)
            self.manager.consume("api", amount=10, current_time=t)
        last_t = now + timedelta(hours=9)
        prediction = self.manager.predict_trend("api", current_time=last_t)
        self.assertAlmostEqual(prediction.current_rate, 0.1, places=1)
        self.assertGreater(prediction.avg_consumption_per_hour, 0)
        self.assertFalse(prediction.will_exhaust)

    def test_predict_will_exhaust(self):
        self.manager.create_quota(
            "api", 100, QuotaPeriod.DAILY,
            ResetStrategy.FIXED_TIME
        )
        now = datetime(2024, 1, 15, 0, 0, tzinfo=timezone.utc)
        for i in range(10):
            t = now + timedelta(minutes=i)
            self.manager.consume("api", amount=8, current_time=t)
        prediction = self.manager.predict_trend("api")
        self.assertTrue(prediction.will_exhaust)

    def test_prediction_to_dict(self):
        self.manager.create_quota(
            "api", 1000, QuotaPeriod.DAILY,
            ResetStrategy.FIXED_TIME
        )
        self.manager.consume("api")
        prediction = self.manager.predict_trend("api")
        d = prediction.to_dict()
        self.assertIn("current_rate", d)
        self.assertIn("predicted_usage_at_reset", d)
        self.assertIn("predicted_rate_at_reset", d)
        self.assertIn("will_exhaust", d)
        self.assertIn("estimated_exhaust_time", d)
        self.assertIn("avg_consumption_per_hour", d)

    def test_predict_nonexistent_quota(self):
        with self.assertRaises(ValueError):
            self.manager.predict_trend("nonexistent")


class TestUsageHistory(unittest.TestCase):
    def setUp(self):
        self.manager = HierarchicalQuotaManager()

    def test_history_recorded_on_consume(self):
        self.manager.create_quota(
            "api", 100, QuotaPeriod.DAILY,
            ResetStrategy.FIXED_TIME
        )
        self.manager.consume("api")
        self.manager.consume("api")
        history = self.manager.get_usage_history("api")
        self.assertEqual(len(history), 2)
        self.assertEqual(history[0]["amount"], 1)
        self.assertEqual(history[1]["cumulative_used"], 2)

    def test_history_recorded_on_borrow(self):
        self.manager.create_quota(
            "api", 2, QuotaPeriod.DAILY,
            ResetStrategy.FIXED_TIME,
            max_borrow=5
        )
        self.manager.consume("api")
        self.manager.borrow("api", 1)
        history = self.manager.get_usage_history("api")
        self.assertEqual(len(history), 2)


class TestExistingQuotaFeatures(unittest.TestCase):
    def setUp(self):
        self.manager = HierarchicalQuotaManager()

    def test_create_quota(self):
        self.manager.create_quota(
            quota_id="test", limit=100,
            period=QuotaPeriod.DAILY,
            strategy=ResetStrategy.FIXED_TIME
        )
        config = self.manager.get_quota_config("test")
        self.assertIsNotNone(config)
        self.assertEqual(config.limit, 100)

    def test_consume_quota(self):
        self.manager.create_quota("test", 10, QuotaPeriod.DAILY,
                                  ResetStrategy.FIXED_TIME)
        result = self.manager.consume("test")
        self.assertTrue(result.allowed)
        self.assertEqual(result.used, 1)
        self.assertEqual(result.remaining, 9)

    def test_quota_exceeded(self):
        self.manager.create_quota("test", 3, QuotaPeriod.DAILY,
                                  ResetStrategy.FIXED_TIME)
        for i in range(3):
            result = self.manager.consume("test")
            self.assertTrue(result.allowed)
        result = self.manager.consume("test")
        self.assertFalse(result.allowed)

    def test_reset_quota(self):
        self.manager.create_quota("test", 10, QuotaPeriod.DAILY,
                                  ResetStrategy.FIXED_TIME)
        self.manager.consume("test")
        self.manager.consume("test")
        self.manager.reset("test")
        result = self.manager.check("test")
        self.assertEqual(result.used, 0)

    def test_boundary_precise_second(self):
        self.manager.create_quota(
            "boundary", 5, QuotaPeriod.DAILY,
            ResetStrategy.FIXED_TIME,
            fixed_reset_hour=0, fixed_reset_minute=0
        )
        before = datetime(2024, 1, 15, 23, 59, 59, tzinfo=timezone.utc)
        for _ in range(5):
            self.manager.consume("boundary", current_time=before)
        result = self.manager.check("boundary", current_time=before)
        self.assertEqual(result.remaining, 0)
        after = datetime(2024, 1, 16, 0, 0, 0, tzinfo=timezone.utc)
        result = self.manager.consume("boundary", current_time=after)
        self.assertEqual(result.used, 1)


class TestTimezoneHelpers(unittest.TestCase):
    def test_ensure_timezone_naive(self):
        dt = datetime(2024, 1, 15, 12, 0, 0)
        dt_tz = _ensure_timezone(dt)
        self.assertIsNotNone(dt_tz.tzinfo)

    def test_to_timezone_conversion(self):
        dt_utc = datetime(2024, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
        tz_cst = timezone(timedelta(hours=8))
        dt_cst = _to_timezone(dt_utc, tz_cst)
        self.assertEqual(dt_cst.hour, 20)


if __name__ == "__main__":
    unittest.main()
