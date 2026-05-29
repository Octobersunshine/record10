from quota_manager import (
    HierarchicalQuotaManager, QuotaPeriod, ResetStrategy, QuotaLevel
)
from datetime import datetime, timedelta, timezone


def example_hierarchical_quotas():
    print("=" * 60)
    print("示例1: 配额分层（全局 → 租户 → 用户）")
    print("=" * 60)

    manager = HierarchicalQuotaManager()

    manager.create_quota(
        "global_api", 10000, QuotaPeriod.DAILY,
        ResetStrategy.FIXED_TIME,
        level=QuotaLevel.GLOBAL,
        fixed_reset_hour=0
    )

    manager.create_quota(
        "tenant_acme", 5000, QuotaPeriod.DAILY,
        ResetStrategy.FIXED_TIME,
        level=QuotaLevel.TENANT,
        parent_id="global_api"
    )

    manager.create_quota(
        "tenant_beta", 3000, QuotaPeriod.DAILY,
        ResetStrategy.FIXED_TIME,
        level=QuotaLevel.TENANT,
        parent_id="global_api"
    )

    manager.create_quota(
        "user_alice", 1000, QuotaPeriod.DAILY,
        ResetStrategy.FIXED_TIME,
        level=QuotaLevel.USER,
        parent_id="tenant_acme"
    )

    manager.create_quota(
        "user_bob", 500, QuotaPeriod.DAILY,
        ResetStrategy.FIXED_TIME,
        level=QuotaLevel.USER,
        parent_id="tenant_acme"
    )

    for _ in range(5):
        result = manager.consume("user_alice")
    print(f"Alice消费5次:")
    print(f"  Alice: 已用={result.used}, 剩余={result.remaining}")
    for key, lr in result.level_results.items():
        print(f"  {key}: 已用={lr.used}, 剩余={lr.remaining}")

    print(f"\n租户ACME的子配额: {manager.get_children('tenant_acme')}")
    print(f"全局的子配额: {manager.get_children('global_api')}")
    print()


def example_parent_limit_blocks_child():
    print("=" * 60)
    print("示例2: 父级配额限制子级消费")
    print("=" * 60)

    manager = HierarchicalQuotaManager()

    manager.create_quota(
        "global", 5, QuotaPeriod.DAILY,
        ResetStrategy.FIXED_TIME,
        level=QuotaLevel.GLOBAL
    )

    manager.create_quota(
        "user", 100, QuotaPeriod.DAILY,
        ResetStrategy.FIXED_TIME,
        level=QuotaLevel.USER,
        parent_id="global"
    )

    print(f"用户配额限制: 100, 全局配额限制: 5")
    for i in range(7):
        result = manager.consume("user")
        print(f"  第{i+1}次: 允许={result.allowed}, "
              f"已用={result.used}, 剩余={result.remaining}")
    print(f"  → 全局配额耗尽后，即使子配额还有剩余也会被拒绝")
    print()


def example_borrow_and_repay():
    print("=" * 60)
    print("示例3: 配额预借和还贷")
    print("=" * 60)

    manager = HierarchicalQuotaManager()

    manager.create_quota(
        "api", 10, QuotaPeriod.DAILY,
        ResetStrategy.FIXED_TIME,
        max_borrow=5
    )

    print(f"配额限制: 10, 最大预借: 5")
    print()

    for _ in range(10):
        manager.consume("api")
    result = manager.check("api")
    print(f"正常消费10次: 已用={result.used}, 剩余={result.remaining}")

    result = manager.borrow("api", 3, reason="流量高峰")
    print(f"预借3次: 允许={result.allowed}, "
          f"已用={result.used}, 预借={result.borrowed}, "
          f"有效剩余={result.effective_remaining}")

    result = manager.repay("api", 1)
    print(f"还贷1次: 已用={result.used}, 预借={result.borrowed}, "
          f"有效剩余={result.effective_remaining}")

    result = manager.repay("api", 2)
    print(f"还贷2次: 已用={result.used}, 预借={result.borrowed}, "
          f"有效剩余={result.effective_remaining}")

    records = manager.get_borrow_records("api")
    print(f"\n预借记录:")
    for r in records:
        print(f"  预借={r['amount']}, 原因={r['reason']}, "
              f"已还={r['repaid']}, 未还={r['outstanding']}")
    print()


def example_auto_borrow():
    print("=" * 60)
    print("示例4: 消费时自动预借")
    print("=" * 60)

    manager = HierarchicalQuotaManager()

    manager.create_quota(
        "api", 10, QuotaPeriod.DAILY,
        ResetStrategy.FIXED_TIME,
        max_borrow=5
    )

    for _ in range(10):
        manager.consume("api")

    print(f"正常消费10次后:")
    result = manager.consume("api", amount=3)
    print(f"  再消费3次: 允许={result.allowed}, "
          f"已用={result.used}, 预借={result.borrowed}")

    result = manager.consume("api", amount=5)
    print(f"  再消费5次: 允许={result.allowed} (超出最大预借)")
    print()


def example_trend_prediction():
    print("=" * 60)
    print("示例5: 配额使用率趋势预测")
    print("=" * 60)

    manager = HierarchicalQuotaManager()
    tz = timezone(timedelta(hours=8))

    manager.create_quota(
        "api", 1000, QuotaPeriod.DAILY,
        ResetStrategy.FIXED_TIME,
        tz=tz
    )

    now = datetime(2024, 1, 15, 8, 0, 0, tzinfo=tz)
    for i in range(10):
        t = now + timedelta(hours=i)
        manager.consume("api", amount=20, current_time=t)

    last_t = now + timedelta(hours=9)
    prediction = manager.predict_trend("api", current_time=last_t)

    print(f"配额限制: 1000, 已消费: 200")
    print(f"当前使用率: {prediction.current_rate:.1%}")
    print(f"每小时平均消耗: {prediction.avg_consumption_per_hour:.1f}")
    print(f"预测重置时用量: {prediction.predicted_usage_at_reset}")
    print(f"预测重置时使用率: {prediction.predicted_rate_at_reset:.1%}")
    print(f"是否会耗尽: {'是' if prediction.will_exhaust else '否'}")
    if prediction.estimated_exhaust_time:
        print(f"预计耗尽时间: {prediction.estimated_exhaust_time}")
    print()

    d = prediction.to_dict()
    print("预测结果字典:")
    for k, v in d.items():
        print(f"  {k}: {v}")
    print()


def example_trend_prediction_exhaust():
    print("=" * 60)
    print("示例6: 高消耗场景预测（将会耗尽）")
    print("=" * 60)

    manager = HierarchicalQuotaManager()

    manager.create_quota(
        "api", 100, QuotaPeriod.DAILY,
        ResetStrategy.FIXED_TIME
    )

    now = datetime(2024, 1, 15, 0, 10, 0, tzinfo=timezone.utc)
    for i in range(10):
        t = now + timedelta(minutes=i)
        manager.consume("api", amount=5, current_time=t)

    last_t = now + timedelta(minutes=9)
    prediction = manager.predict_trend("api", current_time=last_t)

    print(f"配额限制: 100, 10分钟内已消费: 50")
    print(f"当前使用率: {prediction.current_rate:.1%}")
    print(f"每小时平均消耗: {prediction.avg_consumption_per_hour:.1f}")
    print(f"是否会耗尽: {'是' if prediction.will_exhaust else '否'}")
    if prediction.estimated_exhaust_time:
        print(f"预计耗尽时间: {prediction.estimated_exhaust_time}")
    print()


def example_usage_history():
    print("=" * 60)
    print("示例7: 使用历史记录")
    print("=" * 60)

    manager = HierarchicalQuotaManager()

    manager.create_quota(
        "api", 100, QuotaPeriod.DAILY,
        ResetStrategy.FIXED_TIME
    )

    now = datetime(2024, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
    for i in range(5):
        t = now + timedelta(hours=i)
        manager.consume("api", current_time=t)

    history = manager.get_usage_history("api")
    print(f"历史记录 ({len(history)} 条):")
    for h in history:
        print(f"  时间={h['timestamp']}, "
              f"数量={h['amount']}, 累计={h['cumulative_used']}")
    print()


def example_combined_features():
    print("=" * 60)
    print("示例8: 综合使用 - 分层 + 预借 + 趋势预测")
    print("=" * 60)

    manager = HierarchicalQuotaManager()
    tz = timezone(timedelta(hours=8))

    manager.create_quota(
        "global", 10000, QuotaPeriod.MONTHLY,
        ResetStrategy.FIXED_TIME,
        level=QuotaLevel.GLOBAL,
        tz=tz
    )

    manager.create_quota(
        "tenant_acme", 3000, QuotaPeriod.MONTHLY,
        ResetStrategy.FIXED_TIME,
        level=QuotaLevel.TENANT,
        parent_id="global",
        max_borrow=200,
        tz=tz
    )

    manager.create_quota(
        "user_alice", 1000, QuotaPeriod.MONTHLY,
        ResetStrategy.FIXED_TIME,
        level=QuotaLevel.USER,
        parent_id="tenant_acme",
        max_borrow=50,
        tz=tz
    )

    now = datetime(2024, 1, 15, 12, 0, 0, tzinfo=tz)
    for i in range(10):
        t = now + timedelta(hours=i)
        manager.consume("user_alice", amount=50, current_time=t)

    last_t = now + timedelta(hours=9)
    result = manager.check("user_alice", current_time=last_t)
    print(f"Alice配额状态:")
    print(f"  已用: {result.used}/{result.limit}")
    print(f"  使用率: {result.usage_rate:.1%}")
    print(f"  预借: {result.borrowed}")
    print(f"  有效剩余: {result.effective_remaining}")

    print(f"\n各层级状态:")
    for key, lr in result.level_results.items():
        print(f"  {key}: 已用={lr.used}/{lr.limit}, "
              f"使用率={lr.usage_rate:.1%}")

    prediction = manager.predict_trend("user_alice", current_time=last_t)
    print(f"\n趋势预测:")
    print(f"  当前使用率: {prediction.current_rate:.1%}")
    print(f"  预测月底用量: {prediction.predicted_usage_at_reset}")
    print(f"  是否会耗尽: {'是' if prediction.will_exhaust else '否'}")
    print()


def example_result_to_dict():
    print("=" * 60)
    print("示例9: 完整结果转字典（用于API响应）")
    print("=" * 60)

    manager = HierarchicalQuotaManager()

    manager.create_quota(
        "global", 1000, QuotaPeriod.DAILY,
        ResetStrategy.FIXED_TIME,
        level=QuotaLevel.GLOBAL
    )
    manager.create_quota(
        "user", 100, QuotaPeriod.DAILY,
        ResetStrategy.FIXED_TIME,
        level=QuotaLevel.USER,
        parent_id="global",
        max_borrow=10
    )

    for _ in range(95):
        manager.consume("user")
    manager.borrow("user", 5, reason="临时需求")

    result = manager.check("user")
    import json
    print(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))
    print()


if __name__ == "__main__":
    example_hierarchical_quotas()
    example_parent_limit_blocks_child()
    example_borrow_and_repay()
    example_auto_borrow()
    example_trend_prediction()
    example_trend_prediction_exhaust()
    example_usage_history()
    example_combined_features()
    example_result_to_dict()
