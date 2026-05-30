from phone_location import PhoneSegmentDB, QueryStats, BATCH_MAX_SIZE


db = PhoneSegmentDB()


print("=== 查询单个手机号 ===")
result = db.query("13900101234")
print(f"手机号: 13900101234")
print(f"  运营商: {result['carrier']}")
print(f"  省份: {result['province']}")
print(f"  城市: {result['city']}")
print(f"  数据源: {result['data_source']}")
print(f"  是否最新: {result['is_latest']}")
print()


print("=== 批量查询（含统计报告） ===")
numbers = [
    "13900101234", "13800103456", "18600107890",
    "13300104567", "15500109876", "13500601234",
    "19900301234", "13800103456", "13900101234",
    "13800103456"
]
results, stats = db.batch_query_with_stats(numbers, show_progress=True)
for num, res in zip(numbers, results):
    carrier = res.get("actual_carrier") or res.get("carrier", "未知")
    ported_mark = " [转网]" if res.get("ported") else ""
    print(f"  {num}: {carrier}{ported_mark} | {res['province']} {res['city']}")
print()


print("=== 统计报告 ===")
stats.print_report()
print()


print("=== 获取饼图数据（JSON格式） ===")
report = stats.full_report()
import json
print(json.dumps(report["carrier_distribution"], ensure_ascii=False, indent=2))
print()


print("=== 携号转网查询 ===")
result = db.query("13900101234", check_mnp=True)
print(f"手机号: 13900101234")
print(f"  号段运营商: {result['carrier']}")
print(f"  实际运营商: {result.get('actual_carrier', result['carrier'])}")
print(f"  是否转网: {result.get('ported', False)}")
print(f"  转网检测来源: {result.get('mnp_source', 'N/A')}")
print()


print("=== 批量查询上限 ===")
print(f"  单次最多: {BATCH_MAX_SIZE} 个号码")
try:
    db.batch_query(["13900101234"] * 101)
except ValueError as e:
    print(f"  超出限制测试: {e}")
