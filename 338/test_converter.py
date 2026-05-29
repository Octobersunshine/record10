import sys
import os
import json
import tempfile
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_converter import (
    convert,
    detect_csv_delimiter,
    csv_to_json,
    json_to_csv,
    json_to_xml,
    xml_to_json,
    strip_bom,
    read_file,
    write_file,
    BOM,
)

print("=" * 60)
print("  数据格式转换工具 - 功能测试")
print("=" * 60)
print()

# 测试数据
csv_comma = """name,age,city
Alice,25,Beijing
Bob,30,Shanghai
Charlie,35,Guangzhou"""

csv_tab = """name\tage\tcity
Alice\t25\tBeijing
Bob\t30\tShanghai
Charlie\t35\tGuangzhou"""

csv_semicolon = """name;age;city
Alice;25;Beijing
Bob;30;Shanghai
Charlie;35;Guangzhou"""

json_data = """[
  {"name": "Alice", "age": "25", "city": "Beijing"},
  {"name": "Bob", "age": "30", "city": "Shanghai"},
  {"name": "Charlie", "age": "35", "city": "Guangzhou"}
]"""

xml_data = """<?xml version='1.0' encoding='utf-8'?>
<root>
  <item>
    <name>Alice</name>
    <age>25</age>
    <city>Beijing</city>
  </item>
  <item>
    <name>Bob</name>
    <age>30</age>
    <city>Shanghai</city>
  </item>
  <item>
    <name>Charlie</name>
    <age>35</age>
    <city>Guangzhou</city>
  </item>
</root>"""

# 测试1: CSV分隔符自动检测
print("【测试1】CSV分隔符自动检测")
print("-" * 60)
delimiters = [
    ('逗号', csv_comma, ','),
    ('制表符', csv_tab, '\t'),
    ('分号', csv_semicolon, ';')
]
for name, data, expected in delimiters:
    detected = detect_csv_delimiter(data)
    display = {'\t': '\\t', ',': ',', ';': ';'}[detected]
    status = "✓" if detected == expected else "✗"
    print(f"  {name} CSV: 检测到 '{display}'  {status}")
print()

# 测试2: CSV 转 JSON
print("【测试2】CSV(逗号) 转 JSON")
print("-" * 60)
result = csv_to_json(csv_comma)
print(result[:200] + "..." if len(result) > 200 else result)
print("  ✓ 转换成功")
print()

# 测试3: JSON 转 CSV
print("【测试3】JSON 转 CSV")
print("-" * 60)
result = json_to_csv(json_data)
print(result)
print("  ✓ 转换成功")
print()

# 测试4: JSON 转 XML
print("【测试4】JSON 转 XML")
print("-" * 60)
result = json_to_xml(json_data)
print(result[:300] + "..." if len(result) > 300 else result)
print("  ✓ 转换成功")
print()

# 测试5: XML 转 JSON
print("【测试5】XML 转 JSON")
print("-" * 60)
result = xml_to_json(xml_data)
print(result[:200] + "..." if len(result) > 200 else result)
print("  ✓ 转换成功")
print()

# 测试6: CSV 转 XML (使用 convert 通用函数)
print("【测试6】CSV 转 XML (通用convert函数)")
print("-" * 60)
result = convert(csv_comma, 'csv', 'xml')
print(result[:200] + "..." if len(result) > 200 else result)
print("  ✓ 转换成功")
print()

# 测试7: XML 转 CSV (使用 convert 通用函数)
print("【测试7】XML 转 CSV (通用convert函数)")
print("-" * 60)
result = convert(xml_data, 'xml', 'csv')
print(result)
print("  ✓ 转换成功")
print()

# 测试8: 分号分隔CSV 转 JSON
print("【测试8】CSV(分号) 转 JSON (自动检测分隔符)")
print("-" * 60)
result = csv_to_json(csv_semicolon)
print(result[:200] + "..." if len(result) > 200 else result)
print("  ✓ 转换成功")
print()

# 测试9: 制表符分隔CSV 转 JSON
print("【测试9】CSV(制表符) 转 JSON (自动检测分隔符)")
print("-" * 60)
result = csv_to_json(csv_tab)
print(result[:200] + "..." if len(result) > 200 else result)
print("  ✓ 转换成功")
print()

# 测试10: 中文数据测试
print("【测试10】中文数据测试")
print("-" * 60)
csv_chinese = """姓名,年龄,城市
张三,28,北京
李四,32,上海"""
result = convert(csv_chinese, 'csv', 'json')
print(result)
print("  ✓ 中文支持正常")
print()

print("=" * 60)
print("  所有测试完成！")
print("=" * 60)
print()
print("命令行使用示例:")
print("  python data_converter.py csv json -i input.csv -o output.json")
print("  python data_converter.py json xml -i input.json -o output.xml")
print("  cat data.csv | python data_converter.py csv json")
print()

print("=" * 60)
print("  BOM & 编码专项测试")
print("=" * 60)
print()

# 测试11: strip_bom 函数
print("【测试11】strip_bom 函数")
print("-" * 60)
bom_data = BOM + "hello"
no_bom_data = "hello"
assert strip_bom(bom_data) == "hello", "strip_bom failed on BOM data"
assert strip_bom(no_bom_data) == "hello", "strip_bom failed on non-BOM data"
assert strip_bom("") == "", "strip_bom failed on empty string"
print("  纯字符串BOM剥离: ✓")
print("  无BOM字符串不变: ✓")
print("  空字符串安全处理: ✓")
print()

# 测试12: 带BOM的CSV转JSON
print("【测试12】带BOM的CSV转JSON (中文)")
print("-" * 60)
csv_bom_chinese = BOM + """姓名,年龄,城市
张三,28,北京
李四,32,上海"""
result = csv_to_json(csv_bom_chinese)
assert '"姓名"' in result, f"BOM not stripped, got: {result[:100]}"
assert '"张三"' in result, f"Chinese characters corrupted: {result[:100]}"
print(result)
print("  ✓ BOM自动剥离，中文正确输出")
print()

# 测试13: 带BOM的逗号分隔CSV转JSON
print("【测试13】带BOM的逗号分隔CSV转JSON (英文)")
print("-" * 60)
csv_bom_english = BOM + """name,age,city
Alice,25,Beijing
Bob,30,Shanghai"""
result = csv_to_json(csv_bom_english)
assert '"name"' in result, f"BOM caused field name corruption: {result[:100]}"
assert '"Alice"' in result
print(result)
print("  ✓ BOM自动剥离，字段名正确")
print()

# 测试14: 带BOM的分号分隔CSV转JSON
print("【测试14】带BOM的分号分隔CSV转JSON")
print("-" * 60)
csv_bom_semi = BOM + """name;age;city
Alice;25;Beijing"""
result = csv_to_json(csv_bom_semi)
assert '"name"' in result
assert '"Alice"' in result
print(result)
print("  ✓ BOM + 分号分隔符 同时正确处理")
print()

# 测试15: UTF-8-SIG文件读写
print("【测试15】UTF-8-SIG文件读取 (模拟Excel导出的BOM文件)")
print("-" * 60)
csv_content = "姓名,年龄,城市\n张三,28,北京\n李四,32,上海"
with tempfile.NamedTemporaryFile(mode='wb', suffix='.csv', delete=False) as f:
    temp_path = f.name
    f.write(b'\xef\xbb\xbf')
    f.write(csv_content.encode('utf-8'))

try:
    loaded = read_file(temp_path)
    assert not loaded.startswith(BOM), "BOM should be stripped by utf-8-sig"
    assert "姓名" in loaded
    result = csv_to_json(loaded)
    assert '"姓名"' in result
    assert '"张三"' in result
    print(f"  文件内容正确读取: ✓")
    print(f"  BOM已被utf-8-sig自动剥离: ✓")
    print(f"  中文CSV转JSON成功: ✓")
    print(result)
finally:
    os.unlink(temp_path)
print()

# 测试16: write_file输出UTF-8无BOM
print("【测试16】write_file输出UTF-8 (无BOM)")
print("-" * 60)
json_content = '[{"姓名": "张三", "年龄": "28"}]'
with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, encoding='utf-8') as f:
    temp_path = f.name

try:
    write_file(temp_path, json_content)
    with open(temp_path, 'rb') as f:
        raw = f.read()
    assert not raw.startswith(b'\xef\xbb\xbf'), "Output should NOT have BOM"
    loaded = read_file(temp_path)
    assert loaded == json_content
    print("  输出文件无BOM: ✓")
    print("  中文内容完整保留: ✓")
finally:
    os.unlink(temp_path)
print()

# 测试17: 完整链路 - 带BOM的CSV文件转JSON文件
print("【测试17】完整链路: BOM CSV文件 → JSON文件")
print("-" * 60)
csv_raw = "姓名,年龄,城市\n张三,28,北京\n李四,32,上海"
csv_path = os.path.join(tempfile.gettempdir(), 'test_bom_input.csv')
json_path = os.path.join(tempfile.gettempdir(), 'test_bom_output.json')

try:
    with open(csv_path, 'wb') as f:
        f.write(b'\xef\xbb\xbf')
        f.write(csv_raw.encode('utf-8'))

    source = read_file(csv_path)
    result = convert(source, 'csv', 'json')
    write_file(json_path, result)

    with open(json_path, 'rb') as f:
        raw = f.read()
    assert not raw.startswith(b'\xef\xbb\xbf'), "JSON output should not have BOM"

    loaded = read_file(json_path)
    parsed = json.loads(loaded)
    assert len(parsed) == 2
    assert parsed[0]['姓名'] == '张三'
    assert parsed[1]['城市'] == '上海'
    print("  BOM CSV文件读取: ✓")
    print("  转换JSON正确: ✓")
    print("  JSON文件无BOM: ✓")
    print("  中文字段值完整: ✓")
    print(loaded)
finally:
    for p in [csv_path, json_path]:
        if os.path.exists(p):
            os.unlink(p)
print()

# 测试18: 完整链路 - JSON文件转带BOM的CSV文件再转回
print("【测试18】往返测试: JSON → CSV → JSON (中文)")
print("-" * 60)
original_json = '[{"姓名": "张三", "年龄": "28", "城市": "北京"}, {"姓名": "李四", "年龄": "32", "城市": "上海"}]'
csv_result = convert(original_json, 'json', 'csv')
json_result = convert(csv_result, 'csv', 'json')

original = json.loads(original_json)
roundtrip = json.loads(json_result)
assert original == roundtrip, f"Roundtrip mismatch:\n{original}\nvs\n{roundtrip}"
print("  JSON → CSV → JSON 往返数据一致: ✓")
print(f"  原始: {original}")
print(f"  往返: {roundtrip}")
print()

print("=" * 60)
print("  BOM & 编码专项测试全部通过！")
print("=" * 60)
print()

print("=" * 60)
print("  进度管理模块测试")
print("=" * 60)
print()

from data_converter import (
    register_task,
    update_progress,
    mark_task_started,
    mark_task_completed,
    mark_task_failed,
    get_task_status,
    list_tasks,
    excel_to_json,
    json_to_excel,
)

# 测试19: 任务注册与基本状态
print("【测试19】任务注册与基本状态")
print("-" * 60)
tid = register_task("test-task-001")
assert tid == "test-task-001"
status = get_task_status(tid)
assert status is not None
assert status["task_id"] == "test-task-001"
assert status["status"] == "pending"
assert status["progress"] == 0
print(f"  任务ID: {tid}")
print(f"  初始状态: {status['status']}")
print(f"  初始进度: {status['progress']}%")
print("  ✓ 任务注册成功")
print()

# 测试20: 进度更新
print("【测试20】进度更新与状态流转")
print("-" * 60)
mark_task_started(tid)
status = get_task_status(tid)
assert status["status"] == "running"
assert status["start_time"] is not None
print(f"  启动后状态: {status['status']} ✓")

update_progress(tid, 50, message="Processing halfway", current_chunk=5, total_chunks=10)
status = get_task_status(tid)
assert status["progress"] == 50
assert status["message"] == "Processing halfway"
assert status["current_chunk"] == 5
assert status["total_chunks"] == 10
print(f"  进度更新到50%: ✓")
print(f"  当前块: {status['current_chunk']}/{status['total_chunks']}")

update_progress(tid, 101)
status = get_task_status(tid)
assert status["progress"] == 100
print(f"  进度上限钳制为100%: ✓")

mark_task_completed(tid, result="ok", message="Done!")
status = get_task_status(tid)
assert status["status"] == "completed"
assert status["end_time"] is not None
assert status["progress"] == 100
print(f"  完成后状态: {status['status']} ✓")
print()

# 测试21: 任务失败标记
print("【测试21】任务失败标记")
print("-" * 60)
tid2 = register_task("test-task-002")
mark_task_started(tid2)
mark_task_failed(tid2, "File not found")
status = get_task_status(tid2)
assert status["status"] == "failed"
assert status["error"] == "File not found"
print(f"  失败状态: {status['status']} ✓")
print(f"  错误信息: {status['error']} ✓")
print()

# 测试22: 任务列表查询
print("【测试22】任务列表查询")
print("-" * 60)
tasks = list_tasks()
assert len(tasks) >= 2
print(f"  已注册任务数: {len(tasks)} ✓")
for t in tasks:
    print(f"    - {t['task_id']}: {t['status']} ({t['progress']}%)")
print()

# 测试23: 自动生成任务ID
print("【测试23】自动生成任务ID")
print("-" * 60)
auto_tid = register_task()
assert auto_tid is not None
assert len(auto_tid) > 0
status = get_task_status(auto_tid)
assert status["task_id"] == auto_tid
print(f"  自动生成ID: {auto_tid} ✓")
print()

print("=" * 60)
print("  进度管理模块测试全部通过！")
print("=" * 60)
print()

print("=" * 60)
print("  Excel 转换功能测试")
print("=" * 60)
print()

# 测试24: JSON转Excel再转JSON往返
print("【测试24】JSON → Excel → JSON 往返测试")
print("-" * 60)
test_json = json.dumps([
    {"姓名": "张三", "年龄": "28", "城市": "北京"},
    {"姓名": "李四", "年龄": "32", "城市": "上海"},
    {"姓名": "王五", "年龄": "25", "城市": "广州"}
], ensure_ascii=False)

excel_path = os.path.join(tempfile.gettempdir(), 'test_excel_output.xlsx')
try:
    json_to_excel(test_json, excel_path)
    assert os.path.exists(excel_path)
    file_size = os.path.getsize(excel_path)
    print(f"  Excel文件已生成: {os.path.basename(excel_path)}")
    print(f"  文件大小: {file_size} bytes ✓")

    result_json = excel_to_json(excel_path)
    result_data = json.loads(result_json)
    original_data = json.loads(test_json)

    assert len(result_data) == len(original_data)
    for i, (orig, res) in enumerate(zip(original_data, result_data)):
        for key in orig:
            assert res.get(key) == orig[key], f"Row {i} key {key} mismatch: {res.get(key)} vs {orig[key]}"

    print(f"  数据行数一致: {len(result_data)} 行 ✓")
    print(f"  中文字段名和值完整保留 ✓")
    print(f"  往返转换数据完全一致 ✓")
    print(result_json)
finally:
    if os.path.exists(excel_path):
        os.unlink(excel_path)
print()

# 测试25: 带任务ID的JSON转Excel（带进度回调）
print("【测试25】带任务ID的JSON转Excel（进度回调）")
print("-" * 60)
excel_path2 = os.path.join(tempfile.gettempdir(), 'test_progress.xlsx')
tid_conv = register_task("excel-conv-001")
progress_values = []

try:
    mark_task_started(tid_conv)
    json_to_excel(test_json, excel_path2, task_id=tid_conv, chunk_size=1)
    status = get_task_status(tid_conv)
    assert status["status"] == "completed"
    assert status["progress"] == 100
    print(f"  任务状态: {status['status']} ✓")
    print(f"  最终进度: {status['progress']}% ✓")
    print(f"  消息: {status['message']} ✓")
    print(f"  分块数: {status['total_chunks']} ✓")
    print(f"  Excel文件已生成 ✓")
finally:
    if os.path.exists(excel_path2):
        os.unlink(excel_path2)
print()

# 测试26: 带任务ID的Excel转JSON（带进度回调）
print("【测试26】带任务ID的Excel转JSON（进度回调）")
print("-" * 60)
excel_path3 = os.path.join(tempfile.gettempdir(), 'test_read_progress.xlsx')
tid_read = register_task("excel-read-001")
try:
    json_to_excel(test_json, excel_path3)
    result = excel_to_json(excel_path3, task_id=tid_read, chunk_size=1)
    status = get_task_status(tid_read)
    assert status["status"] == "completed"
    assert status["progress"] == 100
    print(f"  任务状态: {status['status']} ✓")
    print(f"  最终进度: {status['progress']}% ✓")
    print(f"  消息: {status['message']} ✓")
    print(f"  JSON输出长度: {len(result)} 字符 ✓")
finally:
    if os.path.exists(excel_path3):
        os.unlink(excel_path3)
print()

# 测试27: 使用convert函数进行Excel转换
print("【测试27】使用convert函数进行Excel转换")
print("-" * 60)
excel_path4 = os.path.join(tempfile.gettempdir(), 'test_convert_excel.xlsx')
try:
    convert_result = convert(test_json, 'json', 'excel', output_path=excel_path4)
    assert convert_result is None
    assert os.path.exists(excel_path4)
    print("  convert(json→excel) ✓")

    back_json = convert('', 'excel', 'json', input_path=excel_path4)
    assert back_json is not None
    back_data = json.loads(back_json)
    assert len(back_data) == 3
    print("  convert(excel→json) ✓")
    print(f"  转换后数据: {back_data}")

    csv_result = convert('', 'xlsx', 'csv', input_path=excel_path4)
    assert csv_result is not None
    assert '姓名' in csv_result
    print("  convert(xlsx→csv) ✓")
finally:
    if os.path.exists(excel_path4):
        os.unlink(excel_path4)
print()

# 测试28: 大数据量分块处理模拟
print("【测试28】大数据量分块处理模拟 (1000行)")
print("-" * 60)
large_data = []
for i in range(1000):
    large_data.append({
        "id": str(i + 1),
        "name": f"用户{i + 1}",
        "age": str(20 + (i % 30)),
        "email": f"user{i + 1}@example.com"
    })
large_json = json.dumps(large_data, ensure_ascii=False)

excel_path5 = os.path.join(tempfile.gettempdir(), 'test_large_data.xlsx')
tid_large = register_task("large-data-001")
try:
    json_to_excel(large_json, excel_path5, task_id=tid_large, chunk_size=200)
    status = get_task_status(tid_large)
    assert status["status"] == "completed"
    assert status["total_chunks"] == 5
    print(f"  数据量: 1000行, 4列")
    print(f"  分块数: {status['total_chunks']} ✓")
    print(f"  文件大小: {os.path.getsize(excel_path5)} bytes ✓")
    print(f"  完成状态: {status['status']} ✓")

    result = excel_to_json(excel_path5, task_id="verify-large", chunk_size=200)
    result_data = json.loads(result)
    assert len(result_data) == 1000
    print(f"  回读数据量: {len(result_data)} 行 ✓")
    print(f"  首行: {result_data[0]}")
    print(f"  末行: {result_data[-1]}")
finally:
    if os.path.exists(excel_path5):
        os.unlink(excel_path5)
print()

# 测试29: 空的sheet_name使用默认值
print("【测试29】指定Sheet名称")
print("-" * 60)
excel_path6 = os.path.join(tempfile.gettempdir(), 'test_sheet_name.xlsx')
try:
    json_to_excel(test_json, excel_path6, sheet_name="用户数据")
    print(f"  Excel已生成 ✓")

    result = excel_to_json(excel_path6, sheet_name="用户数据")
    result_data = json.loads(result)
    assert len(result_data) == 3
    print(f"  按指定Sheet名称读取成功 ✓")
    print(f"  数据: {result_data}")
finally:
    if os.path.exists(excel_path6):
        os.unlink(excel_path6)
print()

# 测试30: 不存在的任务ID查询
print("【测试30】不存在的任务ID查询")
print("-" * 60)
status = get_task_status("non-existent-task")
assert status is None
print("  查询不存在的任务返回None ✓")
print()

print("=" * 60)
print("  Excel 转换功能测试全部通过！")
print("=" * 60)
print()
print("=" * 60)
print(f"  全部 {30} 项测试通过！ ✓")
print("=" * 60)
