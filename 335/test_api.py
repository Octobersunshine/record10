import requests
import json

BASE_URL = 'http://127.0.0.1:5000'

def print_response(title, response):
    print(f'\n{"="*60}')
    print(f'{title}')
    print(f'{"="*60}')
    print(json.dumps(response.json(), ensure_ascii=False, indent=2))

def test_simple_match():
    data = {
        'pattern': r'\d+',
        'test_string': '订单号: 12345, 价格: 99.99元'
    }
    response = requests.post(f'{BASE_URL}/api/regex/test', json=data)
    print_response('测试1: 简单数字匹配', response)

def test_capture_groups():
    data = {
        'pattern': r'(\d{4})-(\d{2})-(\d{2})',
        'test_string': '日期: 2024-01-15, 截止日期: 2024-12-31'
    }
    response = requests.post(f'{BASE_URL}/api/regex/test', json=data)
    print_response('测试2: 捕获分组 (日期匹配)', response)

def test_named_groups():
    data = {
        'pattern': r'(?P<name>\w+): (?P<age>\d+)岁',
        'test_string': '张三: 25岁, 李四: 30岁, 王五: 28岁'
    }
    response = requests.post(f'{BASE_URL}/api/regex/test', json=data)
    print_response('测试3: 命名分组', response)

def test_with_flags():
    data = {
        'pattern': r'hello',
        'test_string': 'Hello world! HELLO everyone! hello test!',
        'flags': ['IGNORECASE']
    }
    response = requests.post(f'{BASE_URL}/api/regex/test', json=data)
    print_response('测试4: 使用标志 (忽略大小写)', response)

def test_multiline_flag():
    data = {
        'pattern': r'^start',
        'test_string': 'start line1\nnot start\nstart line2',
        'flags': ['MULTILINE']
    }
    response = requests.post(f'{BASE_URL}/api/regex/test', json=data)
    print_response('测试5: MULTILINE 标志', response)

def test_invalid_regex():
    data = {
        'pattern': r'[unclosed',
        'test_string': 'test string'
    }
    response = requests.post(f'{BASE_URL}/api/regex/test', json=data)
    print_response('测试6: 无效正则表达式 (错误处理)', response)

def test_email_regex():
    data = {
        'pattern': r'[\w.-]+@[\w.-]+\.\w+',
        'test_string': '联系我们: support@example.com 或 sales@company.org'
    }
    response = requests.post(f'{BASE_URL}/api/regex/test', json=data)
    print_response('测试7: 邮箱匹配', response)

def test_validate_regex():
    print(f'\n{"="*60}')
    print('测试8: 正则验证接口')
    print(f'{"="*60}')
    
    valid_data = {'pattern': r'\d+'}
    response = requests.post(f'{BASE_URL}/api/regex/validate', json=valid_data)
    print('验证有效正则:', json.dumps(response.json(), ensure_ascii=False, indent=2))
    
    invalid_data = {'pattern': r'[unclosed'}
    response = requests.post(f'{BASE_URL}/api/regex/validate', json=invalid_data)
    print('验证无效正则:', json.dumps(response.json(), ensure_ascii=False, indent=2))

def test_no_match():
    data = {
        'pattern': r'notfound',
        'test_string': '这里没有匹配的内容'
    }
    response = requests.post(f'{BASE_URL}/api/regex/test', json=data)
    print_response('测试9: 无匹配情况', response)

def test_empty_string():
    data = {
        'pattern': r'\d*',
        'test_string': ''
    }
    response = requests.post(f'{BASE_URL}/api/regex/test', json=data)
    print_response('测试10: 空字符串匹配', response)

def test_catastrophic_backtracking():
    print(f'\n{"="*60}')
    print('测试11: 灾难性回溯超时保护')
    print(f'{"="*60}')
    
    evil_pattern = r'((((((a+)+)+)+)+)+)b'
    long_string = 'a' * 200
    
    import time
    start_time = time.time()
    
    data = {
        'pattern': evil_pattern,
        'test_string': long_string
    }
    response = requests.post(f'{BASE_URL}/api/regex/test', json=data)
    
    elapsed = time.time() - start_time
    print(f'请求耗时: {elapsed:.2f} 秒')
    print(f'响应状态码: {response.status_code}')
    print(json.dumps(response.json(), ensure_ascii=False, indent=2))
    print(f'✓ 超时保护生效，耗时约 {elapsed:.2f} 秒 (预期约 2 秒)')

def test_catastrophic_backtracking_validate():
    print(f'\n{"="*60}')
    print('测试12: 验证接口灾难性回溯保护')
    print(f'{"="*60}')
    
    evil_pattern = r'((((((((a+)+)+)+)+)+)+)+)b'
    
    import time
    start_time = time.time()
    
    data = {
        'pattern': evil_pattern
    }
    response = requests.post(f'{BASE_URL}/api/regex/validate', json=data)
    
    elapsed = time.time() - start_time
    print(f'请求耗时: {elapsed:.2f} 秒')
    print(json.dumps(response.json(), ensure_ascii=False, indent=2))

def test_long_string_safe_regex():
    data = {
        'pattern': r'a{3,5}',
        'test_string': 'a' * 1000
    }
    response = requests.post(f'{BASE_URL}/api/regex/test', json=data)
    print_response('测试13: 长字符串安全正则 (不会超时)', response)

def test_generate_phone_number():
    data = {
        'positive': ['13812345678', '15987654321', '18611112222'],
        'max_candidates': 5
    }
    response = requests.post(f'{BASE_URL}/api/regex/generate', json=data)
    print_response('测试14: 生成手机号码正则', response)

def test_generate_email():
    data = {
        'positive': ['test@example.com', 'user.name@company.org', 'admin@test.cn'],
        'max_candidates': 5
    }
    response = requests.post(f'{BASE_URL}/api/regex/generate', json=data)
    print_response('测试15: 生成邮箱正则', response)

def test_generate_date():
    data = {
        'positive': ['2024-01-15', '2023-12-31', '2025-06-01'],
        'negative': ['2024/01/15', '2024年1月15日', '01-15-2024'],
        'max_candidates': 5
    }
    response = requests.post(f'{BASE_URL}/api/regex/generate', json=data)
    print_response('测试16: 生成日期正则 (带反例)', response)

def test_generate_chinese_pattern():
    data = {
        'positive': ['张三: 25岁', '李四: 30岁', '王五: 28岁'],
        'max_candidates': 5
    }
    response = requests.post(f'{BASE_URL}/api/regex/generate', json=data)
    print_response('测试17: 生成中文格式正则', response)

def test_generate_with_negative_filter():
    data = {
        'positive': ['订单号: 12345', '订单号: 67890', '订单号: 11111'],
        'negative': ['订单号: abcde', '订单: 12345', '订单号:'],
        'max_candidates': 5
    }
    response = requests.post(f'{BASE_URL}/api/regex/generate', json=data)
    print_response('测试18: 带反例过滤的正则生成', response)

def test_generate_single_example():
    data = {
        'positive': ['AB-1234'],
        'max_candidates': 5
    }
    response = requests.post(f'{BASE_URL}/api/regex/generate', json=data)
    print_response('测试19: 单个样本生成正则', response)

def test_generate_invalid_input():
    data = {
        'positive': 'not a list'
    }
    response = requests.post(f'{BASE_URL}/api/regex/generate', json=data)
    print_response('测试20: 无效输入处理', response)

def test_generate_empty_positive():
    data = {
        'positive': [],
        'negative': ['test']
    }
    response = requests.post(f'{BASE_URL}/api/regex/generate', json=data)
    print_response('测试21: 空正例处理', response)

def test_generate_mixed_pattern():
    data = {
        'positive': ['ID: 2024-001', 'ID: 2023-999', 'ID: 2025-050'],
        'negative': ['ID: 2024-abc', 'id: 2024-001', 'ID:2024-001'],
        'max_candidates': 5
    }
    response = requests.post(f'{BASE_URL}/api/regex/generate', json=data)
    print_response('测试22: 混合格式正则生成', response)

if __name__ == '__main__':
    print('开始测试正则表达式 API...')
    
    test_simple_match()
    test_capture_groups()
    test_named_groups()
    test_with_flags()
    test_multiline_flag()
    test_invalid_regex()
    test_email_regex()
    test_validate_regex()
    test_no_match()
    test_empty_string()
    test_catastrophic_backtracking()
    test_catastrophic_backtracking_validate()
    test_long_string_safe_regex()
    test_generate_phone_number()
    test_generate_email()
    test_generate_date()
    test_generate_chinese_pattern()
    test_generate_with_negative_filter()
    test_generate_single_example()
    test_generate_invalid_input()
    test_generate_empty_positive()
    test_generate_mixed_pattern()
    
    print(f'\n{"="*60}')
    print('所有测试完成!')
    print(f'{"="*60}')
