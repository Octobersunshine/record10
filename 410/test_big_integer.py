from big_integer import (
    add, subtract, multiply, karatsuba_multiply,
    benchmark_multiply, divide, divide_decimal
)
from decimal import Decimal, getcontext


def test_add():
    print("=== 测试加法 ===")
    print(f"123 + 456 = {add('123', '456')} (预期: 579)")
    print(f"-123 + 456 = {add('-123', '456')} (预期: 333)")
    print(f"-123 + -456 = {add('-123', '-456')} (预期: -579)")


def test_subtract():
    print("\n=== 测试减法 ===")
    print(f"123 - 456 = {subtract('123', '456')} (预期: -333)")
    print(f"456 - 123 = {subtract('456', '123')} (预期: 333)")
    print(f"-123 - -456 = {subtract('-123', '-456')} (预期: 333)")


def test_multiply():
    print("\n=== 测试竖式乘法 ===")
    print(f"123 * 456 = {multiply('123', '456')} (预期: 56088)")
    print(f"-123 * 456 = {multiply('-123', '456')} (预期: -56088)")
    print(f"-123 * -456 = {multiply('-123', '-456')} (预期: 56088)")


def test_karatsuba_correctness():
    print("\n=== 测试 Karatsuba 乘法正确性 ===")
    test_cases = [
        ('123', '456'),
        ('999', '999'),
        ('123456789', '987654321'),
        ('-123', '456'),
        ('123', '-456'),
        ('-123', '-456'),
        ('0', '123'),
        ('1', '999999999'),
    ]
    
    all_pass = True
    for a, b in test_cases:
        result_col = multiply(a, b)
        result_kar = karatsuba_multiply(a, b)
        passed = result_col == result_kar
        all_pass = all_pass and passed
        status = '✓' if passed else '✗'
        print(f"  {status} {a} * {b} = {result_kar} (竖式: {result_col})")
    
    print(f"\nKaratsuba 正确性: {'全部通过' if all_pass else '存在错误'}")


def test_karatsuba_vs_column_large():
    print("\n=== Karatsuba vs 竖式 正确性验证（大数） ===")
    test_cases = [
        ('123456789012345678901234567890', '98765432109876543210987654321'),
        ('-' + '1' * 200, '9' * 200),
    ]
    
    for a, b in test_cases:
        result_col = multiply(a, b)
        result_kar = karatsuba_multiply(a, b)
        passed = result_col == result_kar
        status = '✓' if passed else '✗'
        print(f"  {status} 长度 {len(a.lstrip('-'))} x {len(b.lstrip('-'))}: 结果一致 = {passed}")


def test_benchmark():
    print("\n=== 性能对比（数字长度 > 1000） ===")
    
    for n in [500, 1000, 2000]:
        a = '9' * n
        b = '9' * n
        
        print(f"\n  数字长度: {n} 位")
        results = benchmark_multiply(a, b)
        
        print(f"    竖式乘法:   {results['column']['time']:.4f} 秒")
        print(f"    Karatsuba:  {results['karatsuba']['time']:.4f} 秒")
        print(f"    结果一致:   {'✓' if results['match'] else '✗'}")
        print(f"    加速比:     {results['speedup']:.2f}x")


def test_divide():
    print("\n=== 测试整数除法 ===")
    q, r = divide('1234', '12')
    print(f"1234 / 12 = 商: {q}, 余数: {r} (预期: 商: 102, 余数: 10)")
    
    q, r = divide('-1234', '12')
    print(f"-1234 / 12 = 商: {q}, 余数: {r}")


def test_divide_decimal():
    print("\n=== 测试除法小数扩展 ===")
    
    test_cases = [
        ('10', '3', 5),
        ('1', '7', 10),
        ('22', '7', 20),
        ('100', '8', 5),
        ('1', '3', 15),
        ('123456789', '987654321', 10),
        ('-10', '3', 5),
        ('10', '-3', 5),
        ('-10', '-3', 5),
        ('0', '123', 5),
    ]
    
    getcontext().prec = 50
    
    for dividend, divisor, places in test_cases:
        result = divide_decimal(dividend, divisor, places)
        
        d_dividend = Decimal(dividend)
        d_divisor = Decimal(divisor)
        expected = str(d_dividend / d_divisor)
        
        if '.' in expected:
            exp_decimals = expected.split('.')[1]
            exp_decimals = exp_decimals[:places] + '0' * max(0, places - len(exp_decimals))
            expected_str = expected.split('.')[0] + '.' + exp_decimals
        else:
            expected_str = expected + '.' + '0' * places
        
        passed = result == expected_str
        status = '✓' if passed else '✗'
        print(f"  {status} {dividend} / {divisor} ({places}位小数) = {result}")
        if not passed:
            print(f"      预期: {expected_str}")


def test_divide_decimal_precision():
    print("\n=== 除法小数扩展精度验证 ===")
    
    result = divide_decimal('1', '7', 30)
    print(f"  1/7 (30位小数): {result}")
    
    known_pattern = "142857"
    decimal_part = result.split('.')[1]
    pattern_ok = all(decimal_part[i:i+6] == known_pattern 
                     for i in range(0, len(decimal_part) - 6, 6))
    print(f"  循环节 142857 验证: {'✓' if pattern_ok else '✗'}")
    
    result = divide_decimal('22', '7', 20)
    print(f"  22/7 (20位小数): {result}")
    print(f"  整数部分: {result.split('.')[0]} (预期: 3)")


if __name__ == '__main__':
    test_add()
    test_subtract()
    test_multiply()
    test_karatsuba_correctness()
    test_karatsuba_vs_column_large()
    test_benchmark()
    test_divide()
    test_divide_decimal()
    test_divide_decimal_precision()
    print("\n=== 测试完成 ===")
