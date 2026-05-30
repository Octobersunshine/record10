from fractions import Fraction
import re

def parse_fraction(input_str):
    input_str = input_str.strip()
    mixed_pattern = r'^(\d+)\s+(\d+)/(\d+)$'
    simple_pattern = r'^(\d+)/(\d+)$'
    integer_pattern = r'^(\d+)$'
    
    mixed_match = re.match(mixed_pattern, input_str)
    if mixed_match:
        integer = int(mixed_match.group(1))
        numerator = int(mixed_match.group(2))
        denominator = int(mixed_match.group(3))
        return Fraction(integer * denominator + numerator, denominator)
    
    simple_match = re.match(simple_pattern, input_str)
    if simple_match:
        numerator = int(simple_match.group(1))
        denominator = int(simple_match.group(2))
        return Fraction(numerator, denominator)
    
    integer_match = re.match(integer_pattern, input_str)
    if integer_match:
        return Fraction(int(integer_match.group(1)), 1)
    
    raise ValueError(f"无法解析分数: {input_str}")

def to_mixed_fraction(frac):
    if frac.denominator == 1:
        return str(frac.numerator)
    
    abs_num = abs(frac.numerator)
    sign = -1 if frac.numerator < 0 else 1
    integer = abs_num // frac.denominator
    remainder = abs_num % frac.denominator
    
    if integer == 0:
        return f"{sign * remainder}/{frac.denominator}"
    else:
        return f"{sign * integer} {remainder}/{frac.denominator}"

def add_fractions(f1, f2):
    return f1 + f2

def subtract_fractions(f1, f2):
    return f1 - f2

def multiply_fractions(f1, f2):
    return f1 * f2

def divide_fractions(f1, f2):
    if f2 == 0:
        raise ZeroDivisionError("除数不能为零")
    return f1 / f2

def main():
    print("=" * 50)
    print("分数四则运算计算器")
    print("=" * 50)
    print("支持的输入格式:")
    print("  - 真分数: 3/4")
    print("  - 假分数: 7/3")
    print("  - 带分数: 2 1/3")
    print("  - 整数: 5")
    print("=" * 50)
    
    try:
        frac1_str = input("\n请输入第一个分数: ")
        frac2_str = input("请输入第二个分数: ")
        
        frac1 = parse_fraction(frac1_str)
        frac2 = parse_fraction(frac2_str)
        
        print(f"\n解析结果:")
        print(f"  第一个分数: {to_mixed_fraction(frac1)} ({frac1.numerator}/{frac1.denominator})")
        print(f"  第二个分数: {to_mixed_fraction(frac2)} ({frac2.numerator}/{frac2.denominator})")
        
        print("\n" + "=" * 50)
        print("运算结果:")
        print("=" * 50)
        
        add_result = add_fractions(frac1, frac2)
        print(f"\n加法: {to_mixed_fraction(frac1)} + {to_mixed_fraction(frac2)} = {to_mixed_fraction(add_result)}")
        print(f"  最简分数形式: {add_result.numerator}/{add_result.denominator}")
        
        sub_result = subtract_fractions(frac1, frac2)
        print(f"\n减法: {to_mixed_fraction(frac1)} - {to_mixed_fraction(frac2)} = {to_mixed_fraction(sub_result)}")
        print(f"  最简分数形式: {sub_result.numerator}/{sub_result.denominator}")
        
        mul_result = multiply_fractions(frac1, frac2)
        print(f"\n乘法: {to_mixed_fraction(frac1)} * {to_mixed_fraction(frac2)} = {to_mixed_fraction(mul_result)}")
        print(f"  最简分数形式: {mul_result.numerator}/{mul_result.denominator}")
        
        div_result = divide_fractions(frac1, frac2)
        print(f"\n除法: {to_mixed_fraction(frac1)} ÷ {to_mixed_fraction(frac2)} = {to_mixed_fraction(div_result)}")
        print(f"  最简分数形式: {div_result.numerator}/{div_result.denominator}")
        
    except ValueError as e:
        print(f"\n错误: {e}")
    except ZeroDivisionError as e:
        print(f"\n错误: {e}")
    except KeyboardInterrupt:
        print("\n\n程序已退出")

if __name__ == "__main__":
    main()
