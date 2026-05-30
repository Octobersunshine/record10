import re

def gcd(a, b):
    a = abs(a)
    b = abs(b)
    if a == 0 and b == 0:
        return 1
    while b:
        a, b = b, a % b
    return a

def lcm(a, b):
    return abs(a * b) // gcd(a, b)

class Fraction:
    def __init__(self, numerator, denominator=1):
        if denominator == 0:
            raise ZeroDivisionError("分母不能为零")
        
        if numerator == 0:
            self.numerator = 0
            self.denominator = 1
            return
        
        if denominator < 0:
            numerator = -numerator
            denominator = -denominator
        
        common_divisor = gcd(numerator, denominator)
        self.numerator = numerator // common_divisor
        self.denominator = denominator // common_divisor
    
    @staticmethod
    def _to_fraction(other):
        if isinstance(other, Fraction):
            return other
        elif isinstance(other, int):
            return Fraction(other, 1)
        elif isinstance(other, float):
            if other == 0:
                return Fraction(0, 1)
            denominator = 10 ** 10
            numerator = int(round(other * denominator))
            return Fraction(numerator, denominator)
        else:
            raise TypeError(f"不支持的类型: {type(other).__name__}")
    
    def to_float(self, ndigits=None):
        value = self.numerator / self.denominator
        if ndigits is not None:
            return round(value, ndigits)
        return value
    
    def __add__(self, other):
        other = Fraction._to_fraction(other)
        common_denominator = lcm(self.denominator, other.denominator)
        new_numerator = (self.numerator * (common_denominator // self.denominator) +
                        other.numerator * (common_denominator // other.denominator))
        return Fraction(new_numerator, common_denominator)
    
    def __sub__(self, other):
        other = Fraction._to_fraction(other)
        common_denominator = lcm(self.denominator, other.denominator)
        new_numerator = (self.numerator * (common_denominator // self.denominator) -
                        other.numerator * (common_denominator // other.denominator))
        return Fraction(new_numerator, common_denominator)
    
    def __mul__(self, other):
        other = Fraction._to_fraction(other)
        new_numerator = self.numerator * other.numerator
        new_denominator = self.denominator * other.denominator
        return Fraction(new_numerator, new_denominator)
    
    def __truediv__(self, other):
        other = Fraction._to_fraction(other)
        if other.numerator == 0:
            raise ZeroDivisionError("除数不能为零")
        new_numerator = self.numerator * other.denominator
        new_denominator = self.denominator * other.numerator
        if new_denominator == 0:
            raise ZeroDivisionError("除法结果分母为零")
        return Fraction(new_numerator, new_denominator)
    
    def __eq__(self, other):
        try:
            other = Fraction._to_fraction(other)
        except TypeError:
            return NotImplemented
        return self.numerator == other.numerator and self.denominator == other.denominator
    
    def __ne__(self, other):
        result = self.__eq__(other)
        if result is NotImplemented:
            return NotImplemented
        return not result
    
    def __lt__(self, other):
        try:
            other = Fraction._to_fraction(other)
        except TypeError:
            return NotImplemented
        return self.numerator * other.denominator < other.numerator * self.denominator
    
    def __le__(self, other):
        try:
            other = Fraction._to_fraction(other)
        except TypeError:
            return NotImplemented
        return self.numerator * other.denominator <= other.numerator * self.denominator
    
    def __gt__(self, other):
        try:
            other = Fraction._to_fraction(other)
        except TypeError:
            return NotImplemented
        return self.numerator * other.denominator > other.numerator * self.denominator
    
    def __ge__(self, other):
        try:
            other = Fraction._to_fraction(other)
        except TypeError:
            return NotImplemented
        return self.numerator * other.denominator >= other.numerator * self.denominator
    
    def __radd__(self, other):
        return self.__add__(other)
    
    def __rsub__(self, other):
        other = Fraction._to_fraction(other)
        return other.__sub__(self)
    
    def __rmul__(self, other):
        return self.__mul__(other)
    
    def __rtruediv__(self, other):
        other = Fraction._to_fraction(other)
        return other.__truediv__(self)
    
    def __str__(self):
        return f"{self.numerator}/{self.denominator}"

def parse_fraction(input_str):
    input_str = input_str.strip()
    mixed_pattern = r'^(-?\d+)\s+(\d+)/(\d+)$'
    simple_pattern = r'^(-?\d+)/(-?\d+)$'
    integer_pattern = r'^(-?\d+)$'
    
    mixed_match = re.match(mixed_pattern, input_str)
    if mixed_match:
        integer = int(mixed_match.group(1))
        numerator = int(mixed_match.group(2))
        denominator = int(mixed_match.group(3))
        if integer < 0:
            numerator = -numerator
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
    if frac.numerator == 0:
        return "0"
    
    if frac.denominator == 1:
        return str(frac.numerator)
    
    abs_num = abs(frac.numerator)
    sign = -1 if frac.numerator < 0 else 1
    integer = abs_num // frac.denominator
    remainder = abs_num % frac.denominator
    
    if integer == 0:
        return f"{sign * remainder}/{frac.denominator}"
    elif remainder == 0:
        return str(sign * integer)
    else:
        return f"{sign * integer} {remainder}/{frac.denominator}"

def main():
    print("=" * 50)
    print("分数四则运算计算器 (手动实现版)")
    print("=" * 50)
    print("支持的输入格式:")
    print("  - 真分数: 3/4")
    print("  - 假分数: 7/3")
    print("  - 带分数: 2 1/3")
    print("  - 负分数: -3/4, 3/-4")
    print("  - 负带分数: -2 1/3")
    print("  - 整数: 5, -3")
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
        
        add_result = frac1 + frac2
        print(f"\n加法: {to_mixed_fraction(frac1)} + {to_mixed_fraction(frac2)} = {to_mixed_fraction(add_result)}")
        print(f"  最简分数形式: {add_result.numerator}/{add_result.denominator}")
        
        sub_result = frac1 - frac2
        print(f"\n减法: {to_mixed_fraction(frac1)} - {to_mixed_fraction(frac2)} = {to_mixed_fraction(sub_result)}")
        print(f"  最简分数形式: {sub_result.numerator}/{sub_result.denominator}")
        
        mul_result = frac1 * frac2
        print(f"\n乘法: {to_mixed_fraction(frac1)} * {to_mixed_fraction(frac2)} = {to_mixed_fraction(mul_result)}")
        print(f"  最简分数形式: {mul_result.numerator}/{mul_result.denominator}")
        
        div_result = frac1 / frac2
        print(f"\n除法: {to_mixed_fraction(frac1)} ÷ {to_mixed_fraction(frac2)} = {to_mixed_fraction(div_result)}")
        print(f"  最简分数形式: {div_result.numerator}/{div_result.denominator}")
        
        print("\n" + "=" * 50)
        print("浮点数近似值:")
        print("=" * 50)
        print(f"\n  {to_mixed_fraction(frac1)} ≈ {frac1.to_float()}")
        print(f"  {to_mixed_fraction(frac1)} ≈ {frac1.to_float(4)} (保留4位小数)")
        print(f"  {to_mixed_fraction(frac2)} ≈ {frac2.to_float()}")
        print(f"  {to_mixed_fraction(frac2)} ≈ {frac2.to_float(4)} (保留4位小数)")
        
        print("\n" + "=" * 50)
        print("比较运算:")
        print("=" * 50)
        print(f"\n  {to_mixed_fraction(frac1)} == {to_mixed_fraction(frac2)} : {frac1 == frac2}")
        print(f"  {to_mixed_fraction(frac1)} != {to_mixed_fraction(frac2)} : {frac1 != frac2}")
        print(f"  {to_mixed_fraction(frac1)} < {to_mixed_fraction(frac2)}  : {frac1 < frac2}")
        print(f"  {to_mixed_fraction(frac1)} <= {to_mixed_fraction(frac2)} : {frac1 <= frac2}")
        print(f"  {to_mixed_fraction(frac1)} > {to_mixed_fraction(frac2)}  : {frac1 > frac2}")
        print(f"  {to_mixed_fraction(frac1)} >= {to_mixed_fraction(frac2)} : {frac1 >= frac2}")
        
        print("\n" + "=" * 50)
        print("混合运算示例 (分数与整数/浮点数):")
        print("=" * 50)
        print(f"\n  {to_mixed_fraction(frac1)} + 3 = {to_mixed_fraction(frac1 + 3)}")
        print(f"  3 + {to_mixed_fraction(frac1)} = {to_mixed_fraction(3 + frac1)}")
        print(f"  {to_mixed_fraction(frac1)} - 2 = {to_mixed_fraction(frac1 - 2)}")
        print(f"  2 - {to_mixed_fraction(frac1)} = {to_mixed_fraction(2 - frac1)}")
        print(f"  {to_mixed_fraction(frac1)} * 4 = {to_mixed_fraction(frac1 * 4)}")
        print(f"  4 * {to_mixed_fraction(frac1)} = {to_mixed_fraction(4 * frac1)}")
        print(f"  {to_mixed_fraction(frac1)} / 2 = {to_mixed_fraction(frac1 / 2)}")
        print(f"  2 / {to_mixed_fraction(frac1)} = {to_mixed_fraction(2 / frac1)}")
        print(f"\n  {to_mixed_fraction(frac1)} + 0.5 = {to_mixed_fraction(frac1 + 0.5)}")
        print(f"  0.5 + {to_mixed_fraction(frac1)} = {to_mixed_fraction(0.5 + frac1)}")
        print(f"  {to_mixed_fraction(frac1)} * 1.5 = {to_mixed_fraction(frac1 * 1.5)}")
        
    except ValueError as e:
        print(f"\n错误: {e}")
    except ZeroDivisionError as e:
        print(f"\n错误: {e}")
    except TypeError as e:
        print(f"\n错误: {e}")
    except KeyboardInterrupt:
        print("\n\n程序已退出")

if __name__ == "__main__":
    main()
