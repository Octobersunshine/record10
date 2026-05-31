INT32_MIN = -2 ** 31
INT32_MAX = 2 ** 31 - 1


def _is_float(num):
    if isinstance(num, float):
        return True
    if isinstance(num, str):
        return '.' in num
    return False


def _reverse_float(num):
    if isinstance(num, float):
        num_str = str(num)
    else:
        num_str = num.strip()

    sign = -1 if num_str.startswith('-') else 1
    abs_str = num_str.lstrip('-')

    if '.' in abs_str:
        int_part, dec_part = abs_str.split('.', 1)
        reversed_str = dec_part + '.' + int_part
    else:
        reversed_str = abs_str[::-1]

    result = sign * float(reversed_str)
    return result


def reverse_number(num, base=10, return_palindrome=False):
    if not isinstance(base, int) or base not in [2, 8, 10, 16]:
        raise ValueError("base must be 2, 8, 10, or 16")

    if _is_float(num):
        if base != 10:
            raise ValueError("float numbers only support base 10")
        result = _reverse_float(num)
        if return_palindrome:
            return result, is_palindrome(result)
        return result

    if isinstance(num, str):
        num_str = num.strip()
        sign = -1 if num_str.startswith('-') else 1
        abs_str = num_str.lstrip('-')
        try:
            abs_num = int(abs_str, base)
        except ValueError:
            raise ValueError(f"invalid number for base {base}: {num}")
    elif isinstance(num, int):
        sign = -1 if num < 0 else 1
        abs_num = abs(num)
    else:
        raise TypeError("num must be int, float, or str")

    if base == 10:
        reversed_abs = int(str(abs_num)[::-1])
    else:
        if base == 2:
            num_str = bin(abs_num)[2:]
        elif base == 8:
            num_str = oct(abs_num)[2:]
        elif base == 16:
            num_str = hex(abs_num)[2:]

        reversed_str = num_str[::-1]
        reversed_abs = int(reversed_str, base)

    result = sign * reversed_abs

    if result < INT32_MIN or result > INT32_MAX:
        result = 0

    if return_palindrome:
        return result, is_palindrome(result)
    return result


def is_palindrome(num):
    if isinstance(num, float):
        num_str = str(num).lstrip('-')
        return num_str == num_str[::-1]
    elif isinstance(num, int):
        if num < 0:
            return False
        return str(num) == str(num)[::-1]
    elif isinstance(num, str):
        clean_str = num.strip().lstrip('-')
        return clean_str == clean_str[::-1]
    else:
        raise TypeError("num must be int, float, or str")


def reverse_batch(items, base=10, return_palindrome=False):
    results = []
    for item in items:
        try:
            result = reverse_number(item, base, return_palindrome)
            results.append(result)
        except (ValueError, TypeError) as e:
            results.append(None)
    return results


if __name__ == "__main__":
    print("=" * 60)
    print("1. 整数反转测试")
    print("=" * 60)
    print("十进制:")
    print(f"  123 -> {reverse_number(123)}")
    print(f"  -123 -> {reverse_number(-123)}")
    print(f"  100 -> {reverse_number(100)}")

    print("\n二进制:")
    print(f"  0b101 (5) -> {reverse_number('101', 2)} (反转后: 101=5)")
    print(f"  0b1010 (10) -> {reverse_number('1010', 2)} (反转后: 0101=5)")

    print("\n八进制:")
    print(f"  0o123 (83) -> {reverse_number('123', 8)} (反转后: 321=209)")

    print("\n十六进制:")
    print(f"  0x1a3 (419) -> {reverse_number('1a3', 16)} (反转后: 3a1=929)")

    print("\n" + "=" * 60)
    print("2. 32位整数溢出测试")
    print("=" * 60)
    print(f"  1534236469 -> {reverse_number(1534236469)} (溢出，返回0)")
    print(f"  2147483647 -> {reverse_number(2147483647)} (溢出，返回0)")
    print(f"  1111111111 -> {reverse_number(1111111111)} (不溢出)")

    print("\n" + "=" * 60)
    print("3. 浮点数反转测试")
    print("=" * 60)
    print(f"  3.14 -> {reverse_number(3.14)}")
    print(f"  123.456 -> {reverse_number(123.456)}")
    print(f"  -7.89 -> {reverse_number(-7.89)}")
    print(f"  '0.001' -> {reverse_number('0.001')}")
    print(f"  '100.0' -> {reverse_number('100.0')}")

    print("\n" + "=" * 60)
    print("4. 回文数判断测试")
    print("=" * 60)
    print(f"  121 反转 -> {reverse_number(121, return_palindrome=True)}")
    print(f"  123 反转 -> {reverse_number(123, return_palindrome=True)}")
    print(f"  12.21 反转 -> {reverse_number(12.21, return_palindrome=True)}")
    print(f"  3.14 反转 -> {reverse_number(3.14, return_palindrome=True)}")

    print("\n直接判断回文数:")
    print(f"  is_palindrome(121) -> {is_palindrome(121)}")
    print(f"  is_palindrome(123) -> {is_palindrome(123)}")
    print(f"  is_palindrome(-121) -> {is_palindrome(-121)}")
    print(f"  is_palindrome(12.21) -> {is_palindrome(12.21)}")
    print(f"  is_palindrome('12321') -> {is_palindrome('12321')}")

    print("\n" + "=" * 60)
    print("5. 批量反转测试")
    print("=" * 60)
    test_items = [123, -456, 789, 3.14, 121, '1010', 'abc', None]
    results = reverse_batch(test_items, return_palindrome=True)
    for item, result in zip(test_items, results):
        print(f"  {item!r:>10} -> {result}")

    print("\n不同进制批量反转:")
    hex_items = ['1a3', 'ff', 'abc', '-12']
    hex_results = reverse_batch(hex_items, base=16)
    for item, result in zip(hex_items, hex_results):
        print(f"  {item!r:>10} (hex) -> {result}")
