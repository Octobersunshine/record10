import time


def _process_sign(num: str) -> tuple:
    negative = False
    if num.startswith('-'):
        negative = True
        num = num[1:]
    return negative, _remove_leading_zeros(num)


def _is_zero(num: str) -> bool:
    return _remove_leading_zeros(num.lstrip('-')) == '0'


def _add_unsigned(a: str, b: str) -> str:
    a_list = [int(c) for c in a]
    b_list = [int(c) for c in b]
    
    i, j = len(a_list) - 1, len(b_list) - 1
    carry = 0
    result = []
    
    while i >= 0 or j >= 0 or carry > 0:
        sum_digits = carry
        if i >= 0:
            sum_digits += a_list[i]
            i -= 1
        if j >= 0:
            sum_digits += b_list[j]
            j -= 1
        carry = sum_digits // 10
        result.append(sum_digits % 10)
    
    return ''.join(str(d) for d in reversed(result))


def _subtract_unsigned(a: str, b: str) -> str:
    cmp_result = _compare(a, b)
    if cmp_result < 0:
        return '-' + _subtract_unsigned(b, a)
    if cmp_result == 0:
        return '0'
    
    a_list = [int(c) for c in a]
    b_list = [int(c) for c in b]
    
    i, j = len(a_list) - 1, len(b_list) - 1
    borrow = 0
    result = []
    
    while i >= 0:
        digit_a = a_list[i] - borrow
        digit_b = b_list[j] if j >= 0 else 0
        borrow = 0
        
        if digit_a < digit_b:
            digit_a += 10
            borrow = 1
        
        result.append(digit_a - digit_b)
        i -= 1
        j -= 1
    
    while len(result) > 1 and result[-1] == 0:
        result.pop()
    
    return ''.join(str(d) for d in reversed(result))


def add(a: str, b: str) -> str:
    neg_a, abs_a = _process_sign(a)
    neg_b, abs_b = _process_sign(b)
    
    if not neg_a and not neg_b:
        return _add_unsigned(abs_a, abs_b)
    if neg_a and neg_b:
        return '-' + _add_unsigned(abs_a, abs_b)
    if neg_a and not neg_b:
        return subtract(b, abs_a)
    if not neg_a and neg_b:
        return subtract(a, abs_b)


def subtract(a: str, b: str) -> str:
    neg_a, abs_a = _process_sign(a)
    neg_b, abs_b = _process_sign(b)
    
    if not neg_a and not neg_b:
        cmp_result = _compare(abs_a, abs_b)
        if cmp_result < 0:
            return '-' + _subtract_unsigned(abs_b, abs_a)
        if cmp_result == 0:
            return '0'
        return _subtract_unsigned(abs_a, abs_b)
    if neg_a and neg_b:
        return subtract(abs_b, abs_a)
    if neg_a and not neg_b:
        return '-' + _add_unsigned(abs_a, abs_b)
    if not neg_a and neg_b:
        return _add_unsigned(abs_a, abs_b)


def multiply(a: str, b: str) -> str:
    if _is_zero(a) or _is_zero(b):
        return '0'
    
    neg_a, abs_a = _process_sign(a)
    neg_b, abs_b = _process_sign(b)
    
    negative = neg_a ^ neg_b
    
    m, n = len(abs_a), len(abs_b)
    result = [0] * (m + n)
    
    for i in range(m - 1, -1, -1):
        for j in range(n - 1, -1, -1):
            product = int(abs_a[i]) * int(abs_b[j])
            p1, p2 = i + j, i + j + 1
            total = product + result[p2]
            result[p1] += total // 10
            result[p2] = total % 10
    
    start = 0
    while start < len(result) and result[start] == 0:
        start += 1
    
    result_str = ''.join(str(d) for d in result[start:])
    return '-' + result_str if negative else result_str


KARATSUBA_THRESHOLD = 64


def _multiply_unsigned_karatsuba(a: str, b: str) -> str:
    if len(a) <= KARATSUBA_THRESHOLD or len(b) <= KARATSUBA_THRESHOLD:
        return _multiply_unsigned_column(a, b)
    
    max_len = max(len(a), len(b))
    half = max_len // 2
    
    a_high, a_low = _split_at(a, half)
    b_high, b_low = _split_at(b, half)
    
    z2 = _multiply_unsigned_karatsuba(a_high, b_high)
    z0 = _multiply_unsigned_karatsuba(a_low, b_low)
    
    sum_a = _add_unsigned(a_high, a_low)
    sum_b = _add_unsigned(b_high, b_low)
    z1_full = _multiply_unsigned_karatsuba(sum_a, sum_b)
    z1 = _subtract_unsigned(_subtract_unsigned(z1_full, z2), z0)
    if z1.startswith('-'):
        z1 = z1[1:]
    
    z2_shifted = z2 + '0' * (2 * half)
    z1_shifted = z1 + '0' * half
    
    result = _add_unsigned(_add_unsigned(z2_shifted, z1_shifted), z0)
    return result


def _split_at(num: str, pos: int) -> tuple:
    if len(num) <= pos:
        return ('0', num)
    high = num[:len(num) - pos]
    low = num[len(num) - pos:]
    return (high, low)


def _multiply_unsigned_column(a: str, b: str) -> str:
    if a == '0' or b == '0':
        return '0'
    
    m, n = len(a), len(b)
    result = [0] * (m + n)
    
    for i in range(m - 1, -1, -1):
        for j in range(n - 1, -1, -1):
            product = int(a[i]) * int(b[j])
            p1, p2 = i + j, i + j + 1
            total = product + result[p2]
            result[p1] += total // 10
            result[p2] = total % 10
    
    start = 0
    while start < len(result) and result[start] == 0:
        start += 1
    
    return ''.join(str(d) for d in result[start:])


def karatsuba_multiply(a: str, b: str) -> str:
    if _is_zero(a) or _is_zero(b):
        return '0'
    
    neg_a, abs_a = _process_sign(a)
    neg_b, abs_b = _process_sign(b)
    
    negative = neg_a ^ neg_b
    
    result_str = _multiply_unsigned_karatsuba(abs_a, abs_b)
    return '-' + result_str if negative else result_str


def benchmark_multiply(a: str, b: str) -> dict:
    results = {}
    
    start = time.perf_counter()
    result_column = multiply(a, b)
    time_column = time.perf_counter() - start
    results['column'] = {'result': result_column, 'time': time_column}
    
    start = time.perf_counter()
    result_karatsuba = karatsuba_multiply(a, b)
    time_karatsuba = time.perf_counter() - start
    results['karatsuba'] = {'result': result_karatsuba, 'time': time_karatsuba}
    
    results['match'] = (result_column == result_karatsuba)
    results['speedup'] = time_column / time_karatsuba if time_karatsuba > 0 else float('inf')
    
    return results


def divide(dividend: str, divisor: str) -> tuple:
    if _is_zero(divisor):
        raise ZeroDivisionError("integer division or modulo by zero")
    
    if _is_zero(dividend):
        return ('0', '0')
    
    neg_dividend, abs_dividend = _process_sign(dividend)
    neg_divisor, abs_divisor = _process_sign(divisor)
    
    negative = neg_dividend ^ neg_divisor
    
    if _compare(abs_dividend, abs_divisor) < 0:
        if not negative:
            return ('0', abs_dividend)
        else:
            return ('-1', _subtract_unsigned(abs_divisor, abs_dividend))
    
    quotient = []
    remainder = '0'
    
    for digit in abs_dividend:
        remainder = _remove_leading_zeros(remainder + digit)
        
        q = 0
        while _compare(remainder, abs_divisor) >= 0:
            remainder = _subtract_unsigned(remainder, abs_divisor)
            if remainder.startswith('-'):
                remainder = remainder[1:]
            q += 1
        
        quotient.append(str(q))
    
    quotient_str = _remove_leading_zeros(''.join(quotient))
    if not quotient_str:
        quotient_str = '0'
    
    if negative and quotient_str != '0':
        if _is_zero(remainder):
            quotient_str = '-' + quotient_str
        else:
            quotient_str = '-' + _add_unsigned(quotient_str, '1')
            remainder = _subtract_unsigned(abs_divisor, remainder)
    
    if neg_divisor and not _is_zero(remainder):
        remainder = '-' + remainder
    
    return (quotient_str, remainder)


def divide_decimal(dividend: str, divisor: str, decimal_places: int = 10) -> str:
    if _is_zero(divisor):
        raise ZeroDivisionError("integer division or modulo by zero")
    
    if _is_zero(dividend):
        return '0' + ('.' + '0' * decimal_places if decimal_places > 0 else '')
    
    neg_dividend, abs_dividend = _process_sign(dividend)
    neg_divisor, abs_divisor = _process_sign(divisor)
    
    negative = neg_dividend ^ neg_divisor
    
    scaled = abs_dividend + '0' * decimal_places
    
    quotient_digits = []
    remainder = '0'
    
    for digit in scaled:
        remainder = _remove_leading_zeros(remainder + digit)
        
        q = 0
        while _compare(remainder, abs_divisor) >= 0:
            remainder = _subtract_unsigned(remainder, abs_divisor)
            if remainder.startswith('-'):
                remainder = remainder[1:]
            q += 1
        
        quotient_digits.append(str(q))
    
    integer_len = len(abs_dividend)
    quotient_str = ''.join(quotient_digits)
    
    if len(quotient_str) <= integer_len:
        quotient_str = '0' * (integer_len - len(quotient_str) + 1) + quotient_str
    
    integer_part = quotient_str[:len(quotient_str) - decimal_places] if decimal_places > 0 else quotient_str
    decimal_part = quotient_str[len(quotient_str) - decimal_places:] if decimal_places > 0 else ''
    
    integer_part = _remove_leading_zeros(integer_part) if integer_part else '0'
    if not integer_part:
        integer_part = '0'
    
    if decimal_places > 0:
        result = integer_part + '.' + decimal_part
    else:
        result = integer_part
    
    if negative and result != '0':
        result = '-' + result
    
    return result


def _compare(a: str, b: str) -> int:
    a = _remove_leading_zeros(a)
    b = _remove_leading_zeros(b)
    
    if len(a) > len(b):
        return 1
    if len(a) < len(b):
        return -1
    
    for i in range(len(a)):
        if a[i] > b[i]:
            return 1
        if a[i] < b[i]:
            return -1
    
    return 0


def _remove_leading_zeros(s: str) -> str:
    if not s:
        return '0'
    i = 0
    while i < len(s) - 1 and s[i] == '0':
        i += 1
    return s[i:]
