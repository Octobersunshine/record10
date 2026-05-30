def gray_code_reflection(n):
    if n == 0:
        return []
    if n == 1:
        return ['0', '1']
    
    prev_gray = gray_code_reflection(n - 1)
    
    result = []
    for code in prev_gray:
        result.append('0' + code)
    for code in reversed(prev_gray):
        result.append('1' + code)
    
    return result


def gray_code_int(n):
    gray_strings = gray_code_reflection(n)
    return [int(code, 2) for code in gray_strings]


def cyclic_gray_code(n):
    if n == 0:
        return []
    return gray_code_int(n)


def cyclic_gray_code_str(n):
    codes = cyclic_gray_code(n)
    if n == 0:
        return []
    return [format(c, f'0{n}b') for c in codes]


def verify_cyclic_gray_code(codes):
    if len(codes) <= 1:
        return True
    for i in range(len(codes)):
        j = (i + 1) % len(codes)
        xor = codes[i] ^ codes[j]
        if bin(xor).count('1') != 1:
            return False
    return True


def partial_gray_code(n, k):
    if n == 0 or k <= 0:
        return []
    if k >= 2 ** n:
        return gray_code_int(n)
    
    result = [0]
    used = {0}
    
    def hamming_distance(a, b):
        return bin(a ^ b).count('1')
    
    while len(result) < k:
        last = result[-1]
        found = False
        for i in range(n):
            candidate = last ^ (1 << i)
            if candidate not in used:
                result.append(candidate)
                used.add(candidate)
                found = True
                break
        if not found:
            break
    
    return result


def partial_gray_code_str(n, k):
    codes = partial_gray_code(n, k)
    if n == 0:
        return []
    return [format(c, f'0{n}b') for c in codes]


def hamming_distance_distribution(codes):
    if len(codes) < 2:
        return {}
    
    distribution = {}
    for i in range(len(codes) - 1):
        dist = bin(codes[i] ^ codes[i + 1]).count('1')
        distribution[dist] = distribution.get(dist, 0) + 1
    
    return dict(sorted(distribution.items()))


def cyclic_hamming_distance_distribution(codes):
    if len(codes) < 2:
        return {}
    
    distribution = {}
    n = len(codes)
    for i in range(n):
        j = (i + 1) % n
        dist = bin(codes[i] ^ codes[j]).count('1')
        distribution[dist] = distribution.get(dist, 0) + 1
    
    return dict(sorted(distribution.items()))


def binary_to_gray(binary):
    if isinstance(binary, str):
        binary = int(binary, 2)
    return binary ^ (binary >> 1)


def gray_to_binary(gray):
    if isinstance(gray, str):
        gray = int(gray, 2)
    binary = gray
    while gray > 0:
        gray >>= 1
        binary ^= gray
    return binary


def gray_to_binary_str(gray_str):
    gray_int = int(gray_str, 2)
    binary_int = gray_to_binary(gray_int)
    return format(binary_int, f'0{len(gray_str)}b')


def binary_to_gray_str(binary_str):
    binary_int = int(binary_str, 2)
    gray_int = binary_to_gray(binary_int)
    return format(gray_int, f'0{len(binary_str)}b')


def verify_gray_code(codes):
    for i in range(len(codes) - 1):
        xor = codes[i] ^ codes[i + 1]
        if bin(xor).count('1') != 1:
            return False
    return True


if __name__ == '__main__':
    print("=== 测试 n=0 ===")
    print(f"n=0 格雷码: {gray_code_reflection(0)}")
    print(f"n=0 格雷码(整数): {gray_code_int(0)}")
    
    n = 3
    print(f"\n=== 生成 {n} 位格雷码（字符串形式）===")
    gray_str = gray_code_reflection(n)
    for i, code in enumerate(gray_str):
        print(f"{i:2d}: {code}")
    
    print(f"\n=== 生成 {n} 位格雷码（整数形式）===")
    gray_int = gray_code_int(n)
    for i, code in enumerate(gray_int):
        print(f"{i:2d}: {code}")
    
    print(f"\n=== 验证格雷码正确性 ===")
    print(f"验证结果: {verify_gray_code(gray_int)}")
    
    print(f"\n=== 双向转换测试 ===")
    test_values = [0, 1, 2, 3, 4, 5, 6, 7]
    print(f"{'十进制':>6} | {'二进制':>8} | {'格雷码(整数)':>10} | {'格雷码(字符串)':>12} | {'转回二进制':>10}")
    print("-" * 65)
    for val in test_values:
        binary_str = format(val, '03b')
        gray_int = binary_to_gray(val)
        gray_str = binary_to_gray_str(binary_str)
        back_binary = gray_to_binary(gray_int)
        print(f"{val:>6} | {binary_str:>8} | {gray_int:>10} | {gray_str:>12} | {back_binary:>10}")
    
    print(f"\n=== 字符串转换测试 ===")
    gray_code = '101'
    binary_code = gray_to_binary_str(gray_code)
    print(f"格雷码 {gray_code} -> 二进制 {binary_code}")
    print(f"二进制 {binary_code} -> 格雷码 {binary_to_gray_str(binary_code)}")
    
    print(f"\n=== 循环格雷码测试 ===")
    for n_bits in [2, 3, 4]:
        cyclic_codes = cyclic_gray_code(n_bits)
        cyclic_str = cyclic_gray_code_str(n_bits)
        print(f"{n_bits}位循环格雷码: {cyclic_str}")
        print(f"  循环验证: {verify_cyclic_gray_code(cyclic_codes)}")
        wrap_dist = bin(cyclic_codes[0] ^ cyclic_codes[-1]).count('1')
        print(f"  首尾汉明距离: {wrap_dist}")
    
    print(f"\n=== 非整数位（部分）格雷码测试 ===")
    for n_bits, k_len in [(3, 5), (4, 6), (3, 3)]:
        partial_codes = partial_gray_code(n_bits, k_len)
        partial_str = partial_gray_code_str(n_bits, k_len)
        print(f"{n_bits}位，取{k_len}个: {partial_str}")
        print(f"  相邻验证: {verify_gray_code(partial_codes)}")
    
    print(f"\n=== 汉明距离分布测试 ===")
    for n_bits in [3, 4]:
        codes = gray_code_int(n_bits)
        dist = hamming_distance_distribution(codes)
        print(f"{n_bits}位标准格雷码相邻距离分布: {dist}")
        cyclic_dist = cyclic_hamming_distance_distribution(codes)
        print(f"{n_bits}位格雷码循环距离分布: {cyclic_dist}")
    
    print(f"\n=== 部分格雷码距离分布 ===")
    partial_codes = partial_gray_code(4, 10)
    partial_dist = hamming_distance_distribution(partial_codes)
    print(f"4位取10个的相邻距离分布: {partial_dist}")
