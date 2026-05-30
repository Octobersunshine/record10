import time
import random


def manacher_core(s: str) -> tuple:
    if not s:
        return "", 0, 0

    if len(s) == 1:
        return s, 1, 0

    t = '#' + '#'.join(s) + '#'
    n = len(t)
    P = [0] * n
    C = R = 0
    max_len = 0
    center_idx = 0

    for i in range(n):
        mirror = 2 * C - i
        if i < R:
            P[i] = min(R - i, P[mirror])

        a = i + P[i] + 1
        b = i - P[i] - 1
        while a < n and b >= 0 and t[a] == t[b]:
            P[i] += 1
            a += 1
            b -= 1

        if i + P[i] > R:
            C = i
            R = i + P[i]

        if P[i] > max_len:
            max_len = P[i]
            center_idx = i

    start = (center_idx - max_len) // 2
    longest = s[start:start + max_len]
    return longest, max_len, start, P


def manacher(s: str) -> dict:
    if not s:
        return {"longest": "", "length": 0, "start": 0, "all_palindromes": set(), "count": 0}

    if len(s) == 1:
        return {"longest": s, "length": 1, "start": 0, "all_palindromes": {s}, "count": 1}

    longest, max_len, start, P = manacher_core(s)

    t = '#' + '#'.join(s) + '#'
    n = len(t)
    all_palindromes = set()

    for ch in s:
        all_palindromes.add(ch)

    for i in range(n):
        if i % 2 == 1:
            for L in range(3, P[i] + 1, 2):
                orig_start = (i - L) // 2
                all_palindromes.add(s[orig_start:orig_start + L])
        else:
            for L in range(2, P[i] + 1, 2):
                orig_start = (i - L) // 2
                all_palindromes.add(s[orig_start:orig_start + L])

    return {
        "longest": longest,
        "length": max_len,
        "start": start,
        "all_palindromes": all_palindromes,
        "count": len(all_palindromes),
    }


def center_expand(s: str) -> dict:
    if not s:
        return {"longest": "", "length": 0, "start": 0, "all_palindromes": set(), "count": 0}

    if len(s) == 1:
        return {"longest": s, "length": 1, "start": 0, "all_palindromes": {s}, "count": 1}

    all_palindromes = set()
    max_len = 1
    start = 0

    for ch in s:
        all_palindromes.add(ch)

    for i in range(len(s)):
        l, r = i - 1, i + 1
        while l >= 0 and r < len(s) and s[l] == s[r]:
            sub = s[l:r + 1]
            all_palindromes.add(sub)
            if r - l + 1 > max_len:
                max_len = r - l + 1
                start = l
            l -= 1
            r += 1

        l, r = i, i + 1
        while l >= 0 and r < len(s) and s[l] == s[r]:
            sub = s[l:r + 1]
            all_palindromes.add(sub)
            if r - l + 1 > max_len:
                max_len = r - l + 1
                start = l
            l -= 1
            r += 1

    longest = s[start:start + max_len]
    return {
        "longest": longest,
        "length": max_len,
        "start": start,
        "all_palindromes": all_palindromes,
        "count": len(all_palindromes),
    }


def benchmark_core(s: str, iterations: int = 100):
    manacher_time = 0.0
    center_time = 0.0

    for _ in range(iterations):
        t0 = time.perf_counter()
        manacher_core(s)
        manacher_time += time.perf_counter() - t0

        t0 = time.perf_counter()
        _center_expand_core(s)
        center_time += time.perf_counter() - t0

    return manacher_time / iterations, center_time / iterations


def _center_expand_core(s: str) -> tuple:
    if not s:
        return "", 0, 0

    max_len = 1
    start = 0

    for i in range(len(s)):
        l, r = i - 1, i + 1
        while l >= 0 and r < len(s) and s[l] == s[r]:
            if r - l + 1 > max_len:
                max_len = r - l + 1
                start = l
            l -= 1
            r += 1

        l, r = i, i + 1
        while l >= 0 and r < len(s) and s[l] == s[r]:
            if r - l + 1 > max_len:
                max_len = r - l + 1
                start = l
            l -= 1
            r += 1

    return s[start:start + max_len], max_len, start


def benchmark_full(s: str, iterations: int = 100):
    manacher_time = 0.0
    center_time = 0.0

    for _ in range(iterations):
        t0 = time.perf_counter()
        manacher(s)
        manacher_time += time.perf_counter() - t0

        t0 = time.perf_counter()
        center_expand(s)
        center_time += time.perf_counter() - t0

    return manacher_time / iterations, center_time / iterations


def generate_test_string(length: int, charset: str = "abc") -> str:
    return ''.join(random.choice(charset) for _ in range(length))


if __name__ == "__main__":
    print("=" * 60)
    print("功能验证")
    print("=" * 60)

    test_cases = [
        "babad",
        "cbbd",
        "a",
        "ac",
        "aaaa",
        "abcba",
        "abacdfgdcaba",
        "",
    ]

    for s in test_cases:
        m = manacher(s)
        c = center_expand(s)
        print(f"字符串: '{s}'")
        print(f"  Manacher     -> 最长回文: '{m['longest']}', 长度: {m['length']}, 起始: {m['start']}")
        print(f"  中心扩展法   -> 最长回文: '{c['longest']}', 长度: {c['length']}, 起始: {c['start']}")
        print(f"  所有回文子串(Manacher):   {sorted(m['all_palindromes'])}")
        print(f"  所有回文子串(中心扩展法): {sorted(c['all_palindromes'])}")
        print(f"  不重复回文数(Manacher): {m['count']}, (中心扩展法): {c['count']}")
        match = m['all_palindromes'] == c['all_palindromes']
        print(f"  结果一致性: {'OK' if match else 'MISMATCH'}")
        print()

    print("=" * 60)
    print("性能对比1: 随机字符串 - 仅查找最长回文")
    print("  (随机串回文短, 中心扩展未触发最坏情况)")
    print("=" * 60)

    sizes = [100, 500, 1000, 3000, 5000, 10000]
    print(f"{'长度':>8} | {'Manacher (ms)':>14} | {'中心扩展法 (ms)':>16} | {'加速比':>8}")
    print("-" * 60)

    for size in sizes:
        s = generate_test_string(size, "abc")
        m_time, c_time = benchmark_core(s, iterations=100)
        ratio = c_time / m_time if m_time > 0 else float('inf')
        print(f"{size:>8} | {m_time * 1000:>14.4f} | {c_time * 1000:>16.4f} | {ratio:>7.2f}x")

    print()
    print("=" * 60)
    print("性能对比2: 最坏情况 'a'*n - 中心扩展真正 O(n^2)")
    print("  (全相同字符, 每个中心扩展到边界)")
    print("=" * 60)

    sizes_worst = [100, 500, 1000, 3000, 5000, 10000, 20000]
    print(f"{'长度':>8} | {'Manacher (ms)':>14} | {'中心扩展法 (ms)':>16} | {'加速比':>8}")
    print("-" * 60)

    for size in sizes_worst:
        s = 'a' * size
        iters = max(5, 100 // max(1, size // 100))
        m_time, c_time = benchmark_core(s, iterations=iters)
        ratio = c_time / m_time if m_time > 0 else float('inf')
        print(f"{size:>8} | {m_time * 1000:>14.4f} | {c_time * 1000:>16.4f} | {ratio:>7.2f}x")

    print()
    print("=" * 60)
    print("性能对比3: 含所有回文子串提取 (两者均为 O(n^2))")
    print("=" * 60)

    sizes_full = [100, 500, 1000, 2000]
    print(f"{'长度':>8} | {'Manacher (ms)':>14} | {'中心扩展法 (ms)':>16} | {'加速比':>8}")
    print("-" * 60)

    for size in sizes_full:
        s = generate_test_string(size, "abc")
        m_time, c_time = benchmark_full(s, iterations=50)
        ratio = c_time / m_time if m_time > 0 else float('inf')
        print(f"{size:>8} | {m_time * 1000:>14.4f} | {c_time * 1000:>16.4f} | {ratio:>7.2f}x")
