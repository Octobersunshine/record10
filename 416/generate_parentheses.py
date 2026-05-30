import time
from math import comb


def generate_parentheses(n):
    if not isinstance(n, int):
        raise TypeError('n must be an integer')
    if n < 0:
        raise ValueError('n must be non-negative')
    if n == 0:
        return []

    result = set()

    def backtrack(s, left, right):
        if len(s) == 2 * n:
            result.add(s)
            return
        if left < n:
            backtrack(s + '(', left + 1, right)
        if right < left:
            backtrack(s + ')', left, right + 1)

    backtrack('', 0, 0)
    return sorted(result)


def generate_multi_brackets(n):
    if not isinstance(n, int):
        raise TypeError('n must be an integer')
    if n < 0:
        raise ValueError('n must be non-negative')
    if n == 0:
        return []

    pairs = [('(', ')'), ('[', ']'), ('{', '}')]
    result = set()

    def backtrack(s, counts):
        if len(s) == 2 * n:
            result.add(s)
            return
        for open_b, close_b in pairs:
            if counts[open_b] < n:
                new_counts = dict(counts)
                new_counts[open_b] += 1
                backtrack(s + open_b, new_counts)
        for open_b, close_b in pairs:
            if counts[close_b] < counts[open_b]:
                new_counts = dict(counts)
                new_counts[close_b] += 1
                backtrack(s + close_b, new_counts)

    initial = {'(': 0, ')': 0, '[': 0, ']': 0, '{': 0, '}': 0}
    backtrack('', initial)
    return sorted(result)


def catalan_count(n):
    if not isinstance(n, int):
        raise TypeError('n must be an integer')
    if n < 0:
        raise ValueError('n must be non-negative')
    if n == 0:
        return 0
    return comb(2 * n, n) // (n + 1)


if __name__ == '__main__':
    print('=' * 60)
    print('单括号生成测试')
    print('=' * 60)
    for n in [0, 1, 2, 3]:
        combos = generate_parentheses(n)
        print(f'n = {n}: 共 {len(combos)} 种')
        for c in combos:
            print(f'  {c}')
    print()

    print('=' * 60)
    print('多括号生成测试 ((), [], {})')
    print('=' * 60)
    for n in [1, 2]:
        combos = generate_multi_brackets(n)
        print(f'n = {n}: 共 {len(combos)} 种')
        for c in combos:
            print(f'  {c}')
    print()

    print('=' * 60)
    print('卡特兰数计数测试')
    print('=' * 60)
    for n in range(11):
        print(f'C_{n} = {catalan_count(n)}')
    print()

    print('=' * 60)
    print('生成时间对比')
    print('=' * 60)
    print(f'{"n":>3} | {"单括号生成(s)":>14} | {"多括号生成(s)":>14} | {"卡特兰计数(s)":>14} | {"C_n":>10}')
    print('-' * 70)
    for n in range(1, 8):
        t1 = time.perf_counter()
        result_single = generate_parentheses(n)
        t_single = time.perf_counter() - t1

        if n <= 4:
            t1 = time.perf_counter()
            result_multi = generate_multi_brackets(n)
            t_multi = time.perf_counter() - t1
        else:
            t_multi = float('inf')

        t1 = time.perf_counter()
        c = catalan_count(n)
        t_catalan = time.perf_counter() - t1

        multi_str = f'{t_multi:.6f}' if t_multi != float('inf') else '    >10s    '
        print(f'{n:>3} | {t_single:>14.6f} | {multi_str:>14} | {t_catalan:>14.6f} | {c:>10}')
        print(f'      验证: 单括号={len(result_single)}, '
              f'多括号={len(result_multi) if t_multi != float("inf") else "skipped"}, '
              f'卡特兰={c}')
