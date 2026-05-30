import time
from functools import lru_cache


def matrix_chain_order(p):
    n = len(p) - 1
    m = [[0] * (n + 1) for _ in range(n + 1)]
    s = [[0] * (n + 1) for _ in range(n + 1)]

    for l in range(2, n + 1):
        for i in range(1, n - l + 2):
            j = i + l - 1
            m[i][j] = float('inf')
            for k in range(i, j):
                q = m[i][k] + m[k + 1][j] + p[i - 1] * p[k] * p[j]
                if q < m[i][j]:
                    m[i][j] = q
                    s[i][j] = k

    return m, s


def build_optimal_parens(s, i, j):
    if i == j:
        return f"A{i}"
    k = s[i][j]
    return f"({build_optimal_parens(s, i, k)}{build_optimal_parens(s, k + 1, j)})"


def build_nested_list(s, i, j):
    if i == j:
        return i
    k = s[i][j]
    return [build_nested_list(s, i, k), build_nested_list(s, k + 1, j)]


def build_multiply_order(s, i, j):
    steps = []
    if i == j:
        return i, steps

    k = s[i][j]
    left, left_steps = build_multiply_order(s, i, k)
    right, right_steps = build_multiply_order(s, k + 1, j)
    steps.extend(left_steps)
    steps.extend(right_steps)
    steps.append((left, right))
    result = (left, right)
    return result, steps


def matrix_chain_dp(p, output_format="string"):
    n = len(p) - 1
    if n < 2:
        if n == 1:
            return 0, "A" if output_format == "string" else (1, []) if output_format == "order" else 1
        else:
            return 0, "" if output_format == "string" else (None, []) if output_format == "order" else None

    m, s = matrix_chain_order(p)
    min_cost = m[1][n]

    if output_format == "string":
        parens = build_optimal_parens(s, 1, n)
        return min_cost, parens
    elif output_format == "list":
        nested = build_nested_list(s, 1, n)
        return min_cost, nested
    elif output_format == "order":
        _, steps = build_multiply_order(s, 1, n)
        return min_cost, steps
    else:
        raise ValueError(f"Unsupported output_format: {output_format}. Use 'string', 'list', or 'order'.")


def matrix_chain_memoized(p):
    n = len(p) - 1
    memo = {}

    def rec(i, j):
        if i == j:
            return 0
        if (i, j) in memo:
            return memo[(i, j)]
        memo[(i, j)] = float('inf')
        for k in range(i, j):
            cost = rec(i, k) + rec(k + 1, j) + p[i - 1] * p[k] * p[j]
            if cost < memo[(i, j)]:
                memo[(i, j)] = cost
        return memo[(i, j)]

    if n < 2:
        return 0

    return rec(1, n)


def matrix_chain_recursive_naive(p):
    n = len(p) - 1

    def rec(i, j):
        if i == j:
            return 0
        min_cost = float('inf')
        for k in range(i, j):
            cost = rec(i, k) + rec(k + 1, j) + p[i - 1] * p[k] * p[j]
            if cost < min_cost:
                min_cost = cost
        return min_cost

    if n < 2:
        return 0

    return rec(1, n)


def format_order_steps(steps):
    names = {}
    result = []
    counter = 1

    def get_name(expr):
        nonlocal counter
        if isinstance(expr, int):
            return f"A{expr}"
        key = expr
        if key not in names:
            names[key] = f"M{counter}"
            counter += 1
        return names[key]

    for left, right in steps:
        left_name = get_name(left)
        right_name = get_name(right)
        result.append(f"{left_name} × {right_name}")

    return result


def benchmark(p, repeat=5):
    print(f"\n{'=' * 60}")
    print(f"性能对比测试 (维度序列长度: {len(p)}, 矩阵数: {len(p)-1})")
    print(f"维度序列: {p}")
    print(f"{'=' * 60}")

    dp_times = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        cost_dp, _ = matrix_chain_dp(p, "string")
        t1 = time.perf_counter()
        dp_times.append((t1 - t0) * 1000)

    memo_times = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        cost_memo = matrix_chain_memoized(p)
        t1 = time.perf_counter()
        memo_times.append((t1 - t0) * 1000)

    n = len(p) - 1
    if n <= 12:
        naive_times = []
        for _ in range(max(1, repeat // 2)):
            t0 = time.perf_counter()
            cost_naive = matrix_chain_recursive_naive(p)
            t1 = time.perf_counter()
            naive_times.append((t1 - t0) * 1000)
        avg_naive = sum(naive_times) / len(naive_times)
        print(f"朴素递归   : {avg_naive:8.3f} ms (结果: {cost_naive})")
    else:
        avg_naive = None
        print(f"朴素递归   : 跳过 (n={n} 太大会超时)")

    avg_dp = sum(dp_times) / len(dp_times)
    avg_memo = sum(memo_times) / len(memo_times)

    print(f"动态规划   : {avg_dp:8.3f} ms (结果: {cost_dp})")
    print(f"备忘录递归 : {avg_memo:8.3f} ms (结果: {cost_memo})")

    if avg_naive and avg_naive > 0:
        print(f"\nDP 比朴素递归快: {avg_naive / avg_dp:.1f} 倍")
        print(f"备忘录 比朴素递归快: {avg_naive / avg_memo:.1f} 倍")
    print(f"DP 与备忘录 速度比: {avg_memo / avg_dp:.1f} 倍")

    assert cost_dp == cost_memo, "结果不一致!"
    if avg_naive:
        assert cost_dp == cost_naive, "结果不一致!"
    print(f"\n所有方法结果一致 ✓")


def main():
    p = [30, 35, 15, 5, 10, 20, 25]

    print(f"{'=' * 60}")
    print(f"矩阵链乘最优括号化 - 动态规划")
    print(f"{'=' * 60}")
    print(f"维度序列: {p}")
    print(f"矩阵数量: {len(p) - 1}")
    print()

    cost_str, parens_str = matrix_chain_dp(p, output_format="string")
    print(f"【字符串格式】")
    print(f"  最小乘法次数: {cost_str}")
    print(f"  括号化方案  : {parens_str}")
    print()

    cost_list, parens_list = matrix_chain_dp(p, output_format="list")
    print(f"【嵌套列表格式】")
    print(f"  最小乘法次数: {cost_list}")
    print(f"  括号化方案  : {parens_list}")
    print()

    cost_order, steps = matrix_chain_dp(p, output_format="order")
    print(f"【乘法顺序】")
    print(f"  最小乘法次数: {cost_order}")
    print(f"  计算步骤 ({len(steps)} 步):")
    formatted_steps = format_order_steps(steps)
    for idx, step in enumerate(formatted_steps, 1):
        print(f"    第 {idx} 步: {step}")
    print()

    print(f"原始步骤元组:")
    for idx, step in enumerate(steps, 1):
        print(f"    第 {idx} 步: {step}")
    print()

    print(f"\n{'=' * 60}")
    print(f"边界情况测试")
    print(f"{'=' * 60}")

    p0 = [10]
    cost0, par0 = matrix_chain_dp(p0, "string")
    print(f"0个矩阵 [10]: cost={cost0}, parens='{par0}'")

    p1 = [10, 20]
    cost1, par1 = matrix_chain_dp(p1, "string")
    print(f"1个矩阵 [10,20]: cost={cost1}, parens='{par1}'")

    p2 = [10, 20, 30]
    cost2, par2 = matrix_chain_dp(p2, "string")
    print(f"2个矩阵 [10,20,30]: cost={cost2}, parens='{par2}'")

    benchmark([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
    benchmark([5, 10, 3, 12, 5, 50, 6, 8, 2, 15, 7, 4, 9])
    benchmark([30, 35, 15, 5, 10, 20, 25, 18, 22, 12, 8, 16, 10, 5, 20])


if __name__ == "__main__":
    main()
