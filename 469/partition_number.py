def partition_number(n):
    if n < 0:
        return 0
    if n == 0:
        return 1

    dp = [0] * (n + 1)
    dp[0] = 1

    for i in range(1, n + 1):
        k = 1
        while True:
            p1 = k * (3 * k - 1) // 2
            if p1 > i:
                break
            sign = 1 if k % 2 == 1 else -1
            dp[i] += sign * dp[i - p1]

            p2 = k * (3 * k + 1) // 2
            if p2 > i:
                k += 1
                continue
            dp[i] += sign * dp[i - p2]
            k += 1

    return dp[n]


def partition_number_restricted(n, m):
    if n < 0:
        return 0
    if n == 0:
        return 1
    if m <= 0:
        return 0
    if m > n:
        m = n

    dp = [0] * (n + 1)
    dp[0] = 1

    for j in range(1, m + 1):
        for i in range(j, n + 1):
            dp[i] += dp[i - j]

    return dp[n]


def partition_number_k_parts(n, k):
    if n < 0 or k <= 0 or k > n:
        return 0
    if k == n:
        return 1
    if k == 1:
        return 1

    dp_prev = [0] * (n + 1)
    dp_curr = [0] * (n + 1)

    for i in range(1, n + 1):
        dp_prev[i] = 1

    for parts in range(2, k + 1):
        dp_curr = [0] * (n + 1)
        for i in range(parts, n + 1):
            dp_curr[i] = dp_curr[i - parts] + dp_prev[i - 1]
        dp_prev = dp_curr

    return dp_prev[n]


def partition_number_distinct(n):
    if n < 0:
        return 0
    if n == 0:
        return 1

    dp = [0] * (n + 1)
    dp[0] = 1

    for j in range(1, n + 1):
        for i in range(n, j - 1, -1):
            dp[i] += dp[i - j]

    return dp[n]


def partition_enumerate(n, max_part=None, distinct=False):
    if n < 0:
        return []
    if n == 0:
        return [[]]
    if max_part is None:
        max_part = n

    results = []

    def backtrack(remaining, max_val, current):
        if remaining == 0:
            results.append(current[:])
            return
        upper = min(remaining, max_val)
        for part in range(upper, 0, -1):
            current.append(part)
            next_max = part - 1 if distinct else part
            backtrack(remaining - part, next_max, current)
            current.pop()

    backtrack(n, max_part, [])
    return results


def partition_enumerate_k_parts(n, k, distinct=False):
    if n < 0 or k <= 0 or k > n:
        return []

    results = []

    def backtrack(remaining, parts_left, max_val, current):
        if parts_left == 0:
            if remaining == 0:
                results.append(current[:])
            return
        if remaining < parts_left:
            return
        upper = min(remaining, max_val)
        for part in range(upper, 0, -1):
            current.append(part)
            next_max = part - 1 if distinct else part
            backtrack(remaining - part, parts_left - 1, next_max, current)
            current.pop()

    backtrack(n, k, n, [])
    return results


if __name__ == "__main__":
    print("=" * 60)
    print("1. 基本划分数 p(n)")
    print("=" * 60)
    for n in [0, 1, 2, 3, 4, 5, 10, 50, 100]:
        print(f"  p({n}) = {partition_number(n)}")

    print()
    print("=" * 60)
    print("2. 带限制的划分数 p(n, m)：各部分 <= m")
    print("=" * 60)
    for n, m in [(5, 2), (5, 3), (5, 5), (10, 3), (10, 5), (20, 5)]:
        print(f"  p({n}, max_part={m}) = {partition_number_restricted(n, m)}")

    print()
    print("=" * 60)
    print("3. 恰好 k 个部分的划分数 p_k(n, k)")
    print("=" * 60)
    for n, k in [(5, 1), (5, 2), (5, 3), (5, 5), (10, 3), (10, 4)]:
        print(f"  p_{k}({n}) = {partition_number_k_parts(n, k)}")

    print()
    print("=" * 60)
    print("4. 拆分为不同整数的划分数 q(n)：无重复部分")
    print("=" * 60)
    for n in [0, 1, 2, 3, 4, 5, 10, 20, 50, 100]:
        print(f"  q({n}) = {partition_number_distinct(n)}")

    print()
    print("=" * 60)
    print("5. 具体划分方案（小 n）")
    print("=" * 60)

    print("\n  --- 5 的所有划分 ---")
    for p in partition_enumerate(5):
        print(f"    {p}")

    print("\n  --- 5 的划分（各部分 <= 3）---")
    for p in partition_enumerate(5, max_part=3):
        print(f"    {p}")

    print("\n  --- 5 的划分（不同部分）---")
    for p in partition_enumerate(5, distinct=True):
        print(f"    {p}")

    print("\n  --- 5 的划分（恰好 2 个部分）---")
    for p in partition_enumerate_k_parts(5, 2):
        print(f"    {p}")

    print("\n  --- 5 的划分（恰好 2 个不同部分）---")
    for p in partition_enumerate_k_parts(5, 2, distinct=True):
        print(f"    {p}")

    print()
    print("=" * 60)
    print("6. 一致性验证")
    print("=" * 60)
    ok = True
    for n in range(1, 51):
        p1 = partition_number(n)
        p2 = partition_number_restricted(n, n)
        p3 = partition_number_distinct(n)
        if p1 != p2:
            print(f"  MISMATCH: p({n})={p1} vs restricted({n},{n})={p2}")
            ok = False
    print(f"  p(n) == p(n, max_part=n) for n=1..50: {'PASS' if ok else 'FAIL'}")

    ok = True
    for n in range(1, 30):
        total_k = sum(partition_number_k_parts(n, k) for k in range(1, n + 1))
        if total_k != partition_number(n):
            print(f"  MISMATCH: sum p_k({n})={total_k} vs p({n})={partition_number(n)}")
            ok = False
    print(f"  sum(p_k(n)) == p(n) for n=1..29: {'PASS' if ok else 'FAIL'}")

    ok = True
    for n in range(0, 20):
        enum_count = len(partition_enumerate(n))
        if enum_count != partition_number(n):
            print(f"  MISMATCH: enum({n})={enum_count} vs p({n})={partition_number(n)}")
            ok = False
    print(f"  enumerate(n) count == p(n) for n=0..19: {'PASS' if ok else 'FAIL'}")

    ok = True
    for n in range(0, 20):
        enum_dist = len(partition_enumerate(n, distinct=True))
        if enum_dist != partition_number_distinct(n):
            print(f"  MISMATCH: enum_dist({n})={enum_dist} vs q({n})={partition_number_distinct(n)}")
            ok = False
    print(f"  enumerate(n, distinct) count == q(n) for n=0..19: {'PASS' if ok else 'FAIL'}")
