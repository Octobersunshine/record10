from enum import Enum


class Op(Enum):
    MATCH = 0
    DELETE = 1
    INSERT = 2
    REPLACE = 3
    TRANSPOSE = 4


def levenshtein(s1, s2):
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    path = [[Op.MATCH] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        dp[i][0] = i
        path[i][0] = Op.DELETE
    for j in range(n + 1):
        dp[0][j] = j
        path[0][j] = Op.INSERT

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
                path[i][j] = Op.MATCH
            else:
                delete_cost = dp[i - 1][j] + 1
                insert_cost = dp[i][j - 1] + 1
                replace_cost = dp[i - 1][j - 1] + 1

                min_cost = min(delete_cost, insert_cost, replace_cost)
                dp[i][j] = min_cost

                if min_cost == replace_cost:
                    path[i][j] = Op.REPLACE
                elif min_cost == insert_cost:
                    path[i][j] = Op.INSERT
                else:
                    path[i][j] = Op.DELETE

    ops = []
    i, j = m, n
    while i > 0 or j > 0:
        if i == 0:
            ops.append(("INSERT", 0, s2[j - 1]))
            j -= 1
        elif j == 0:
            ops.append(("DELETE", i - 1, s1[i - 1]))
            i -= 1
        else:
            op = path[i][j]
            if op == Op.MATCH:
                ops.append(("MATCH", i - 1, s1[i - 1]))
                i -= 1
                j -= 1
            elif op == Op.REPLACE:
                ops.append(("REPLACE", i - 1, s1[i - 1], s2[j - 1]))
                i -= 1
                j -= 1
            elif op == Op.INSERT:
                ops.append(("INSERT", i, s2[j - 1]))
                j -= 1
            elif op == Op.DELETE:
                ops.append(("DELETE", i - 1, s1[i - 1]))
                i -= 1

    ops.reverse()
    return dp[m][n], ops


def levenshtein_optimized(s1, s2):
    if len(s1) < len(s2):
        return levenshtein_optimized(s2, s1)

    m, n = len(s1), len(s2)
    prev = list(range(n + 1))
    curr = [0] * (n + 1)

    for i in range(1, m + 1):
        curr[0] = i
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                curr[j] = prev[j - 1]
            else:
                curr[j] = 1 + min(prev[j], curr[j - 1], prev[j - 1])
        prev, curr = curr, prev

    return prev[n]


def weighted_levenshtein(s1, s2, insert_cost=1, delete_cost=1, replace_cost=1):
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    path = [[Op.MATCH] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        dp[i][0] = i * delete_cost
        path[i][0] = Op.DELETE
    for j in range(n + 1):
        dp[0][j] = j * insert_cost
        path[0][j] = Op.INSERT

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
                path[i][j] = Op.MATCH
            else:
                del_c = dp[i - 1][j] + delete_cost
                ins_c = dp[i][j - 1] + insert_cost
                rep_c = dp[i - 1][j - 1] + replace_cost

                min_cost = min(del_c, ins_c, rep_c)
                dp[i][j] = min_cost

                if min_cost == rep_c:
                    path[i][j] = Op.REPLACE
                elif min_cost == ins_c:
                    path[i][j] = Op.INSERT
                else:
                    path[i][j] = Op.DELETE

    ops = []
    i, j = m, n
    while i > 0 or j > 0:
        if i == 0:
            ops.append(("INSERT", 0, s2[j - 1]))
            j -= 1
        elif j == 0:
            ops.append(("DELETE", i - 1, s1[i - 1]))
            i -= 1
        else:
            op = path[i][j]
            if op == Op.MATCH:
                ops.append(("MATCH", i - 1, s1[i - 1]))
                i -= 1
                j -= 1
            elif op == Op.REPLACE:
                ops.append(("REPLACE", i - 1, s1[i - 1], s2[j - 1]))
                i -= 1
                j -= 1
            elif op == Op.INSERT:
                ops.append(("INSERT", i, s2[j - 1]))
                j -= 1
            elif op == Op.DELETE:
                ops.append(("DELETE", i - 1, s1[i - 1]))
                i -= 1

    ops.reverse()
    return dp[m][n], ops


def damerau_levenshtein(s1, s2, insert_cost=1, delete_cost=1, replace_cost=1, transpose_cost=1):
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    path = [[Op.MATCH] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        dp[i][0] = i * delete_cost
        path[i][0] = Op.DELETE
    for j in range(n + 1):
        dp[0][j] = j * insert_cost
        path[0][j] = Op.INSERT

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
                path[i][j] = Op.MATCH
            else:
                candidates = [
                    (dp[i - 1][j] + delete_cost, Op.DELETE),
                    (dp[i][j - 1] + insert_cost, Op.INSERT),
                    (dp[i - 1][j - 1] + replace_cost, Op.REPLACE),
                ]

                if (i > 1 and j > 1
                        and s1[i - 1] == s2[j - 2]
                        and s1[i - 2] == s2[j - 1]):
                    candidates.append(
                        (dp[i - 2][j - 2] + transpose_cost, Op.TRANSPOSE)
                    )

                min_cost, best_op = min(candidates, key=lambda x: x[0])
                dp[i][j] = min_cost
                path[i][j] = best_op

    ops = []
    i, j = m, n
    while i > 0 or j > 0:
        if i == 0:
            ops.append(("INSERT", 0, s2[j - 1]))
            j -= 1
        elif j == 0:
            ops.append(("DELETE", i - 1, s1[i - 1]))
            i -= 1
        else:
            op = path[i][j]
            if op == Op.MATCH:
                ops.append(("MATCH", i - 1, s1[i - 1]))
                i -= 1
                j -= 1
            elif op == Op.REPLACE:
                ops.append(("REPLACE", i - 1, s1[i - 1], s2[j - 1]))
                i -= 1
                j -= 1
            elif op == Op.INSERT:
                ops.append(("INSERT", i, s2[j - 1]))
                j -= 1
            elif op == Op.DELETE:
                ops.append(("DELETE", i - 1, s1[i - 1]))
                i -= 1
            elif op == Op.TRANSPOSE:
                ops.append(("TRANSPOSE", i - 2, s1[i - 2], s1[i - 1]))
                i -= 2
                j -= 2

    ops.reverse()
    return dp[m][n], ops


def visualize_alignment(s1, s2, ops):
    s1_aligned = []
    s2_aligned = []
    op_symbols = []

    p1, p2 = 0, 0
    for op in ops:
        if op[0] == "MATCH":
            s1_aligned.append(s1[p1])
            s2_aligned.append(s2[p2])
            op_symbols.append("|")
            p1 += 1
            p2 += 1
        elif op[0] == "REPLACE":
            s1_aligned.append(s1[p1])
            s2_aligned.append(s2[p2])
            op_symbols.append("~")
            p1 += 1
            p2 += 1
        elif op[0] == "INSERT":
            s1_aligned.append("_")
            s2_aligned.append(s2[p2])
            op_symbols.append("+")
            p2 += 1
        elif op[0] == "DELETE":
            s1_aligned.append(s1[p1])
            s2_aligned.append("_")
            op_symbols.append("-")
            p1 += 1
        elif op[0] == "TRANSPOSE":
            s1_aligned.append(s1[p1])
            s1_aligned.append(s1[p1 + 1])
            s2_aligned.append(s2[p2])
            s2_aligned.append(s2[p2 + 1])
            op_symbols.append("X")
            op_symbols.append("X")
            p1 += 2
            p2 += 2

    cw = 3
    line_s1 = "s1: " + "".join(f"{c:<{cw}}" for c in s1_aligned)
    line_op = "    " + "".join(f"{s:<{cw}}" for s in op_symbols)
    line_s2 = "s2: " + "".join(f"{c:<{cw}}" for c in s2_aligned)

    legend = "    |=匹配  ~=替换  +=插入  -=删除  X=交换  _=空位"

    return "\n".join([line_s1, line_op, line_s2, legend])


def print_ops(ops):
    for op in ops:
        if op[0] == "MATCH":
            continue
        elif op[0] == "REPLACE":
            print(f"    替换: 位置{op[1]} '{op[2]}' -> '{op[3]}'")
        elif op[0] == "INSERT":
            print(f"    插入: 位置{op[1]} 插入'{op[2]}'")
        elif op[0] == "DELETE":
            print(f"    删除: 位置{op[1]} 删除'{op[2]}'")
        elif op[0] == "TRANSPOSE":
            print(f"    交换: 位置{op[1]} '{op[2]}'<->'{op[3]}'")


if __name__ == "__main__":
    pairs = [
        ("kitten", "sitting"),
        ("intention", "execution"),
        ("sunday", "saturday"),
        ("abcde", "ace"),
        ("ca", "ac"),
    ]

    print("=" * 60)
    print("1. Levenshtein 标准编辑距离 (带路径矩阵)")
    print("=" * 60)
    for s1, s2 in pairs:
        dist, ops = levenshtein(s1, s2)
        print(f"'{s1}' -> '{s2}'  距离={dist}")
        print_ops(ops)
        print()

    print("=" * 60)
    print("2. 带权重编辑距离")
    print("=" * 60)
    for s1, s2 in pairs:
        dist_std, _ = weighted_levenshtein(s1, s2)
        dist_cheap_insert, _ = weighted_levenshtein(s1, s2, insert_cost=0.5)
        dist_expensive_replace, _ = weighted_levenshtein(s1, s2, replace_cost=3)
        print(f"'{s1}' -> '{s2}'")
        print(f"  标准权重(insert=1, delete=1, replace=1):  距离={dist_std}")
        print(f"  插入半价(insert=0.5, delete=1, replace=1): 距离={dist_cheap_insert}")
        print(f"  替换昂贵(insert=1, delete=1, replace=3):   距离={dist_expensive_replace}")
        print()

    print("=" * 60)
    print("3. Damerau-Levenshtein 距离 (允许相邻交换)")
    print("=" * 60)
    for s1, s2 in pairs:
        dist_l, _ = levenshtein(s1, s2)
        dist_d, ops_d = damerau_levenshtein(s1, s2)
        print(f"'{s1}' -> '{s2}'")
        print(f"  Levenshtein:         距离={dist_l}")
        print(f"  Damerau-Levenshtein: 距离={dist_d}")
        if dist_d < dist_l:
            print(f"  >>> Damerau更优! 节省{dist_l - dist_d}步")
        print(f"  操作序列:")
        print_ops(ops_d)
        print()

    print("=" * 60)
    print("4. 可视化对齐结果")
    print("=" * 60)
    test_cases = [
        ("kitten", "sitting", False),
        ("ca", "ac", True),
        ("sunday", "saturday", False),
        ("intention", "execution", False),
        ("abcde", "ace", False),
    ]
    for s1, s2, use_damerau in test_cases:
        if use_damerau:
            dist, ops = damerau_levenshtein(s1, s2)
            label = "Damerau-Levenshtein"
        else:
            dist, ops = levenshtein(s1, s2)
            label = "Levenshtein"
        print(f"'{s1}' -> '{s2}'  [{label} 距离={dist}]")
        print(visualize_alignment(s1, s2, ops))
        print()
