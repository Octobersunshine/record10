def solve_n_queens(n, deduplicate=False, limit=None):
    if n <= 0:
        return []
    if n == 1:
        return [[0]]
    if n == 2 or n == 3:
        return []

    solutions = []
    cols = [0] * n
    full_mask = (1 << n) - 1

    def backtrack(row, cols_mask, diag1_mask, diag2_mask):
        if limit is not None and len(solutions) >= limit:
            return
        if row == n:
            solutions.append(cols[:])
            return
        available = full_mask & ~(cols_mask | diag1_mask | diag2_mask)
        while available:
            bit = available & -available
            available ^= bit
            col = bit.bit_length() - 1
            cols[row] = col
            backtrack(
                row + 1,
                cols_mask | bit,
                (diag1_mask | bit) << 1,
                (diag2_mask | bit) >> 1,
            )

    backtrack(0, 0, 0, 0)

    if deduplicate and limit is None:
        solutions = _deduplicate_symmetric(solutions)

    return solutions


def find_one_solution(n):
    if n <= 0:
        return None
    if n == 1:
        return [0]
    if n == 2 or n == 3:
        return None

    cols = [0] * n
    full_mask = (1 << n) - 1

    def backtrack(row, cols_mask, diag1_mask, diag2_mask):
        if row == n:
            return True
        available = full_mask & ~(cols_mask | diag1_mask | diag2_mask)
        while available:
            bit = available & -available
            available ^= bit
            col = bit.bit_length() - 1
            cols[row] = col
            if backtrack(
                row + 1,
                cols_mask | bit,
                (diag1_mask | bit) << 1,
                (diag2_mask | bit) >> 1,
            ):
                return True
        return False

    backtrack(0, 0, 0, 0)
    return cols


def _get_symmetries(sol):
    n = len(sol)
    result = []
    result.append(list(sol))
    rot90 = [0] * n
    for i in range(n):
        rot90[sol[i]] = n - 1 - i
    result.append(rot90)
    rot180 = [0] * n
    for i in range(n):
        rot180[n - 1 - i] = n - 1 - sol[i]
    result.append(rot180)
    rot270 = [0] * n
    for i in range(n):
        rot270[n - 1 - sol[i]] = i
    result.append(rot270)
    ref_v = [n - 1 - sol[i] for i in range(n)]
    result.append(ref_v)
    ref_h = [0] * n
    for i in range(n):
        ref_h[n - 1 - i] = sol[i]
    result.append(ref_h)
    ref_d1 = [0] * n
    for i in range(n):
        ref_d1[sol[i]] = i
    result.append(ref_d1)
    ref_d2 = [0] * n
    for i in range(n):
        ref_d2[n - 1 - sol[i]] = n - 1 - i
    result.append(ref_d2)
    return result


def _canonical_form(sol):
    return min(tuple(s) for s in _get_symmetries(sol))


def _deduplicate_symmetric(solutions):
    seen = set()
    result = []
    for sol in solutions:
        cf = _canonical_form(sol)
        if cf not in seen:
            seen.add(cf)
            result.append(sol)
    return result


def print_board(sol):
    n = len(sol)
    for row in range(n):
        line = []
        for col in range(n):
            line.append("Q" if sol[row] == col else ".")
        print(" ".join(line))


if __name__ == "__main__":
    n = int(input("请输入棋盘大小 n: "))
    mode_input = input("选择模式 (1-全部解, 2-前N个解, 3-单个解): ").strip()
    mode = int(mode_input) if mode_input.isdigit() else 1

    if mode == 2:
        limit_input = input("请输入要返回的解的数量: ").strip()
        limit = int(limit_input) if limit_input.isdigit() else 10
        dedup = False
    elif mode == 3:
        sol = find_one_solution(n)
        if sol:
            print(f"找到一个解: {sol}")
            print_board(sol)
        else:
            print("无解")
        exit()
    else:
        limit = None
        dedup_input = input("是否去除对称方案？(y/n): ").strip().lower()
        dedup = dedup_input == "y"

    solutions = solve_n_queens(n, deduplicate=dedup, limit=limit)
    for i, sol in enumerate(solutions, 1):
        print(f"方案 {i}: {sol}")
    print(f"共 {len(solutions)} 种方案")
