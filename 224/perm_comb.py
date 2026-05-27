from itertools import permutations, combinations


def _validate_input(elements, k, allow_duplicates=False):
    if not isinstance(k, int):
        raise TypeError(f"k 必须是整数，当前类型: {type(k).__name__}")
    if k < 0:
        raise ValueError(f"k 不能为负数，当前值: {k}")
    n = len(elements)
    if not allow_duplicates and len(set(elements)) != n:
        raise ValueError("elements 包含重复元素，请使用带_duplicates后缀的函数，或确保无重复")
    return n


def generate_permutations(elements, k):
    n = _validate_input(elements, k)
    if k == 0 or k > n:
        return [], 0
    elements = sorted(elements)
    result = [list(p) for p in permutations(elements, k)]
    return result, len(result)


def generate_combinations(elements, k):
    n = _validate_input(elements, k)
    if k == 0 or k > n:
        return [], 0
    elements = sorted(elements)
    result = [list(c) for c in combinations(elements, k)]
    return result, len(result)


def generate_permutations_recursive(elements, k):
    n = _validate_input(elements, k)
    if k == 0 or k > n:
        return [], 0
    elements = sorted(elements)
    result = []
    used = [False] * n
    current = []

    def backtrack():
        if len(current) == k:
            result.append(current[:])
            return
        for i in range(n):
            if used[i]:
                continue
            used[i] = True
            current.append(elements[i])
            backtrack()
            current.pop()
            used[i] = False

    backtrack()
    return result, len(result)


def generate_combinations_recursive(elements, k):
    n = _validate_input(elements, k)
    if k == 0 or k > n:
        return [], 0
    elements = sorted(elements)
    result = []
    current = []

    def backtrack(start):
        if len(current) == k:
            result.append(current[:])
            return
        for i in range(start, n):
            current.append(elements[i])
            backtrack(i + 1)
            current.pop()

    backtrack(0)
    return result, len(result)


def generate_permutations_stream(elements, k):
    n = _validate_input(elements, k)
    if k == 0 or k > n:
        return
    elements = sorted(elements)
    used = [False] * n
    current = []

    def backtrack():
        if len(current) == k:
            yield current[:]
            return
        for i in range(n):
            if used[i]:
                continue
            used[i] = True
            current.append(elements[i])
            yield from backtrack()
            current.pop()
            used[i] = False

    yield from backtrack()


def generate_combinations_stream(elements, k):
    n = _validate_input(elements, k)
    if k == 0 or k > n:
        return
    elements = sorted(elements)
    current = []

    def backtrack(start):
        if len(current) == k:
            yield current[:]
            return
        for i in range(start, n):
            current.append(elements[i])
            yield from backtrack(i + 1)
            current.pop()

    yield from backtrack(0)


def generate_permutations_with_duplicates(elements, k):
    n = _validate_input(elements, k, allow_duplicates=True)
    if k == 0 or k > n:
        return [], 0
    elements = sorted(elements)
    result = []
    used = [False] * n
    current = []

    def backtrack():
        if len(current) == k:
            result.append(current[:])
            return
        for i in range(n):
            if used[i]:
                continue
            if i > 0 and elements[i] == elements[i - 1] and not used[i - 1]:
                continue
            used[i] = True
            current.append(elements[i])
            backtrack()
            current.pop()
            used[i] = False

    backtrack()
    return result, len(result)


def generate_combinations_with_duplicates(elements, k):
    n = _validate_input(elements, k, allow_duplicates=True)
    if k == 0 or k > n:
        return [], 0
    elements = sorted(elements)
    result = []
    current = []

    def backtrack(start):
        if len(current) == k:
            result.append(current[:])
            return
        for i in range(start, n):
            if i > start and elements[i] == elements[i - 1]:
                continue
            current.append(elements[i])
            backtrack(i + 1)
            current.pop()

    backtrack(0)
    return result, len(result)


def generate_permutations_with_duplicates_stream(elements, k):
    n = _validate_input(elements, k, allow_duplicates=True)
    if k == 0 or k > n:
        return
    elements = sorted(elements)
    used = [False] * n
    current = []

    def backtrack():
        if len(current) == k:
            yield current[:]
            return
        for i in range(n):
            if used[i]:
                continue
            if i > 0 and elements[i] == elements[i - 1] and not used[i - 1]:
                continue
            used[i] = True
            current.append(elements[i])
            yield from backtrack()
            current.pop()
            used[i] = False

    yield from backtrack()


def generate_combinations_with_duplicates_stream(elements, k):
    n = _validate_input(elements, k, allow_duplicates=True)
    if k == 0 or k > n:
        return
    elements = sorted(elements)
    current = []

    def backtrack(start):
        if len(current) == k:
            yield current[:]
            return
        for i in range(start, n):
            if i > start and elements[i] == elements[i - 1]:
                continue
            current.append(elements[i])
            yield from backtrack(i + 1)
            current.pop()

    yield from backtrack(0)


def count_permutations_with_duplicates(elements, k):
    from math import factorial
    n = _validate_input(elements, k, allow_duplicates=True)
    if k == 0 or k > n:
        return 0
    elements = sorted(elements)
    used = [False] * n
    count = 0

    def backtrack(depth):
        nonlocal count
        if depth == k:
            count += 1
            return
        for i in range(n):
            if used[i]:
                continue
            if i > 0 and elements[i] == elements[i - 1] and not used[i - 1]:
                continue
            used[i] = True
            backtrack(depth + 1)
            used[i] = False

    backtrack(0)
    return count


def count_combinations_with_duplicates(elements, k):
    n = _validate_input(elements, k, allow_duplicates=True)
    if k == 0 or k > n:
        return 0
    elements = sorted(elements)
    count = 0

    def backtrack(start, depth):
        nonlocal count
        if depth == k:
            count += 1
            return
        for i in range(start, n):
            if i > start and elements[i] == elements[i - 1]:
                continue
            backtrack(i + 1, depth + 1)

    backtrack(0, 0)
    return count


if __name__ == "__main__":
    data = ['C', 'A', 'B']
    n_data = len(data)
    dup_data = ['A', 'B', 'A', 'C']

    print("=" * 50)
    print("1. 正常用例测试 (无重复): k=2")
    print("=" * 50)
    k = 2
    perms, p_count = generate_permutations(data, k)
    print(f"\n排列 (P({n_data},{k}) = {p_count}):")
    for p in perms:
        print(f"  {p}")

    combs, c_count = generate_combinations(data, k)
    print(f"\n组合 (C({n_data},{k}) = {c_count}):")
    for c in combs:
        print(f"  {c}")

    print("\n" + "=" * 50)
    print("2. 流式生成测试 (无重复): k=2")
    print("=" * 50)
    k = 2
    print("\n排列 (流式):")
    count = 0
    for p in generate_permutations_stream(data, k):
        print(f"  {p}")
        count += 1
    print(f"共 {count} 个")

    print("\n组合 (流式):")
    count = 0
    for c in generate_combinations_stream(data, k):
        print(f"  {c}")
        count += 1
    print(f"共 {count} 个")

    print("\n" + "=" * 50)
    print(f"3. 带重复元素测试: {dup_data}, k=2")
    print("=" * 50)
    k = 2
    perms_dup, p_count_dup = generate_permutations_with_duplicates(dup_data, k)
    p_count_calc = count_permutations_with_duplicates(dup_data, k)
    print(f"\n带重复排列 (共 {p_count_dup} 个, 计数函数验证: {p_count_calc}):")
    for p in perms_dup:
        print(f"  {p}")

    combs_dup, c_count_dup = generate_combinations_with_duplicates(dup_data, k)
    c_count_calc = count_combinations_with_duplicates(dup_data, k)
    print(f"\n带重复组合 (共 {c_count_dup} 个, 计数函数验证: {c_count_calc}):")
    for c in combs_dup:
        print(f"  {c}")

    print("\n" + "=" * 50)
    print(f"4. 带重复元素流式测试: {dup_data}, k=3")
    print("=" * 50)
    k = 3
    print("\n带重复排列 (流式):")
    count = 0
    for p in generate_permutations_with_duplicates_stream(dup_data, k):
        print(f"  {p}")
        count += 1
    print(f"共 {count} 个")

    print("\n带重复组合 (流式):")
    count = 0
    for c in generate_combinations_with_duplicates_stream(dup_data, k):
        print(f"  {c}")
        count += 1
    print(f"共 {count} 个")

    print("\n" + "=" * 50)
    print("5. 边界测试: k=0")
    print("=" * 50)
    k = 0
    perms0, p_count0 = generate_permutations(data, k)
    print(f"排列结果: {perms0}, 总数: {p_count0}")
    combs0, c_count0 = generate_combinations(data, k)
    print(f"组合结果: {combs0}, 总数: {c_count0}")
    perms_dup0, _ = generate_permutations_with_duplicates(dup_data, k)
    print(f"带重复排列结果: {perms_dup0}")

    print("\n流式 k=0 测试:")
    list_stream0 = list(generate_permutations_stream(data, k))
    print(f"流式排列 k=0: {list_stream0}")

    print("\n" + "=" * 50)
    print("6. 边界测试: k > n (k=5)")
    print("=" * 50)
    k = 5
    perms5, p_count5 = generate_permutations(data, k)
    print(f"排列结果: {perms5}, 总数: {p_count5}")
    combs5, c_count5 = generate_combinations(data, k)
    print(f"组合结果: {combs5}, 总数: {c_count5}")
    perms_dup5, _ = generate_permutations_with_duplicates(dup_data, k)
    print(f"带重复排列结果: {perms_dup5}")

    print("\n" + "=" * 50)
    print("7. 错误输入测试")
    print("=" * 50)
    try:
        generate_permutations(data, -1)
    except ValueError as e:
        print(f"k=-1 报错: {e}")

    try:
        generate_permutations(data, 2.5)
    except TypeError as e:
        print(f"k=2.5 报错: {e}")

    try:
        generate_permutations([1, 1, 2], 2)
    except ValueError as e:
        print(f"普通函数传重复元素 报错: {e}")

    try:
        generate_permutations_stream([1, 1, 2], 2)
    except ValueError as e:
        print(f"流式函数传重复元素 报错: {e}")

    print("\n" + "=" * 50)
    print("8. 递归实现边界验证")
    print("=" * 50)
    k = 0
    perms_r0, _ = generate_permutations_recursive(data, k)
    combs_r0, _ = generate_combinations_recursive(data, k)
    print(f"k=0 递归排列: {perms_r0}")
    print(f"k=0 递归组合: {combs_r0}")

    k = 5
    perms_r5, _ = generate_permutations_recursive(data, k)
    combs_r5, _ = generate_combinations_recursive(data, k)
    print(f"k=5 递归排列: {perms_r5}")
    print(f"k=5 递归组合: {combs_r5}")

    print("\n" + "=" * 50)
    print("9. 流式 vs 列表 一致性验证")
    print("=" * 50)
    list_perms, _ = generate_permutations(data, 3)
    stream_perms = list(generate_permutations_stream(data, 3))
    print(f"无重复排列一致性: {list_perms == stream_perms}")

    list_combs, _ = generate_combinations(data, 2)
    stream_combs = list(generate_combinations_stream(data, 2))
    print(f"无重复组合一致性: {list_combs == stream_combs}")

    list_perms_d, _ = generate_permutations_with_duplicates(dup_data, 2)
    stream_perms_d = list(generate_permutations_with_duplicates_stream(dup_data, 2))
    print(f"带重复排列一致性: {list_perms_d == stream_perms_d}")

    list_combs_d, _ = generate_combinations_with_duplicates(dup_data, 2)
    stream_combs_d = list(generate_combinations_with_duplicates_stream(dup_data, 2))
    print(f"带重复组合一致性: {list_combs_d == stream_combs_d}")
