def xor_two_numbers(a, b):
    return a ^ b

def xor_array(arr):
    if not arr:
        return None
    result = 0
    for num in arr:
        result ^= num
    return result

def find_odd_occurrence(arr):
    if not arr:
        return None
    result = 0
    for num in arr:
        result ^= num
    return result

def find_two_odd_occurrence(arr):
    if len(arr) < 2:
        return None

    xor_all = 0
    for num in arr:
        xor_all ^= num

    if xor_all == 0:
        return None

    rightmost_set_bit = xor_all & (-xor_all)

    num1 = 0
    num2 = 0
    for num in arr:
        if num & rightmost_set_bit:
            num1 ^= num
        else:
            num2 ^= num

    return sorted([num1, num2])

def find_all_odd_occurrence(arr):
    if not arr:
        return []

    arr = arr[:]
    result = []
    n = len(arr)
    i = 0

    while i < n:
        xor_all = 0
        for j in range(i, n):
            xor_all ^= arr[j]

        if xor_all != 0:
            count = 0
            for j in range(i, n):
                if arr[j] == xor_all:
                    count += 1

            if count % 2 == 1:
                result.append(xor_all)
                k = i
                for j in range(i, n):
                    if arr[j] != xor_all:
                        arr[k], arr[j] = arr[j], arr[k]
                        k += 1
                n = k
                continue

            rightmost_set_bit = xor_all & (-xor_all)

            xor1 = 0
            xor2 = 0
            for j in range(i, n):
                if arr[j] & rightmost_set_bit:
                    xor1 ^= arr[j]
                else:
                    xor2 ^= arr[j]

            candidates = []
            if xor1 != 0:
                candidates.append(xor1)
            if xor2 != 0:
                candidates.append(xor2)

            found = False
            for cand in candidates:
                cnt = 0
                for j in range(i, n):
                    if arr[j] == cand:
                        cnt += 1
                if cnt % 2 == 1:
                    result.append(cand)
                    k = i
                    for j in range(i, n):
                        if arr[j] != cand:
                            arr[k], arr[j] = arr[j], arr[k]
                            k += 1
                    n = k
                    found = True
                    break

            if found:
                continue

        found = False
        for j in range(i, n):
            count = 0
            for m in range(i, n):
                if arr[m] == arr[j]:
                    count += 1
            if count % 2 == 1:
                target = arr[j]
                result.append(target)
                k = i
                for m in range(i, n):
                    if arr[m] != target:
                        arr[k], arr[m] = arr[m], arr[k]
                        k += 1
                n = k
                found = True
                break

        if not found:
            break

    return sorted(set(result))

def count_xor_subarrays(arr, target=0):
    if not arr:
        return 0, []

    prefix_count = {0: 1}
    prefix_xor = 0
    total_count = 0
    subarrays = []

    for r in range(len(arr)):
        prefix_xor ^= arr[r]
        key = prefix_xor ^ target

        if key in prefix_count:
            total_count += prefix_count[key]

        prefix_count[prefix_xor] = prefix_count.get(prefix_xor, 0) + 1

    prefix_xor = 0
    prefix_indices = {0: [-1]}
    for r in range(len(arr)):
        prefix_xor ^= arr[r]
        key = prefix_xor ^ target

        if key in prefix_indices:
            for l in prefix_indices[key]:
                subarrays.append((l + 1, r))

        if prefix_xor not in prefix_indices:
            prefix_indices[prefix_xor] = []
        prefix_indices[prefix_xor].append(r)

    return total_count, subarrays

def process_multiple_test_cases(test_cases):
    results = []
    for arr in test_cases:
        results.append(find_odd_occurrence(arr))
    return results

def process_multiple_two_odd_cases(test_cases):
    results = []
    for arr in test_cases:
        results.append(find_two_odd_occurrence(arr))
    return results

def process_multiple_all_odd_cases(test_cases):
    results = []
    for arr in test_cases:
        results.append(find_all_odd_occurrence(arr))
    return results

def process_multiple_count_cases(test_cases):
    results = []
    for arr, target in test_cases:
        results.append(count_xor_subarrays(arr, target))
    return results

def main():
    print("=== 异或运算常见操作演示 ===")

    a, b = 5, 3
    print(f"\n1. 两个数的异或: {a} ^ {b} = {xor_two_numbers(a, b)}")
    print(f"   二进制: {bin(a)} ^ {bin(b)} = {bin(xor_two_numbers(a, b))}")

    arr = [1, 2, 3, 4, 5]
    print(f"\n2. 数组的异或: {arr}")
    print(f"   结果: {xor_array(arr)}")

    empty_arr = []
    print(f"\n   空数组的异或: {empty_arr}")
    print(f"   结果: {xor_array(empty_arr)}")

    test_cases = [
        [4, 3, 2, 4, 1, 3, 2],
        [7, 9, 7, 9, 7],
        [100, 200, 200, 100, 50],
        [1, 1, 2, 2, 3, 3, 4],
        [],
    ]

    print("\n3. 找出1个出现奇数次的元素（多组数据）:")
    results = process_multiple_test_cases(test_cases)
    for i, (arr, res) in enumerate(zip(test_cases, results), 1):
        print(f"   测试用例 {i}: 数组 {arr} -> 落单元素: {res}")

    two_odd_cases = [
        [4, 3, 2, 4, 1, 3, 2, 5],
        [7, 9, 7, 9, 7, 11],
        [100, 200, 200, 100, 50, 60],
        [1, 1, 2, 3],
    ]

    print("\n4. 找出2个出现奇数次的元素（分组异或）:")
    results = process_multiple_two_odd_cases(two_odd_cases)
    for i, (arr, res) in enumerate(zip(two_odd_cases, results), 1):
        print(f"   测试用例 {i}: 数组 {arr} -> 落单元素: {res}")

    print("\n   --- 分组异或原理说明 ---")
    example = [4, 3, 2, 4, 1, 3, 2, 5]
    xor_all = 0
    for num in example:
        xor_all ^= num
    print(f"   示例数组: {example}")
    print(f"   步骤1 - 全部异或: xor_all = 1 ^ 5 = {xor_all} (二进制: {bin(xor_all)})")
    split_bit = xor_all & (-xor_all)
    print(f"   步骤2 - 找最低位1: {bin(xor_all)} & {bin(-xor_all)} = {bin(split_bit)} (第{split_bit.bit_length() - 1}位不同)")
    print(f"   步骤3 - 按该位分组:")
    group_a = [num for num in example if num & split_bit]
    group_b = [num for num in example if not (num & split_bit)]
    print(f"      该位为1的组: {group_a}")
    print(f"      该位为0的组: {group_b}")
    print(f"   步骤4 - 分别异或: {find_two_odd_occurrence(example)}")

    all_odd_cases = [
        [4, 3, 2, 4, 1, 3, 2],
        [4, 3, 2, 4, 1, 3, 2, 5],
        [1, 2, 3, 1, 2, 3, 4, 5, 6],
        [1, 1, 2, 2, 3, 3, 4, 5, 6, 7],
        [7, 7, 7],
        [],
    ]

    print("\n5. 找出所有出现奇数次的元素（通用分组法，不使用额外空间）:")
    results = process_multiple_all_odd_cases(all_odd_cases)
    for i, (arr, res) in enumerate(zip(all_odd_cases, results), 1):
        print(f"   测试用例 {i}: 数组 {arr} -> 落单元素: {res}")

    print("\n   --- 通用分组法原理 ---")
    example2 = [1, 2, 3, 1, 2, 3, 4, 5, 6]
    print(f"   示例数组: {example2}")
    print(f"   预期结果: [4, 5, 6] (三个各出现1次)")
    xor_all2 = xor_array(example2)
    print(f"   全部异或: {xor_all2} (二进制: {bin(xor_all2)})")
    print(f"   递归分组思想: 每次用最低位1分组，直到每组只剩1个奇数次元素")

    count_cases = [
        ([1, 2, 3, 0, 3, 2, 1], 0),
        ([1, 1, 2, 2, 3, 3], 0),
        ([1, 2, 3], 0),
        ([4, 2, 2, 6, 4], 6),
    ]

    print("\n6. 统计异或子数组数量（前缀异或 + 哈希表）:")
    results = process_multiple_count_cases(count_cases)
    for i, ((arr, target), (cnt, subs)) in enumerate(zip(count_cases, results), 1):
        print(f"   测试用例 {i}: 数组 {arr}, target={target}")
        print(f"      数量: {cnt}")
        if cnt > 0 and len(subs) <= 10:
            print(f"      子数组索引: {subs}")
            for l, r in subs:
                print(f"         [{l}:{r}] = {arr[l:r+1]} (xor={xor_array(arr[l:r+1])})")

    print("\n   --- 前缀异或原理 ---")
    example3 = [1, 2, 3, 0, 3, 2, 1]
    print(f"   示例数组: {example3}, target=0")
    prefix = [0]
    for num in example3:
        prefix.append(prefix[-1] ^ num)
    print(f"   前缀异或: {prefix}")
    print(f"   公式: subarray(l..r) = prefix[r+1] ^ prefix[l]")
    print(f"   找异或=0 即找 prefix[r+1] == prefix[l]")
    from collections import defaultdict
    freq = defaultdict(list)
    for idx, val in enumerate(prefix):
        freq[val].append(idx)
    print(f"   重复的前缀异或值: {dict([(k, v) for k, v in freq.items() if len(v) > 1])}")
    print(f"   每组贡献 C(k,2) = k*(k-1)/2 个子数组")

if __name__ == "__main__":
    main()
