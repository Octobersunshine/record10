def _process_input(input_set):
    elements = list(dict.fromkeys(input_set))
    original_len = len(list(input_set))
    if len(elements) < original_len:
        print(f"提示: 输入包含 {original_len - len(elements)} 个重复元素，已自动去重")
    return elements


def subset_count(input_set):
    elements = _process_input(input_set)
    return 1 << len(elements)


def group_by_size(power_set_result):
    grouped = {}
    for subset in power_set_result:
        size = len(subset)
        if size not in grouped:
            grouped[size] = []
        grouped[size].append(subset)
    return dict(sorted(grouped.items()))


def power_set_stream(input_set):
    elements = _process_input(input_set)
    n = len(elements)
    for mask in range(1 << n):
        subset = []
        for i in range(n):
            if mask & (1 << i):
                subset.append(elements[i])
        yield subset


def power_set_bitwise(input_set):
    elements = _process_input(input_set)
    n = len(elements)
    result = []
    
    for mask in range(1 << n):
        subset = []
        for i in range(n):
            if mask & (1 << i):
                subset.append(elements[i])
        result.append(subset)
    
    return result


def power_set_iterative(input_set):
    elements = _process_input(input_set)
    result = [[]]
    
    for elem in elements:
        new_subsets = []
        for subset in result:
            new_subsets.append(subset + [elem])
        result.extend(new_subsets)
    
    return result


if __name__ == "__main__":
    test_input = [1, 2, 3]

    print(f"{'='*50}")
    print("功能1: 基本幂集生成")
    print(f"输入: {test_input}")
    result = power_set_bitwise(test_input)
    print(f"结果: {result}")
    print(f"子集数量: {len(result)}")

    print(f"\n{'='*50}")
    print("功能2: 按子集大小分组输出")
    grouped = group_by_size(result)
    for size, subsets in grouped.items():
        print(f"  长度 {size}: {subsets}")

    print(f"\n{'='*50}")
    print("功能3: 流式生成（逐个yield子集）")
    print("逐个输出: ", end="")
    for subset in power_set_stream(test_input):
        print(subset, end=" ")
    print()

    print(f"\n{'='*50}")
    print("功能4: 子集总数")
    count = subset_count(test_input)
    print(f"输入 {test_input} 的子集总数: {count} (= 2^{len(test_input)})")

    print(f"\n{'='*50}")
    print("功能5: 边界情况验证")

    empty_result = power_set_bitwise([])
    assert empty_result == [[]], f"空列表应返回[[]], 实际: {empty_result}"
    print(f"  空列表 -> {empty_result} ✓")

    assert subset_count([]) == 1, "空列表子集总数应为1"
    print(f"  空列表子集总数 -> {subset_count([])} ✓")

    empty_grouped = group_by_size(empty_result)
    assert 0 in empty_grouped and empty_grouped[0] == [[]], "空列表分组应包含长度0的[[]]"
    print(f"  空列表分组 -> {empty_grouped} ✓")

    stream_list = list(power_set_stream([]))
    assert stream_list == [[]], "空列表流式应生成[[]]"
    print(f"  空列表流式 -> {stream_list} ✓")

    dup_result = power_set_bitwise([1, 1, 2, 2, 3])
    assert len(dup_result) == 8, f"去重后子集数应为8, 实际: {len(dup_result)}"
    print(f"  [1,1,2,2,3] 去重后子集数 -> {len(dup_result)} ✓")

    large_input = list(range(20))
    count_20 = subset_count(large_input)
    assert count_20 == 1 << 20, "20个元素子集总数应为2^20"
    print(f"  20个元素子集总数 -> {count_20} (= 2^20) ✓")

    stream_count = sum(1 for _ in power_set_stream([1, 2, 3, 4]))
    assert stream_count == 16, f"流式生成子集数应为16, 实际: {stream_count}"
    print(f"  [1,2,3,4] 流式子集数 -> {stream_count} ✓")

    print(f"\n{'='*50}")
    print("所有测试通过!")
