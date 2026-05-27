def _is_hashable(item):
    try:
        hash(item)
        return True
    except TypeError:
        return False


def _make_key(item):
    if _is_hashable(item):
        return item
    return repr(item)


def _build_key_set(lst):
    result = set()
    for item in lst:
        result.add(_make_key(item))
    return result


def union(list1, list2):
    seen = set()
    result = []
    for item in list1 + list2:
        key = _make_key(item)
        if key not in seen:
            seen.add(key)
            result.append(item)
    return result


def intersection(list1, list2):
    set2_keys = _build_key_set(list2)
    seen = set()
    result = []
    for item in list1:
        key = _make_key(item)
        if key in set2_keys and key not in seen:
            seen.add(key)
            result.append(item)
    return result


def difference(list1, list2):
    set2_keys = _build_key_set(list2)
    seen = set()
    result = []
    for item in list1:
        key = _make_key(item)
        if key not in set2_keys and key not in seen:
            seen.add(key)
            result.append(item)
    return result


def symmetric_difference(list1, list2):
    set1_keys = _build_key_set(list1)
    set2_keys = _build_key_set(list2)
    seen = set()
    result = []
    for item in list1:
        key = _make_key(item)
        if key not in set2_keys and key not in seen:
            seen.add(key)
            result.append(item)
    for item in list2:
        key = _make_key(item)
        if key not in set1_keys and key not in seen:
            seen.add(key)
            result.append(item)
    return result


def _build_freq_map(lst):
    freq = {}
    order = []
    for item in lst:
        key = _make_key(item)
        if key in freq:
            freq[key] = (freq[key][0], freq[key][1] + 1)
        else:
            freq[key] = (item, 1)
            order.append(key)
    return freq, order


def multiset_union(list1, list2):
    freq1, order1 = _build_freq_map(list1)
    freq2, order2 = _build_freq_map(list2)
    seen = set()
    result = []
    for key in order1:
        count1 = freq1[key][1]
        count2 = freq2[key][1] if key in freq2 else 0
        result.append((freq1[key][0], max(count1, count2)))
        seen.add(key)
    for key in order2:
        if key not in seen:
            result.append((freq2[key][0], freq2[key][1]))
    return result


def multiset_intersection(list1, list2):
    freq1, order1 = _build_freq_map(list1)
    freq2, _ = _build_freq_map(list2)
    result = []
    for key in order1:
        if key in freq2:
            min_count = min(freq1[key][1], freq2[key][1])
            if min_count > 0:
                result.append((freq1[key][0], min_count))
    return result


def multiset_difference(list1, list2):
    freq1, order1 = _build_freq_map(list1)
    freq2, _ = _build_freq_map(list2)
    result = []
    for key in order1:
        count2 = freq2[key][1] if key in freq2 else 0
        diff = freq1[key][1] - count2
        if diff > 0:
            result.append((freq1[key][0], diff))
    return result


def multiset_symmetric_difference(list1, list2):
    freq1, order1 = _build_freq_map(list1)
    freq2, order2 = _build_freq_map(list2)
    seen = set()
    result = []
    for key in order1:
        count1 = freq1[key][1]
        count2 = freq2[key][1] if key in freq2 else 0
        abs_diff = abs(count1 - count2)
        if abs_diff > 0:
            result.append((freq1[key][0], abs_diff))
        seen.add(key)
    for key in order2:
        if key not in seen:
            result.append((freq2[key][0], freq2[key][1]))
    return result


def multiset_expand(result):
    expanded = []
    for item, count in result:
        expanded.extend([item] * count)
    return expanded


if __name__ == "__main__":
    a = [1, 2, 2, 3, 4, 5]
    b = [4, 5, 5, 6, 7, 8]

    print("=== 可哈希元素测试 ===")
    print("列表 a:", a)
    print("列表 b:", b)
    print("-" * 40)
    print("并集:", union(a, b))
    print("交集:", intersection(a, b))
    print("差集 (a - b):", difference(a, b))
    print("差集 (b - a):", difference(b, a))
    print("对称差集:", symmetric_difference(a, b))

    print()
    print("=== 不可哈希元素测试 ===")
    c = [1, [2, 3], [2, 3], 4]
    d = [[2, 3], 4, [5, 6], 4]
    print("列表 c:", c)
    print("列表 d:", d)
    print("-" * 40)
    print("并集:", union(c, d))
    print("交集:", intersection(c, d))
    print("差集 (c - d):", difference(c, d))
    print("差集 (d - c):", difference(d, c))
    print("对称差集:", symmetric_difference(c, d))

    print()
    print("=== 混合元素测试 ===")
    e = [1, "hello", {"a": 1}, {"a": 1}, [1, 2]]
    f = ["hello", {"a": 2}, [1, 2], 3]
    print("列表 e:", e)
    print("列表 f:", f)
    print("-" * 40)
    print("并集:", union(e, f))
    print("交集:", intersection(e, f))
    print("差集 (e - f):", difference(e, f))
    print("对称差集:", symmetric_difference(e, f))

    print()
    print("=== 多重集合（Multiset）运算测试 ===")
    m1 = [1, 1, 2, 2, 2, 3, 4, 4]
    m2 = [2, 2, 3, 3, 3, 4, 5, 5]
    print("列表 m1:", m1)
    print("列表 m2:", m2)
    print("-" * 40)
    mu = multiset_union(m1, m2)
    mi = multiset_intersection(m1, m2)
    md = multiset_difference(m1, m2)
    msd = multiset_symmetric_difference(m1, m2)
    print("多重并集 (取max频次):", mu, "→", multiset_expand(mu))
    print("多重交集 (取min频次):", mi, "→", multiset_expand(mi))
    print("多重差集 m1-m2 (频次相减):", md, "→", multiset_expand(md))
    print("多重差集 m2-m1:", multiset_difference(m2, m1), "→", multiset_expand(multiset_difference(m2, m1)))
    print("多重对称差集 (频次差的绝对值):", msd, "→", multiset_expand(msd))
