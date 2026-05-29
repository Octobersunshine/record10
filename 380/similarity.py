def jaccard_similarity(a, b, empty_set_return=1.0):
    set_a = set(a)
    set_b = set(b)
    if not set_a and not set_b:
        return float(empty_set_return)
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union if union != 0 else 0.0


def dice_coefficient(a, b, empty_set_return=1.0):
    set_a = set(a)
    set_b = set(b)
    if not set_a and not set_b:
        return float(empty_set_return)
    intersection = len(set_a & set_b)
    total = len(set_a) + len(set_b)
    return (2 * intersection) / total if total != 0 else 0.0


def weighted_jaccard_similarity(a, b, empty_set_return=1.0):
    if not a and not b:
        return float(empty_set_return)
    all_keys = set(a.keys()) | set(b.keys())
    min_sum = sum(min(a.get(k, 0), b.get(k, 0)) for k in all_keys)
    max_sum = sum(max(a.get(k, 0), b.get(k, 0)) for k in all_keys)
    return min_sum / max_sum if max_sum != 0 else 0.0


def multiset_jaccard_similarity(sets, empty_set_return=1.0):
    if len(sets) < 2:
        return float(empty_set_return)
    n = len(sets)
    total = 0.0
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            total += jaccard_similarity(sets[i], sets[j], empty_set_return)
            count += 1
    return total / count if count != 0 else float(empty_set_return)


def check_inclusion(a, b):
    set_a = set(a)
    set_b = set(b)
    is_a_subset_b = set_a.issubset(set_b)
    is_a_superset_b = set_a.issuperset(set_b)
    if is_a_subset_b and is_a_superset_b:
        relation = 'equal'
    elif is_a_subset_b:
        relation = 'a_subset_b'
    elif is_a_superset_b:
        relation = 'a_superset_b'
    else:
        relation = 'none'
    return {
        'a_subset_b': is_a_subset_b,
        'a_superset_b': is_a_superset_b,
        'relation': relation
    }


def calculate_similarities(a, b, empty_set_return=1.0):
    return {
        'jaccard': jaccard_similarity(a, b, empty_set_return),
        'dice': dice_coefficient(a, b, empty_set_return),
        'inclusion': check_inclusion(a, b)
    }


if __name__ == '__main__':
    list1 = ['apple', 'banana', 'orange', 'grape']
    list2 = ['apple', 'banana', 'kiwi', 'mango']
    result = calculate_similarities(list1, list2)
    print(f"集合A: {list1}")
    print(f"集合B: {list2}")
    print(f"杰卡德相似度: {result['jaccard']:.4f}")
    print(f"骰子系数: {result['dice']:.4f}")
    print(f"包含关系: {result['inclusion']}")
    print()

    freq_a = {'apple': 3, 'banana': 2, 'orange': 1}
    freq_b = {'apple': 1, 'banana': 4, 'kiwi': 2}
    wj = weighted_jaccard_similarity(freq_a, freq_b)
    print(f"词频A: {freq_a}")
    print(f"词频B: {freq_b}")
    print(f"加权杰卡德相似度: {wj:.4f}")
    print()

    sets = [
        ['apple', 'banana', 'orange'],
        ['apple', 'banana', 'kiwi'],
        ['apple', 'mango', 'grape']
    ]
    avg = multiset_jaccard_similarity(sets)
    print(f"多集合: {sets}")
    print(f"多集合平均杰卡德相似度: {avg:.4f}")
    print()

    subset_result = check_inclusion(['apple', 'banana'], ['apple', 'banana', 'orange'])
    print(f"包含关系测试: {subset_result}")
    equal_result = check_inclusion(['apple', 'banana'], ['banana', 'apple'])
    print(f"相等集合测试: {equal_result}")
