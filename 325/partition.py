import random


def partition_lomuto(arr, pivot_idx):
    arr = arr.copy()
    lo, hi = 0, len(arr) - 1
    arr[pivot_idx], arr[hi] = arr[hi], arr[pivot_idx]
    pivot = arr[hi]
    i = lo
    for j in range(lo, hi):
        if arr[j] <= pivot:
            arr[i], arr[j] = arr[j], arr[i]
            i += 1
    arr[i], arr[hi] = arr[hi], arr[i]
    return arr, i


def partition_hoare(arr, pivot_idx):
    arr = arr.copy()
    lo, hi = 0, len(arr) - 1
    arr[pivot_idx], arr[lo] = arr[lo], arr[pivot_idx]
    pivot = arr[lo]
    i, j = lo - 1, hi + 1
    while True:
        i += 1
        while arr[i] < pivot:
            i += 1
        j -= 1
        while arr[j] > pivot:
            j -= 1
        if i >= j:
            return arr, j
        arr[i], arr[j] = arr[j], arr[i]


def partition_three_way(arr, pivot_idx):
    arr = arr.copy()
    lo, hi = 0, len(arr) - 1
    pivot = arr[pivot_idx]
    lt, i, gt = lo, lo, hi
    while i <= gt:
        if arr[i] < pivot:
            arr[lt], arr[i] = arr[i], arr[lt]
            lt += 1
            i += 1
        elif arr[i] > pivot:
            arr[i], arr[gt] = arr[gt], arr[i]
            gt -= 1
        else:
            i += 1
    return arr, lt, gt


def select_pivot_first(arr):
    return 0


def select_pivot_last(arr):
    return len(arr) - 1


def select_pivot_random(arr):
    return random.randint(0, len(arr) - 1)


def select_pivot_median_of_three(arr):
    lo, hi = 0, len(arr) - 1
    mid = (lo + hi) // 2
    candidates = [(arr[lo], lo), (arr[mid], mid), (arr[hi], hi)]
    candidates.sort()
    return candidates[1][1]


def calculate_balance_metric(pivot_pos, n):
    left_size = pivot_pos
    right_size = n - 1 - pivot_pos
    min_size = min(left_size, right_size)
    max_size = max(left_size, right_size)
    if max_size == 0:
        balance_ratio = 1.0
    else:
        balance_ratio = min_size / max_size
    imbalance = max_size / n
    return {
        "left_size": left_size,
        "right_size": right_size,
        "balance_ratio": round(balance_ratio, 4),
        "imbalance_factor": round(imbalance, 4)
    }


def compare_pivot_strategies(arr, partition_func=partition_lomuto):
    strategies = [
        ("first", select_pivot_first),
        ("last", select_pivot_last),
        ("random", select_pivot_random),
        ("median_of_three", select_pivot_median_of_three)
    ]
    results = {}
    for name, selector in strategies:
        pivot_idx = selector(arr)
        result_arr, pivot_pos = partition_func(arr, pivot_idx)
        metric = calculate_balance_metric(pivot_pos, len(arr))
        results[name] = {
            "pivot_idx": pivot_idx,
            "pivot_value": arr[pivot_idx],
            "pivot_pos": pivot_pos,
            "metric": metric
        }
    return results


if __name__ == "__main__":
    a = [3, 6, 8, 10, 1, 2, 1]
    result, pos = partition_lomuto(a, 0)
    print(f"Lomuto: arr={result}, pivot_pos={pos}")

    result, pos = partition_hoare(a, 0)
    print(f"Hoare:  arr={result}, pivot_pos={pos}")

    result, lt, gt = partition_three_way(a, 0)
    print(f"Three-way: arr={result}, lt={lt}, gt={gt}")

    b = [2, 2, 2, 2, 2]
    result, lt, gt = partition_three_way(b, 0)
    print(f"Three-way all equal: arr={result}, lt={lt}, gt={gt}")

    print("\n=== Pivot Strategy Comparison ===")
    test_arrays = [
        [3, 6, 8, 10, 1, 2, 1],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        [10, 9, 8, 7, 6, 5, 4, 3, 2, 1],
        [5, 3, 8, 5, 2, 5, 1, 5, 7, 5, 4]
    ]
    for idx, test_arr in enumerate(test_arrays):
        print(f"\nTest case {idx + 1}: {test_arr}")
        results = compare_pivot_strategies(test_arr)
        for strategy, data in results.items():
            m = data["metric"]
            print(f"  {strategy:18}: pivot={data['pivot_value']:2} (idx {data['pivot_idx']:2}) -> pos {data['pivot_pos']:2}, "
                  f"L={m['left_size']:2}, R={m['right_size']:2}, "
                  f"ratio={m['balance_ratio']:.2f}, imbalance={m['imbalance_factor']:.2f}")
