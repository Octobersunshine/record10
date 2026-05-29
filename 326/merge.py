import heapq


def merge_sorted(arr1, arr2):
    n1, n2 = len(arr1), len(arr2)
    merged = [None] * (n1 + n2)
    i = j = k = 0
    comparisons = 0

    while i < n1 and j < n2:
        comparisons += 1
        if arr1[i] <= arr2[j]:
            merged[k] = arr1[i]
            i += 1
        else:
            merged[k] = arr2[j]
            j += 1
        k += 1

    if i < n1:
        merged[k:] = arr1[i:]
    if j < n2:
        merged[k:] = arr2[j:]

    return merged, comparisons


def kway_merge(arrays):
    heap = []
    pop_sequence = []
    total_len = 0

    for idx, arr in enumerate(arrays):
        total_len += len(arr)
        if arr:
            heapq.heappush(heap, (arr[0], idx, 0))

    merged = [None] * total_len
    k = 0

    while heap:
        val, arr_idx, elem_idx = heapq.heappop(heap)
        pop_sequence.append(val)
        merged[k] = val
        k += 1

        if elem_idx + 1 < len(arrays[arr_idx]):
            next_val = arrays[arr_idx][elem_idx + 1]
            heapq.heappush(heap, (next_val, arr_idx, elem_idx + 1))

    return merged, pop_sequence


if __name__ == "__main__":
    a = [1, 3, 5, 7]
    b = [2, 4, 6, 8, 10]
    result, comps = merge_sorted(a, b)
    print("=== 两路归并 ===")
    print(f"arr1: {a}")
    print(f"arr2: {b}")
    print(f"merged: {result}")
    print(f"comparisons: {comps}")
    print()

    arrays = [
        [1, 4, 7, 10],
        [2, 5, 8, 11],
        [3, 6, 9, 12],
        [0, 13, 14],
    ]
    k_result, pop_seq = kway_merge(arrays)
    print("=== 多路归并 (k=4) ===")
    for i, arr in enumerate(arrays):
        print(f"arr{i+1}: {arr}")
    print(f"merged: {k_result}")
    print(f"pop sequence: {pop_seq}")
