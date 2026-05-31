import heapq
import time
import random
from typing import List, Tuple, Dict


def find_min_max(matrix: List[List[int]]) -> Tuple[int, int]:
    min_val = matrix[0][0]
    max_val = matrix[0][0]
    for row in matrix:
        for val in row:
            if val < min_val:
                min_val = val
            if val > max_val:
                max_val = val
    return min_val, max_val


def count_less_or_equal(matrix: List[List[int]], target: int) -> int:
    rows = len(matrix)
    cols = len(matrix[0])
    count = 0
    row, col = rows - 1, 0

    while row >= 0 and col < cols:
        if matrix[row][col] <= target:
            count += row + 1
            col += 1
        else:
            row -= 1

    return count


def find_position(matrix: List[List[int]], target: int, k: int) -> Tuple[int, int]:
    rows = len(matrix)
    cols = len(matrix[0])
    less_count = count_less_or_equal(matrix, target - 1)
    need_rank = k - less_count
    row, col = 0, cols - 1
    found = 0
    while row < rows and col >= 0:
        if matrix[row][col] > target:
            col -= 1
        elif matrix[row][col] < target:
            row += 1
        else:
            end_row = row
            while end_row + 1 < rows and matrix[end_row + 1][col] == target:
                end_row += 1
            num_targets = end_row - row + 1
            if found + num_targets >= need_rank:
                return row + (need_rank - found - 1), col
            found += num_targets
            col -= 1
    return -1, -1


def kth_smallest_binary_search(matrix: List[List[int]], k: int) -> Tuple[int, int, int]:
    rows = len(matrix)
    cols = len(matrix[0])
    low, high = find_min_max(matrix)

    while low < high:
        mid = low + (high - low) // 2
        count = count_less_or_equal(matrix, mid)
        if count < k:
            low = mid + 1
        else:
            high = mid

    r, c = find_position(matrix, low, k)
    return low, r, c


def kth_smallest_heap(matrix: List[List[int]], k: int) -> Tuple[int, int, int]:
    rows = len(matrix)
    cols = len(matrix[0])
    min_heap = []
    visited = set()

    for col in range(min(k, cols)):
        heapq.heappush(min_heap, (matrix[0][col], 0, col))
        visited.add((0, col))

    result_val = 0
    result_row = 0
    result_col = 0
    for i in range(k):
        val, row, col = heapq.heappop(min_heap)
        result_val = val
        result_row = row
        result_col = col
        if row + 1 < rows and (row + 1, col) not in visited:
            heapq.heappush(min_heap, (matrix[row + 1][col], row + 1, col))
            visited.add((row + 1, col))

    return result_val, result_row, result_col


def batch_kth_smallest_bs(matrix: List[List[int]], ks: List[int]) -> Dict[int, Tuple[int, int, int]]:
    results = {}
    sorted_ks = sorted(set(ks))
    for k in sorted_ks:
        results[k] = kth_smallest_binary_search(matrix, k)
    return results


def batch_kth_smallest_heap(matrix: List[List[int]], ks: List[int]) -> Dict[int, Tuple[int, int, int]]:
    results = {}
    max_k = max(ks) if ks else 0
    if max_k == 0:
        return results

    rows = len(matrix)
    cols = len(matrix[0])
    min_heap = []
    visited = set()

    for col in range(min(max_k, cols)):
        heapq.heappush(min_heap, (matrix[0][col], 0, col))
        visited.add((0, col))

    k_set = set(ks)
    for i in range(1, max_k + 1):
        val, row, col = heapq.heappop(min_heap)
        if i in k_set:
            results[i] = (val, row, col)
        if row + 1 < rows and (row + 1, col) not in visited:
            heapq.heappush(min_heap, (matrix[row + 1][col], row + 1, col))
            visited.add((row + 1, col))

    return results


def benchmark(matrix: List[List[int]], ks: List[int]) -> None:
    print("=" * 70)
    print("算法时间复杂度对比与性能基准测试")
    print("=" * 70)

    rows = len(matrix)
    cols = len(matrix[0])
    total = rows * cols
    max_k = max(ks)

    print(f"\n矩阵规模: {rows} x {cols} = {total} 个元素")
    print(f"查询 K 值: {ks}")

    min_val, max_val = find_min_max(matrix)
    val_span = max_val - min_val

    print("\n" + "-" * 70)
    print("【二分查找法（值域二分）】")
    print("-" * 70)
    print(f"  时间复杂度: O((rows + cols) * log(max - min))")
    print(f"    - find_min_max:        O(rows * cols)")
    print(f"    - 二分循环次数:        O(log({val_span})) = {val_span.bit_length()} 次")
    print(f"    - 每次循环计数:        O(rows + cols) = O({rows} + {cols})")
    print(f"    - find_position:       O(rows + cols) = O({rows} + {cols})")
    print(f"    - 批量查询 m 个 K:     O(m * (rows + cols) * log(max - min))")
    print(f"    - 每次查询独立,无法复用中间结果")

    start = time.perf_counter()
    bs_results = batch_kth_smallest_bs(matrix, ks)
    bs_time = time.perf_counter() - start

    for k in ks:
        val, r, c = bs_results[k]
        assert matrix[r][c] == val, f"位置验证失败: matrix[{r}][{c}]={matrix[r][c]} != {val}"
        print(f"    k={k}: 值={val}, 位置=({r}, {c})")
    print(f"  批量查询耗时: {bs_time * 1000:.4f} ms")

    print("\n" + "-" * 70)
    print("【最小堆法（优先队列）】")
    print("-" * 70)
    print(f"  时间复杂度: O(max_k * log(min(max_k, cols)))")
    print(f"    - 初始化堆:            O(min(max_k, cols)) = O({min(max_k, cols)})")
    print(f"    - 弹出 max_k 次:       每次 O(log(min(max_k, cols))) = O(log({min(max_k, cols)}))")
    print(f"    - 批量查询 m 个 K:     O(max_k * log(min(max_k, cols)))")
    print(f"    - 只需遍历到最大K值,较小K值自动获得")
    print(f"    - 天然返回元素位置,无需额外查找")

    start = time.perf_counter()
    heap_results = batch_kth_smallest_heap(matrix, ks)
    heap_time = time.perf_counter() - start

    for k in ks:
        val, r, c = heap_results[k]
        assert matrix[r][c] == val, f"位置验证失败: matrix[{r}][{c}]={matrix[r][c]} != {val}"
        print(f"    k={k}: 值={val}, 位置=({r}, {c})")
    print(f"  批量查询耗时: {heap_time * 1000:.4f} ms")

    print("\n" + "-" * 70)
    print("【对比总结】")
    print("-" * 70)
    print(f"  {'指标':<20} {'二分查找':<25} {'最小堆':<25}")
    print(f"  {'-' * 70}")
    print(f"  {'单次查询':<20} {'O(n*log(max-min))':<25} {'O(k*log(min(k,n)))':<25}")
    print(f"  {'批量查询(m个K)':<20} {'O(m*n*log(max-min))':<25} {'O(maxK*log(min(maxK,n)))':<25}")
    print(f"  {'空间复杂度':<20} {'O(1)':<25} {'O(min(maxK,n))':<25}")
    print(f"  {'返回位置':<20} {'需额外O(n)查找':<25} {'天然支持':<25}")
    print(f"  {'适用场景':<20} {'k较大或值域小':<25} {'k较小或批量查询':<25}")
    speedup = bs_time / heap_time if heap_time > 0 else float('inf')
    print(f"  {'本次耗时(ms)':<20} {bs_time * 1000:<25.4f} {heap_time * 1000:<25.4f}")
    print(f"  {'堆法加速比':<20} {'-':<25} {speedup:<25.2f}x")


def generate_sorted_matrix(rows: int, cols: int, val_range: int = 1000) -> List[List[int]]:
    matrix = [[0] * cols for _ in range(rows)]
    for r in range(rows):
        for c in range(cols):
            candidates = []
            if r > 0:
                candidates.append(matrix[r - 1][c])
            if c > 0:
                candidates.append(matrix[r][c - 1])
            low_bound = max(candidates) if candidates else 0
            matrix[r][c] = low_bound + random.randint(1, max(1, val_range // (rows + cols)))
    return matrix


def test_kth_smallest():
    test_cases = [
        {
            "name": "基础测试 3x3",
            "matrix": [[1, 5, 9], [10, 11, 13], [12, 13, 15]],
            "k": 8,
            "expected_val": 13
        },
        {
            "name": "单元素矩阵",
            "matrix": [[-5]],
            "k": 1,
            "expected_val": -5
        },
        {
            "name": "含重复元素 3x3",
            "matrix": [[1, 3, 5], [2, 3, 6], [4, 5, 7]],
            "k": 5,
            "expected_val": 4
        },
        {
            "name": "非方阵 3x4",
            "matrix": [[1, 3, 5, 7], [2, 4, 6, 8], [5, 6, 7, 9]],
            "k": 6,
            "expected_val": 5
        },
        {
            "name": "首列含重复",
            "matrix": [[1, 2], [1, 3]],
            "k": 2,
            "expected_val": 1
        },
        {
            "name": "全重复元素",
            "matrix": [[2, 2, 2], [2, 2, 2], [2, 2, 2]],
            "k": 5,
            "expected_val": 2
        },
        {
            "name": "大量重复 k=1",
            "matrix": [[1, 1, 2], [1, 2, 3], [2, 3, 4]],
            "k": 1,
            "expected_val": 1
        },
        {
            "name": "大量重复 k=4",
            "matrix": [[1, 1, 2], [1, 2, 3], [2, 3, 4]],
            "k": 4,
            "expected_val": 2
        },
        {
            "name": "第1小元素",
            "matrix": [[1, 5, 9], [10, 11, 13], [12, 13, 15]],
            "k": 1,
            "expected_val": 1
        },
        {
            "name": "最后1小元素",
            "matrix": [[1, 5, 9], [10, 11, 13], [12, 13, 15]],
            "k": 9,
            "expected_val": 15
        },
        {
            "name": "find_min_max 验证 1x5",
            "matrix": [[-10, -5, 0, 5, 10]],
            "k": 3,
            "expected_val": 0
        },
        {
            "name": "find_min_max 验证 5x1",
            "matrix": [[-3], [-1], [0], [2], [4]],
            "k": 4,
            "expected_val": 2
        },
        {
            "name": "重复元素分布广",
            "matrix": [[1, 2, 3], [2, 3, 4], [3, 4, 5]],
            "k": 5,
            "expected_val": 3
        },
        {
            "name": "4x4 矩阵中间k",
            "matrix": [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]],
            "k": 8,
            "expected_val": 8
        }
    ]

    passed = 0
    failed = 0

    for tc in test_cases:
        name = tc["name"]
        matrix = tc["matrix"]
        k = tc["k"]
        expected_val = tc["expected_val"]

        bs_val, bs_row, bs_col = kth_smallest_binary_search(matrix, k)
        heap_val, heap_row, heap_col = kth_smallest_heap(matrix, k)

        bs_pass = bs_val == expected_val and matrix[bs_row][bs_col] == expected_val
        heap_pass = heap_val == expected_val and matrix[heap_row][heap_col] == expected_val

        if bs_pass and heap_pass:
            print(f"✓ {name}: k={k}, val={expected_val}, "
                  f"bs=({bs_val},{bs_row},{bs_col}), heap=({heap_val},{heap_row},{heap_col})")
            passed += 1
        else:
            print(f"✗ {name}: k={k}, expected_val={expected_val}")
            if not bs_pass:
                print(f"    二分查找失败: 得到 ({bs_val}, {bs_row}, {bs_col})")
                failed += 1
            if not heap_pass:
                print(f"    最小堆失败: 得到 ({heap_val}, {heap_row}, {heap_col})")
                failed += 1

    print(f"\n测试结果: 通过 {passed}/{len(test_cases)}, 失败 {failed}")
    return failed == 0


def test_batch_query():
    print("\n" + "=" * 70)
    print("批量查询测试")
    print("=" * 70)

    matrix = [
        [1, 5, 9],
        [10, 11, 13],
        [12, 13, 15]
    ]
    ks = [1, 3, 5, 7, 9]

    print(f"\n矩阵:")
    for row in matrix:
        print(f"  {row}")
    print(f"查询 K 值: {ks}")

    bs_results = batch_kth_smallest_bs(matrix, ks)
    heap_results = batch_kth_smallest_heap(matrix, ks)

    all_match = True
    for k in ks:
        bs_val, bs_row, bs_col = bs_results[k]
        heap_val, heap_row, heap_col = heap_results[k]
        match = bs_val == heap_val and matrix[bs_row][bs_col] == bs_val and matrix[heap_row][heap_col] == heap_val
        status = "✓" if match else "✗"
        print(f"  {status} k={k}: bs=({bs_val},{bs_row},{bs_col}), heap=({heap_val},{heap_row},{heap_col})")
        if not match:
            all_match = False

    return all_match


if __name__ == "__main__":
    random.seed(42)
    success = test_kth_smallest()
    batch_ok = test_batch_query()

    matrix = generate_sorted_matrix(100, 100, 10000)
    ks = [1, 10, 50, 500, 5000, 10000]
    benchmark(matrix, ks)

    print("\n" + "=" * 70)
    print("结论")
    print("=" * 70)
    print("""
  当 K 较小或需要批量查询多个 K 值时,最小堆法更优:
    - 单次只需遍历到 max_k,自动获得所有较小 K 的结果
    - 天然返回元素位置,无需额外查找

  当值域范围较小或 K 接近 n² 时,二分查找法更优:
    - 每次查询独立,不受 K 大小影响
    - 值域小时 log(max-min) 很小,效率高
    - 空间复杂度 O(1),无需额外存储

  批量查询场景:
    - 二分查找:每个 K 独立二分,总复杂度 O(m * n * log(max-min))
    - 最小堆:只需遍历到 max_k,总复杂度 O(max_k * log(min(max_k, n)))
    - 当 m 个 K 值较集中时,最小堆批量查询优势明显
""")

    if not success or not batch_ok:
        exit(1)
