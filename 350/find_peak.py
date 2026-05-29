from typing import List, Tuple


def find_peak_element(nums: List[int]) -> int:
    n = len(nums)

    if n == 1:
        return 0

    if nums[0] > nums[1]:
        return 0

    if nums[-1] > nums[-2]:
        return n - 1

    left, right = 1, n - 2

    while left < right:
        mid = (left + right) // 2
        if nums[mid] > nums[mid + 1]:
            right = mid
        else:
            left = mid + 1

    return left


def find_all_peaks(nums: List[int]) -> List[int]:
    n = len(nums)
    if n == 0:
        return []
    if n == 1:
        return [0]

    peaks = []
    if nums[0] > nums[1]:
        peaks.append(0)
    for i in range(1, n - 1):
        if nums[i] > nums[i - 1] and nums[i] > nums[i + 1]:
            peaks.append(i)
    if nums[-1] > nums[-2]:
        peaks.append(n - 1)
    return peaks


def find_peak_2d(matrix: List[List[int]]) -> Tuple[int, int]:
    rows = len(matrix)
    cols = len(matrix[0])

    top, bottom = 0, rows - 1

    while top < bottom:
        mid = (top + bottom) // 2
        max_col = 0
        for j in range(1, cols):
            if matrix[mid][j] > matrix[mid][max_col]:
                max_col = j

        if matrix[mid][max_col] < matrix[mid + 1][max_col]:
            top = mid + 1
        else:
            bottom = mid

    max_col = 0
    for j in range(1, cols):
        if matrix[top][j] > matrix[top][max_col]:
            max_col = j

    return (top, max_col)


def find_all_peaks_2d(matrix: List[List[int]]) -> List[Tuple[int, int]]:
    rows = len(matrix)
    cols = len(matrix[0])
    peaks = []

    for i in range(rows):
        for j in range(cols):
            val = matrix[i][j]
            up = matrix[i - 1][j] if i > 0 else float('-inf')
            down = matrix[i + 1][j] if i < rows - 1 else float('-inf')
            left = matrix[i][j - 1] if j > 0 else float('-inf')
            right = matrix[i][j + 1] if j < cols - 1 else float('-inf')
            if val > up and val > down and val > left and val > right:
                peaks.append((i, j))
    return peaks


if __name__ == "__main__":
    print("=== find_peak_element (1D, single peak) ===")
    test_cases_1d = [
        ([1, 2, 3, 1], 2),
        ([1, 2, 1, 3, 5, 6, 4], (1, 5)),
        ([1], 0),
        ([1, 2], 1),
        ([2, 1], 0),
        ([1, 3, 2, 1], 1),
        ([1, 2, 3, 4, 5], 4),
        ([5, 4, 3, 2, 1], 0),
    ]
    for nums, expected in test_cases_1d:
        result = find_peak_element(nums)
        ok = result == expected if isinstance(expected, int) else result in expected
        print(f"nums={nums}  peak_idx={result}  val={nums[result]}  ok={ok}")

    print("\n=== find_all_peaks (1D, all peaks) ===")
    all_peaks_cases = [
        ([1, 2, 3, 1], [2]),
        ([1, 2, 1, 3, 5, 6, 4], [1, 5]),
        ([1], [0]),
        ([1, 2, 3, 4, 5], [4]),
        ([5, 4, 3, 2, 1], [0]),
        ([1, 3, 2, 4, 1, 5, 2], [1, 3, 5]),
        ([2, 2, 2], []),
    ]
    for nums, expected in all_peaks_cases:
        result = find_all_peaks(nums)
        ok = result == expected
        print(f"nums={nums}  peaks={result}  expected={expected}  ok={ok}")

    print("\n=== find_peak_2d (2D, single peak via binary search) ===")
    matrices = [
        [[1, 4, 3],
         [3, 5, 2],
         [2, 1, 0]],
        [[10, 8, 5],
         [6,  7, 4],
         [3,  2, 1]],
        [[1, 2, 3],
         [4, 5, 6],
         [7, 8, 9]],
    ]
    for mat in matrices:
        pos = find_peak_2d(mat)
        val = mat[pos[0]][pos[1]]
        i, j = pos
        up = mat[i - 1][j] if i > 0 else float('-inf')
        down = mat[i + 1][j] if i < len(mat) - 1 else float('-inf')
        left = mat[i][j - 1] if j > 0 else float('-inf')
        right = mat[i][j + 1] if j < len(mat[0]) - 1 else float('-inf')
        ok = val > up and val > down and val > left and val > right
        print(f"matrix={mat}  peak_pos={pos}  val={val}  ok={ok}")

    print("\n=== find_all_peaks_2d (2D, all peaks) ===")
    for mat in matrices:
        peaks = find_all_peaks_2d(mat)
        print(f"matrix={mat}  all_peaks={peaks}")
