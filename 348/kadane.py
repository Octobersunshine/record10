def max_subarray(nums):
    if not nums:
        return 0, -1, -1

    max_sum = nums[0]
    cur_sum = nums[0]
    max_idx = 0
    start = 0
    end = 0
    temp_start = 0

    for i in range(1, len(nums)):
        if nums[i] > nums[max_idx]:
            max_idx = i

        if cur_sum + nums[i] > nums[i]:
            cur_sum += nums[i]
        else:
            cur_sum = nums[i]
            temp_start = i

        if cur_sum > max_sum:
            max_sum = cur_sum
            start = temp_start
            end = i

    if max_sum < 0:
        return nums[max_idx], max_idx, max_idx

    return max_sum, start, end


def min_subarray(nums):
    if not nums:
        return 0, -1, -1

    min_sum = nums[0]
    cur_sum = nums[0]
    min_idx = 0
    start = 0
    end = 0
    temp_start = 0

    for i in range(1, len(nums)):
        if nums[i] < nums[min_idx]:
            min_idx = i

        if cur_sum + nums[i] < nums[i]:
            cur_sum += nums[i]
        else:
            cur_sum = nums[i]
            temp_start = i

        if cur_sum < min_sum:
            min_sum = cur_sum
            start = temp_start
            end = i

    if min_sum > 0:
        return nums[min_idx], min_idx, min_idx

    return min_sum, start, end


def max_subarray_circular(nums):
    if not nums:
        return 0, -1, -1, False

    n = len(nums)
    max_normal, s_normal, e_normal = max_subarray(nums)

    if n == 1:
        return max_normal, s_normal, e_normal, False

    total = sum(nums)
    min_wrap, s_wrap, e_wrap = min_subarray(nums)
    max_circular = total - min_wrap

    if max_circular <= max_normal:
        return max_normal, s_normal, e_normal, False

    if s_wrap == 0 and e_wrap == n - 1:
        return max_normal, s_normal, e_normal, False

    start = e_wrap + 1
    end = s_wrap - 1

    if start >= n:
        start = 0
    if end < 0:
        end = n - 1

    return max_circular, start, end, True


if __name__ == "__main__":
    test_cases = [
        [-2, 1, -3, 4, -1, 2, 1, -5, 4],
        [1, 2, 3, -2, 5],
        [-1, -2, -3, -4],
        [5, -9, 6, -2, 3],
        [1],
        [0, 0, 0],
        [2, -1, 2],
        [1, -2, 3, -2],
        [5, -3, 5],
        [-3, -2, -3],
        [3, -1, 2, -1],
        [3, -2, 2, -3],
    ]

    print("=== 普通最大子数组和 ===")
    for nums in test_cases:
        max_sum, s, e = max_subarray(nums)
        print(f"数组: {nums}")
        print(f"最大和: {max_sum}, 子数组: {nums[s:e+1]}, 索引: [{s}, {e}]\n")

    print("=== 循环数组最大子数组和 ===")
    for nums in test_cases:
        max_sum, s, e, wrapped = max_subarray_circular(nums)
        if wrapped:
            subarray = nums[s:] + nums[:e + 1]
        else:
            subarray = nums[s:e + 1]
        wrap_tag = " [跨越边界]" if wrapped else ""
        print(f"数组: {nums}")
        print(f"最大和: {max_sum}, 子数组: {subarray}, 索引: [{s}, {e}]{wrap_tag}\n")
