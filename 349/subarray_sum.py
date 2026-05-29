def subarray_sum(nums, k):
    """统计和为 K 的连续子数组个数（支持包含负数的数组）。

    算法：前缀和 + 哈希表
        设 prefix_sum[i] 为 nums[0..i] 的累加和，
        子数组 nums[j+1..i] 的和 = prefix_sum[i] - prefix_sum[j]
        令 prefix_sum[i] - prefix_sum[j] = k，
        则对于每个 i，只需统计有多少个 j (< i) 满足 prefix_sum[j] = prefix_sum[i] - k。

    边界说明：
        1. 空数组返回 0，没有非空子数组。
        2. 初始化 {0: 1} 表示前缀和为 0 出现 1 次（对应 j = -1 的"空前缀"），
           用于统计从数组开头到当前位置的子数组，而非计数空子数组。
        3. 由于先查询哈希表再更新，j 始终 < i，因此不会统计长度为 0 的空子数组。
        4. K = 0 时，正确统计所有和为 0 的非空子数组。

    Args:
        nums (list[int]): 输入数组，可包含负数
        k (int): 目标和

    Returns:
        int: 和为 k 的连续非空子数组的个数

    时间复杂度：O(n)，一次遍历
    空间复杂度：O(n)，哈希表最坏存储 n 个不同前缀和
    """
    if not nums:
        return 0

    prefix_sum_count = {0: 1}
    prefix_sum = 0
    count = 0

    for num in nums:
        prefix_sum += num
        count += prefix_sum_count.get(prefix_sum - k, 0)
        prefix_sum_count[prefix_sum] = prefix_sum_count.get(prefix_sum, 0) + 1

    return count


def find_subarray_sum_indices(nums, k, min_length=1):
    """找出所有和为 K 的连续子数组的起止索引，支持最小长度限制。

    算法：前缀和 + 哈希表（存储索引列表）
        设 prefix_sum[i] 为 nums[0..i-1] 的累加和（prefix_sum[0] = 0），
        子数组 nums[start..end]（含两端）的和 = prefix_sum[end+1] - prefix_sum[start]
        令 prefix_sum[end+1] - prefix_sum[start] = k，
        则对于每个 end+1 = i，只需查找所有 start 满足 prefix_sum[start] = prefix_sum[i] - k。

    Args:
        nums (list[int]): 输入数组，可包含负数
        k (int): 目标和
        min_length (int): 子数组最小长度，默认为 1（至少一个元素）

    Returns:
        list[tuple[int, int]]: 所有符合条件的子数组的 (start, end) 索引列表
            索引为闭区间 [start, end]，子数组长度 = end - start + 1

    时间复杂度：O(n) ~ O(n^2)，最坏情况所有前缀和相同且都满足条件
    空间复杂度：O(n)，哈希表存储索引列表

    示例：
        >>> find_subarray_sum_indices([1, 1, 1], 2)
        [(0, 1), (1, 2)]
        >>> find_subarray_sum_indices([1, 1, 1], 2, min_length=2)
        [(0, 1), (1, 2)]
        >>> find_subarray_sum_indices([0, 0], 0, min_length=2)
        [(0, 1)]
    """
    if not nums or min_length < 1:
        return []

    from collections import defaultdict
    prefix_sum_indices = defaultdict(list)
    prefix_sum_indices[0].append(0)
    prefix_sum = 0
    result = []

    for i in range(len(nums)):
        prefix_sum += nums[i]
        target = prefix_sum - k

        if target in prefix_sum_indices:
            for start in prefix_sum_indices[target]:
                end = i
                length = end - start + 1
                if length >= min_length:
                    result.append((start, end))

        prefix_sum_indices[prefix_sum].append(i + 1)

    return result


def print_indices_test(nums, k, min_length=1):
    indices = find_subarray_sum_indices(nums, k, min_length)
    subarrays = [nums[s:e+1] for s, e in indices]
    print(f"nums={nums}, k={k}, min_len={min_length}")
    print(f"  索引列表: {indices}")
    print(f"  子数组:   {subarrays}")
    print(f"  个数:     {len(indices)}")
    print()


if __name__ == "__main__":
    print("=" * 60)
    print("subarray_sum - 统计个数")
    print("=" * 60)
    print("空数组, k=0:", subarray_sum([], 0))
    print("空数组, k=5:", subarray_sum([], 5))
    print("[1,1,1], k=2:", subarray_sum([1, 1, 1], 2))
    print("[1,2,3], k=3:", subarray_sum([1, 2, 3], 3))
    print("[1,-1,0], k=0:", subarray_sum([1, -1, 0], 0))
    print("[0], k=0:", subarray_sum([0], 0))
    print("[0,0], k=0:", subarray_sum([0, 0], 0))
    print("[3,4,7,2,-3,1,4,2], k=7:", subarray_sum([3, 4, 7, 2, -3, 1, 4, 2], 7))
    print()

    print("=" * 60)
    print("find_subarray_sum_indices - 返回索引列表")
    print("=" * 60)
    print_indices_test([1, 1, 1], 2)
    print_indices_test([1, 2, 3], 3)
    print_indices_test([1, -1, 0], 0)
    print_indices_test([0, 0], 0)

    print("=" * 60)
    print("find_subarray_sum_indices - 最小长度限制")
    print("=" * 60)
    print_indices_test([0, 0], 0, min_length=2)
    print_indices_test([1, -1, 0], 0, min_length=2)
    print_indices_test([3, 4, 7, 2, -3, 1, 4, 2], 7, min_length=2)
    print_indices_test([1, 2, 3, 4, 5], 9, min_length=3)
