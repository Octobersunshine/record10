def rotated_binary_search(nums, target):
    left, right = 0, len(nums) - 1
    
    while left <= right:
        mid = (left + right) // 2
        if nums[mid] == target:
            return mid
        
        if nums[left] == nums[mid] == nums[right]:
            for i in range(left, right + 1):
                if nums[i] == target:
                    return i
            return -1
        
        if nums[left] <= nums[mid]:
            if nums[left] <= target < nums[mid]:
                right = mid - 1
            else:
                left = mid + 1
        else:
            if nums[mid] < target <= nums[right]:
                left = mid + 1
            else:
                right = mid - 1
    
    return -1


def find_min_rotated(nums):
    left, right = 0, len(nums) - 1
    
    while left < right:
        if nums[left] < nums[right]:
            return nums[left], left
        
        mid = (left + right) // 2
        
        if nums[left] == nums[mid] == nums[right]:
            min_val = nums[left]
            min_idx = left
            for i in range(left + 1, right + 1):
                if nums[i] < min_val:
                    min_val = nums[i]
                    min_idx = i
            return min_val, min_idx
        
        if nums[mid] > nums[right]:
            left = mid + 1
        else:
            right = mid
    
    return nums[left], left


if __name__ == "__main__":
    test_cases = [
        ([4, 5, 6, 1, 2, 3], 1),
        ([4, 5, 6, 1, 2, 3], 6),
        ([4, 5, 6, 1, 2, 3], 7),
        ([1], 1),
        ([3, 1], 1),
        ([5, 1, 3], 3),
        ([1, 1, 1, 2, 1], 2),
        ([1, 1, 1, 2, 1], 1),
        ([2, 2, 2, 0, 2], 0),
        ([2, 2, 2, 0, 2], 3),
        ([1, 0, 1, 1, 1], 0),
        ([1, 1, 1, 1, 1], 1),
        ([1, 1, 1, 1, 1], 2),
    ]
    
    print("=== 二分查找测试 ===")
    for nums, target in test_cases:
        result = rotated_binary_search(nums, target)
        print(f"nums = {nums}, target = {target}, index = {result}")
    
    print("\n=== 查找最小值测试 ===")
    min_test_cases = [
        [4, 5, 6, 1, 2, 3],
        [1, 2, 3, 4, 5],
        [1],
        [3, 1],
        [5, 1, 3],
        [2, 2, 2, 0, 1],
        [1, 0, 1, 1, 1],
        [1, 1, 1, 1, 1],
        [3, 3, 1, 3],
        [2, 3, 4, 5, 1],
    ]
    
    for nums in min_test_cases:
        min_val, pivot_idx = find_min_rotated(nums)
        print(f"nums = {nums}, min = {min_val}, pivot_index = {pivot_idx}")
