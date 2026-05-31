def three_sum(nums):
    n = len(nums)
    if n < 3:
        return []

    result = []
    nums.sort()

    for i in range(n - 2):
        if nums[i] > 0:
            break
        if i > 0 and nums[i] == nums[i - 1]:
            continue

        left, right = i + 1, n - 1
        while left < right:
            current_sum = nums[i] + nums[left] + nums[right]

            if current_sum == 0:
                result.append([nums[i], nums[left], nums[right]])

                while left < right and nums[left] == nums[left + 1]:
                    left += 1
                while left < right and nums[right] == nums[right - 1]:
                    right -= 1

                left += 1
                right -= 1
            elif current_sum < 0:
                left += 1
            else:
                right -= 1

    return result


def four_sum(nums, target):
    n = len(nums)
    if n < 4:
        return []

    result = []
    nums.sort()

    for i in range(n - 3):
        if i > 0 and nums[i] == nums[i - 1]:
            continue
        for j in range(i + 1, n - 2):
            if j > i + 1 and nums[j] == nums[j - 1]:
                continue

            left, right = j + 1, n - 1
            while left < right:
                current_sum = nums[i] + nums[j] + nums[left] + nums[right]

                if current_sum == target:
                    result.append([nums[i], nums[j], nums[left], nums[right]])

                    while left < right and nums[left] == nums[left + 1]:
                        left += 1
                    while left < right and nums[right] == nums[right - 1]:
                        right -= 1

                    left += 1
                    right -= 1
                elif current_sum < target:
                    left += 1
                else:
                    right -= 1

    return result


def k_sum(nums, target, k):
    n = len(nums)
    if n < k or k < 2:
        return []

    nums.sort()
    result = []
    path = []

    def _k_sum(start, target, k):
        if k == 2:
            left, right = start, n - 1
            while left < right:
                current_sum = nums[left] + nums[right]
                if current_sum == target:
                    result.append(path + [nums[left], nums[right]])

                    while left < right and nums[left] == nums[left + 1]:
                        left += 1
                    while left < right and nums[right] == nums[right - 1]:
                        right -= 1

                    left += 1
                    right -= 1
                elif current_sum < target:
                    left += 1
                else:
                    right -= 1
            return

        for i in range(start, n - k + 1):
            if i > start and nums[i] == nums[i - 1]:
                continue

            path.append(nums[i])
            _k_sum(i + 1, target - nums[i], k - 1)
            path.pop()

    _k_sum(0, target, k)
    return result


def three_sum_closest(nums, target):
    n = len(nums)
    if n < 3:
        return None

    nums.sort()
    closest = nums[0] + nums[1] + nums[2]

    for i in range(n - 2):
        if i > 0 and nums[i] == nums[i - 1]:
            continue

        left, right = i + 1, n - 1
        while left < right:
            current_sum = nums[i] + nums[left] + nums[right]

            if abs(current_sum - target) < abs(closest - target):
                closest = current_sum

            if current_sum == target:
                return current_sum
            elif current_sum < target:
                left += 1
            else:
                right -= 1

    return closest


if __name__ == "__main__":
    print("=" * 50)
    print("三数之和 (three_sum)")
    print("=" * 50)
    test_cases_3 = [
        [-1, 0, 1, 2, -1, -4],
        [],
        [0],
        [0, 0],
        [0, 0, 0],
        [-2, 0, 1, 1, 2],
        [-1, 0, 1, 0],
        [-2, -2, -2, 0, 0, 2, 2, 2],
    ]
    for i, nums in enumerate(test_cases_3, 1):
        print(f"  用例 {i}: nums={nums} => {three_sum(nums)}")

    print()
    print("=" * 50)
    print("四数之和 (four_sum)")
    print("=" * 50)
    test_cases_4 = [
        ([1, 0, -1, 0, -2, 2], 0),
        ([2, 2, 2, 2, 2], 8),
        ([1, 0, -1, 0, -2, 2], 1),
        ([], 0),
        ([0, 0], 0),
    ]
    for i, (nums, target) in enumerate(test_cases_4, 1):
        print(f"  用例 {i}: nums={nums}, target={target} => {four_sum(nums, target)}")

    print()
    print("=" * 50)
    print("K数之和 (k_sum)")
    print("=" * 50)
    test_cases_k = [
        ([1, 0, -1, 0, -2, 2], 0, 4),
        ([1, 0, -1, 0, -2, 2], 0, 3),
        ([-1, 0, 1, 2, -1, -4], 0, 3),
        ([2, 2, 2, 2, 2], 6, 3),
        ([1, 1, 1, 1, 1], 4, 4),
    ]
    for i, (nums, target, k) in enumerate(test_cases_k, 1):
        print(f"  用例 {i}: nums={nums}, target={target}, k={k} => {k_sum(nums, target, k)}")

    print()
    print("=" * 50)
    print("最接近目标的三数之和 (three_sum_closest)")
    print("=" * 50)
    test_cases_closest = [
        ([-1, 2, 1, -4], 1),
        ([0, 0, 0], 1),
        ([1, 1, 1, 0], -100),
        ([-1, 0, 1, 2, -1, -4], 3),
        ([0, 2, 1, -3], 1),
    ]
    for i, (nums, target) in enumerate(test_cases_closest, 1):
        result = three_sum_closest(nums, target)
        diff = abs(result - target) if result is not None else None
        print(f"  用例 {i}: nums={nums}, target={target} => sum={result}, diff={diff}")
