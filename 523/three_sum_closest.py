from itertools import combinations
import time
import random


def k_sum_closest(nums, target, k):
    n = len(nums)

    if n < k or k < 1:
        return None

    nums.sort()

    best_sum = sum(nums[:k])
    best_combo = tuple(nums[:k])

    def dfs(start, k_rem, curr_sum, curr_combo):
        nonlocal best_sum, best_combo

        if k_rem == 2:
            left, right = start, n - 1

            min_two = nums[left] + nums[left + 1]
            if curr_sum + min_two > target:
                s = curr_sum + min_two
                if abs(s - target) < abs(best_sum - target):
                    best_sum = s
                    best_combo = tuple(curr_combo) + (nums[left], nums[left + 1])
                return

            max_two = nums[right - 1] + nums[right]
            if curr_sum + max_two < target:
                s = curr_sum + max_two
                if abs(s - target) < abs(best_sum - target):
                    best_sum = s
                    best_combo = tuple(curr_combo) + (nums[right - 1], nums[right])
                return

            while left < right:
                s = curr_sum + nums[left] + nums[right]

                if s == target:
                    best_sum = s
                    best_combo = tuple(curr_combo) + (nums[left], nums[right])
                    return

                if abs(s - target) < abs(best_sum - target):
                    best_sum = s
                    best_combo = tuple(curr_combo) + (nums[left], nums[right])

                if s < target:
                    left += 1
                else:
                    right -= 1
            return

        for i in range(start, n - k_rem + 1):
            if i > start and nums[i] == nums[i - 1]:
                continue

            min_remaining = sum(nums[i + 1: i + k_rem])
            if curr_sum + nums[i] + min_remaining > target:
                s = curr_sum + nums[i] + min_remaining
                if abs(s - target) < abs(best_sum - target):
                    best_sum = s
                    best_combo = tuple(curr_combo) + (nums[i],) + tuple(nums[i + 1: i + k_rem])
                break

            max_remaining = sum(nums[n - k_rem + 1: n])
            if curr_sum + nums[i] + max_remaining < target:
                s = curr_sum + nums[i] + max_remaining
                if abs(s - target) < abs(best_sum - target):
                    best_sum = s
                    best_combo = tuple(curr_combo) + (nums[i],) + tuple(nums[n - k_rem + 1: n])
                continue

            dfs(i + 1, k_rem - 1, curr_sum + nums[i], curr_combo + (nums[i],))

    dfs(0, k, 0, ())
    return best_sum, best_combo


def k_sum_closest_brute(nums, target, k):
    n = len(nums)

    if n < k or k < 1:
        return None

    best_sum = None
    best_combo = None

    for combo in combinations(nums, k):
        s = sum(combo)
        if best_sum is None or abs(s - target) < abs(best_sum - target):
            best_sum = s
            best_combo = combo

    return best_sum, best_combo


def three_sum_closest(nums, target):
    result = k_sum_closest(nums, target, 3)
    return result


if __name__ == "__main__":
    print("=" * 60)
    print("正确性测试（仅验证最接近的和值）")
    print("=" * 60)

    test_cases = [
        ([-1, 2, 1, -4], 1, 3, 2),
        ([0, 0, 0], 1, 3, 0),
        ([1, 1, 1, 1], 0, 3, 3),
        ([1, 2, 3, 4, 5], 10, 3, 10),
        ([-1, -2, -3, -4], -10, 3, -9),
        ([], 0, 3, None),
        ([1], 0, 3, None),
        ([1, 2], 0, 3, None),
        ([1, 2, 3], 6, 3, 6),
        ([-1, 0, 1, 1, 55], 3, 3, 2),
        ([-3, -2, -1, 0, 1, 2, 3], 0, 3, 0),
        ([-5, -4, -3, -2, -1], -10, 3, -10),
    ]

    all_passed = True
    for nums, target, k, expected_sum in test_cases:
        result = k_sum_closest(nums, target, k)
        if expected_sum is None:
            ok = result is None
        else:
            ok = result is not None and result[0] == expected_sum
        status = "PASS" if ok else "FAIL"
        if not ok:
            all_passed = False
        if result is None:
            print(f"{status}  nums={nums}, target={target}, k={k} => None (expect None)")
        else:
            print(f"{status}  nums={nums}, target={target}, k={k}")
            print(f"       closest_sum={result[0]} (expect {expected_sum}), combo={result[1]}")

    print()
    print("=" * 60)
    print("k=2, k=4 扩展测试")
    print("=" * 60)

    ext_cases = [
        ([1, 2, 3, 4, 5], 8, 2, 8),
        ([-1, 0, 1, 2], -2, 2, -1),
        ([10, 20, 30], 25, 2, 30),
        ([1, 2, 3, 4, 5], 10, 4, 10),
        ([0, 0, 0, 0], 1, 4, 0),
        ([-1, 0, 1, 2, 3], 5, 4, 5),
    ]

    for nums, target, k, expected_sum in ext_cases:
        result_sum, result_combo = k_sum_closest(nums, target, k)
        ok = result_sum == expected_sum
        status = "PASS" if ok else "FAIL"
        if not ok:
            all_passed = False
        print(f"{status}  nums={nums}, target={target}, k={k}")
        print(f"       closest_sum={result_sum} (expect {expected_sum}), combo={result_combo}")

    print()
    print("=" * 60)
    print("双指针 vs 暴力枚举 正确性交叉验证（比较差值绝对值）")
    print("=" * 60)

    random.seed(42)
    cross_fail = False
    for trial in range(500):
        size = random.randint(3, 15)
        nums = [random.randint(-20, 20) for _ in range(size)]
        target = random.randint(-30, 30)
        k = random.randint(2, min(4, size))

        r1 = k_sum_closest(nums[:], target, k)
        r2 = k_sum_closest_brute(nums[:], target, k)

        if r1 is None and r2 is None:
            continue
        if r1 is None or r2 is None:
            print(f"FAIL  trial={trial} nums={nums} target={target} k={k}")
            print(f"       two_pointer={r1}, brute={r2}")
            cross_fail = True
            break

        diff1 = abs(r1[0] - target)
        diff2 = abs(r2[0] - target)

        if diff1 != diff2:
            print(f"FAIL  trial={trial} nums={nums} target={target} k={k}")
            print(f"       two_pointer: sum={r1[0]} diff={diff1}")
            print(f"       brute:       sum={r2[0]} diff={diff2}")
            cross_fail = True
            break
    else:
        print(f"PASS  500次随机交叉验证全部一致 (k=2~4, 比较差值绝对值)")

    print()
    print("=" * 60)
    print("性能对比：双指针 vs 暴力枚举 (k=3)")
    print("=" * 60)

    sizes = [50, 100, 200, 500]

    for size in sizes:
        nums = [random.randint(-10000, 10000) for _ in range(size)]
        target = random.randint(-10000, 10000)

        t0 = time.perf_counter()
        r_tp = k_sum_closest(nums[:], target, 3)
        t_tp = time.perf_counter() - t0

        if size <= 200:
            t0 = time.perf_counter()
            r_brute = k_sum_closest_brute(nums[:], target, 3)
            t_brute = time.perf_counter() - t0
            tp_sum = r_tp[0] if r_tp else None
            brute_sum = r_brute[0] if r_brute else None
            match = abs(tp_sum - target) == abs(brute_sum - target) if tp_sum and brute_sum else tp_sum == brute_sum
            print(f"n={size:>5}  双指针: {t_tp:>8.4f}s  暴力: {t_brute:>8.4f}s  "
                  f"加速比: {t_brute / t_tp:>7.1f}x  差值一致: {match}")
        else:
            t0 = time.perf_counter()
            k_sum_closest_brute(nums[:50], target, 3)
            t_brute_50 = time.perf_counter() - t0
            est_brute = t_brute_50 * (size / 50) ** 3
            print(f"n={size:>5}  双指针: {t_tp:>8.4f}s  暴力: ~{est_brute:>7.1f}s(估算)  "
                  f"加速比: ~{est_brute / t_tp:>7.0f}x")

    print()
    print("=" * 60)
    print("性能对比：不同 k 值 (双指针n=200, 暴力同数组验证n=30)")
    print("=" * 60)

    for k in [2, 3, 4]:
        nums_small = [random.randint(-10000, 10000) for _ in range(30)]
        nums_large = [random.randint(-10000, 10000) for _ in range(200)]
        target = random.randint(-10000, 10000)

        r_brute = k_sum_closest_brute(nums_small[:], target, k)
        r_tp_small = k_sum_closest(nums_small[:], target, k)
        tp_small_sum = r_tp_small[0] if r_tp_small else None
        brute_sum = r_brute[0] if r_brute else None
        match_small = abs(tp_small_sum - target) == abs(brute_sum - target) if tp_small_sum and brute_sum else tp_small_sum == brute_sum

        t0 = time.perf_counter()
        k_sum_closest(nums_large[:], target, k)
        t_tp_large = time.perf_counter() - t0

        t0 = time.perf_counter()
        k_sum_closest_brute(nums_small[:], target, k)
        t_brute = time.perf_counter() - t0

        print(f"k={k}  双指针(n=200): {t_tp_large:>8.4f}s  暴力(n=30): {t_brute:>8.4f}s  "
              f"同数组验证(n=30): {'PASS' if match_small else 'FAIL'}")

    print()
    print("=" * 60)
    print("复杂度对比分析")
    print("=" * 60)
    print(f"{'方法':<25} {'时间复杂度':<20} {'空间复杂度':<15}")
    print(f"{'-'*25} {'-'*20} {'-'*15}")
    print(f"{'暴力枚举 C(n,k)':<25} {'O(C(n,k)*k)':<20} {'O(k)':<15}")
    print(f"{'排序+递归+双指针':<25} {'O(n^(k-1))':<20} {'O(k*n)':<15}")
    print(f"{'  k=2 双指针':<25} {'O(n log n)':<20} {'O(log n)':<15}")
    print(f"{'  k=3 递归+双指针':<25} {'O(n^2)':<20} {'O(n)':<15}")
    print(f"{'  k=4 递归+双指针':<25} {'O(n^3)':<20} {'O(n)':<15}")

    print()
    if all_passed and not cross_fail:
        print("所有测试通过！")
    else:
        print("存在失败测试！")
