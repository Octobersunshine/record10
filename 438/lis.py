import bisect
import random
import time


def longest_increasing_subsequence(nums):
    n = len(nums)
    if n == 0:
        return 0, []

    dp = [1] * n
    prev = [-1] * n

    for i in range(n):
        for j in range(i):
            if nums[j] < nums[i] and dp[j] + 1 > dp[i]:
                dp[i] = dp[j] + 1
                prev[i] = j

    max_len = max(dp)
    max_idx = dp.index(max_len)

    lis = []
    while max_idx != -1:
        lis.append(nums[max_idx])
        max_idx = prev[max_idx]
    lis.reverse()

    return max_len, lis


def longest_increasing_subsequence_fast(nums, return_sequence=False):
    n = len(nums)
    if n == 0:
        if return_sequence:
            return 0, []
        return 0

    tails = []
    if return_sequence:
        tails_idx = []
        prev = [-1] * n
        insert_pos = [0] * n

    for i, x in enumerate(nums):
        pos = bisect.bisect_left(tails, x)
        if pos == len(tails):
            tails.append(x)
            if return_sequence:
                tails_idx.append(i)
        else:
            tails[pos] = x
            if return_sequence:
                tails_idx[pos] = i

        if return_sequence:
            insert_pos[i] = pos
            if pos > 0:
                prev[i] = tails_idx[pos - 1]

    max_len = len(tails)

    if not return_sequence:
        return max_len

    lis = []
    curr = tails_idx[-1]
    while curr != -1:
        lis.append(nums[curr])
        curr = prev[curr]
    lis.reverse()

    return max_len, lis


def longest_non_decreasing_subsequence(nums, return_sequence=False):
    n = len(nums)
    if n == 0:
        if return_sequence:
            return 0, []
        return 0

    tails = []
    if return_sequence:
        tails_idx = []
        prev = [-1] * n

    for i, x in enumerate(nums):
        pos = bisect.bisect_right(tails, x)
        if pos == len(tails):
            tails.append(x)
            if return_sequence:
                tails_idx.append(i)
        else:
            tails[pos] = x
            if return_sequence:
                tails_idx[pos] = i

        if return_sequence:
            if pos > 0:
                prev[i] = tails_idx[pos - 1]

    max_len = len(tails)

    if not return_sequence:
        return max_len

    seq = []
    curr = tails_idx[-1]
    while curr != -1:
        seq.append(nums[curr])
        curr = prev[curr]
    seq.reverse()

    return max_len, seq


def longest_common_subsequence(s1, s2, return_sequence=False):
    m, n = len(s1), len(s2)
    if m == 0 or n == 0:
        if return_sequence:
            return 0, []
        return 0

    if return_sequence:
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if s1[i - 1] == s2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

        max_len = dp[m][n]
        seq = []
        i, j = m, n
        while i > 0 and j > 0:
            if s1[i - 1] == s2[j - 1]:
                seq.append(s1[i - 1])
                i -= 1
                j -= 1
            elif dp[i - 1][j] >= dp[i][j - 1]:
                i -= 1
            else:
                j -= 1
        seq.reverse()

        return max_len, seq
    else:
        prev_row = [0] * (n + 1)
        curr_row = [0] * (n + 1)
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if s1[i - 1] == s2[j - 1]:
                    curr_row[j] = prev_row[j - 1] + 1
                else:
                    curr_row[j] = max(prev_row[j], curr_row[j - 1])
            prev_row, curr_row = curr_row, [0] * (n + 1)

        return prev_row[n]


def longest_common_subsequence_multi(sequences, return_sequence=False):
    if not sequences:
        if return_sequence:
            return 0, []
        return 0

    non_empty = [s for s in sequences if len(s) > 0]
    if not non_empty:
        if return_sequence:
            return 0, []
        return 0

    if len(non_empty) == 1:
        length = len(non_empty[0])
        if return_sequence:
            return length, list(non_empty[0])
        return length

    current = list(non_empty[0])
    for i in range(1, len(non_empty)):
        _, current = longest_common_subsequence(current, non_empty[i], return_sequence=True)
        if not current:
            break

    max_len = len(current)
    if return_sequence:
        return max_len, current
    return max_len


if __name__ == "__main__":
    print("=" * 60)
    print("LNDS 最长非降子序列测试")
    print("=" * 60)

    lnds_cases = [
        [10, 9, 2, 5, 3, 7, 101, 18],
        [0, 1, 0, 3, 2, 3],
        [7, 7, 7, 7, 7, 7, 7],
        [4, 10, 4, 3, 8, 9],
        [1, 2, 2, 3, 3, 4],
        [],
        [5],
    ]

    for i, nums in enumerate(lnds_cases, 1):
        lis_len = longest_increasing_subsequence_fast(nums)
        lnds_len, lnds_seq = longest_non_decreasing_subsequence(nums, return_sequence=True)
        print(f"测试用例 {i}: {nums}")
        print(f"  LIS(严格) 长度: {lis_len}")
        print(f"  LNDS(非降)长度: {lnds_len}, 序列: {lnds_seq}")
        is_non_decreasing = all(lnds_seq[k] <= lnds_seq[k + 1] for k in range(len(lnds_seq) - 1))
        print(f"  序列合法性验证(非降): {is_non_decreasing}")
        print()

    print("=" * 60)
    print("LCS 最长公共子序列测试（两序列）")
    print("=" * 60)

    lcs_cases = [
        ("ABCBDAB", "BDCAB"),
        ("AGGTAB", "GXTXAYB"),
        ("abcdef", "abc"),
        ("", "abc"),
        ("abc", "abc"),
        ("abc", "def"),
        ([1, 2, 3, 4, 5], [2, 4, 5, 6]),
        ([10, 20, 30, 20, 30], [10, 20, 30]),
    ]

    for i, (s1, s2) in enumerate(lcs_cases, 1):
        lcs_len = longest_common_subsequence(s1, s2)
        lcs_len_seq, lcs_seq = longest_common_subsequence(s1, s2, return_sequence=True)
        print(f"测试用例 {i}:")
        print(f"  S1: {s1}")
        print(f"  S2: {s2}")
        print(f"  LCS长度: {lcs_len_seq}, 序列: {lcs_seq}")
        assert lcs_len == lcs_len_seq, f"长度不一致: {lcs_len} vs {lcs_len_seq}"
        print()

    print("=" * 60)
    print("多序列LCS测试")
    print("=" * 60)

    multi_lcs_cases = [
        (["ABCBDAB", "BDCAB", "BADCB"], "3序列"),
        (["AGGTAB", "GXTXAYB", "AGTB"], "3序列"),
        (["abcde", "ace", "ae"], "3序列(含子集)"),
        (["abc", "abc", "abc"], "3序列(相同)"),
        (["abc", "def", "ghi"], "3序列(无公共)"),
        (["ABCBDAB", "BDCAB"], "2序列(回退两序列)"),
        ([], "空列表"),
        ([["abc"], ["abc"]], "单元素列表"),
    ]

    for i, (seqs, desc) in enumerate(multi_lcs_cases, 1):
        lcs_len = longest_common_subsequence_multi(seqs)
        lcs_len_seq, lcs_seq = longest_common_subsequence_multi(seqs, return_sequence=True)
        print(f"测试用例 {i} ({desc}):")
        print(f"  输入: {seqs}")
        print(f"  多序列LCS长度: {lcs_len_seq}, 序列: {lcs_seq}")
        assert lcs_len == lcs_len_seq, f"长度不一致: {lcs_len} vs {lcs_len_seq}"
        print()

    print("=" * 60)
    print("性能测试: LCS滚动数组 vs 全表")
    print("=" * 60)

    random.seed(42)
    s1 = [random.randint(0, 100) for _ in range(2000)]
    s2 = [random.randint(0, 100) for _ in range(2000)]

    start = time.time()
    lcs_len_only = longest_common_subsequence(s1, s2, return_sequence=False)
    time_rolling = time.time() - start

    start = time.time()
    lcs_len_full, lcs_seq = longest_common_subsequence(s1, s2, return_sequence=True)
    time_full = time.time() - start

    print(f"n = m = 2000:")
    print(f"  仅长度(滚动数组): {time_rolling:.4f}s, LCS长度={lcs_len_only}")
    print(f"  含序列(全表DP)  : {time_full:.4f}s, LCS长度={lcs_len_full}")
    assert lcs_len_only == lcs_len_full
    print()
