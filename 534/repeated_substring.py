def detect_repeated_substring_kmp(s):
    n = len(s)
    if n == 0:
        return None, 0
    if n == 1:
        return s, 1
    
    next_arr = [0] * n
    j = 0
    for i in range(1, n):
        while j > 0 and s[i] != s[j]:
            j = next_arr[j - 1]
        if s[i] == s[j]:
            j += 1
        next_arr[i] = j
    
    l = next_arr[-1]
    if l > 0 and n % (n - l) == 0:
        repeat_count = n // (n - l)
        repeat_substring = s[:n - l]
        return repeat_substring, repeat_count
    return None, 1


def detect_repeated_substring_divisor(s):
    n = len(s)
    if n == 0:
        return None, 0
    if n == 1:
        return s, 1
    
    for length in range(1, n // 2 + 1):
        if n % length == 0:
            repeat_count = n // length
            substring = s[:length]
            if substring * repeat_count == s:
                return substring, repeat_count
    return None, 1


def find_longest_repeated_substring(s, ignore_case=False, min_repeats=2):
    n = len(s)
    if n < 2:
        return None, 0, []
    
    s_compare = s.lower() if ignore_case else s
    max_length = 0
    best_substring = None
    best_count = 0
    best_positions = []
    
    for length in range(n - 1, 0, -1):
        substring_positions = {}
        
        for start in range(n - length + 1):
            substr = s_compare[start:start + length]
            if substr not in substring_positions:
                substring_positions[substr] = []
            substring_positions[substr].append(start)
        
        for substr, positions in substring_positions.items():
            if len(positions) >= min_repeats:
                if length > max_length:
                    max_length = length
                    best_substring = s[positions[0]:positions[0] + length]
                    best_count = len(positions)
                    best_positions = positions
        
        if max_length > 0:
            break
    
    if max_length > 0:
        return best_substring, best_count, best_positions
    return None, 0, []


def find_all_repeated_substrings(s, ignore_case=False, min_repeats=2, min_length=1):
    n = len(s)
    if n < 2:
        return []
    
    s_compare = s.lower() if ignore_case else s
    results = []
    
    for length in range(n - 1, min_length - 1, -1):
        substring_positions = {}
        
        for start in range(n - length + 1):
            substr = s_compare[start:start + length]
            if substr not in substring_positions:
                substring_positions[substr] = []
            substring_positions[substr].append(start)
        
        for substr, positions in substring_positions.items():
            if len(positions) >= min_repeats:
                original_substr = s[positions[0]:positions[0] + length]
                results.append({
                    'substring': original_substr,
                    'length': length,
                    'count': len(positions),
                    'positions': positions
                })
    
    return results


def main():
    test_cases_full = [
        "abcabc",
        "ababab",
        "aaaaa",
        "abcd",
        "abababab",
        "xyzxyzxyz",
        "",
        "a",
        "aa",
        "abababc"
    ]
    
    print("=" * 70)
    print("检测字符串是否由重复子串组成（完全重复）")
    print("=" * 70)
    
    for s in test_cases_full:
        print(f"\n测试字符串: '{s}'")
        
        substr_kmp, count_kmp = detect_repeated_substring_kmp(s)
        substr_div, count_div = detect_repeated_substring_divisor(s)
        
        print(f"  KMP next数组法: ", end="")
        if substr_kmp is not None:
            print(f"由 '{substr_kmp}' 重复 {count_kmp} 次组成")
        else:
            print("不是由重复子串组成 (返回False)")
        
        print(f"  整除检查法:     ", end="")
        if substr_div is not None:
            print(f"由 '{substr_div}' 重复 {count_div} 次组成")
        else:
            print("不是由重复子串组成 (返回False)")
        
        assert substr_kmp == substr_div and count_kmp == count_div, "两种方法结果不一致！"
    
    print("\n")
    print("=" * 70)
    print("查找最长重复子串（非完全重复）")
    print("=" * 70)
    
    test_cases_longest = [
        "ababa",
        "abcabcab",
        "ABabAB",
        "mississippi",
        "banana",
        "testtesttest",
        "abcdef",
        "aabbaabbaabb"
    ]
    
    for s in test_cases_longest:
        print(f"\n测试字符串: '{s}'")
        
        substr, count, positions = find_longest_repeated_substring(s)
        print(f"  最长重复子串: ", end="")
        if substr:
            print(f"'{substr}' (长度={len(substr)}, 重复{count}次)")
            print(f"  出现位置: {positions}")
        else:
            print("无")
        
        substr_ci, count_ci, positions_ci = find_longest_repeated_substring(s, ignore_case=True)
        print(f"  忽略大小写: ", end="")
        if substr_ci:
            print(f"'{substr_ci}' (长度={len(substr_ci)}, 重复{count_ci}次)")
        else:
            print("无")
    
    print("\n")
    print("=" * 70)
    print("查找所有重复子串（按长度降序）")
    print("=" * 70)
    
    test_all = "ababa"
    print(f"\n测试字符串: '{test_all}'")
    all_repeats = find_all_repeated_substrings(test_all, min_repeats=2, min_length=2)
    
    if all_repeats:
        print(f"  找到 {len(all_repeats)} 个重复子串:")
        for i, result in enumerate(all_repeats[:5], 1):
            print(f"    {i}. '{result['substring']}' (长度={result['length']}, "
                  f"重复{result['count']}次, 位置={result['positions']})")
        if len(all_repeats) > 5:
            print(f"    ... 还有 {len(all_repeats) - 5} 个")
    else:
        print("  无重复子串")


if __name__ == "__main__":
    main()
