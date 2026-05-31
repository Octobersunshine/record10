def is_isomorphic(s: str, t: str, wildcard: str = None) -> bool:
    if len(s) != len(t):
        return False
    
    s_to_t = {}
    t_to_s = {}
    
    for char_s, char_t in zip(s, t):
        if wildcard is not None and (char_s == wildcard or char_t == wildcard):
            continue
        
        if char_s in s_to_t and s_to_t[char_s] != char_t:
            return False
        if char_t in t_to_s and t_to_s[char_t] != char_s:
            return False
        if char_s not in s_to_t and char_t not in t_to_s:
            s_to_t[char_s] = char_t
            t_to_s[char_t] = char_s
    
    return True


def get_isomorphic_mapping(s: str, t: str, wildcard: str = None) -> dict | None:
    if len(s) != len(t):
        return None
    
    s_to_t = {}
    t_to_s = {}
    
    for char_s, char_t in zip(s, t):
        if wildcard is not None and (char_s == wildcard or char_t == wildcard):
            continue
        
        if char_s in s_to_t:
            if s_to_t[char_s] != char_t:
                return None
        elif char_t in t_to_s:
            if t_to_s[char_t] != char_s:
                return None
        else:
            s_to_t[char_s] = char_t
            t_to_s[char_t] = char_s
    
    return s_to_t


def find_isomorphic_substrings(text: str, pattern: str, wildcard: str = None) -> list[dict]:
    results = []
    pattern_len = len(pattern)
    text_len = len(text)
    
    if pattern_len > text_len or pattern_len == 0:
        return results
    
    for i in range(text_len - pattern_len + 1):
        substring = text[i:i + pattern_len]
        mapping = get_isomorphic_mapping(substring, pattern, wildcard)
        if mapping is not None:
            results.append({
                "start_index": i,
                "end_index": i + pattern_len - 1,
                "substring": substring,
                "mapping": mapping
            })
    
    return results


if __name__ == "__main__":
    all_passed = True
    
    print("=" * 60)
    print("测试 1: 基础同构判断 (is_isomorphic)")
    print("=" * 60)
    
    basic_test_cases = [
        ("egg", "add", None, True),
        ("foo", "bar", None, False),
        ("paper", "title", None, True),
        ("", "", None, True),
        ("a", "", None, False),
        ("", "a", None, False),
        ("ab", "aa", None, False),
        ("abc", "def", None, True),
        ("badc", "baba", None, False),
        ("hello", "world", None, False),
    ]
    
    for i, (s, t, wc, expected) in enumerate(basic_test_cases, 1):
        result = is_isomorphic(s, t, wc)
        status = "✓ 通过" if result == expected else "✗ 失败"
        if result != expected:
            all_passed = False
        print(f"用例 {i}: s='{s}', t='{t}'")
        print(f"  预期: {expected}, 实际: {result} {status}")
        print()
    
    print("=" * 60)
    print("测试 2: 通配符同构判断 (wildcard='?')")
    print("=" * 60)
    
    wildcard_test_cases = [
        ("egg", "a?d", "?", True),
        ("foo", "?a?", "?", True),
        ("paper", "t?t?e", "?", True),
        ("paper", "t?x?e", "?", False),
        ("abc", "?b?", "?", True),
        ("ab", "?a", "?", True),
        ("aba", "?x?", "?", True),
        ("abcde", "a?c?e", "?", True),
        ("hello", "h?ll?", "?", True),
        ("aba", "?xx", "?", False),
    ]
    
    for i, (s, t, wc, expected) in enumerate(wildcard_test_cases, 1):
        result = is_isomorphic(s, t, wc)
        status = "✓ 通过" if result == expected else "✗ 失败"
        if result != expected:
            all_passed = False
        print(f"用例 {i}: s='{s}', t='{t}', wildcard='{wc}'")
        print(f"  预期: {expected}, 实际: {result} {status}")
        print()
    
    print("=" * 60)
    print("测试 3: 获取同构映射 (get_isomorphic_mapping)")
    print("=" * 60)
    
    mapping_test_cases = [
        ("egg", "add", None, {"e": "a", "g": "d"}),
        ("paper", "title", None, {"p": "t", "a": "i", "e": "l", "r": "e"}),
        ("foo", "bar", None, None),
        ("ab", "aa", None, None),
        ("egg", "a?d", "?", {"e": "a", "g": "d"}),
        ("abc", "?b?", "?", {"b": "b"}),
    ]
    
    for i, (s, t, wc, expected) in enumerate(mapping_test_cases, 1):
        result = get_isomorphic_mapping(s, t, wc)
        passed = result == expected
        status = "✓ 通过" if passed else "✗ 失败"
        if not passed:
            all_passed = False
        print(f"用例 {i}: s='{s}', t='{t}', wildcard={repr(wc)}")
        print(f"  预期: {expected}")
        print(f"  实际: {result} {status}")
        print()
    
    print("=" * 60)
    print("测试 4: 查找同构子串 (find_isomorphic_substrings)")
    print("=" * 60)
    
    substr_test_cases = [
        {
            "text": "abcxyzeggaddpaper",
            "pattern": "egg",
            "wildcard": None,
            "expected_count": 2,
            "desc": "查找 'egg' 模式 (字符重复模式)"
        },
        {
            "text": "abcxyzeggaddpaper",
            "pattern": "g?g",
            "wildcard": "?",
            "expected_count": 1,
            "desc": "通配符查找 'g?g' 模式 (首尾相同)"
        },
        {
            "text": "hello world, hello python",
            "pattern": "h?ll?",
            "wildcard": "?",
            "expected_count": 2,
            "desc": "通配符查找 'h?ll?' 模式"
        },
        {
            "text": "abcdefghij",
            "pattern": "xxy",
            "wildcard": None,
            "expected_count": 0,
            "desc": "无匹配子串 (需要重复字符)"
        },
        {
            "text": "ab",
            "pattern": "abc",
            "wildcard": None,
            "expected_count": 0,
            "desc": "模式比文本长"
        },
        {
            "text": "aabbaabbab",
            "pattern": "xyy",
            "wildcard": None,
            "expected_count": 3,
            "desc": "查找 'xyy' 模式 (ABB结构)"
        },
    ]
    
    for i, test in enumerate(substr_test_cases, 1):
        results = find_isomorphic_substrings(
            test["text"], test["pattern"], test["wildcard"]
        )
        passed = len(results) == test["expected_count"]
        status = "✓ 通过" if passed else "✗ 失败"
        if not passed:
            all_passed = False
        print(f"用例 {i}: {test['desc']}")
        print(f"  text='{test['text']}', pattern='{test['pattern']}'")
        print(f"  预期数量: {test['expected_count']}, 实际数量: {len(results)} {status}")
        for j, r in enumerate(results, 1):
            print(f"    匹配 {j}: 位置[{r['start_index']}:{r['end_index']}], "
                  f"子串='{r['substring']}', 映射={r['mapping']}")
        print()
    
    print("=" * 60)
    if all_passed:
        print("所有测试用例通过！")
    else:
        print("部分测试用例失败！")
    print("=" * 60)
