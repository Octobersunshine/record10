import time
from collections import deque

def build_next(pattern):
    n = len(pattern)
    next_arr = [0] * n
    j = 0
    for i in range(1, n):
        while j > 0 and pattern[i] != pattern[j]:
            j = next_arr[j - 1]
        if pattern[i] == pattern[j]:
            j += 1
        next_arr[i] = j
    return next_arr

def kmp_search(text, pattern):
    if not pattern:
        return [], []
    
    m, n = len(text), len(pattern)
    if m < n:
        return [], []
    
    next_arr = build_next(pattern)
    j = 0
    positions = []
    
    for i in range(m):
        while j > 0 and text[i] != pattern[j]:
            j = next_arr[j - 1]
        if text[i] == pattern[j]:
            j += 1
        if j == n:
            positions.append(i - n + 1)
            j = next_arr[j - 1]
    
    return positions, next_arr

class ACNode:
    def __init__(self):
        self.children = {}
        self.fail = None
        self.outputs = []

class ACAutomaton:
    def __init__(self, patterns):
        self.root = ACNode()
        self.patterns = patterns
        self._build_trie()
        self._build_fail()
    
    def _build_trie(self):
        for idx, pattern in enumerate(self.patterns):
            if not pattern:
                continue
            node = self.root
            for char in pattern:
                if char not in node.children:
                    node.children[char] = ACNode()
                node = node.children[char]
            node.outputs.append(idx)
    
    def _build_fail(self):
        queue = deque()
        for child in self.root.children.values():
            child.fail = self.root
            queue.append(child)
        
        while queue:
            current_node = queue.popleft()
            for char, child in current_node.children.items():
                queue.append(child)
                fail_node = current_node.fail
                while fail_node is not None and char not in fail_node.children:
                    fail_node = fail_node.fail
                child.fail = fail_node.children[char] if fail_node else self.root
                if child.fail:
                    child.outputs.extend(child.fail.outputs)
    
    def search(self, text):
        results = {pattern: [] for pattern in self.patterns}
        node = self.root
        
        for i, char in enumerate(text):
            while node is not self.root and char not in node.children:
                node = node.fail
            if char in node.children:
                node = node.children[char]
            
            for pattern_idx in node.outputs:
                pattern = self.patterns[pattern_idx]
                start_pos = i - len(pattern) + 1
                results[pattern].append(start_pos)
        
        return results

def wildcard_match(text, pattern):
    m, n = len(text), len(pattern)
    dp = [[False] * (n + 1) for _ in range(m + 1)]
    dp[0][0] = True
    
    for j in range(1, n + 1):
        if pattern[j - 1] == '*':
            dp[0][j] = dp[0][j - 1]
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if pattern[j - 1] == '*':
                dp[i][j] = dp[i][j - 1] or dp[i - 1][j]
            elif pattern[j - 1] == '?' or text[i - 1] == pattern[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
    
    return dp[m][n]

def wildcard_search_all(text, pattern):
    if not pattern:
        return []
    if '*' not in pattern and '?' not in pattern:
        positions, _ = kmp_search(text, pattern)
        return positions
    
    positions = []
    m, n = len(text), len(pattern)
    
    min_len = sum(1 for c in pattern if c != '*')
    if m < min_len:
        return positions
    
    if pattern.startswith('*') and pattern.endswith('*'):
        for i in range(m + 1):
            for j in range(i, m + 1):
                if wildcard_match(text[i:j], pattern.strip('*')):
                    if j - i >= min_len - 2 * (pattern.count('*') - 1):
                        if i not in positions:
                            positions.append(i)
    elif pattern.endswith('*'):
        prefix = pattern.rstrip('*')
        for i in range(m):
            if wildcard_match(text[i:], pattern):
                positions.append(i)
                if '*' not in prefix:
                    break
    elif pattern.startswith('*'):
        suffix = pattern.lstrip('*')
        suffix_len = len(suffix)
        for i in range(m - suffix_len + 1):
            if wildcard_match(text[i:i+suffix_len], suffix):
                positions.append(i)
    else:
        for i in range(m):
            for j in range(i + 1, m + 1):
                if wildcard_match(text[i:j], pattern):
                    positions.append(i)
                    break
    
    return sorted(list(set(positions)))

def builtin_find_all(text, pattern):
    positions = []
    start = 0
    while True:
        pos = text.find(pattern, start)
        if pos == -1:
            break
        positions.append(pos)
        start = pos + 1
    return positions

def performance_test():
    print("\n" + "="*70)
    print("性能对比测试")
    print("="*70)
    
    text = "ABABABCABABABCABABABC" * 1000
    pattern = "ABABC"
    
    print(f"\n文本长度: {len(text)}")
    print(f"模式串: '{pattern}'")
    print()
    
    start = time.time()
    for _ in range(100):
        kmp_search(text, pattern)
    kmp_time = (time.time() - start) / 100 * 1000
    
    start = time.time()
    for _ in range(100):
        builtin_find_all(text, pattern)
    builtin_time = (time.time() - start) / 100 * 1000
    
    patterns = ["ABABC", "ABC", "BCA", "AB"]
    ac = ACAutomaton(patterns)
    start = time.time()
    for _ in range(100):
        ac.search(text)
    ac_time = (time.time() - start) / 100 * 1000
    
    print(f"{'算法':<15} {'平均耗时 (ms)':<15} {'匹配数':<10}")
    print("-" * 50)
    
    kmp_positions, _ = kmp_search(text, pattern)
    print(f"{'KMP':<15} {kmp_time:.4f}{'':<11} {len(kmp_positions):<10}")
    
    builtin_positions = builtin_find_all(text, pattern)
    print(f"{'内置 str.find':<15} {builtin_time:.4f}{'':<11} {len(builtin_positions):<10}")
    
    ac_results = ac.search(text)
    ac_total = sum(len(v) for v in ac_results.values())
    print(f"{'AC自动机(4模式)':<15} {ac_time:.4f}{'':<11} {ac_total:<10}")
    
    print(f"\n性能比率 (KMP vs 内置): {kmp_time/builtin_time:.2f}x")
    print(f"性能比率 (AC自动机 vs KMP): {ac_time/kmp_time:.2f}x")

if __name__ == "__main__":
    print("="*70)
    print("1. KMP 算法测试")
    print("="*70)
    
    test_cases = [
        ("ABABABCABABABCABABABC", "ABABC", "正常匹配测试"),
        ("ABCDE", "XYZ", "无匹配测试"),
        ("ABCDE", "ABCDE", "完全匹配测试"),
        ("", "ABC", "空文本串测试"),
        ("ABC", "", "空模式串测试"),
        ("", "", "两者都为空测试"),
        ("ABC", "ABCD", "文本短于模式测试"),
        ("AAAAA", "AA", "多次重叠匹配测试"),
    ]
    
    for text, pattern, desc in test_cases:
        print(f"\n{'='*50}")
        print(f"测试: {desc}")
        print(f"{'='*50}")
        print(f"文本串: '{text}' (长度: {len(text)})")
        print(f"模式串: '{pattern}' (长度: {len(pattern)})")
        
        positions, next_arr = kmp_search(text, pattern)
        
        print(f"部分匹配表 (next数组): {next_arr}")
        print(f"匹配位置: {positions}")
        
        for idx, pos in enumerate(positions, 1):
            print(f"  第{idx}次: 位置 {pos} -> '{text[pos:pos+len(pattern)]}'")
        
        if not positions:
            print("  (无匹配)")
    
    print("\n" + "="*70)
    print("2. AC自动机 - 多模式串匹配测试")
    print("="*70)
    
    text = "ABABABCABABABCABABABC"
    patterns = ["ABABC", "ABC", "BCA", "AB"]
    print(f"\n文本串: '{text}'")
    print(f"模式串列表: {patterns}")
    
    ac = ACAutomaton(patterns)
    results = ac.search(text)
    
    print("\n匹配结果:")
    for pattern, positions in results.items():
        print(f"  '{pattern}': {positions}")
    
    print("\n" + "="*70)
    print("3. 通配符匹配测试 (?匹配单字符, *匹配任意字符)")
    print("="*70)
    
    wildcard_tests = [
        ("hello world", "he?lo*", "匹配测试1"),
        ("hello world", "h*o?d", "匹配测试2"),
        ("hello world", "*world", "后缀匹配"),
        ("hello world", "hello*", "前缀匹配"),
        ("hello world", "*ell*", "包含匹配"),
        ("hello world", "he?lo", "精确匹配(?)"),
        ("hello world", "xyz*", "不匹配测试"),
        ("abcde", "a?c?e", "全问号匹配"),
        ("", "*", "空字符串匹配*"),
        ("abc", "???", "问号匹配全部"),
    ]
    
    for text, pattern, desc in wildcard_tests:
        is_match = wildcard_match(text, pattern)
        positions = wildcard_search_all(text, pattern)
        print(f"\n测试: {desc}")
        print(f"  文本: '{text}', 模式: '{pattern}'")
        print(f"  是否匹配: {is_match}, 匹配位置: {positions}")
    
    performance_test()
    
    print("\n" + "="*70)
    print("所有测试完成!")
    print("="*70)
