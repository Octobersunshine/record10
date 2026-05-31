from dataclasses import dataclass
import time
import random


@dataclass
class MatchResult:
    length: int
    start: int
    end: int
    substring: str

    def __repr__(self):
        if self.length == 0:
            return "MatchResult(length=0, 无有效匹配)"
        return (f"MatchResult(length={self.length}, "
                f"range=[{self.start},{self.end}], "
                f'substring="{self.substring}")')


EMPTY_RESULT = MatchResult(0, -1, -1, "")


def longest_valid_stack_single(s: str) -> MatchResult:
    stack = [-1]
    max_len = 0
    best_start = -1
    best_end = -1
    for i, char in enumerate(s):
        if char == '(':
            stack.append(i)
        else:
            stack.pop()
            if not stack:
                stack.append(i)
            else:
                current_len = i - stack[-1]
                if current_len > max_len:
                    max_len = current_len
                    best_start = stack[-1] + 1
                    best_end = i
    if max_len == 0:
        return EMPTY_RESULT
    return MatchResult(max_len, best_start, best_end, s[best_start:best_end + 1])


def longest_valid_dp_single(s: str) -> MatchResult:
    n = len(s)
    if n < 2:
        return EMPTY_RESULT
    dp = [0] * n
    max_len = 0
    best_end = -1
    for i in range(1, n):
        if s[i] == ')':
            if s[i - 1] == '(':
                dp[i] = dp[i - 2] + 2 if i >= 2 else 2
            elif i - dp[i - 1] > 0 and s[i - dp[i - 1] - 1] == '(':
                if i - dp[i - 1] >= 2:
                    dp[i] = dp[i - 1] + dp[i - dp[i - 1] - 2] + 2
                else:
                    dp[i] = dp[i - 1] + 2
            if dp[i] > max_len:
                max_len = dp[i]
                best_end = i
    if max_len == 0:
        return EMPTY_RESULT
    best_start = best_end - max_len + 1
    return MatchResult(max_len, best_start, best_end, s[best_start:best_end + 1])


def longest_valid_two_pass_single(s: str) -> MatchResult:
    n = len(s)
    left = right = 0
    max_len = 0
    best_start = -1
    best_end = -1

    for i, char in enumerate(s):
        if char == '(':
            left += 1
        else:
            right += 1
        if left == right:
            current_len = 2 * right
            if current_len > max_len:
                max_len = current_len
                best_end = i
                best_start = i - current_len + 1
        elif right > left:
            left = right = 0

    left = right = 0
    for i in range(n - 1, -1, -1):
        if s[i] == '(':
            left += 1
        else:
            right += 1
        if left == right:
            current_len = 2 * left
            if current_len > max_len:
                max_len = current_len
                best_start = i
                best_end = i + current_len - 1
        elif left > right:
            left = right = 0

    if max_len == 0:
        return EMPTY_RESULT
    return MatchResult(max_len, best_start, best_end, s[best_start:best_end + 1])


OPEN_BRACKETS = {'(': ')', '[': ']', '{': '}'}
CLOSE_BRACKETS = {')': '(', ']': '[', '}': '{'}


def longest_valid_mixed(s: str) -> MatchResult:
    stack = [-1]
    max_len = 0
    best_start = -1
    best_end = -1
    for i, char in enumerate(s):
        if char in OPEN_BRACKETS:
            stack.append(i)
        elif char in CLOSE_BRACKETS:
            if stack[-1] != -1 and s[stack[-1]] == CLOSE_BRACKETS[char]:
                stack.pop()
                current_len = i - stack[-1]
                if current_len > max_len:
                    max_len = current_len
                    best_start = stack[-1] + 1
                    best_end = i
            else:
                stack.append(i)
        else:
            stack.append(i)
    if max_len == 0:
        return EMPTY_RESULT
    return MatchResult(max_len, best_start, best_end, s[best_start:best_end + 1])


def benchmark(func, s: str, iterations: int = 100) -> float:
    start = time.perf_counter()
    for _ in range(iterations):
        func(s)
    elapsed = time.perf_counter() - start
    return elapsed / iterations * 1000


def run_single_tests():
    test_cases = [
        ("(()", 2),
        (")()())", 4),
        ("", 0),
        ("()", 2),
        ("(()()", 4),
        ("()(())", 6),
        ("()()()", 6),
        ("())()()", 4),
        ("(()))()()()", 6),
        ("((((", 0),
        ("))))", 0),
        ("()()", 4),
        ("()(())()", 8),
        ("(()))(", 4),
        ("((()()))", 8),
        (")(", 0),
        ("((((()))))", 10),
        ("()()()()", 8),
        ("(()(()(", 2),
    ]

    print("=" * 80)
    print("【单括号类型测试 - 仅含()】")
    print("=" * 80)
    all_pass = True
    funcs = [
        ("栈方法", longest_valid_stack_single),
        ("DP方法", longest_valid_dp_single),
        ("两次遍历", longest_valid_two_pass_single),
    ]

    for s, expected in test_cases:
        results = []
        for name, func in funcs:
            r = func(s)
            results.append((name, r))
        passed = all(r.length == expected for _, r in results)
        if not passed:
            all_pass = False

        status = "✓" if passed else "✗"
        print(f'{status} 输入: "{s:12}" 期望: {expected:2d}', end="")
        for name, r in results:
            print(f" | {name}: {r.length:2d}", end="")
        if expected > 0:
            r0 = results[0][1]
            print(f' | 子串: "{r0.substring}" [{r0.start},{r0.end}]', end="")
        print()

    print("-" * 80)
    print(f"单括号测试: {'全部通过 ✓' if all_pass else '存在失败 ✗'}")
    return all_pass


def run_mixed_tests():
    test_cases = [
        ("()[]{}", 6),
        ("([)]", 0),
        ("({[]})", 6),
        ("([{}])", 6),
        ("(]", 0),
        ("([)]{}", 2),
        ("{[()]}", 6),
        ("()[]{()}", 8),
        ("({)}[]", 2),
        ("[(])", 0),
        ("(([]))", 6),
        ("{()[]()}", 8),
        ("([{)]}", 0),
        ("((()", 2),
        ("([{", 0),
        ("}]{[", 0),
        ("()([{}])[]", 10),
        ("(([]){})", 8),
        ("([)()]", 2),
        ("{([]()[])}", 10),
    ]

    print()
    print("=" * 80)
    print("【多括号混合测试 - 含()、[]、{}】")
    print("=" * 80)
    all_pass = True

    for s, expected in test_cases:
        r = longest_valid_mixed(s)
        passed = r.length == expected
        if not passed:
            all_pass = False
        status = "✓" if passed else "✗"
        substr_info = f' | 子串: "{r.substring}" [{r.start},{r.end}]' if r.length > 0 else ""
        print(f'{status} 输入: "{s:14}" 期望: {expected:2d} | 结果: {r.length:2d}{substr_info}')

    print("-" * 80)
    print(f"多括号混合测试: {'全部通过 ✓' if all_pass else '存在失败 ✗'}")
    return all_pass


def run_benchmark():
    print()
    print("=" * 80)
    print("【时间复杂度对比】")
    print("=" * 80)

    sizes = [100, 1000, 10000, 50000]
    random.seed(42)

    print(f"{'规模':>8} | {'栈方法(ms)':>10} | {'DP方法(ms)':>10} | {'两次遍历(ms)':>12} | {'混合括号(ms)':>12}")
    print("-" * 80)

    for size in sizes:
        s = ''.join(random.choice('()') for _ in range(size))
        s_mixed = ''.join(random.choice('()[]{}') for _ in range(size))
        iters = max(1, 50000 // size)

        t_stack = benchmark(longest_valid_stack_single, s, iters)
        t_dp = benchmark(longest_valid_dp_single, s, iters)
        t_two = benchmark(longest_valid_two_pass_single, s, iters)
        t_mixed = benchmark(longest_valid_mixed, s_mixed, iters)

        print(f"{size:>8} | {t_stack:>10.4f} | {t_dp:>10.4f} | {t_two:>12.4f} | {t_mixed:>12.4f}")

    print("-" * 80)
    print("复杂度分析:")
    print("  栈方法:     时间 O(n)  空间 O(n)  - 通用性强，支持多括号扩展")
    print("  DP方法:     时间 O(n)  空间 O(n)  - 状态转移清晰，仅适用单括号类型")
    print("  两次遍历法: 时间 O(n)  空间 O(1)  - 空间最优，仅适用单括号类型")
    print("  混合括号法: 时间 O(n)  空间 O(n)  - 基于栈扩展，支持()[]{}混合匹配")


if __name__ == "__main__":
    single_ok = run_single_tests()
    mixed_ok = run_mixed_tests()
    run_benchmark()
    print()
    print("=" * 80)
    print(f"总结果: {'全部测试通过 ✓' if single_ok and mixed_ok else '存在测试失败 ✗'}")
