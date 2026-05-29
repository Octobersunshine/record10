class BracketMatcher:
    def __init__(self):
        self.opening = '([{'
        self.closing = ')]}'
        self.matching = {')': '(', ']': '[', '}': '{'}

    def is_valid(self, s: str, verbose: bool = False) -> dict:
        stack = []
        steps = []
        max_depth = 0
        current_depth = 0
        error_pos = -1
        error_char = ''
        error_type = ''

        for idx, char in enumerate(s):
            if char not in self.opening and char not in self.closing:
                continue

            if char in self.opening:
                stack.append((char, idx))
                current_depth += 1
                max_depth = max(max_depth, current_depth)
                steps.append(f"步骤 {len(steps) + 1}: 字符 '{char}' (位置 {idx}) 入栈，当前深度: {current_depth}，当前栈: {[c for c, _ in stack]}")
            else:
                if not stack:
                    error_pos = idx
                    error_char = char
                    error_type = '无匹配左括号'
                    steps.append(f"步骤 {len(steps) + 1}: 字符 '{char}' (位置 {idx}) 没有匹配的左括号，匹配失败")
                    if verbose:
                        self._print_steps(steps)
                    return {
                        'is_valid': False,
                        'max_depth': max_depth,
                        'error_pos': error_pos,
                        'error_char': error_char,
                        'error_type': error_type
                    }
                top_char, top_idx = stack.pop()
                if top_char == self.matching[char]:
                    current_depth -= 1
                    steps.append(f"步骤 {len(steps) + 1}: 字符 '{char}' (位置 {idx}) 与栈顶 '{top_char}' (位置 {top_idx}) 匹配，弹出栈顶，当前深度: {current_depth}，当前栈: {[c for c, _ in stack]}")
                else:
                    error_pos = idx
                    error_char = char
                    error_type = f'与左括号 "{top_char}" (位置 {top_idx}) 不匹配'
                    steps.append(f"步骤 {len(steps) + 1}: 字符 '{char}' (位置 {idx}) 与栈顶 '{top_char}' (位置 {top_idx}) 不匹配，匹配失败")
                    if verbose:
                        self._print_steps(steps)
                    return {
                        'is_valid': False,
                        'max_depth': max_depth,
                        'error_pos': error_pos,
                        'error_char': error_char,
                        'error_type': error_type
                    }

        if not stack:
            steps.append(f"步骤 {len(steps) + 1}: 栈为空，所有括号匹配成功，最大嵌套深度: {max_depth}")
            if verbose:
                self._print_steps(steps)
            return {
                'is_valid': True,
                'max_depth': max_depth,
                'error_pos': -1,
                'error_char': '',
                'error_type': ''
            }
        else:
            leftover_char, leftover_idx = stack[0]
            error_pos = leftover_idx
            error_char = leftover_char
            error_type = '无匹配右括号'
            steps.append(f"步骤 {len(steps) + 1}: 栈不为空，剩余未匹配的左括号: {[c for c, _ in stack]}，匹配失败")
            if verbose:
                self._print_steps(steps)
            return {
                'is_valid': False,
                'max_depth': max_depth,
                'error_pos': error_pos,
                'error_char': error_char,
                'error_type': error_type
            }

    def _print_steps(self, steps):
        print("\n" + "=" * 60)
        print("匹配过程:")
        print("=" * 60)
        for step in steps:
            print(step)
        print("=" * 60 + "\n")


def bracket_matching_api(s: str, verbose: bool = False) -> dict:
    matcher = BracketMatcher()
    result = matcher.is_valid(s, verbose)
    return {
        "input": s,
        "is_valid": result['is_valid'],
        "max_depth": result['max_depth'],
        "error_pos": result['error_pos'],
        "error_char": result['error_char'],
        "error_type": result['error_type'],
        "message": "括号匹配合法" if result['is_valid'] else "括号匹配不合法"
    }


if __name__ == "__main__":
    test_cases = [
        "()",
        "()[]{}",
        "(]",
        "([)]",
        "{[]}",
        "((()))",
        "({[)]}",
        "",
        "abc(def[ghi]{jkl})",
        "((())",
        "())",
        "{{[[(())]]}}",
        "a(b)c[d]e{f}g"
    ]

    print("=" * 60)
    print("括号匹配API测试")
    print("=" * 60)

    for test in test_cases:
        print(f"\n测试字符串: '{test}'")
        result = bracket_matching_api(test, verbose=True)
        print(f"结果: {result['message']}")
        print(f"最大嵌套深度: {result['max_depth']}")
        if not result['is_valid']:
            print(f"错误位置: {result['error_pos']}")
            print(f"错误字符: '{result['error_char']}'")
            print(f"错误类型: {result['error_type']}")
