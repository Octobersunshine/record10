import math
import time

FUNCTIONS = {
    'sin': math.sin,
    'cos': math.cos,
    'log': math.log,
    'sqrt': math.sqrt,
}

FUNCTION_NAMES = set(FUNCTIONS.keys())


def tokenize(expression):
    tokens = []
    current = ''
    i = 0
    while i < len(expression):
        char = expression[i]
        if char.isdigit() or char == '.':
            current += char
        else:
            if current:
                tokens.append(current)
                current = ''
            if char.isalpha():
                func_name = ''
                while i < len(expression) and expression[i].isalpha():
                    func_name += expression[i]
                    i += 1
                tokens.append(func_name)
                continue
            elif char in '+-*/(),':
                tokens.append(char)
        i += 1
    if current:
        tokens.append(current)
    return tokens


def infix_to_postfix(expression, variables=None):
    variables = variables or {}
    precedence = {'+': 1, '-': 1, '*': 2, '/': 2, 'u+': 3, 'u-': 3}
    precedence.update({fn: 4 for fn in FUNCTION_NAMES})
    right_associative = {'u+', 'u-'}
    right_associative.update(FUNCTION_NAMES)
    output = []
    stack = []
    steps = []

    tokens = tokenize(expression)
    steps.append(f"原始表达式: {expression}")
    steps.append(f"变量: {variables}")
    steps.append(f"分词结果: {tokens}")

    for i, token in enumerate(tokens):
        is_number = token.replace('.', '', 1).isdigit() and token.count('.') <= 1
        is_variable = token.isalpha() and token not in FUNCTION_NAMES and token not in '+-*/()'

        if is_number:
            output.append(token)
            steps.append(f"遇到数字 {token}，加入输出队列: {output}")
        elif is_variable:
            if token in variables:
                output.append(str(variables[token]))
                steps.append(f"遇到变量 {token}={variables[token]}，替换后加入输出: {output}")
            else:
                raise ValueError(f"未定义的变量: {token}")
        elif token in FUNCTION_NAMES:
            stack.append(token)
            steps.append(f"遇到函数 {token}，压入栈: {stack}")
        elif token == '(':
            stack.append(token)
            steps.append(f"遇到左括号，压入栈: {stack}")
        elif token == ')':
            while stack and stack[-1] != '(':
                op = stack.pop()
                output.append(op)
                steps.append(f"遇到右括号，弹出栈顶 {op} 到输出: {output}")
            if stack and stack[-1] == '(':
                stack.pop()
                steps.append(f"弹出左括号，当前栈: {stack}")
            if stack and stack[-1] in FUNCTION_NAMES:
                fn = stack.pop()
                output.append(fn)
                steps.append(f"弹出函数 {fn} 到输出: {output}")
        else:
            is_unary = False
            if token in '+-':
                if i == 0:
                    is_unary = True
                elif tokens[i - 1] in '+-*/(':
                    is_unary = True

            if is_unary:
                token = 'u' + token
                steps.append(f"识别一元运算符 {token}")

            while stack and stack[-1] != '(':
                if token in right_associative:
                    if precedence.get(stack[-1], 0) > precedence.get(token, 0):
                        op = stack.pop()
                        output.append(op)
                        steps.append(f"栈顶 {op} 优先级 > {token}，弹出到输出: {output}")
                    else:
                        break
                else:
                    if precedence.get(stack[-1], 0) >= precedence.get(token, 0):
                        op = stack.pop()
                        output.append(op)
                        steps.append(f"栈顶 {op} 优先级 >= {token}，弹出到输出: {output}")
                    else:
                        break
            stack.append(token)
            steps.append(f"将 {token} 压入栈: {stack}")

    while stack:
        op = stack.pop()
        if op == '(':
            raise ValueError("括号不匹配：存在未闭合的左括号")
        output.append(op)
        steps.append(f"弹出剩余栈中 {op} 到输出: {output}")

    return output, steps


def evaluate_postfix(postfix):
    stack = []
    steps = []

    steps.append(f"后缀表达式: {postfix}")

    for token in postfix:
        is_number = token.replace('.', '', 1).isdigit() and token.count('.') <= 1
        if is_number:
            stack.append(float(token))
            steps.append(f"遇到数字 {token}，压入栈: {stack}")
        elif token in ('u+', 'u-'):
            a = stack.pop()
            steps.append(f"遇到一元运算符 {token}，弹出 {a}")
            result = a if token == 'u+' else -a
            stack.append(result)
            steps.append(f"计算 {token[1:]}{a} = {result}，压入栈: {stack}")
        elif token in FUNCTION_NAMES:
            a = stack.pop()
            steps.append(f"遇到函数 {token}，弹出 {a}")
            result = FUNCTIONS[token](a)
            stack.append(result)
            steps.append(f"计算 {token}({a}) = {result}，压入栈: {stack}")
        else:
            b = stack.pop()
            a = stack.pop()
            steps.append(f"遇到运算符 {token}，弹出 {a} 和 {b}")
            if token == '+':
                result = a + b
            elif token == '-':
                result = a - b
            elif token == '*':
                result = a * b
            elif token == '/':
                result = a / b
            else:
                raise ValueError(f"未知运算符: {token}")
            stack.append(result)
            steps.append(f"计算 {a} {token} {b} = {result}，压入栈: {stack}")

    final_result = stack[0]
    steps.append(f"最终结果: {final_result}")
    return final_result, steps


def evaluate_expression(expression, variables=None):
    expression = expression.replace(' ', '')
    variables = variables or {}
    postfix, conversion_steps = infix_to_postfix(expression, variables)
    result, evaluation_steps = evaluate_postfix(postfix)
    all_steps = conversion_steps + ['\n=== 计算后缀表达式 ==='] + evaluation_steps
    return result, all_steps


def _direct_tokenize(expression):
    tokens = []
    current = ''
    i = 0
    while i < len(expression):
        char = expression[i]
        if char.isdigit() or char == '.':
            current += char
        else:
            if current:
                tokens.append(current)
                current = ''
            if char.isalpha():
                func_name = ''
                while i < len(expression) and expression[i].isalpha():
                    func_name += expression[i]
                    i += 1
                tokens.append(func_name)
                continue
            elif char in '+-*/(),':
                tokens.append(char)
        i += 1
    if current:
        tokens.append(current)
    return tokens


def evaluate_direct(expression, variables=None):
    expression = expression.replace(' ', '')
    variables = variables or {}
    tokens = _direct_tokenize(expression)
    pos = [0]

    def peek():
        if pos[0] < len(tokens):
            return tokens[pos[0]]
        return None

    def consume():
        token = tokens[pos[0]]
        pos[0] += 1
        return token

    def parse_expr():
        left = parse_term()
        while peek() in ('+', '-'):
            op = consume()
            right = parse_term()
            if op == '+':
                left = left + right
            else:
                left = left - right
        return left

    def parse_term():
        left = parse_factor()
        while peek() in ('*', '/'):
            op = consume()
            right = parse_factor()
            if op == '*':
                left = left * right
            else:
                left = left / right
        return left

    def parse_factor():
        token = peek()

        if token in FUNCTION_NAMES:
            fn_name = consume()
            consume()  # (
            arg = parse_expr()
            consume()  # )
            return FUNCTIONS[fn_name](arg)

        if token == '(':
            consume()
            result = parse_expr()
            consume()  # )
            return result

        if token in ('+', '-'):
            op = consume()
            val = parse_factor()
            return val if op == '+' else -val

        token = consume()
        is_number = token.replace('.', '', 1).isdigit() and token.count('.') <= 1
        if is_number:
            return float(token)
        if token.isalpha() and token in variables:
            return float(variables[token])
        raise ValueError(f"意外的token: {token}")

    result = parse_expr()
    return result


def benchmark(expression, variables=None, iterations=10000):
    variables = variables or {}
    clean_expr = expression.replace(' ', '')

    start = time.perf_counter()
    for _ in range(iterations):
        infix_to_postfix(clean_expr, variables)
        evaluate_postfix(infix_to_postfix(clean_expr, variables)[0])
    postfix_time = time.perf_counter() - start

    start = time.perf_counter()
    for _ in range(iterations):
        evaluate_direct(clean_expr, variables)
    direct_time = time.perf_counter() - start

    return {
        'expression': expression,
        'variables': variables,
        'iterations': iterations,
        'postfix_time': postfix_time,
        'direct_time': direct_time,
        'ratio': postfix_time / direct_time if direct_time > 0 else float('inf'),
        'faster': '直接求值' if direct_time < postfix_time else '后缀求值',
    }


if __name__ == "__main__":
    print("=" * 60)
    print("1. 基本表达式测试")
    print("=" * 60)
    result, steps = evaluate_expression("3+4*2/(1-5)")
    print(f"3+4*2/(1-5) = {result}")

    print("\n" + "=" * 60)
    print("2. 函数测试")
    print("=" * 60)
    result, steps = evaluate_expression("sin(0)+cos(0)")
    print(f"sin(0)+cos(0) = {result}")

    result, steps = evaluate_expression("sqrt(16)*2")
    print(f"sqrt(16)*2 = {result}")

    result, steps = evaluate_expression("log(1)+3")
    print(f"log(1)+3 = {result}")

    result, steps = evaluate_expression("2*sin(3.14159/2)")
    print(f"2*sin(3.14159/2) = {result}")

    print("\n" + "=" * 60)
    print("3. 变量替换测试")
    print("=" * 60)
    result, steps = evaluate_expression("x+y*2", variables={"x": 3, "y": 4})
    print(f"x+y*2 (x=3, y=4) = {result}")

    result, steps = evaluate_expression("a*sin(b)", variables={"a": 2, "b": 1.5708})
    print(f"a*sin(b) (a=2, b=pi/2) = {result}")

    result, steps = evaluate_expression("sqrt(x)+y", variables={"x": 9, "y": 1})
    print(f"sqrt(x)+y (x=9, y=1) = {result}")

    print("\n" + "=" * 60)
    print("4. 效率对比")
    print("=" * 60)
    for expr, var, iters in [
        ("3+4*2/(1-5)", None, 50000),
        ("sin(x)+cos(y)*sqrt(z)", {"x": 1, "y": 2, "z": 4}, 20000),
        ("a*b+c/d-e", {"a": 1, "b": 2, "c": 3, "d": 4, "e": 5}, 50000),
    ]:
        bench = benchmark(expr, var, iters)
        print(f"\n表达式: {bench['expression']}")
        if bench['variables']:
            print(f"变量: {bench['variables']}")
        print(f"迭代次数: {bench['iterations']}")
        print(f"后缀求值耗时: {bench['postfix_time']:.4f}s")
        print(f"直接求值耗时: {bench['direct_time']:.4f}s")
        print(f"后缀/直接 比率: {bench['ratio']:.2f}x")
        print(f"更快方式: {bench['faster']}")
