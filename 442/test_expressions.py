from expression_evaluator import evaluate_expression

def test_expression(expr, expected):
    result, steps = evaluate_expression(expr.replace(' ', ''))
    print(f"\n{'='*50}")
    print(f"测试表达式: {expr}")
    print(f"预期结果: {expected}")
    print(f"实际结果: {result}")
    print(f"测试通过: {abs(result - expected) < 0.0001}")
    print("转换步骤:")
    for step in steps:
        print(f"  {step}")

if __name__ == "__main__":
    test_expression("3 + 4 * 2", 11)
    test_expression("(3 + 4) * 2", 14)
    test_expression("10 / 2 + 3", 8)
    test_expression("100 - 20 * 3", 40)
    test_expression("( ( 2 + 3 ) * ( 4 - 1 ) ) / 5", 3)
