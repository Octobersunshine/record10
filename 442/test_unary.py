from expression_evaluator import evaluate_expression

def test_expression(expr, expected):
    result, steps = evaluate_expression(expr)
    passed = abs(result - expected) < 0.0001
    status = "✓ 通过" if passed else "✗ 失败"
    print(f"\n{'='*60}")
    print(f"表达式: {expr}")
    print(f"预期: {expected}, 实际: {result} - {status}")
    if not passed:
        print("步骤:")
        for step in steps:
            print(f"  {step}")
    return passed

if __name__ == "__main__":
    tests = [
        ("2*-3", -6),
        ("-5", -5),
        ("+5", 5),
        ("--5", 5),
        ("-+5", -5),
        ("3*-2+1", -5),
        ("2*(-3+4)", 2),
        ("(-5)*(-2)", 10),
        ("3+-4", -1),
        ("3--4", 7),
        ("10/2*3", 15),
        ("8-3-2", 3),
        ("8/2/2", 2),
        ("3+4*2", 11),
        ("(3+4)*2", 14),
    ]
    
    passed_count = 0
    for expr, expected in tests:
        if test_expression(expr, expected):
            passed_count += 1
    
    print(f"\n{'='*60}")
    print(f"总计: {passed_count}/{len(tests)} 测试通过")
