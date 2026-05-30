from expression_evaluator import evaluate_expression, evaluate_direct
import math

def test(expr, expected, variables=None):
    variables = variables or {}
    result_postfix, steps = evaluate_expression(expr, variables)
    result_direct = evaluate_direct(expr, variables)
    pf_ok = abs(result_postfix - expected) < 1e-6
    dt_ok = abs(result_direct - expected) < 1e-6
    match_ok = abs(result_postfix - result_direct) < 1e-6
    status = "✓" if (pf_ok and dt_ok and match_ok) else "✗"
    print(f"{status} {expr}  预期={expected}  后缀={result_postfix:.6f}  直接={result_direct:.6f}  一致={match_ok}")
    if not (pf_ok and dt_ok and match_ok):
        for s in steps:
            print(f"    {s}")
    return pf_ok and dt_ok and match_ok

if __name__ == "__main__":
    passed = 0
    total = 0

    print("=== 四则运算 ===")
    cases = [
        ("3+4*2", 11),
        ("(3+4)*2", 14),
        ("10/2+3", 8),
        ("8-3-2", 3),
        ("2*-3", -6),
        ("-5+3", -2),
        ("3+-4", -1),
        ("3--4", 7),
        ("--5", 5),
        ("-+5", -5),
        ("(-5)*(-2)", 10),
    ]
    for expr, expected in cases:
        total += 1
        if test(expr, expected):
            passed += 1

    print("\n=== 函数 ===")
    cases = [
        ("sin(0)", 0),
        ("cos(0)", 1),
        ("sqrt(16)", 4),
        ("log(1)", 0),
        ("sqrt(16)+cos(0)", 5),
        ("2*sin(1.5708)", 2.0),
        ("sin(0)+cos(0)*sqrt(4)", 2.0),
        ("-sqrt(9)", -3.0),
        ("sqrt(4)*sqrt(9)", 6.0),
    ]
    for expr, expected in cases:
        total += 1
        if test(expr, expected):
            passed += 1

    print("\n=== 变量 ===")
    cases = [
        ("x+y", 7, {"x": 3, "y": 4}),
        ("x*y+1", 7, {"x": 2, "y": 3}),
        ("sqrt(x)", 3, {"x": 9}),
        ("a*sin(b)", 2.0, {"a": 2, "b": math.pi / 2}),
        ("x*x+y*y", 25, {"x": 3, "y": 4}),
        ("sqrt(x)+y*z", 7, {"x": 9, "y": 2, "z": 2}),
    ]
    for expr, expected, var in cases:
        total += 1
        if test(expr, expected, var):
            passed += 1

    print(f"\n总计: {passed}/{total} 通过")
