import py_compile
import sys

try:
    py_compile.compile('gic_calculator.py', doraise=True)
    print("语法检查通过！代码没有语法错误。")
    sys.exit(0)
except py_compile.PyCompileError as e:
    print(f"语法错误: {e}")
    sys.exit(1)
except Exception as e:
    print(f"检查时出错: {e}")
    sys.exit(1)
