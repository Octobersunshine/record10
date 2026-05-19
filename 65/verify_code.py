import ast
import sys

def verify_python_file(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            code = f.read()
        
        tree = ast.parse(code)
        print(f"✓ {filepath} 语法正确")
        
        classes = [node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
        functions = [node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
        
        print(f"\n类定义: {classes}")
        print(f"函数定义: {functions}")
        
        return True
    except SyntaxError as e:
        print(f"✗ {filepath} 语法错误: {e}")
        return False
    except Exception as e:
        print(f"✗ {filepath} 错误: {e}")
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("Python代码语法验证")
    print("=" * 60)
    
    result = verify_python_file("gene_regulation_ssa.py")
    
    print("\n" + "=" * 60)
    if result:
        print("验证通过！代码结构完整。")
    else:
        print("验证失败！请检查代码。")
    print("=" * 60)
