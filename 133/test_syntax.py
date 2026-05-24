"""
语法测试脚本
验证novikov_thorne_disk.py的语法正确性
"""
import ast
import sys

def check_syntax(filename):
    """检查Python文件语法"""
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            source = f.read()
        ast.parse(source)
        print(f"✓ {filename} 语法正确")
        return True
    except SyntaxError as e:
        print(f"✗ {filename} 语法错误:")
        print(f"  行 {e.lineno}: {e.msg}")
        print(f"  {e.text}")
        return False

if __name__ == '__main__':
    print("=" * 50)
    print("语法检查")
    print("=" * 50)
    
    success = check_syntax('novikov_thorne_disk.py')
    
    print("=" * 50)
    if success:
        print("所有检查通过!")
        sys.exit(0)
    else:
        print("发现错误!")
        sys.exit(1)
