import sys

def verify_syntax():
    print("正在验证代码语法...")
    try:
        with open('shooting_method.py', 'r', encoding='utf-8') as f:
            code = f.read()
        compile(code, 'shooting_method.py', 'exec')
        print("✓ 语法检查通过！")
        return True
    except SyntaxError as e:
        print(f"✗ 语法错误: {e}")
        return False
    except Exception as e:
        print(f"✗ 错误: {e}")
        return False

def check_imports():
    print("\n正在检查依赖包...")
    packages = ['numpy', 'scipy', 'matplotlib']
    for pkg in packages:
        try:
            __import__(pkg)
            print(f"✓ {pkg} 已安装")
        except ImportError:
            print(f"✗ {pkg} 未安装，请运行: pip install {pkg}")

if __name__ == "__main__":
    print("=" * 50)
    print("打靶法代码验证")
    print("=" * 50)
    syntax_ok = verify_syntax()
    check_imports()
    print("\n" + "=" * 50)
    if syntax_ok:
        print("代码验证完成！可以运行: python shooting_method.py")
    else:
        print("请修复上述错误后再运行")
    print("=" * 50)
