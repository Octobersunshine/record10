from image_base64_converter import (
    strip_data_url_prefix,
    validate_base64_format,
    base64_to_image,
    image_to_base64_with_data_uri
)

def test_data_url_prefix():
    print("=" * 60)
    print("测试1: Data URL前缀识别与去除")
    print("=" * 60)
    
    test_cases = [
        ("data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==", "标准PNG Data URL"),
        ("data:image/jpeg;base64,/9j/4AAQSkZJRgABAQEAYABgAAD/2wBDAAgGBgcGBQgHBwcJCQgKDBQNDAsLDBkSEw8UHRofHh0aHBwgJC4nICIsIxwcKDcpLDAxNDQ0Hyc5PTgyPC4zNDL/2wBDAQkJCQwLDBgNDRgyIRwhMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjL/wAARCAABAAEDASIAAhEBAxEB/8QAHwAAAQUBAQEBAQEAAAAAAAAAAAECAwQFBgcICQoL/8QAtRAAAgEDAwIEAwUFBAQAAAF9AQIDAAQRBRIhMUEGE1FhByJxFDKBkaEII0KxwRVS0fAkM2JyggkKFhcYGRolJicoKSo0NTY3ODk6Q0RFRkdISUpTVFVWV1hZWmNkZWZnaGlqc3R1dnd4eXqDhIWGh4iJipKTlJWWl5iZmqKjpKWmp6ipqrKztLW2t7i5usLDxMXGx8jJytLT1NXW19jZ2uHi4+Tl5ufo6erx8vP09fb3+Pn6/8QAHwEAAwEBAQEBAQEBAQAAAAAAAAECAwQFBgcICQoL/8QAtREAAgECBAQDBAcFBAQAAQJ3AAECAxEEBSExBhJBUQdhcRMiMoEIFEKRobHBCSMzUvAVYnLRChYkNOEl8RcYGRomJygpKjU2Nzg5OkNERUZHSElKU1RVVldYWVpjZGVmZ2hpanN0dXZ3eHl6goOEhYaHiImKkpOUlZaXmJmaoqOkpaanqKmqsrO0tba3uLm6wsPExcbHyMnK0tPU1dbX2Nna4uPk5ebn6Onq8vP09fb3+Pn6/9oADAMBAAIRAxEAPwD3+iiigD//2Q==", "标准JPEG Data URL"),
        ("data:image/gif;base64,R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7", "标准GIF Data URL"),
        ("DATA:IMAGE/PNG;BASE64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==", "大写前缀"),
        ("data:image/webp;base64,UklGRkoAAABXRUJQVlA4WAoAAAAQAAAAAAAAAAAAQUxQSAwAAAABBxAR/Q9ERP8DAABWUDggGAAAABQBAJ0BKgEAAQAAAP4AAA3AAP7mtQAAAA==", "WebP Data URL"),
        ("   data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==   ", "带前后空格"),
        ("data:image/png;name=test.png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==", "带额外参数"),
        ("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==", "纯Base64（无前缀）"),
    ]
    
    for i, (test_str, description) in enumerate(test_cases, 1):
        try:
            cleaned, had_prefix = strip_data_url_prefix(test_str)
            status = "✓ 前缀已去除" if had_prefix else "✓ 无前缀"
            print(f"\n测试{i}: {description}")
            print(f"  {status}")
            print(f"  处理后长度: {len(cleaned)} 字符")
            if len(cleaned) > 30:
                print(f"  内容预览: {cleaned[:30]}...")
        except Exception as e:
            print(f"\n测试{i}: {description}")
            print(f"  ✗ 错误: {e}")

def test_base64_validation():
    print("\n" + "=" * 60)
    print("测试2: Base64格式验证与错误提示")
    print("=" * 60)
    
    test_cases = [
        ("", "空字符串"),
        ("abc", "过短字符串"),
        ("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg", "缺少填充字符"),
        ("iVBORw0KGgo!@#$%^&*AAAAA", "包含特殊字符"),
        ("iVBORw0KGgo\nAAAANSUhEUg\nAAAAEAAAABCAY", "包含换行符"),
        ("iVBORw0KGgo AAAANSUhEUg AAAA EAAAABCAY", "包含空格"),
        ("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==", "有效Base64"),
    ]
    
    for i, (test_str, description) in enumerate(test_cases, 1):
        print(f"\n测试{i}: {description}")
        try:
            validate_base64_format(test_str)
            print(f"  ✓ 格式验证通过")
        except ValueError as e:
            print(f"  ✗ 验证失败（预期）")
            print(f"    提示信息: {e}")

def test_roundtrip():
    print("\n" + "=" * 60)
    print("测试3: 转换完整性验证")
    print("=" * 60)
    
    valid_base64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
    data_uri = f"data:image/png;base64,{valid_base64}"
    
    print("\n测试纯Base64转图片:")
    try:
        base64_to_image(valid_base64, "test_output1.png")
        print("  ✓ 成功")
    except Exception as e:
        print(f"  ✗ 失败: {e}")
    
    print("\n测试Data URI转图片:")
    try:
        base64_to_image(data_uri, "test_output2.png")
        print("  ✓ 成功")
    except Exception as e:
        print(f"  ✗ 失败: {e}")

def test_invalid_image_data():
    print("\n" + "=" * 60)
    print("测试4: 无效图片数据的错误提示")
    print("=" * 60)
    
    invalid_cases = [
        ("SGVsbG8gV29ybGQh", "普通文本的Base64", "非图片数据"),
        ("data:image/png;base64,SGVsbG8gV29ybGQh", "带前缀的非图片数据", "Data URL包装的非图片数据"),
    ]
    
    for i, (base64_str, description, case_name) in enumerate(invalid_cases, 1):
        print(f"\n测试{i}: {case_name}")
        try:
            base64_to_image(base64_str, "test_invalid.png")
            print("  ✗ 应该失败但成功了")
        except ValueError as e:
            print(f"  ✓ 正确抛出错误")
            print(f"    错误信息:\n{e}")

if __name__ == "__main__":
    print("Base64兼容性修复 - 测试套件")
    print("=" * 60)
    
    test_data_url_prefix()
    test_base64_validation()
    test_roundtrip()
    test_invalid_image_data()
    
    print("\n" + "=" * 60)
    print("所有测试完成！")
    print("=" * 60)
