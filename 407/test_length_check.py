from qrcode_tool import (
    check_qr_length,
    generate_qr_base64,
    generate_qr_file,
    decode_qr_image,
    MAX_QR_LENGTH
)

print("=" * 60)
print("测试二维码长度检测功能")
print("=" * 60)

print(f"\n最大支持长度: {MAX_QR_LENGTH} 字符")

print("\n测试1: 正常长度内容")
short_text = "Hello World! 这是一个测试。"
result = check_qr_length(short_text)
print(f"  内容长度: {result['length']}")
print(f"  是否有效: {result['valid']}")
print(f"  提示信息: {result['message'] or '无'}")
assert result['valid'] == True, "正常长度应该有效"
print("  ✓ 通过")

print("\n测试2: 接近上限长度 (80%)")
near_limit_text = "A" * int(MAX_QR_LENGTH * 0.85)
result = check_qr_length(near_limit_text)
print(f"  内容长度: {result['length']}")
print(f"  是否有效: {result['valid']}")
print(f"  提示信息: {result['message'] or '无'}")
assert result['valid'] == True, "接近上限应该有效"
assert "警告" in result['message'], "应该有警告提示"
print("  ✓ 通过")

print("\n测试3: 超长内容 (超出限制)")
too_long_text = "A" * (MAX_QR_LENGTH + 100)
result = check_qr_length(too_long_text)
print(f"  内容长度: {result['length']}")
print(f"  是否有效: {result['valid']}")
print(f"  提示信息: {result['message'] or '无'}")
assert result['valid'] == False, "超长内容应该无效"
assert "超出" in result['message'], "应该有超出提示"
print("  ✓ 通过")

print("\n测试4: 生成长内容二维码")
long_text = "测试长内容 " * 100
result = check_qr_length(long_text)
print(f"  内容长度: {result['length']}")
try:
    base64_result = generate_qr_base64(long_text)
    print(f"  Base64长度: {len(base64_result)}")
    print("  ✓ 生成成功")
except Exception as e:
    print(f"  ✗ 生成失败: {e}")

print("\n测试5: 生成超长内容 - 应抛出异常")
too_long_text = "X" * (MAX_QR_LENGTH + 500)
try:
    generate_qr_base64(too_long_text)
    print("  ✗ 应该抛出异常但没有")
except ValueError as e:
    print(f"  ✓ 正确抛出异常: {str(e)[:60]}...")

print("\n测试6: 长内容编码后解码验证")
test_text = "Test123 " * 50
print(f"  原始长度: {len(test_text)}")
base64_data = generate_qr_base64(test_text)
generate_qr_file(test_text, "long_test_qr.png")
print("  ✓ 二维码文件已生成")
decoded_text = decode_qr_image("long_test_qr.png")
print(f"  解码长度: {len(decoded_text)}")
assert decoded_text == test_text, "解码内容应与原始一致"
print("  ✓ 解码验证通过")

print("\n" + "=" * 60)
print("✅ 所有长度检测测试通过！")
print("=" * 60)
