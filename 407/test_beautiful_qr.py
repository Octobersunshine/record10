from qrcode_tool import (
    generate_beautiful_qr_base64,
    generate_beautiful_qr_file,
    decode_qr_image,
    ERROR_CORRECTION_LEVELS
)

print("=" * 60)
print("测试美化二维码功能")
print("=" * 60)

test_data = "https://www.example.com"

print("\n测试1: 自定义颜色二维码")
try:
    generate_beautiful_qr_file(
        data=test_data,
        output_path="color_qr.png",
        fill_color="#1E88E5",
        back_color="#F5F5F5",
        error_correction='H'
    )
    print("  ✓ 蓝色二维码生成成功: color_qr.png")
except Exception as e:
    print(f"  ✗ 失败: {e}")

print("\n测试2: 圆角二维码")
try:
    generate_beautiful_qr_file(
        data=test_data,
        output_path="rounded_qr.png",
        fill_color="#43A047",
        back_color="#E8F5E9",
        rounded_corners=True,
        corner_radius=30
    )
    print("  ✓ 圆角绿色二维码生成成功: rounded_qr.png")
except Exception as e:
    print(f"  ✗ 失败: {e}")

print("\n测试3: 不同容错级别测试")
for level in ['L', 'M', 'Q', 'H']:
    try:
        base64_result = generate_beautiful_qr_base64(
            data=test_data,
            error_correction=level
        )
        print(f"  ✓ 容错级别 {level} 生成成功, Base64长度: {len(base64_result)}")
    except Exception as e:
        print(f"  ✗ 容错级别 {level} 失败: {e}")

print("\n测试4: 美化二维码Base64生成")
try:
    base64_result = generate_beautiful_qr_base64(
        data=test_data,
        fill_color="#E53935",
        back_color="#FFEBEE",
        error_correction='Q'
    )
    print(f"  ✓ Base64生成成功, 长度: {len(base64_result)}")
    print(f"    前缀: {base64_result[:80]}...")
except Exception as e:
    print(f"  ✗ 失败: {e}")

print("\n测试5: 美化后二维码解码验证")
try:
    decoded = decode_qr_image("color_qr.png")
    if decoded == test_data:
        print("  ✓ 彩色二维码解码验证通过")
    else:
        print(f"  ✗ 解码内容不匹配: {decoded}")
except Exception as e:
    print(f"  ✗ 解码失败: {e}")

print("\n测试6: 圆角二维码解码验证")
try:
    decoded = decode_qr_image("rounded_qr.png")
    if decoded == test_data:
        print("  ✓ 圆角二维码解码验证通过")
    else:
        print(f"  ✗ 解码内容不匹配: {decoded}")
except Exception as e:
    print(f"  ✗ 解码失败: {e}")

print("\n" + "=" * 60)
print("✅ 美化二维码功能测试完成！")
print("=" * 60)
print("\n提示: 如需测试带Logo的二维码，请准备一张Logo图片后使用:")
print("  generate_beautiful_qr_file('内容', 'output.png', logo_path='logo.png')")
