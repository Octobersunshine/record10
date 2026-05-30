from qrcode_tool import generate_qr_base64, generate_qr_file, decode_qr_image

print("测试1: 生成二维码Base64...")
test_data = "https://www.example.com"
base64_result = generate_qr_base64(test_data)
print(f"✓ Base64生成成功，长度: {len(base64_result)} 字符")
print(f"  前缀: {base64_result[:50]}...")

print("\n测试2: 生成二维码图片文件...")
generate_qr_file(test_data, "test_qr.png")
print("✓ 二维码文件已保存: test_qr.png")

print("\n测试3: 解码二维码图片...")
decoded_text = decode_qr_image("test_qr.png")
print(f"✓ 解码成功: {decoded_text}")

if decoded_text == test_data:
    print("\n✅ 所有测试通过！编码解码一致")
else:
    print("\n❌ 测试失败：编码解码不一致")
