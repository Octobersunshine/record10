import qrcode
import base64
import io
from PIL import Image
import os

def generate_qr_base64(data: str, size: int = 300) -> str:
    qr = qrcode.QRCode(
        version=1,
        error_correction=qrcode.constants.ERROR_CORRECT_L,
        box_size=10,
        border=4,
    )
    qr.add_data(data)
    qr.make(fit=True)
    
    img = qr.make_image(fill_color="black", back_color="white")
    img = img.resize((size, size), Image.LANCZOS)
    
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    img_bytes = buffer.getvalue()
    
    base64_str = base64.b64encode(img_bytes).decode("utf-8")
    return f"data:image/png;base64,{base64_str}"

print("测试: 生成二维码Base64...")
test_data = "https://www.example.com"
base64_result = generate_qr_base64(test_data)
print(f"✓ Base64生成成功，长度: {len(base64_result)} 字符")
print(f"  前缀: {base64_result[:80]}...")

def generate_qr_file(data: str, output_path: str, size: int = 300) -> None:
    qr = qrcode.QRCode(
        version=1,
        error_correction=qrcode.constants.ERROR_CORRECT_L,
        box_size=10,
        border=4,
    )
    qr.add_data(data)
    qr.make(fit=True)
    
    img = qr.make_image(fill_color="black", back_color="white")
    img = img.resize((size, size), Image.LANCZOS)
    img.save(output_path)

generate_qr_file(test_data, "test_qr.png")
print(f"✓ 二维码文件已保存: {os.path.abspath('test_qr.png')}")
print("\n✅ 二维码生成功能测试通过！")
