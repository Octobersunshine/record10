import qrcode
import base64
import io
from PIL import Image, ImageDraw
import cv2
import numpy as np
import os

MAX_QR_LENGTH = 3000

ERROR_CORRECTION_LEVELS = {
    'L': qrcode.constants.ERROR_CORRECT_L,
    'M': qrcode.constants.ERROR_CORRECT_M,
    'Q': qrcode.constants.ERROR_CORRECT_Q,
    'H': qrcode.constants.ERROR_CORRECT_H,
}


def check_qr_length(data: str) -> dict:
    data_length = len(data)
    is_valid = data_length <= MAX_QR_LENGTH
    
    result = {
        "valid": is_valid,
        "length": data_length,
        "max_length": MAX_QR_LENGTH,
        "message": ""
    }
    
    if not is_valid:
        excess = data_length - MAX_QR_LENGTH
        result["message"] = (
            f"二维码内容过长！当前长度: {data_length} 字符，"
            f"最大支持: {MAX_QR_LENGTH} 字符，超出: {excess} 字符。"
            f"建议精简文本内容或使用短链接。"
        )
    elif data_length > MAX_QR_LENGTH * 0.8:
        result["message"] = (
            f"警告：内容长度接近上限 ({data_length}/{MAX_QR_LENGTH} 字符)，"
            f"建议精简以确保扫码成功率。"
        )
    
    return result


def generate_qr_base64(data: str, size: int = 300, auto_fit: bool = True) -> str:
    check_result = check_qr_length(data)
    if not check_result["valid"]:
        raise ValueError(check_result["message"])
    
    if auto_fit:
        qr = qrcode.QRCode(
            version=None,
            error_correction=qrcode.constants.ERROR_CORRECT_M,
            box_size=10,
            border=4,
        )
    else:
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


def generate_qr_file(data: str, output_path: str, size: int = 300, auto_fit: bool = True) -> None:
    check_result = check_qr_length(data)
    if not check_result["valid"]:
        raise ValueError(check_result["message"])
    
    if auto_fit:
        qr = qrcode.QRCode(
            version=None,
            error_correction=qrcode.constants.ERROR_CORRECT_M,
            box_size=10,
            border=4,
        )
    else:
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


def decode_qr_image(image_path: str) -> str:
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"文件不存在: {image_path}")
    
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"无法读取图片文件: {image_path}")
    
    detector = cv2.QRCodeDetector()
    data, _, _ = detector.detectAndDecode(img)
    
    if not data:
        raise ValueError("未能识别二维码内容")
    
    return data


def decode_qr_base64(base64_data: str) -> str:
    if base64_data.startswith("data:image"):
        base64_data = base64_data.split(",")[1]
    
    img_bytes = base64.b64decode(base64_data)
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if img is None:
        raise ValueError("无法解码Base64图片数据")
    
    detector = cv2.QRCodeDetector()
    data, _, _ = detector.detectAndDecode(img)
    
    if not data:
        raise ValueError("未能识别二维码内容")
    
    return data


def _add_logo_to_qr(qr_img: Image.Image, logo_path: str, logo_size_ratio: float = 0.2) -> Image.Image:
    if not os.path.exists(logo_path):
        raise FileNotFoundError(f"Logo文件不存在: {logo_path}")
    
    qr_width, qr_height = qr_img.size
    logo_size = int(min(qr_width, qr_height) * logo_size_ratio)
    
    logo = Image.open(logo_path).convert("RGBA")
    logo = logo.resize((logo_size, logo_size), Image.LANCZOS)
    
    logo_bg = Image.new('RGBA', (logo_size + 10, logo_size + 10), 'white')
    bg_width, bg_height = logo_bg.size
    logo_pos = ((bg_width - logo_size) // 2, (bg_height - logo_size) // 2)
    logo_bg.paste(logo, logo_pos, logo)
    
    pos = ((qr_width - bg_width) // 2, (qr_height - bg_height) // 2)
    qr_img = qr_img.convert("RGBA")
    qr_img.paste(logo_bg, pos, logo_bg)
    
    return qr_img.convert("RGB")


def _add_rounded_corners(img: Image.Image, radius: int = 20) -> Image.Image:
    mask = Image.new('L', img.size, 0)
    draw = ImageDraw.Draw(mask)
    width, height = img.size
    draw.rounded_rectangle([(0, 0), (width, height)], radius=radius, fill=255)
    
    result = Image.new('RGBA', img.size)
    result.paste(img, (0, 0), mask)
    return result


def generate_beautiful_qr_base64(
    data: str,
    size: int = 400,
    fill_color: str = "#000000",
    back_color: str = "#FFFFFF",
    error_correction: str = 'H',
    logo_path: str = None,
    logo_size_ratio: float = 0.2,
    rounded_corners: bool = False,
    corner_radius: int = 20
) -> str:
    check_result = check_qr_length(data)
    if not check_result["valid"]:
        raise ValueError(check_result["message"])
    
    error_level = ERROR_CORRECTION_LEVELS.get(error_correction.upper(), qrcode.constants.ERROR_CORRECT_H)
    
    qr = qrcode.QRCode(
        version=None,
        error_correction=error_level,
        box_size=10,
        border=4,
    )
    qr.add_data(data)
    qr.make(fit=True)
    
    img = qr.make_image(fill_color=fill_color, back_color=back_color)
    img = img.resize((size, size), Image.LANCZOS)
    
    if logo_path:
        img = _add_logo_to_qr(img, logo_path, logo_size_ratio)
    
    if rounded_corners:
        img = _add_rounded_corners(img, corner_radius)
    
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    img_bytes = buffer.getvalue()
    
    base64_str = base64.b64encode(img_bytes).decode("utf-8")
    return f"data:image/png;base64,{base64_str}"


def generate_beautiful_qr_file(
    data: str,
    output_path: str,
    size: int = 400,
    fill_color: str = "#000000",
    back_color: str = "#FFFFFF",
    error_correction: str = 'H',
    logo_path: str = None,
    logo_size_ratio: float = 0.2,
    rounded_corners: bool = False,
    corner_radius: int = 20
) -> None:
    check_result = check_qr_length(data)
    if not check_result["valid"]:
        raise ValueError(check_result["message"])
    
    error_level = ERROR_CORRECTION_LEVELS.get(error_correction.upper(), qrcode.constants.ERROR_CORRECT_H)
    
    qr = qrcode.QRCode(
        version=None,
        error_correction=error_level,
        box_size=10,
        border=4,
    )
    qr.add_data(data)
    qr.make(fit=True)
    
    img = qr.make_image(fill_color=fill_color, back_color=back_color)
    img = img.resize((size, size), Image.LANCZOS)
    
    if logo_path:
        img = _add_logo_to_qr(img, logo_path, logo_size_ratio)
    
    if rounded_corners:
        img = _add_rounded_corners(img, corner_radius)
    
    img.save(output_path)


if __name__ == "__main__":
    import sys
    
    print("=" * 60)
    print("二维码工具 - 支持生成(Base64)/解码/美化/Logo")
    print("=" * 60)
    
    while True:
        print("\n请选择操作:")
        print("1. 生成普通二维码 (返回Base64)")
        print("2. 生成普通二维码 (保存为文件)")
        print("3. 生成美化二维码 (支持Logo/自定义颜色)")
        print("4. 解码二维码图片文件")
        print("5. 退出")
        
        choice = input("\n请输入选项 (1-5): ").strip()
        
        if choice == "1":
            text = input("请输入文本或URL: ").strip()
            if text:
                check_result = check_qr_length(text)
                print(f"\n内容长度: {check_result['length']} 字符 (最大 {MAX_QR_LENGTH})")
                if check_result["message"]:
                    print(check_result["message"])
                
                if check_result["valid"]:
                    try:
                        base64_result = generate_qr_base64(text)
                        print(f"\n生成成功! Base64长度: {len(base64_result)} 字符")
                        print(f"\nBase64内容 (前100字符):")
                        print(base64_result[:100] + "...")
                        save = input("\n是否保存Base64到文件? (y/n): ").strip().lower()
                        if save == "y":
                            with open("qr_base64.txt", "w", encoding="utf-8") as f:
                                f.write(base64_result)
                            print("已保存到 qr_base64.txt")
                    except Exception as e:
                        print(f"生成失败: {e}")
            else:
                print("内容不能为空!")
        
        elif choice == "2":
            text = input("请输入文本或URL: ").strip()
            filename = input("请输入输出文件名 (如 qr.png): ").strip()
            if text and filename:
                check_result = check_qr_length(text)
                print(f"\n内容长度: {check_result['length']} 字符 (最大 {MAX_QR_LENGTH})")
                if check_result["message"]:
                    print(check_result["message"])
                
                if check_result["valid"]:
                    try:
                        generate_qr_file(text, filename)
                        print(f"二维码已保存到: {os.path.abspath(filename)}")
                    except Exception as e:
                        print(f"生成失败: {e}")
            else:
                print("内容和文件名不能为空!")
        
        elif choice == "3":
            text = input("请输入文本或URL: ").strip()
            if not text:
                print("内容不能为空!")
                continue
            
            check_result = check_qr_length(text)
            print(f"\n内容长度: {check_result['length']} 字符 (最大 {MAX_QR_LENGTH})")
            if check_result["message"]:
                print(check_result["message"])
            
            if not check_result["valid"]:
                continue
            
            print("\n美化设置 (直接回车使用默认值):")
            fill_color = input("二维码颜色 (默认 #000000): ").strip() or "#000000"
            back_color = input("背景颜色 (默认 #FFFFFF): ").strip() or "#FFFFFF"
            error_correction = input("容错级别 L/M/Q/H (默认 H): ").strip().upper() or "H"
            logo_path = input("Logo图片路径 (可选，回车跳过): ").strip() or None
            rounded = input("是否圆角? (y/n, 默认 n): ").strip().lower() == 'y'
            filename = input("输出文件名 (默认 beautiful_qr.png): ").strip() or "beautiful_qr.png"
            
            try:
                generate_beautiful_qr_file(
                    data=text,
                    output_path=filename,
                    fill_color=fill_color,
                    back_color=back_color,
                    error_correction=error_correction,
                    logo_path=logo_path,
                    rounded_corners=rounded
                )
                print(f"\n美化二维码已保存到: {os.path.abspath(filename)}")
                print(f"配置: 颜色={fill_color}/{back_color}, 容错={error_correction}, "
                      f"Logo={'有' if logo_path else '无'}, 圆角={'是' if rounded else '否'}")
            except Exception as e:
                print(f"生成失败: {e}")
        
        elif choice == "4":
            filepath = input("请输入二维码图片路径: ").strip()
            if filepath:
                try:
                    result = decode_qr_image(filepath)
                    print(f"\n解码成功! 内容:")
                    print("-" * 50)
                    print(result)
                    print("-" * 50)
                except Exception as e:
                    print(f"解码失败: {e}")
            else:
                print("文件路径不能为空!")
        
        elif choice == "5":
            print("再见!")
            sys.exit(0)
        
        else:
            print("无效选项，请重新输入!")
