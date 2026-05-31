from image_base64_converter import (
    image_to_base64,
    image_to_base64_with_data_uri,
    base64_to_image,
    get_image_from_base64
)

def demo_image_to_base64():
    print("=" * 50)
    print("示例1: 图片转Base64")
    print("=" * 50)
    
    try:
        base64_str = image_to_base64("test_image.png", "output_base64.txt")
        print(f"Base64字符串长度: {len(base64_str)} 字符")
    except FileNotFoundError:
        print("提示: 请确保存在 test_image.png 文件，或修改为实际图片路径")

def demo_image_to_data_uri():
    print("\n" + "=" * 50)
    print("示例2: 图片转Data URI (可直接用于HTML)")
    print("=" * 50)
    
    try:
        data_uri = image_to_base64_with_data_uri("test_image.png")
        print(f"Data URI前100字符: {data_uri[:100]}...")
    except FileNotFoundError:
        print("提示: 请确保存在 test_image.png 文件")

def demo_base64_to_image():
    print("\n" + "=" * 50)
    print("示例3: Base64转图片并保存")
    print("=" * 50)
    
    try:
        with open("output_base64.txt", "r") as f:
            base64_str = f.read()
        base64_to_image(base64_str, "restored_image.png")
        print("图片已恢复为 restored_image.png")
    except FileNotFoundError:
        print("提示: 请先运行示例1生成Base64文件")

def demo_get_pil_image():
    print("\n" + "=" * 50)
    print("示例4: Base64转PIL Image对象 (内存中使用)")
    print("=" * 50)
    
    try:
        with open("output_base64.txt", "r") as f:
            base64_str = f.read()
        img = get_image_from_base64(base64_str)
        print(f"图片尺寸: {img.size}")
        print(f"图片模式: {img.mode}")
    except FileNotFoundError:
        print("提示: 请先运行示例1生成Base64文件")

if __name__ == "__main__":
    print("图片与Base64双向转换 - 使用示例")
    print("支持格式: JPG, PNG, GIF, BMP, WebP, TIFF")
    
    demo_image_to_base64()
    demo_image_to_data_uri()
    demo_base64_to_image()
    demo_get_pil_image()
