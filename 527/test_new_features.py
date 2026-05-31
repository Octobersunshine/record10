import os
import json
from image_base64_converter import (
    get_image_info_from_base64,
    image_to_base64_compressed,
    batch_image_to_base64,
    batch_base64_to_image,
    image_to_base64,
    base64_to_image
)

def create_test_images():
    print("创建测试图片...")
    from PIL import Image
    
    test_dir = "test_images"
    os.makedirs(test_dir, exist_ok=True)
    
    colors = [
        ((255, 0, 0), "red"),
        ((0, 255, 0), "green"),
        ((0, 0, 255), "blue"),
    ]
    
    for i, (color, name) in enumerate(colors, 1):
        img = Image.new('RGB', (800, 600), color)
        path = os.path.join(test_dir, f"{name}_{i}.jpg")
        img.save(path, quality=95)
        print(f"  创建: {path} ({os.path.getsize(path)//1024} KB)")
    
    img_png = Image.new('RGBA', (400, 300), (255, 0, 0, 128))
    png_path = os.path.join(test_dir, "transparent.png")
    img_png.save(png_path)
    print(f"  创建: {png_path} ({os.path.getsize(png_path)//1024} KB)")
    
    return test_dir

def test_get_image_info():
    print("\n" + "=" * 60)
    print("测试1: 提取Base64图片信息")
    print("=" * 60)
    
    test_base64 = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
    
    try:
        info = get_image_info_from_base64(test_base64)
        print("\n提取到的图片信息:")
        print("-" * 40)
        for key, value in info.items():
            print(f"  {key:25s}: {value}")
        print("-" * 40)
        print("✓ 信息提取成功")
    except Exception as e:
        print(f"✗ 失败: {e}")

def test_compressed_conversion():
    print("\n" + "=" * 60)
    print("测试2: 压缩后Base64转换（质量可调）")
    print("=" * 60)
    
    test_image = "test_images/red_1.jpg"
    
    if not os.path.exists(test_image):
        print("跳过测试：测试图片不存在")
        return
    
    qualities = [90, 70, 50, 30]
    
    print(f"\n原始图片: {test_image}")
    print(f"原始大小: {os.path.getsize(test_image)//1024} KB")
    print("-" * 60)
    print(f"{'质量':<8} {'压缩后大小':<12} {'压缩率':<10} {'Base64长度':<12}")
    print("-" * 60)
    
    for quality in qualities:
        try:
            result = image_to_base64_compressed(
                test_image, 
                quality=quality,
                output_file=f"test_compressed_q{quality}.txt"
            )
            print(f"{quality:<8} {result['compressed_size_kb']:<12.2f} KB "
                  f"{result['compression_ratio_percent']:<10.2f}% "
                  f"{result['base64_length']:<12}")
            print(f"  ✓ 已保存到 test_compressed_q{quality}.txt")
        except Exception as e:
            print(f"{quality:<8} ✗ 失败: {e}")
    
    print("-" * 60)

def test_batch_encode():
    print("\n" + "=" * 60)
    print("测试3: 批量图片转Base64")
    print("=" * 60)
    
    if not os.path.exists("test_images"):
        print("跳过测试：测试图片目录不存在")
        return
    
    image_files = [
        os.path.join("test_images", f) 
        for f in os.listdir("test_images")
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ]
    
    if not image_files:
        print("跳过测试：未找到测试图片")
        return
    
    print(f"\n将批量转换 {len(image_files)} 张图片...")
    
    try:
        summary = batch_image_to_base64(
            image_files,
            output_dir="batch_output",
            data_uri=True,
            quality=80
        )
        
        print(f"\n批量转换结果:")
        print(f"  总计: {summary['total']}")
        print(f"  成功: {summary['success']}")
        print(f"  失败: {summary['failed']}")
        
        for i, result in enumerate(summary['results'], 1):
            status = "✓" if result.get('success') else "✗"
            note = f" ({result.get('note', '')})" if result.get('note') else ""
            compressed_note = f" [压缩率 {result.get('compression_ratio_percent', 0):.1f}%]" if result.get('compressed') else " [未压缩]"
            print(f"  {status} [{i}] {os.path.basename(result['source'])}"
                  f"{compressed_note}{note}")
        
        if os.path.exists("batch_output/batch_summary.json"):
            with open("batch_output/batch_summary.json", 'r', encoding='utf-8') as f:
                data = json.load(f)
            print(f"\n✓ 汇总信息已保存: batch_output/batch_summary.json")
        
    except Exception as e:
        print(f"✗ 失败: {e}")

def test_batch_decode():
    print("\n" + "=" * 60)
    print("测试4: 批量Base64转图片")
    print("=" * 60)
    
    if not os.path.exists("batch_output"):
        print("跳过测试：批量输出目录不存在")
        return
    
    base64_files = [
        os.path.join("batch_output", f)
        for f in os.listdir("batch_output")
        if f.endswith('.txt') and not f.startswith('batch_')
    ]
    
    if not base64_files:
        print("跳过测试：未找到Base64文件")
        return
    
    base64_items = []
    for file_path in base64_files:
        with open(file_path, 'r', encoding='utf-8') as f:
            base64_str = f.read().strip()
        name = os.path.splitext(os.path.basename(file_path))[0].replace('_data_uri', '')
        base64_items.append({"base64": base64_str, "name": name})
    
    print(f"\n将批量解码 {len(base64_items)} 个Base64文件...")
    
    try:
        summary = batch_base64_to_image(
            base64_items,
            output_dir="batch_decoded"
        )
        
        print(f"\n批量解码结果:")
        print(f"  总计: {summary['total']}")
        print(f"  成功: {summary['success']}")
        print(f"  失败: {summary['failed']}")
        
        for result in summary['results']:
            print(f"  ✓ {result['name']}.{result['format'].lower()} "
                  f"({result['dimensions']})")
        
    except Exception as e:
        print(f"✗ 失败: {e}")

def test_invalid_compression_format():
    print("\n" + "=" * 60)
    print("测试5: 不支持压缩格式的错误提示")
    print("=" * 60)
    
    test_png = "test_images/transparent.png"
    
    if not os.path.exists(test_png):
        print("跳过测试：测试图片不存在")
        return
    
    try:
        image_to_base64_compressed(test_png, quality=80)
        print("✗ 应该抛出错误但没有")
    except ValueError as e:
        print(f"✓ 正确抛出错误:")
        print(f"  {e}")

def cleanup_test_files():
    print("\n" + "=" * 60)
    print("清理测试文件")
    print("=" * 60)
    
    files_to_remove = [
        "test_compressed_q90.txt",
        "test_compressed_q70.txt",
        "test_compressed_q50.txt",
        "test_compressed_q30.txt",
    ]
    
    for f in files_to_remove:
        if os.path.exists(f):
            os.remove(f)
            print(f"  已删除: {f}")
    
    print("✓ 清理完成（保留测试图片和批量输出目录供参考）")

if __name__ == "__main__":
    print("新功能测试套件")
    print("=" * 60)
    
    test_dir = create_test_images()
    
    test_get_image_info()
    test_compressed_conversion()
    test_batch_encode()
    test_batch_decode()
    test_invalid_compression_format()
    cleanup_test_files()
    
    print("\n" + "=" * 60)
    print("所有测试完成！")
    print("=" * 60)
